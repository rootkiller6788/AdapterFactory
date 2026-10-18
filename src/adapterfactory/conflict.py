"""冲突：**能力空间实测**（路径 B）与**参数空间预测**（路径 A）的对质。

### 为什么必须有两条路

只有实测的话，「``upper`` 与 ``sort`` 互相干扰 0.56」是一个**孤立的事实**：
它说不出干扰从哪来，也说不出换一对 Adapter 会不会还这样。
只有预测的话，预测不被检验就只是一个说法。

两条路做的是两件不同的事：

- **路径 B（实测）**：把两个适配器同时挂上，量它在各自能力上还剩多少。
  它直接就是答案，但**只能在已经训练完的适配器上做** —— 对一份还没训的
  Adapter 组合，它一个字都说不出。
- **路径 A（预测）**：只看两个 delta 张量本身（``cos`` 与符号一致率），
  不需要跑任何前向。它便宜，而且**能在组合之前**给出判断。

两者必须同向。**同向 ⇒ 「冲突」这个说法有来源；不同向 ⇒ 那本身就是最有
信息量的输出**，因为它说明我们对「为什么会互相干扰」的理解是错的。

### 本项目的实测结果：路径 A 被推翻（同向率 1/6）

跑下来 delta 之间几乎正交（``cos`` 在 −0.08…+0.03），符号一致率贴着 0.50
（也就是**随机**），而实测干扰**每一对都接近满值**。于是下面这条约定
（``cos < 0`` ⇒ 预测有害）给出的判断接近抛硬币。

这不是「预测器调参不够」，而是**机制选错了**。实测数据指向另一个机制：
每一个 delta 都是在「只有它自己」的条件下训练出来的，而 stack 把
k 个这样的 delta **原样相加**，于是在共用的挂点上做了 k 倍于训练时见过的
更新。这个机制里**决定干扰的是幅度，不是 delta 之间的夹角** ——
两个完全正交的 delta 相加，照样把权重推出原来那个解的邻域。

所以本项目保留 ``cos``/符号一致率这两列（它们是设计要求，且**被推翻了**
是有价值的结论），同时把 ``expansion``（**相加后的更新幅度相对于单个的倍数**）
作为第二个候选预测器一起报出来，让报告里的相关表替读者分辨
「哪个量真的与干扰有关」。**不预先宣布哪个对** —— 那正是两条路对质的意思。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import compose as P
from . import evaluate as E
from . import net as N
from .schema import ConflictEdge

__all__ = [
    "Predictors",
    "predictors_of",
    "pair_conflict",
    "all_pairs",
    "direction_sign",
    "predictor_table",
]


@dataclass(frozen=True)
class Predictors:
    """一对适配器在**参数空间**上的全部候选预测量。**都不需要跑前向。**"""

    a: str
    b: str
    #: 共同挂点上的余弦相似度。正交或某一侧全零时是 ``None``。
    cos: float | None
    #: 共同挂点上的符号一致率（只看严格同号／严格反号的坐标）。
    sign_rate: float | None
    #: 共同挂点个数。``2`` 与 ``7`` 的可信度不同，所以它必须一起报。
    n_common: int
    #: ``‖Δa + Δb‖ / max(‖Δa‖, ‖Δb‖)``，只在共同挂点上看。
    #: 完全相同的两个 delta ⇒ 2.0；正交同幅 ⇒ √2 ≈ 1.414；恰好相反 ⇒ 0。
    expansion: float | None
    #: ``b`` 更新的基座元素里，有多少比例也被 ``a`` 更新 —— 干扰的「接触面」。
    overlap_ab: float | None
    overlap_ba: float | None
    #: 单侧范数之比 ``‖Δb‖/‖Δa‖``。幅度机制的另一个候选量。
    norm_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "cos": self.cos,
            "sign_rate": self.sign_rate,
            "n_common": self.n_common,
            "expansion": self.expansion,
            "overlap_ab": self.overlap_ab,
            "overlap_ba": self.overlap_ba,
            "norm_ratio": self.norm_ratio,
        }


def _norm_and_overlap(
    da: Mapping[str, list[list[float]]],
    db: Mapping[str, list[list[float]]],
) -> tuple[float, float, float, float, float]:
    """返回 ``(‖A‖², ‖B‖², ‖A+B‖², A∩B 上的‖A‖², A∩B 上的‖B‖²)``。

    全部只在**共同挂点**上累加。后两个量用来算重叠度：
    「A 有多大比例的改动落在 b 也动的地方」—— 它是干扰的**接触面**。
    """
    sa = 0.0
    sb = 0.0
    sab = 0.0
    common_sq_a = 0.0
    common_sq_b = 0.0
    for k in sorted(set(da) & set(db)):
        for ra, rb in zip(da[k], db[k]):
            for x, y in zip(ra, rb):
                sa += x * x
                sb += y * y
                sab += (x + y) * (x + y)
                common_sq_a += x * x
                common_sq_b += y * y
    return sa, sb, sab, common_sq_a, common_sq_b


def _total_sq(d: Mapping[str, list[list[float]]], keys: Sequence[str]) -> float:
    tot = 0.0
    for k in keys:
        for row in d[k]:
            tot += sum(x * x for x in row)
    return tot


def predictors_of(
    label_a: str,
    ad_a: N.Adapter,
    label_b: str,
    ad_b: N.Adapter,
    weights: Mapping[str, Any],
    shape: N.ModelShape,
) -> Predictors:
    """算出 ``a``/``b`` 这一对在参数空间上的全部预测量。

    ``cos`` 只在**共同挂点**上算（见 :func:`~adapterfactory.compose.cosine_and_sign`
    的说明：把不共享的挂点也算进去会结构性地把 ``cos`` 拉向 0，
    而那正好与「不重叠所以不干扰」这个看似合理的错误结论同向）。
    """
    da = P.delta_of_pair(ad_a, weights, shape)
    db = P.delta_of_pair(ad_b, weights, shape)
    cos, rate, n_common = P.cosine_and_sign(da, db)
    sa, sb, sab, common_sq_a, common_sq_b = _norm_and_overlap(da, db)
    total_sq_a = _total_sq(da, sorted(da))
    total_sq_b = _total_sq(db, sorted(db))

    expansion = None
    if n_common and sa > 0.0 and sb > 0.0:
        expansion = math.sqrt(sab) / math.sqrt(max(sa, sb))
    overlap_ab = (common_sq_a / total_sq_a) if total_sq_a > 0.0 else None
    overlap_ba = (common_sq_b / total_sq_b) if total_sq_b > 0.0 else None
    norm_ratio = (math.sqrt(sb) / math.sqrt(sa)) if sa > 0.0 else None
    return Predictors(
        a=label_a,
        b=label_b,
        cos=cos,
        sign_rate=rate,
        n_common=n_common,
        expansion=expansion,
        overlap_ab=overlap_ab,
        overlap_ba=overlap_ba,
        norm_ratio=norm_ratio,
    )


def direction_sign(cos: float | None) -> bool | None:
    """**路径 A 的判据**：``cos < 0`` ⇒ 预测这一对有害。

    这个约定写在一个函数里、而不是散在调用点：判据改了（例如换成
    「``cos`` 低于某个负阈值才叫冲突」）就会一眼看见，而散在调用点的
    判据改动会让「同向率」这个数字在两次运行之间悄悄地换了含义。

    ``cos`` 为 ``None``（正交）时返回 ``None``：**预测不了**。
    返回 ``False`` 会把「预测不了」记成「预测为无害」，于是
    「两路一致率」会把抛硬币的那些算成成功的预测。
    """
    if cos is None:
        return None
    return cos < 0.0


def pair_conflict(
    pred: Predictors,
    solo_a: E.Adjudication,
    solo_b: E.Adjudication,
    combo: E.Adjudication,
    base: E.Adjudication,
    *,
    cap_a: str,
    cap_b: str,
    threshold: float = 0.0,
) -> ConflictEdge:
    """把一对的定义域证据拼成一条 :class:`~adapterfactory.schema.ConflictEdge`。

    定向干扰用**各自的能力**：``I_{b→a}`` 看 ``a`` 在自己能力上还剩多少
    （``cap_a``），``I_{a→b}`` 看 ``b`` 的（``cap_b``）。对称值是两者的平均。
    ``affected`` 记下全部能力上的**组合相对单干**的变化量，按量级降序 ——
    它让「这一对互相干扰」有一个可追溯的落点，而不是一个孤零零的标量。
    """
    i_a_from_b = E.interference(own=solo_a, combo=combo, base=base, capability=cap_a)
    i_b_from_a = E.interference(own=solo_b, combo=combo, base=base, capability=cap_b)
    sym = E.interference_sym(i_a_from_b, i_b_from_a)
    if sym is None:
        # 有一侧测不出来时**不猜**。用一个单侧值冒充对称值是这里最容易犯的错：
        # 数值上完全合理，而「方向」这个信息被静默丢掉了。
        raise ValueError(f"{pred.a} ~ {pred.b}: 定向干扰缺一侧，无法给出对称干扰")

    predicted = direction_sign(pred.cos)
    # 「有害」的判据带上一个门槛：``interference > threshold``。
    # threshold=0 时它退化成「正负号」，而正负号在浮点噪声下不稳定 ——
    # 两个实测都是 0.0000 的场合，符号由 1e-17 决定，于是「同向率」会
    # 随机器而变。调用方可以给一个分辨率下限（例如 1/n_positions）。
    observed = sym > threshold
    agree = None if predicted is None else (predicted == observed)

    affected: dict[str, float] = {}
    for cap in sorted(set(solo_a.measures) | set(solo_b.measures)):
        a_cap = solo_a.acc(cap)
        b_cap = solo_b.acc(cap)
        c_cap = combo.acc(cap)
        if c_cap is None:
            continue
        best = max((v for v in (a_cap, b_cap) if v is not None), default=None)
        if best is None:
            continue
        # 组合相对**更好的那个单干**差了多少。用「更好的那个」当基准，
        # 是因为组合本应至少不差于任一成员 —— 用平均会让「一个强一个弱」
        # 的那一对看起来比实际平和。
        affected[cap] = best - c_cap
    ordered = dict(sorted(affected.items(), key=lambda kv: (-abs(kv[1]), kv[0])))

    return ConflictEdge(
        a=pred.a,
        b=pred.b,
        interference=sym,
        i_a_from_b=i_a_from_b,
        i_b_from_a=i_b_from_a,
        predicted_cos=pred.cos,
        predicted_sign=pred.sign_rate,
        agree=agree,
        affected=ordered,
    )


def all_pairs(
    records: Sequence[Any],
    solo: Mapping[str, E.Adjudication],
    combos: Mapping[str, E.Adjudication],
    base: E.Adjudication,
    weights: Mapping[str, Any],
    shape: N.ModelShape,
    adapters: Mapping[str, N.Adapter],
    *,
    capabilities: Sequence[str],
    threshold: float = 0.0,
) -> tuple[list[ConflictEdge], list[Predictors]]:
    """全部两两组合的冲突边（``C(8,2)=28`` 对）以及它们各自的预测量。

    ``combos`` 的键是 ``compose.cid_of("stack", (a, b))``。**固定用 ``stack``**：
    干扰的定义是「另一个成员把权重空间推开了多少」，
    而 ``merge`` 的 ``1/k`` 是**故意**的稀释（那由 ``merge_loss`` 负责），
    拿 ``merge`` 去量干扰会把两个效应混在一起。
    """
    only = {r.label: r for r in records}
    edges: list[ConflictEdge] = []
    preds: list[Predictors] = []
    labels = sorted(only)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            cid = P.cid_of("stack", (a, b))
            combo = combos.get(cid)
            if combo is None:
                raise KeyError(f"缺少组合 {cid}（{a} + {b}）的评测 —— 冲突矩阵需要它")
            p = predictors_of(a, adapters[a], b, adapters[b], weights, shape)
            preds.append(p)
            edges.append(
                pair_conflict(
                    p,
                    solo[a],
                    solo[b],
                    combo,
                    base,
                    cap_a=only[a].capability,
                    cap_b=only[b].capability,
                    threshold=threshold,
                )
            )
    del capabilities
    return edges, preds


def predictor_table(preds: Sequence[Predictors], edges: Sequence[ConflictEdge]) -> list[dict[str, Any]]:
    """候选预测量 vs 实测干扰的**逐对明细 + 相关系数**。

    这是报告里「哪条路径说得通」那一节的原料。**候选量是并列报出的，
    不预选一个**：若只报 ``cos`` 的相关系数，「幅度才是原因」这个结论就没有
    数字支撑；若只报幅度，那路径 A 被推翻这件事就看不见了。

    相关系数用 **Pearson** 与 **Spearman** 两个：干扰与幅度之间大概率是
    单调而不线性的（例如「幅度超过某一档就开始破坏」），只看 Pearson 会把
    一个完美的单调关系读成「相关性很弱」。
    """
    if len(preds) != len(edges):
        raise ValueError(f"预测 {len(preds)} 条与冲突边 {len(edges)} 条不等")
    rows: list[dict[str, Any]] = []
    for p, e in zip(preds, edges):
        rows.append({**p.to_dict(), "interference": e.interference})

    cands = ("cos", "sign_rate", "expansion", "overlap_ab", "overlap_ba", "norm_ratio")
    obs = [e.interference for e in edges]
    stats: list[dict[str, Any]] = []
    for name in cands:
        xs = [p.to_dict()[name] for p in preds]
        pair = [(x, y) for x, y in zip(xs, obs) if x is not None]
        if len(pair) < 3:
            stats.append({"name": name, "n": len(pair), "pearson": None, "spearman": None})
            continue
        stats.append(
            {
                "name": name,
                "n": len(pair),
                "pearson": _pearson([x for x, _ in pair], [y for _, y in pair]),
                "spearman": _spearman([x for x, _ in pair], [y for _, y in pair]),
            }
        )
    rows.append({"_predictor_stats": stats})
    return rows


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0.0 or syy == 0.0:
        # 某一侧完全没有变化（例如全 zoo 的 overlap 都是 1.0）——
        # 相关系数无定义。返回 None 而不是 0.0：0.0 会被读成
        # 「测过了，没有相关性」，而事实是「这个量在这里是常数，
        # 它无法参与任何排名」。
        return None
    return sxy / math.sqrt(sxx * syy)


def _ranks(xs: Sequence[float]) -> list[float]:
    """平均秩（并列取平均）。并列直接按出现顺序给秩会制造伪相关。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))
