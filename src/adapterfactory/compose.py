"""组合：**在 Adapter 集合的子集格上展开设计空间，再把每个点实例化成一次前向。**

### 设计空间 = 子集格，不是笛卡尔积

8 个 Adapter 的全部 size 1..3 子集是 ``C(8,1)+C(8,2)+C(8,3) = 8+28+56 = 92`` 个。
**子集不是排列** —— ``{upper-qv, shift-qv}`` 与 ``{shift-qv, upper-qv}`` 是同一个
组合，因为 :func:`adapterfactory.net.merge_into` 做的是 ``Σ cᵢΔᵢ``，求和可交换。
把它当排列展开会让 276 变成 8+56+336=400，而报告里「组合矩阵有 400 行」这件事
看起来只是「更全」；重复的行数值几乎相同（只差浮点累加序），
于是任何「取最大值的前沿」都会挑中重复项的一个副本，前沿条数虚高。

所以：**成员在 :func:`cid_of` 与计划表里一律排序**。于是 ``[a,b]`` 与 ``[b,a]``
不只是「碰巧算出一样」，而是**根本就是同一条记录**。

### 三个方法的实例化方式（这是本项目的主对照）

| 方法 | 实例化成什么 | 推理时多付什么 |
| --- | --- | --- |
| ``stack`` | 一张新权重表 ``W + Σ Δᵢ``（系数全 1） | **什么都不多付** |
| ``merge`` | 一张新权重表 ``W + (1/k)Σ Δᵢ`` | 什么都不多付 |
| ``route`` | **基座权重不动**，前向时逐层按门混合 | k 次额外的矩阵乘 + 一次软最大 |

这张表就是「合并省下的是计算，付出的是能力」这句话的可执行形式：
``stack``/``merge`` 的 MACs 与基座**逐位相同**，``route`` 的 MACs 更高。
若三个方法走同一条前向路径，这一列就永远是常数，报告里那一节也就没有内容。

### 一条刻意的实现约束

``stack`` 与 ``merge`` 共用 :func:`~adapterfactory.net.merge_into`，只差系数。
「求平均」不是另一套算法，它只是求和的一个特例。拆成两份实现的话，
``merge`` 的能力损失到底来自**平均**还是来自**别的东西**就永远说不清了 ——
而 ``merge_loss`` 的全部意义就在于回答前者。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from . import config as C
from . import net as N
from .rng import digest_of, short_digest

__all__ = [
    "Plan",
    "Realized",
    "cid_of",
    "iter_subsets",
    "plan_grid",
    "grid_size",
    "coefficients",
    "realize",
    "realize_base",
    "delta_of_pair",
    "cosine_and_sign",
]


def cid_of(method: str, members: Sequence[str]) -> str:
    """组合的内容标识：``方法 + 排序后的成员`` 的摘要前 12 位。

    **成员先排序。** 不排序的话 ``[upper-qv, shift-qv]`` 与
    ``[shift-qv, upper-qv]`` 会得到两个 ``cid``，于是同一份组合在
    ``runs.jsonl`` 里出现两次 —— 而两次的数值只差浮点累加序（~1e-16），
    肉眼完全一致。症状是所有按 ``cid`` 建的索引都会拿到两条，
    ``run_by_cid`` 静默返回其中一条，**另一条上的指标永远读不到**。
    """
    m = sorted(str(x) for x in members)
    if len(set(m)) != len(m):
        raise ValueError(f"组合 {method} 的成员有重复：{members}")
    if not m:
        raise ValueError("空组合不是组合：成员至少要有一个")
    return short_digest(digest_of({"method": str(method), "members": m}))


def iter_subsets(labels: Sequence[str], sizes: Sequence[int]) -> Iterator[tuple[str, ...]]:
    """按 ``sizes`` 里的规模枚举子集，**成员已排序**。

    ``itertools.combinations`` 在**已排序**的输入上给出的成员天然有序 ——
    这里先把输入排序再去组合，而不是事后 ``sorted()`` 掉：事后排序会让
    「枚举顺序」与「成员顺序」变成两件事，而枚举顺序决定 ``runs.jsonl`` 的
    行序，成员顺序决定 ``cid``。两者现在都由同一个 ``labels`` 定。
    """
    base = sorted(str(x) for x in labels)
    for k in sizes:
        if k < 1:
            raise ValueError(f"子集规模必须 >= 1，得到 {k}")
        if k > len(base):
            raise ValueError(f"子集规模 {k} 超过 Adapter 总数 {len(base)}")
        yield from itertools.combinations(base, k)


def grid_size(n_adapters: int, sizes: Sequence[int], n_methods: int) -> int:
    """网格大小 = Σ C(n, k) × 方法数。**276 这个数字从公式里来，不写死。**

    写死的问题不是「不够灵活」，而是「改了 ``max_size`` 之后没人会发现」：
    组合数会在报告里变成一个与配置无关的常量，于是
    ``require_grid_complete`` 那条验收永远通过。
    """
    from math import comb

    return sum(comb(n_adapters, k) for k in sizes) * n_methods


@dataclass(frozen=True)
class Plan:
    """设计空间里的一个点：**还没被算过**的组合。"""

    cid: str
    method: str
    members: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {"cid": self.cid, "method": self.method, "members": list(self.members), "size": self.size}


def plan_grid(
    labels: Sequence[str],
    methods: Sequence[str],
    sizes: Sequence[int],
) -> list[Plan]:
    """展开整个设计空间。**顺序稳定**：先按规模、再按子集字典序、最后按方法。

    顺序稳定的意义是 ``runs.jsonl`` 可 diff。用 ``set`` 迭代或按方法外层循环，
    行序会随解释器与配置顺序变化，于是「两次运行逐位相同」这条验收只能靠
    读回 JSON 再比对 —— 而那正是它想避免的。
    """
    unknown = [m for m in methods if m not in C.METHODS]
    if unknown:
        raise ValueError(f"未知的组合方法 {unknown}（可选：{list(C.METHODS)}）")
    plans: list[Plan] = []
    for members in iter_subsets(labels, sizes):
        for method in methods:
            plans.append(Plan(cid=cid_of(method, members), method=str(method), members=tuple(members)))
    seen: dict[str, Plan] = {}
    for p in plans:
        if p.cid in seen:
            raise ValueError(f"cid 撞车：{p.cid} 同时属于 {seen[p.cid].to_dict()} 与 {p.to_dict()}")
        seen[p.cid] = p
    return plans


def coefficients(method: str, k: int) -> list[float] | None:
    """``stack`` → 全 1；``merge`` → 全 ``1/k``；``route`` → ``None``。

    ``route`` 返回 ``None`` 而不是全 1：它**根本不进权重空间**，
    给它一组系数等于宣布「route 也是一种合并」，而报告的整个第三列
    就是为了反驳这句话。

    ⚠️ 前面那道 ``method not in C.METHODS`` 不是重复检查：
    :func:`~adapterfactory.config.weight_scheme` 用 ``dict.get`` 实现，
    于是**未知方法与 ``route`` 落到同一个 ``None``**。少了这道守卫，
    ``coefficients("sack", 2)`` 会安静地返回 ``None``，调用方读到的
    意思却是「这是一个 route」。症状是打错一个字母的方法名全程无报错。
    """
    if method not in C.METHODS:
        raise ValueError(f"未知的组合方法 {method!r}（可选：{list(C.METHODS)}）")
    scheme = C.weight_scheme(method)
    if scheme is None:
        return None
    if scheme == C._WSUM:
        return [1.0] * k
    if scheme == C._WAVG:
        return [1.0 / float(k)] * k
    raise ValueError(f"未知的系数口径 {scheme!r}（method={method!r}）")


@dataclass
class Realized:
    """一个**已经实例化**的组合：可以直接拿来前向。

    ``weights`` 与 ``adapters`` 的取值取决于方法，这正是「合并」与「不合并」
    的分野：

    - ``stack`` / ``merge``：``weights`` 是**新的**权重表，``adapters`` 为空；
    - ``route``：``weights`` 是**原样的**基座权重（同一份对象），``adapters`` 全在。

    :meth:`logits` 是唯一的前向入口，两条路在这里分岔、在
    :func:`~adapterfactory.net.forward` 里重新汇合。若调用方自己判断
    ``if method == "route"`` 再各调各的，那三处判断必然会有一处写反，
    而写反的症状是 route 被当成 stack 算（数值合理、结论错）。
    """

    plan: Plan
    weights: dict[str, Any]
    adapters: tuple[N.Adapter, ...] = ()
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None
    temperature: float = 1.0
    merge_coefs: list[float] | None = None
    n_delta_floats: int = 0
    _logits_cache: dict[Any, list[list[float]]] = field(default_factory=dict, repr=False)

    @property
    def method(self) -> str:
        return self.plan.method

    @property
    def members(self) -> tuple[str, ...]:
        return self.plan.members

    @property
    def cid(self) -> str:
        return self.plan.cid

    @property
    def weight_scheme(self) -> str | None:
        return C.weight_scheme(self.method)

    def logits(
        self, ids: Sequence[int], shape: N.ModelShape, *, use_cache: bool = True
    ) -> list[list[float]]:
        """前向一次。**按 ``ids`` 做记忆化** —— 同一个组合要被 4 个能力 × n_eval
        个样本评测，而一个组合的权重表在整轮评测里是常数。

        ``ids`` 是 ``tuple`` 才进缓存：传 list 的话它是不可哈希的，
        而「偷偷 tuple 化一下」会让 ``Realized`` 对调用方撒谎 ——
        调用方以为自己控制了缓存键。

        ⚠️ ``use_cache=False`` 是给**计数**用的，不是优化开关。记忆化与
        MACs 计数直接冲突：命中的那次前向一个算子都没跑，于是它「消耗 0 MACs」——
        这对「这次调用花了多少计算」是对的，对「这个组合一次前向要多少计算」
        是错的。本项目踩过这个坑：``mac_ids`` 取的正是**第一个评测样本**的 ids，
        而 ``measure`` 先跑，于是 ``macs_of`` 次次命中缓存，
        整条 latency 轴恒为 0 —— 而 0 看起来只是一个「很小但合理」的值。
        """
        key = tuple(ids)
        if use_cache:
            hit = self._logits_cache.get(key)
            if hit is not None:
                return hit
        if self.method == "route":
            if self.gates is None:
                raise ValueError(f"{self.cid}: route 组合缺门向量")
            out = N.forward_routed(
                self.weights,
                self.adapters,
                key,
                shape,
                gates=self.gates,
                temperature=self.temperature,
            ).data
        else:
            out = N.logits_of(self.weights, key, shape)
        if use_cache:
            self._logits_cache[key] = out
        return out

    def to_dict(self) -> dict[str, Any]:
        d = self.plan.to_dict()
        d.update(
            {
                "weight_scheme": self.weight_scheme,
                "merge_coefs": None if self.merge_coefs is None else [round(c, 12) for c in self.merge_coefs],
                "n_delta_floats": self.n_delta_floats,
                "merged": not self.adapters,
            }
        )
        return d


def realize(
    weights: dict[str, Any],
    adapters: Sequence[N.Adapter],
    plan: Plan,
    shape: N.ModelShape,
    *,
    gates: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    temperature: float = 1.0,
) -> Realized:
    """把 :class:`Plan` 实例化。``adapters`` 必须与 ``plan.members`` 同序同数。

    顺序由 ``plan.members``（已排序）定，调用方负责按它取。**函数里断言**，
    因为「传进来的顺序与 cid 里的顺序不一致」是本项目里最容易发生、
    最难看出来的一类错：``Σ Δᵢ`` 对两个成员可交换，但对三个成员
    累加序不同会有 ~1e-16 的差 —— 于是数值几乎相同，而 ``cid`` 指向的
    是另一个组合。等到 ``member`` 再组合（把组合当适配器用）时，
    这个错位才会突然变成一个真实的错误结论。
    """
    if len(adapters) != len(plan.members):
        raise ValueError(f"{plan.cid}: 适配器 {len(adapters)} 个 != 成员 {len(plan.members)} 个")
    for ad, lab in zip(adapters, plan.members):
        if ad.label and ad.label != lab:
            raise ValueError(f"{plan.cid}: 适配器标签 {ad.label!r} 与成员 {lab!r} 不对应")
    n_floats = sum(ad.n_params for ad in adapters)
    if plan.method == "route":
        return Realized(
            plan=plan,
            weights=weights,
            adapters=tuple(adapters),
            gates=gates,
            temperature=temperature,
            merge_coefs=None,
            n_delta_floats=n_floats,
        )
    coefs = coefficients(plan.method, len(adapters))
    assert coefs is not None  # stack/merge 的系数一定不是 None
    merged = N.merge_into(weights, list(adapters), weights_by_adapter=coefs)
    return Realized(
        plan=plan,
        weights=merged,
        adapters=(),
        gates=None,
        temperature=temperature,
        merge_coefs=coefs,
        n_delta_floats=n_floats,
    )


def realize_base(weights: dict[str, Any], shape: N.ModelShape) -> Realized:
    """基座自身，当"零号组合"用。``cid`` 是常量 ``"base"``。

    它不走 :func:`realize`，因为 :class:`Plan` 要求成员非空 —— 而
    「基座在四个能力上的成绩」是全部 ``capability_gain`` 的减数，
    少了它整表指标只能算出「组合之间差多少」，算不出「组合涨了多少」。
    """
    return Realized(plan=Plan(cid="base", method="base", members=()), weights=weights, adapters=())


# ----------------------------------------------------------------------
# 参数空间：冲突的**预测**路径
# ----------------------------------------------------------------------
def delta_of_pair(ad: N.Adapter, weights: Mapping[str, Any], shape: N.ModelShape) -> dict[str, list[list[float]]]:
    """一个适配器在**全部挂点**上的 delta 矩阵 ``{模块全名: [in,out]}``。

    只取它自己声明挂的模块。两个覆盖率不同的适配器（``qv`` 与 ``all``）
    因此会在不同的坐标集合上被比较 —— 见 :func:`cosine_and_sign` 的说明。
    """
    base = N.param_shapes_of(dict(weights))
    out: dict[str, list[list[float]]] = {}
    for full in N.expand_targets(base, ad.targets):
        out[full] = ad.delta(full, base[full])
    return out


def cosine_and_sign(
    a: Mapping[str, list[list[float]]],
    b: Mapping[str, list[list[float]]],
) -> tuple[float | None, float | None, int]:
    """两个 delta 张量集合的**余弦相似度**与**符号一致率**，只在共同挂点上算。

    返回 ``(cos, sign_rate, n_common)``；没有共同挂点时两个都是 ``None``。

    ### 为什么必须在**共同挂点**上算

    ``upper-qv`` 只挂 ``q,v`` 两个模块，``sort-all`` 挂 7 个。若把
    ``sort-all`` 多出来的 5 个模块也算进去，它那 5 个模块的 delta 会与
    「零」比较（``upper-qv`` 在那里没有 delta）—— 而「与零做余弦」的结果
    是 0 或未定义，于是 ``cos`` 被**结构性地**拉向 0。
    症状是「覆盖率差得越多的一对，预测的干扰就越小」，
    而那恰好与「挂点不重叠所以互不干扰」这个**看起来合理**的结论同向 —— 两条
    路径就会一起同意一个由比较口径制造的结论。所以只在共同挂点上比较，
    并把 ``n_common`` 一起报出来：``n_common=2`` 与 ``n_common=7`` 的
    可信度本来就不一样。

    ### 符号一致率与余弦是两条不同的信息

    余弦看**方向**，符号一致率看**逐坐标的符号**。两个 delta 可以整体近乎正交
    （``cos≈0``）却在多数坐标上同号 —— 那种情形下它们会**互相加强**一部分坐标
    而抵消另一部分，而单看余弦会把它读成「无关」。两条一起报，读者才能分辨
    「不干扰」与「部分加强部分抵消」。
    """
    keys = sorted(set(a) & set(b))
    if not keys:
        return None, None, 0
    dot = 0.0
    na = 0.0
    nb = 0.0
    same = 0
    total = 0
    for k in keys:
        da, db = a[k], b[k]
        if len(da) != len(db):
            raise ValueError(f"{k}: 两个 delta 形状不同 {len(da)} vs {len(db)}")
        for ra, rb in zip(da, db):
            if len(ra) != len(rb):
                raise ValueError(f"{k}: 两个 delta 行宽不同")
            for x, y in zip(ra, rb):
                dot += x * y
                na += x * x
                nb += y * y
                # 严格大于 0 才算「同号」。0 参与计数会让「一方恰好是 0」
                # 这种退化情形被算成一致，而 delta 恰好为 0 的坐标是常见的
                # （B 初始化为 0 的适配器、被裁剪到 0 的坐标）。
                if x * y > 0.0:
                    same += 1
                elif x * y < 0.0:
                    pass
                else:
                    continue
                total += 1
    cos = None if na == 0.0 or nb == 0.0 else dot / (na**0.5 * nb**0.5)
    rate = None if total == 0 else same / float(total)
    return cos, rate, len(keys)
