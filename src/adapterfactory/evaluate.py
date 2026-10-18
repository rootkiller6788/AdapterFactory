"""评测：**每个组合在 4 个能力上的成绩，以及由此派生的全部指标。**

### 只测一次，派生多次

一个组合要在 4 个能力上被测，但**每个能力的题都要用同一个组合去答**。
所以 :func:`measure` 的前向判定结果被 :class:`Adjudication` 存下来
（每个能力 → 每个样本 → 对/错），后面所有指标 —— ``gain``、``interference``、
``subadditivity``、``merge_loss`` —— 都是在这张布尔表上做算术。

**不这样做会怎样**：每个指标各跑一遍前向。看似只是慢，实际会引入一类
不一致 —— ``interference`` 用 48 个样本、``subadditivity`` 用 32 个，
两个数字都「对」，但它们不再是同一个量纲，而报告里它们会被放在同一张表里
相互印证。

### 两条基线，判定只认更强的那条

- ``chance`` = ``1/|payload_chars|`` —— **不是 ``1/vocab``**。后者把
  「输出空间是 8 个字符」与「词表有 26 个 token」混为一谈，会得到一个
  比真实随机猜测低 3 倍多的数字，于是每条 ``gain`` 都被虚增。
- ``heuristic`` = ``max(majority, copy)`` —— 多数类与照抄输入里更好的那个。

``gain = acc − accuracy(base)``，而**验收盯的是 ``gain > heuristic``**：
用更弱的 ``chance`` 当门槛，一个「学会了输出某个固定字符」的适配器就能通过，
而它什么也没学会（README 附录 C 有这一条）。

### 未测 ≠ 0

:class:`Adjudication` 允许某个 (组合, 能力) 缺失，此时 ``acc`` 是 ``None``
而**不是 0.0**。``0.0`` 是一个合法成绩（全错），把它与「没测」合并会让
「没测的组合」在 ``max(gain)`` 里被当成最差的一种，于是前沿上不会出现 ——
而那正是「静默丢掉一部分设计空间」的最省事写法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import data as D
from . import net as N
from .compose import Realized

__all__ = [
    "Measure",
    "Adjudication",
    "measure",
    "accuracy_of",
    "gain_table",
    "gain",
    "interference",
    "interference_sym",
    "subadditivity",
    "merge_identity_error",
    "merge_loss",
    "macs_of",
]


@dataclass
class Measure:
    """一个组合在**一个能力**上的原始成绩。

    ### 两把尺子，都要报，不能合成一个

    - ``acc`` —— **逐 token 准确率**（微平均，剔除 EOS），与 :mod:`data`
      的 ``mean_accuracy`` / ``baselines`` **同口径**。它是主指标，
      因为它能与 ``chance``/``heuristic`` 基线直接相减。
    - ``exact`` —— **整串全对率**（回答的每一个内容位都对）。部署时真正
      关心的是这个：一列 8 个字符错一个，整个回答就是错的。

    两者口径不同，**必须分开报**。混用是会出事的：``heuristic`` 是逐 token
    算的，若拿它去与整串全对率比较，就会得出「适配器打不过照抄输入」这种
    由口径制造的结论（整串全对率天然远低于逐 token 率）。
    这一条不是假想 —— 本项目第一版就是这样比过一次，见 README 附录 C。
    """

    capability: str
    n_eval: int
    #: 逐 token：答对的（内容）位数 / 内容位总数。
    n_correct: int
    n_positions: int
    acc: float
    #: 整串全对：答对的样本数 / 样本总数。
    n_exact: int
    exact: float
    #: 会不会停。**单独一项**，不并进上面任何一个。
    eos_acc: float
    #: 逐样本「整串全对」布尔。留它是为了任何「换一组样本结论还成立吗」
    #: 的复查可以就地做，不必重跑前向。
    hits: tuple[bool, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "n_eval": self.n_eval,
            "n_correct": self.n_correct,
            "n_positions": self.n_positions,
            "acc": self.acc,
            "n_exact": self.n_exact,
            "exact": self.exact,
            "eos_acc": self.eos_acc,
        }


@dataclass
class Adjudication:
    """一个组合在**全部能力**上的判定：``{能力: Measure}``。

    **这个类不报 MACs，哪怕它跑遍了每个样本。** 曾经它有一个 ``macs`` 字段，
    意思是「这一轮评测消耗的 MACs 总量」—— 那是本文件里第二个叫 ``macs``
    的量，而它与 ``CompositionRun.macs``（一次前向的 MACs）**不同量纲**。
    两个同名不同义的数放在同一份产物里，读的人迟早会拿一个去比另一个。

    更糟的是它**算不对**：那个和是在逐样本的缓存命中之上累加出来的，
    命中一次就少算一次，于是它成了缓存状态的函数而不是组合的函数。
    现在 MACs 只有一个来源（:func:`macs_of`，绕开缓存、一次前向），
    这里只留判定本身。
    """

    cid: str
    method: str
    members: tuple[str, ...]
    measures: dict[str, Measure] = field(default_factory=dict)
    #: 每个能力的**探针集大小**。不同能力可以不同（本项目统一，
    #: 但把它记下来，才能让「两个能力的 acc 能不能直接比」有据可查）。
    n_eval_by_cap: dict[str, int] = field(default_factory=dict)

    def acc(self, capability: str) -> float | None:
        m = self.measures.get(capability)
        return None if m is None else m.acc

    def exact(self, capability: str) -> float | None:
        m = self.measures.get(capability)
        return None if m is None else m.exact

    def hit(self, capability: str, i: int) -> bool | None:
        m = self.measures.get(capability)
        if m is None or i >= len(m.hits):
            return None
        return m.hits[i]

    def capabilities(self) -> list[str]:
        return [c for c in self.measures]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "method": self.method,
            "members": list(self.members),
            "mean_acc": None if not self.measures else sum(m.acc for m in self.measures.values()) / len(self.measures),
            "mean_exact": None if not self.measures else sum(m.exact for m in self.measures.values()) / len(self.measures),
            "metrics": {c: m.to_dict() for c, m in sorted(self.measures.items())},
            "n_eval_by_cap": dict(sorted(self.n_eval_by_cap.items())),
        }


def macs_of(realized: Realized, ids: Sequence[int], shape: N.ModelShape) -> int:
    """跑一次前向并返回 MACs 总数（``math.sumprod`` 的累加量代理）。

    **这是主前沿的 ``latency`` 轴，不是 wall-clock。** 实测耗时不可逐位复现
    （坑 25/33）：同一份产物两次运行的 ``wall_ms`` 会差出百分之几，
    于是「两次跑逐位相同」这条验收永远过不了，而绕过它的办法通常是把
    实测轴偷偷排除在比对之外 —— 那等于承认产物不可复现。

    MACs 是**被测量的那一侧**的函数，与机器无关：``stack``/``merge`` 的
    MACs 与基座逐位相同（矩阵已合并，只是一次更大的乘法），
    ``route`` 则要为每个适配器多做一次 ``x @ Δᵢ``。于是
    「合并省下多少计算」这件事在产物里是一个数字，而不是一段论述。

    ⚠️ **必须绕开记忆化**（``use_cache=False``）。计数与缓存直接冲突：
    命中的那次前向一个算子都没跑，``mac_counter`` 于是报 0。
    「0」在这里看起来只是一个偏小的合法值，所以这个 bug 的表现是
    **静默的** —— 它不会抛错，只会让 latency 轴整条失效，
    让 ``pareto_frontier`` 的第三个轴退化成恒等，让 route
    「为每个适配器多做一次乘法」这句话失去证据。
    """
    with N.mac_counter() as mc:
        realized.logits(tuple(ids), shape, use_cache=False)
    return int(mc.to_dict()["total"])


def measure(
    realized: Realized,
    eval_sets: Mapping[str, Sequence[D.Example]],
    shape: N.ModelShape,
    capabilities: Sequence[str],
) -> Adjudication:
    """把 ``realized`` 在全部能力的探针集上测一遍。

    **逐 token 准确率直接调 :func:`adapterfactory.data.mean_accuracy`**，
    不在这里重写一遍。重写的代价是本项目已经踩过的那个坑：评测用一套口径、
    基线用另一套，两者都「算对了」，而 ``gain`` 是一个跨口径的差值。
    一条实现，一处口径，基线那边也走同一个函数。

    **每个能力的样本数记进 ``n_eval_by_cap``**，而不是取一个全局的
    ``n_eval``：全局值的风险是「某个能力的题少了几个」不会在任何地方显示出来，
    而 ``acc`` 的分辨率是按能力各自算的（1/n），两个能力的 ``acc`` 直接相减
    比较，隐含了「它们的分辨率一样」这个前提。
    """
    adj = Adjudication(cid=realized.cid, method=realized.method, members=realized.members)
    for cap in capabilities:
        probes = eval_sets.get(cap)
        if not probes:
            # 没有探针集的能力**不进 measures** —— 进去一个 acc=0.0 的占位项
            # 会污染 mean_acc，并且让「这个能力没测」与「这个能力全错」在外观上相同。
            continue
        logits_list: list[list[list[float]]] = [
            realized.logits(tuple(ex.ids), shape) for ex in probes
        ]
        agg = D.mean_accuracy(logits_list, probes)
        hits = tuple(_exact_match(lg, ex) for lg, ex in zip(logits_list, probes))
        n = len(probes)
        adj.measures[cap] = Measure(
            capability=cap,
            n_eval=n,
            n_correct=int(round(agg["acc"] * agg["n_positions"])),
            n_positions=int(agg["n_positions"]),
            acc=float(agg["acc"]),
            n_exact=sum(1 for h in hits if h),
            exact=sum(1 for h in hits if h) / float(n),
            eos_acc=float(agg["eos_acc"]),
            hits=hits,
        )
        adj.n_eval_by_cap[cap] = n
    return adj


def _exact_match(logits: Sequence[Sequence[float]], ex: D.Example) -> bool:
    """整串全对：**回答的每一个内容位都猜对**。

    只回答内容位，不要求 EOS —— 与 ``mean_accuracy`` 的主指标同口径
    （那里也剔除 EOS）。「会不会停」由 ``Measure.eos_acc`` 单独盯着，
    两者合成一个数会让「答对了但不停」与「停下来但答错了」变成同一个数字。

    **它比逐 token 准确率严格得多**：8 个字符的回答，逐 token 0.875 意味着
    平均错 1 个，而整串全对率此时可能只有 0.2。两个数字都对，
    但只有 ``exact`` 回答「这份 Adapter 能不能用」。
    """
    if ex.response_len <= 0:
        raise ValueError("样本没有回答位：评测无从下手")
    for t in D.content_positions(ex):
        row = logits[t]
        best = max(range(len(row)), key=row.__getitem__)
        if best != ex.targets[t]:
            return False
    return True


def accuracy_of(adj: Adjudication, capability: str) -> float | None:
    return adj.acc(capability)


def gain_table(
    adj: Adjudication,
    base: Adjudication,
    capabilities: Sequence[str],
) -> dict[str, float | None]:
    """``gain_C(c) = acc_C(c) − acc_base(c)``，逐能力。

    **减数是实测的基座成绩，不是一个假设的 0.5 或 chance。** 用 chance 当减数
    会让 gain 里混进「这个能力本来就比乱猜容易多少」—— 而那是任务的性质，
    不是适配器的贡献。基座成绩实测出来（本项目实测：四个能力上都贴着 chance），
    于是 gain 干净地只剩适配器那一份。
    """
    out: dict[str, float | None] = {}
    for cap in capabilities:
        a = adj.acc(cap)
        b = base.acc(cap)
        out[cap] = None if a is None or b is None else a - b
    return out


def gain(adj: Adjudication, base: Adjudication, capability: str) -> float | None:
    return gain_table(adj, base, [capability])[capability]


def interference(
    own: Adjudication,
    combo: Adjudication,
    base: Adjudication,
    capability: str,
) -> float | None:
    """``I_{other→own} = gain_own(c) − gain_combo(c)``：**别的成员破坏了多少。**

    读法：先看 ``own`` 单干时在能力 ``c`` 上涨了多少，再看它和别的适配器
    一起挂上之后同一个能力上还剩多少。两者之差就是同伴的破坏量。
    两次 gain 都相对**同一个** ``base`` 算，所以基座成绩相减时自然消掉。

    **定向、不对称。** ``I_{j→i}`` 与 ``I_{i→j}`` 一般不等：一个强适配器
    可能几乎不受打扰，却让另一个弱的归零。所以这个函数一次只答一个方向，
    对称化交给 :func:`interference_sym` —— 顺序反过来的话（先对称再问方向）
    方向信息已经丢了，而「谁被谁破坏」正是要回答的问题。

    ``capability`` 是**被观测的那个能力**，不是 ``own`` 自己的标签。
    两者在本项目里一致（``own`` 是 ``upper-*`` 就在 ``upper`` 上看），
    但把参数分开写是有意的：跨能力观测（``sort`` 适配器对 ``upper`` 的
    成绩有没有影响）是同一个函数、同一套数学，只是换一个 ``capability``。
    """
    a = gain(own, base, capability)
    b = gain(combo, base, capability)
    return None if a is None or b is None else a - b


def interference_sym(i_to_j: float | None, j_to_i: float | None) -> float | None:
    """``I(i,j) = ½(I_{j→i} + I_{i→j})``：对称化，只用于**汇总与画图**。

    任一侧缺失（``None``）就返回 ``None``，**不取另一侧的绝对值当两者都有**。
    「只有一个方向测过」时，用单侧代表对称值会让一个方向的干扰被读成
    双向的；而报告里逐个 ``conflict`` 行是要标方向的。
    """
    if i_to_j is None or j_to_i is None:
        return None
    return 0.5 * (i_to_j + j_to_i)


def subadditivity(
    solo_i: Adjudication,
    solo_j: Adjudication,
    combo: Adjudication,
    base: Adjudication,
    capability: str,
) -> float | None:
    """``gain_i(c) + gain_j(c) − gain_{ij}(c)``：**收益是相加、相消还是互相破坏。**

    三个结果各有名字，而且必须分开（这是 README 要的三个可判定答案）：

    - ``≈ 0`` ⇒ **相加**：两个适配器做的事互不重叠；
    - ``> 0`` ⇒ **次可加**：合起来不如分别来 —— 同能力的一对必然落在这一档
      （它们在做同一件事，第二个的边际收益递减）；
    - ``< 0`` ⇒ **超可加**：合起来比自己单干还强，一般是一对适配器各自
      学到了一半、拼起来正好互补。

    接受 ``< 0`` 是**故意的**：把超可加当成错误会让这一列只输出非负数，
    而「相加／相消／互相破坏」里第二个答案就永远给不出来。

    三个 ``gain`` 都相对同一个 ``base``，基座成绩在减法里消掉 ——
    这也意味着 ``base`` 缺失时三者一起变 ``None``，而不是留下一个
    「用别的基准算出来的差值」。
    """
    gi = gain(solo_i, base, capability)
    gj = gain(solo_j, base, capability)
    gij = gain(combo, base, capability)
    if gi is None or gj is None or gij is None:
        return None
    return gi + gj - gij


def merge_identity_error(
    weights: dict[str, Any],
    adapters: Sequence[N.Adapter],
    ids: Sequence[int],
    shape: N.ModelShape,
) -> float:
    """``max_abs_diff``：**合并进权重再前向** vs **不合并、逐个施加**。

    **这是实现检查，不是能力指标。** 两者在实数上恒等
    （``x@W + Σx@Δᵢ == x@(W + ΣΔᵢ)``），差别只有累加顺序，实测在 1e-16 量级。
    所以这条断言能查出的是**实现错误**（漏了一层、挂点展开不一致、系数用错），
    而不是浮点噪声 —— 门槛才敢定在 1e-12。

    ⚠️ **它不查「合并没有损失能力」。** 那是 ``merge_loss``。
    两件事在报告里必须分开写：``merge_error≈0`` 与
    ``merge`` 的能力低于 ``stack`` **同时成立**才是事实的全貌。

    ⚠️ **系数必须是全 1。** 用 ``merge`` 的 ``1/k`` 去比会得到一个真实存在的
    差值（那是稀释，不是实现错误），于是这条检查的门槛要么被放宽到失去意义，
    要么把一条正确实现判成失败。默认参数就是为了让它不可能被传错。
    """
    merged = N.merge_into(weights, list(adapters))
    ref = N.logits_of(weights, ids, shape, adapters=list(adapters))
    got = N.logits_of(merged, ids, shape)
    return N.max_abs_diff(got, ref)


def merge_loss(
    stack_run: Adjudication | None,
    merge_run: Adjudication | None,
    capability: str,
) -> float | None:
    """``stack`` 与 ``merge`` 在同一个能力上的成绩差：**平均付出的能力代价。**

    只在**同一组成员、同一个能力**之间有意义。``members`` 不同时返回 ``None``
    —— 把不同成员的两个组合相减会得到一个「差值」，而它没有任何解释。

    这个函数的两个参数类型是 ``Adjudication`` 而不是 ``float``，是为了强制
    调用方把成员一致性检查做在传参之前 —— 若签名收两个 float，
    「这两个数是不是同一组员的」就只能靠注释约定。
    """
    if stack_run is None or merge_run is None:
        return None
    if stack_run.members != merge_run.members:
        return None
    a = stack_run.acc(capability)
    b = merge_run.acc(capability)
    return None if a is None or b is None else a - b
