"""落盘产物的结构契约。

### 一条贯穿全文件的规则：``from_dict`` 必须与 ``to_dict`` 严格互逆

只读命令（``summary`` / ``matrix`` / ``conflict`` / ``graph`` / ``explain`` …）
**不从零重算，而是从 ``out/`` 把产物读回来**。于是「读回来能不能用」是这些
命令唯一的质量指标，而它的失败方式是静默的：

- 少还原一个字段 → 报告里那一列变成 ``—``，看起来像「这一项本来就缺」；
- 多还原一个展示字段 → ``TypeError``，但只在读盘路径上出现；
- 用 ``(**d)`` 而不是显式取字段 → 上面两种都会发生，而且**加一个用于展示的
  字段就会让整条只读路径崩掉**。

所以每个 ``from_dict`` 都显式列出字段名，且会校验必须的那几个。

### 数字的三种"没有值"

它们长得像，含义完全不同，**绝不可互换**：

- ``None`` —— 「**这一项不适用**」。例如 ``route`` 不走权重空间，所以它没有
  ``merge_error``。报告里写成 ``—``。
- ``NaN`` —— 「**算了，但算不出来**」。例如两个适配器在参数空间上完全正交
  时余弦相似度的分母是 0。报告里写成「未知」。
- ``0.0`` —— 「**算了，就是零**」。

用 ``d.get(k) or 0`` 读它们会把前两种一起变成第三种 —— 而那正好是三个
最需要被区分的状态。这也是本工作区明确记过的一个坑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .rng import canonical_json

__all__ = [
    "MISSING",
    "UNKNOWN",
    "fmt_float",
    "CapabilityMetric",
    "AdapterRecord",
    "ArtifactRef",
    "CompositionRun",
    "ConflictEdge",
    "Version",
    "Finding",
    "PipelineResult",
    "dict_diff_paths",
    "VERDICT_TRUE",
    "VERDICT_FALSE",
    "VERDICT_UNDECIDED",
]

#: 「这一项不适用」在报告里的呈现。
MISSING = "—"
#: 「算了但算不出来」在报告里的呈现。
UNKNOWN = "未知"

VERDICT_TRUE = "成立"
VERDICT_FALSE = "被推翻"
VERDICT_UNDECIDED = "未判定"


def fmt_float(v: float | None, digits: int = 4) -> str:
    """把「三种没有值」渲染成文本。**这是它们的唯一出口。**

    每个调用点各写一份 ``f"{v:.4f}" if v is not None else "—"`` 的后果是
    迟早有一处漏掉 ``None`` 判断，于是在报告里打出 ``None`` 四个字母 ——
    而它混在数值列里看起来像一个（很难看的）值。
    """
    if v is None:
        return MISSING
    if isinstance(v, float) and v != v:  # NaN
        return UNKNOWN
    return f"{v:.{digits}f}"


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CapabilityMetric:
    """一个组合在**一个**能力上的实测。

    ``gain`` 是**算出来的**，不是存出来的 —— 但它是相对**基座实测**的差值，
    而基座实测在每个能力上都要重算一遍前向。存下来是为了让只读命令
    （``matrix`` / ``explain``）不必重新跑任何前向。
    """

    capability: str
    acc: float
    gain: float
    #: **整串全对率**：回答的每一个内容位都对。部署时真正关心的是这个 ——
    #: 一列 8 个字符错一个，整个回答就是错的。它与 ``acc`` 是**两把尺子**，
    #: 必须分开报：``acc`` 是逐 token 的微平均（与基线同口径），
    #: ``exact`` 是逐样本的全对率，后者天然远低于前者。
    #: 第一版曾把 ``heuristic``（逐 token）与整串全对率比较，得出
    #: 「适配器打不过照抄输入」—— 那是口径制造的结论，不是实测。
    exact: float
    eos_acc: float
    n_positions: float
    #: 这条组合在这个能力上的实测**干扰**：``gain(成员单独) − gain(组合)``
    #: 里的被减数。只在 size>1 时有意义，size-1 时为 ``None``。
    interference_from: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "acc": self.acc,
            "gain": self.gain,
            "exact": self.exact,
            "eos_acc": self.eos_acc,
            "n_positions": self.n_positions,
            "interference_from": self.interference_from,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CapabilityMetric":
        # ``exact`` 用 ``d.get`` 是为了让旧产物还能读：缺它就填 ``nan``
        # —— 填 0.0 会让「这一列没记过」与「整串全对一个都没有」长得一样，
        # 而后者是一个真实的（很差的）成绩。
        return cls(
            capability=str(d["capability"]),
            acc=float(d["acc"]),
            gain=float(d["gain"]),
            exact=float(d["exact"]) if d.get("exact") is not None else float("nan"),
            eos_acc=float(d["eos_acc"]),
            n_positions=float(d["n_positions"]),
            interference_from=None if d.get("interference_from") is None else float(d["interference_from"]),
        )


# ----------------------------------------------------------------------
# zoo
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterRecord:
    """zoo 里的一个 Adapter。**它是登记（Registry）这件事的具体含义。**

    注意这里**没有** ``precision`` 或 ``method`` 之类的字段：本项目的 8 个
    Adapter 全部是同一种东西（LoRA），差异只在**能力**与**覆盖范围**上。
    加一个恒定不变的字段进产物，会让它看起来像一个自变量 —— 而它不是一个
    被试过的维度。这是「配置里有的东西一定是被试过的」的同一条。
    """

    label: str
    capability: str
    variant: str
    targets: list[str]
    rank: int
    alpha: float
    scale: float
    digest: str
    n_bytes: int
    n_params: int
    base_key: str
    source: str
    #: 每个被改动的基座参数有多少个元素（``parameter_cost`` 的分子）。
    n_base_params_touched: int
    train: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "capability": self.capability,
            "variant": self.variant,
            "targets": list(self.targets),
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "digest": self.digest,
            "n_bytes": self.n_bytes,
            "n_params": self.n_params,
            "base_key": self.base_key,
            "source": self.source,
            "n_base_params_touched": self.n_base_params_touched,
            "train": dict(self.train),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AdapterRecord":
        return cls(
            label=str(d["label"]),
            capability=str(d["capability"]),
            variant=str(d["variant"]),
            targets=[str(t) for t in d["targets"]],
            rank=int(d["rank"]),
            alpha=float(d["alpha"]),
            scale=float(d["scale"]),
            digest=str(d["digest"]),
            n_bytes=int(d["n_bytes"]),
            n_params=int(d["n_params"]),
            base_key=str(d["base_key"]),
            source=str(d["source"]),
            n_base_params_touched=int(d["n_base_params_touched"]),
            train=dict(d.get("train") or {}),
        )


@dataclass(frozen=True)
class ArtifactRef:
    """一份导出的组合产物。**它本身是一个合法的 LoRA**（见 :mod:`zoo`）。"""

    digest: str
    n_bytes: int
    rank_total: int
    n_floats: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "bytes": self.n_bytes,
            "rank_total": self.rank_total,
            "n_floats": self.n_floats,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            digest=str(d["digest"]),
            n_bytes=int(d["bytes"]),
            rank_total=int(d["rank_total"]),
            n_floats=int(d["n_floats"]),
            path=str(d["path"]),
        )


# ----------------------------------------------------------------------
# 组合
# ----------------------------------------------------------------------
@dataclass
class CompositionRun:
    """一个组合的完整记录。**``runs.jsonl`` 的一行就是它的 ``to_dict()``。**"""

    cid: str
    method: str
    members: list[str]
    metrics: dict[str, CapabilityMetric]
    #: 权重空间的合并误差。``route`` 不走权重空间 ⇒ ``None``（不是 0）。
    merge_error: float | None
    n_params_changed: int
    macs: int
    wall_ms: float
    artifact: ArtifactRef | None = None
    #: 这个组合里**每个成员贡献的权重系数**。``stack`` 全 1、``merge`` 全 1/k、
    #: ``route`` 为 ``None``。存下来是因为「merge 的能力损失来自平均还是来自
    #: 别处」这个问题的答案就是这一列。
    coefs: list[float] | None = None
    #: 与同一子集的 ``stack`` 的能力差（``merge`` 才有）。见 :mod:`conflict`。
    merge_loss: float | None = None

    @property
    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "method": self.method,
            "members": list(self.members),
            "size": self.size,
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
            "merge_error": self.merge_error,
            "merge_loss": self.merge_loss,
            "n_params_changed": self.n_params_changed,
            "macs": self.macs,
            "wall_ms": self.wall_ms,
            "coefs": None if self.coefs is None else list(self.coefs),
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CompositionRun":
        metrics = {str(k): CapabilityMetric.from_dict(v) for k, v in (d.get("metrics") or {}).items()}
        return cls(
            cid=str(d["cid"]),
            method=str(d["method"]),
            members=[str(m) for m in d["members"]],
            metrics=metrics,
            merge_error=None if d.get("merge_error") is None else float(d["merge_error"]),
            merge_loss=None if d.get("merge_loss") is None else float(d["merge_loss"]),
            n_params_changed=int(d["n_params_changed"]),
            macs=int(d["macs"]),
            wall_ms=float(d["wall_ms"]),
            artifact=None if d.get("artifact") is None else ArtifactRef.from_dict(d["artifact"]),
            coefs=None if d.get("coefs") is None else [float(c) for c in d["coefs"]],
        )

    # ---- 派生量（**不落盘**，只读路径按需重算）----
    def mean_gain(self, capabilities: Sequence[str]) -> float | None:
        """在**全部**能力上的平均增益。一个组合的「总体好不好」。

        **只在每个能力都有实测时才返回数字。** 缺一个就返回 ``None`` ——
        拿部分能力的平均去和全能力的平均比较，会让「少测了几个」看起来像
        「表现更好」，而两种情形在表里都是一个小数。
        """
        vals = []
        for c in capabilities:
            m = self.metrics.get(c)
            if m is None:
                return None
            vals.append(m.gain)
        return sum(vals) / len(vals) if vals else None

    def own_gain(self) -> float | None:
        """组合在**它自己成员的能力**上的平均增益。

        与 :meth:`mean_gain` 是两个问题，**不能合用一个**：
        ``mean_gain`` 问「整体有没有副作用」，``own_gain`` 问「它想干的事干成了吗」。
        一个组合完全可能自己那件事干得很好而整体是负的（干扰了别的能力）。
        """
        caps = sorted({m.split("-")[0] for m in self.members})
        return self.mean_gain(caps)


# ----------------------------------------------------------------------
# 冲突
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ConflictEdge:
    """一对 Adapter 的干扰，带**两路证据**。

    - ``interference`` —— **能力空间实测**（路径 B）。
    - ``predicted_cos`` / ``predicted_sign`` —— **参数空间预测**（路径 A）：
      两个 delta 张量的余弦相似度与符号一致率。
    - ``agree`` —— 两路是否**同向**。

    ``agree`` 是 ``None`` 而不是 ``False``，当 ``predicted_cos`` 是 NaN
    （正交，分母为 0）。**「预测不了」与「预测错了」是两件事**，把它们
    合并成一个布尔会让「两路一致率」这个数字虚高。
    """

    a: str
    b: str
    interference: float
    #: 定向干扰的一半：``I_{b→a}`` 与 ``I_{a→b}``。
    i_a_from_b: float | None
    i_b_from_a: float | None
    predicted_cos: float | None
    predicted_sign: float | None
    agree: bool | None
    #: 实测这一对在哪几个能力上产生干扰（按干扰量降序）。它让「干扰」这个
    #: 说法有一个可追溯的落点，而不是一个孤零零的标量。
    affected: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "interference": self.interference,
            "i_a_from_b": self.i_a_from_b,
            "i_b_from_a": self.i_b_from_a,
            "predicted_cos": self.predicted_cos,
            "predicted_sign": self.predicted_sign,
            "agree": self.agree,
            "affected": {k: float(v) for k, v in sorted(self.affected.items())},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConflictEdge":
        def opt(key: str) -> float | None:
            v = d.get(key)
            return None if v is None else float(v)

        return cls(
            a=str(d["a"]),
            b=str(d["b"]),
            interference=float(d["interference"]),
            i_a_from_b=opt("i_a_from_b"),
            i_b_from_a=opt("i_b_from_a"),
            predicted_cos=opt("predicted_cos"),
            predicted_sign=opt("predicted_sign"),
            agree=None if d.get("agree") is None else bool(d["agree"]),
            affected={str(k): float(v) for k, v in (d.get("affected") or {}).items()},
        )

    @property
    def harmful(self) -> bool:
        """干扰是否**有害**：组合比成员单独时更差（``interference > 0``）。"""
        return self.interference > 0.0


# ----------------------------------------------------------------------
# 版本与裁决
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Version:
    """内容寻址版本。**唯一不可复现的字段是 ``created_at``。**

    它被单独列出来而不是藏进别处，是为了让「逐位可复现」这条断言有一个
    明确的例外清单 —— 有例外的断言才可能被真正执行。
    """

    version: str
    protocol_digest: str
    version_digest: str
    created_at: str
    n_adapters: int
    n_runs: int
    n_conflicts: int

    #: 明确声明哪些字段不进逐位复现的比较。**它是给测试读的**，
    #: 不是给人读的注释 —— 测试按这个列表排除字段。
    #:
    #: ⚠️ 这一条只管 **``Version`` 自己**的字段。整份产物的豁免单是
    #: :data:`PipelineResult.VOLATILE_PATHS` —— 那张表里有四项，这一条只占其中
    #: 一项（``runs[].wall_ms`` / ``counters.runtime_s`` / ``runs[].artifact.path``
    #: 都不在这一层）。拿这一条去核对整份 ``to_dict`` 会漏三处，而漏掉的那三处
    #: 里有两处是**实测耗时**：漏了它们的比较会在 0.05 秒的粒度上随机成立或失败。
    VOLATILE: tuple[str, ...] = ("created_at",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "protocol_digest": self.protocol_digest,
            "version_digest": self.version_digest,
            "created_at": self.created_at,
            "n_adapters": self.n_adapters,
            "n_runs": self.n_runs,
            "n_conflicts": self.n_conflicts,
            "volatile": list(self.VOLATILE),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Version":
        return cls(
            version=str(d["version"]),
            protocol_digest=str(d["protocol_digest"]),
            version_digest=str(d["version_digest"]),
            created_at=str(d["created_at"]),
            n_adapters=int(d["n_adapters"]),
            n_runs=int(d["n_runs"]),
            n_conflicts=int(d["n_conflicts"]),
        )


@dataclass(frozen=True)
class Finding:
    """一条「设计期望 vs 实测」。**期望被推翻的条目才是报告的价值所在。**

    ``expectation`` 与 ``measured`` 都是**文本**，不是数字 —— 因为期望本身
    往往是一个定性陈述（「同能力组合应当次可加」）。把定性期望硬写成阈值，
    会让「被推翻」变成一个可以靠调阈值避免的结论。
    """

    key: str
    statement: str
    expectation: str
    measured: str
    verdict: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "statement": self.statement,
            "expectation": self.expectation,
            "measured": self.measured,
            "verdict": self.verdict,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Finding":
        return cls(
            key=str(d["key"]),
            statement=str(d["statement"]),
            expectation=str(d["expectation"]),
            measured=str(d["measured"]),
            verdict=str(d["verdict"]),
            note=str(d.get("note", "")),
        )


# ----------------------------------------------------------------------
# 结果总成
# ----------------------------------------------------------------------


def _path_matches(template: str, path: str) -> bool:
    """``runs[].wall_ms`` 这类模板匹不匹配某条具体路径。

    ``[]`` 匹配**恰好一个**下标，不是「任意多段」—— 一条 ``[]`` 只能盖住
    ``runs[0].x`` 这种形状。写成通配会让 ``runs[].metrics`` 这种模板顺带把
    整棵子树都赦免掉，而赦免整棵子树正是这张表最危险的方向。

    于是**模板不匹配它自己**（``runs[]`` 里那个下标不是数字）：本函数的参数
    是**一条具体叶子路径**，传模板进来得到的 ``False`` 是对的 —— 那说明它
    不是一个可以拿去对照的叶子路径。
    """
    t, p = template.split("."), path.split(".")
    if len(t) != len(p):
        return False
    for a, b in zip(t, p):
        if a.endswith("[]"):
            head = a[:-2]
            if not (b.startswith(head + "[") and b.endswith("]")):
                return False
            if not b[len(head) + 1 : -1].isdigit():
                return False
        elif a != b:
            return False
    return True


def dict_diff_paths(a: Any, b: Any, prefix: str = "") -> list[str]:
    """两棵 ``to_dict`` 树里**取值不同**的叶子路径（``runs[3].macs`` 这种形状）。

    存在的理由是一个**反向**断言：两次运行「除声明之外逐位相同」这句话，
    只有在「差异确实一个都没跑出声明之外」时才成立。只写正向那一半的话，
    往 :data:`PipelineResult.VOLATILE_PATHS` 里加一个其实很稳定的字段
    （比如干脆整条 ``runs[]``）能让断言继续绿，而它已经什么都不查了。
    """
    out: list[str] = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b)):
            sub = f"{prefix}.{k}" if prefix else str(k)
            if k in a and k in b:
                out += dict_diff_paths(a[k], b[k], sub)
            else:
                out.append(sub)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{prefix or '[]'}.length")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            out += dict_diff_paths(x, y, f"{prefix}[{i}]")
        return out
    if a != b:
        out.append(prefix or "<root>")
    return out


@dataclass
class PipelineResult:
    """一次 ``run`` 的全部产物。**只读命令从它的 ``to_dict`` / ``from_dict`` 走。**"""

    version: Version
    base_key: str
    base_seed: int
    seq_len: int
    capabilities: list[str]
    adapters: list[AdapterRecord]
    runs: list[CompositionRun]
    conflicts: list[ConflictEdge]
    #: 非支配前沿的 ``cid``（按「平均增益 ↑、参数代价 ↓、MACs ↓」排序）。
    frontier: list[str]
    counters: Mapping[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    notes: Mapping[str, str] = field(default_factory=dict)

    def run_by_cid(self, cid: str) -> CompositionRun:
        for r in self.runs:
            if r.cid == cid:
                return r
        raise KeyError(f"没有组合 {cid!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version.to_dict(),
            "base": {"key": self.base_key, "seed": self.base_seed, "seq_len": self.seq_len},
            "capabilities": list(self.capabilities),
            "adapters": [a.to_dict() for a in self.adapters],
            "runs": [r.to_dict() for r in self.runs],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "frontier": list(self.frontier),
            "counters": dict(self.counters),
            "findings": [f.to_dict() for f in self.findings],
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PipelineResult":
        base = d["base"]
        return cls(
            version=Version.from_dict(d["version"]),
            base_key=str(base["key"]),
            base_seed=int(base["seed"]),
            seq_len=int(base["seq_len"]),
            capabilities=[str(c) for c in d["capabilities"]],
            adapters=[AdapterRecord.from_dict(a) for a in d["adapters"]],
            runs=[CompositionRun.from_dict(r) for r in d["runs"]],
            conflicts=[ConflictEdge.from_dict(c) for c in d["conflicts"]],
            frontier=[str(c) for c in d["frontier"]],
            counters=dict(d.get("counters") or {}),
            findings=[Finding.from_dict(f) for f in d.get("findings") or []],
            notes=dict(d.get("notes") or {}),
        )

    # ---- 摘要 --------------------------------------------------------
    def digest_of_run(self, cid: str) -> str:
        """单条组合的规范摘要。**用于把「这一行的非实测字段两次运行逐位相同」
        这条断言写成一个可执行的比较** —— 逐字段比较要写几十行，
        而且每加一个字段就得改一次。"""
        return canonical_json(self.run_by_cid(cid).to_dict())

    #: 跨运行**不可逐位复现**的叶子路径，``[]`` 匹配任意下标。
    #:
    #: **它是给测试读的，不是给人读的注释**：测试按这张表排除字段，并反向断言
    #: 「两次运行的差异**恰好**落在这些路径上」—— 反向那条才是真正有用的一半，
    #: 因为漏掉一项时，正向比较会随机失败，而随机失败的断言最后通常是被加一个
    #: ``pytest.mark.flaky`` 而不是被修对。
    #:
    #: 这张表是**结论**，不是借口，所以每一项都要说得出理由：
    #:
    #: - ``version.created_at`` —— 墙上时间。
    #: - ``counters.runtime_s`` —— 实测总耗时（秒，四舍五入到 0.1）。**它曾经不在这张表里**，
    #:   于是 ``stable_view`` 的相等断言会随两次运行的时长差在 0.05 秒上下**随机**成立或失败。
    #: - ``runs[].wall_ms`` —— 实测单条耗时（毫秒）。确定性代理量是 ``macs``，
    #:   主前沿只用它（坑 25/33）。
    #: - ``runs[].artifact.path`` —— 导出**位置**（出参目录），不是导出**内容**。
    #:   内容（``digest`` / ``bytes`` / ``rank_total``）照样参与比较，否则
    #:   「两次运行导出了同样的东西」这句话就没人查了。
    VOLATILE_PATHS: tuple[str, ...] = (
        "version.created_at",
        "counters.runtime_s",
        "runs[].wall_ms",
        "runs[].artifact.path",
    )

    @staticmethod
    def is_volatile(path: str) -> bool:
        """某条叶子路径是否在 :data:`VOLATILE_PATHS` 里（``[]`` 匹配任意下标）。"""
        return any(_path_matches(t, path) for t in PipelineResult.VOLATILE_PATHS)

    def stable_view(self) -> dict[str, Any]:
        """**只保留跨运行可复现的那个投影。** 逐位相同这条验收就写在这一句话上。

        删掉哪些字段由 :data:`VOLATILE_PATHS` 一处说了算 —— 这份函数的返回值
        与 ``is_volatile`` 的判据必须是同一张表，否则「两次运行的差异恰好落在
        声明之外」这条反向断言会拿着一张过期的清单去核对。
        """
        d = self.to_dict()
        for run in d["runs"]:
            run.pop("wall_ms", None)
            art = run.get("artifact")
            if isinstance(art, dict):
                art.pop("path", None)
        d["version"].pop("created_at", None)
        if isinstance(d["counters"], dict):
            d["counters"].pop("runtime_s", None)
        return d
