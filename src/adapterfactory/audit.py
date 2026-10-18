"""运行时核对。**每一条都要在每次 ``run`` 上算一遍，结果进报告。**

### 架构墙：为什么查的是「调用」而不是「词」

项目的核心命题是「AdapterFactory 不训练」。这条命题若只写在文档里，
它在第一次需要「顺手加个微调」的时候就会失效。所以 :func:`wall_findings`
每次运行都扫一遍流水线模块的源码。

**第一版查的是标识符 ``train``，那是个错误的设计。**
实测扫出来 15 处：``config["train"]``（超参小节）、``ZooRecord.train``
（旁证元数据）、``AdapterRecord(train=...)``（字段透传）—— 全是**名词**，
没有一处是训练动作。把名词也禁掉的代价不是「更严格」，而是：

- 要么把字段改名成 ``train_meta``，让产物里的键名为了迁就检查而变；
- 要么给检查加白名单，于是**下一次真的写出 ``train_adapter(...)`` 时，
  白名单会把它一起放过**。

所以墙查的是**动作**：训练入口函数被调用、``backward`` 被调用、
``produce`` 被导入。这三件事才是「训练」，而 ``train`` 这个词不是。
**一条会把合法代码判成违规的检查，最终一定会被人关掉。**

### 两条扫描路径，都要

- ``ast``：精确认得出「导入 ``produce``」「调用 ``train_adapter``」。
  它不受文档字符串与注释影响，这是它优于文本扫描的地方。
- ``tokenize``：看见**每一个**标识符，包括 ``ast`` 会归一化掉的那些位置
  （``getattr(m, "back" + "ward")`` 在 token 层面是三个独立标识符）。

两条一起跑，任一条坏掉另一条还能报。**只跑一条的失败方式是静默的**：
``ast`` 解析失败（源码里有个新语法）会抛异常，而异常被 ``except`` 吞掉之后
「墙是绿的」与「墙没跑」长得一模一样。
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import config as C

__all__ = [
    "PIPELINE_MODULES",
    "CONTROL_MODULE",
    "FORBIDDEN_CALLS",
    "DATA_SIDE_CALLS",
    "FORBIDDEN_MODULES",
    "CheckResult",
    "wall_findings",
    "data_side_findings",
    "producer_importers",
    "scan_tokens",
    "checks",
    "run_checks",
    "acceptance_report",
]


#: 受保护的模块 —— 架构墙守着的那一侧。**``net`` / ``data`` / ``produce`` / ``cli`` 不在里面**：
#:
#: - ``net`` 是共享的数值引擎（它的 ``backward`` 只服务 ``produce``）；
#: - ``data`` 是 ``build_train_set`` 的**定义处** —— 把定义处也纳入扫描，
#:   只会让 ``def build_train_set`` 这一行报自己的违规；
#: - ``produce`` 就是墙的另一侧本身；
#: - ``cli`` 是**控制点**，不是被保护的一侧 —— 它必须能提供 ``produce`` 这个子命令。
#:   把控制点也列进受保护清单，得到的不是更严格，而是一条必须被豁免的检查；
#:   而一条**永远被豁免的检查**在第一次有人真把训练写进 ``compose.py`` 时
#:   也会一起放过。控制点由 :data:`CONTROL_MODULE` 那条不变量单独管。
PIPELINE_MODULES: tuple[str, ...] = (
    "rng",
    "schema",
    "config",
    "lineage",
    "zoo",
    "registry",
    "compose",
    "evaluate",
    "conflict",
    "graph",
    "audit",
    "report",
    "visualize",
    "pipeline",
)

#: 唯一允许触碰 producer 的模块。
CONTROL_MODULE = "cli"

#: 一被**调用**就说明这一侧在训练 —— 判据是「它会不会更新权重」。
#:
#: ⚠️ ``build_train_set`` 与 ``batch_order`` **曾经**在里面，那是错的：
#: 前者只是把样本materialize出来，后者只是排个 batch 顺序，两者都不更新任何东西。
#: 而 ``build_train_set`` 还是 ``majority`` 基线**必须**的输入 ——
#: 众数只能在训练集上拟合，在评测集上拟合就是泄漏。
#: 把它们禁掉的后果不是「更安全」，而是让流水线要么算不出基线，
#: 要么为了迁就检查而去找一个语义不同的替代实现。
#: 它们现在进 :data:`DATA_SIDE_CALLS`：**照报，但不否决**。
FORBIDDEN_CALLS: tuple[str, ...] = (
    "train_adapter",
    "produce_zoo",
    "forward_train",
    "make_optimizer",
    "clip_grad_norm_",
    "AdamW",
    "SGD",
    "backward",
    "zero_grad",
)

#: 属于训练那一侧、但**不更新权重**的调用。报告出来是为了让
#: 「流水线确实读了训练集」这件事留在记录里，而不是靠读者去推断。
DATA_SIDE_CALLS: tuple[str, ...] = ("build_train_set", "batch_order", "TrainSettings")

#: 一被**导入**就说明这一侧越了界。
FORBIDDEN_MODULES: tuple[str, ...] = ("produce",)


def scan_tokens(source: str) -> list[tuple[int, str, int]]:
    """``tokenize`` 一遍，返回 ``[(token 类型, 字符串, 行号)]``。

    **注释与字符串不进这个列表** —— 那正是要 ``tokenize`` 而不是扫文本的理由
    （文本扫描会把本函数自己的文档字符串里那些词当成违规）。
    一个 ``TokenError`` 直接抛出去：源码读不成 token 的话，
    「没有发现违规」这句话不成立，静默返回空列表会把它伪装成干净。
    """
    out: list[tuple[int, str, int]] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.NAME, tokenize.OP):
            out.append((tok.type, tok.string, tok.start[0]))
    return out


def _called_names(tree: ast.AST) -> set[str]:
    """源码里被**调用**的那些名字（``f(...)`` 与 ``x.f(...)`` 都算）。"""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _imported_modules(tree: ast.AST) -> set[str]:
    """被导入的模块名（``from . import produce`` 记作 ``produce``）。"""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                out.add(a.name)
            if node.module:
                out.add(node.module.split(".")[-1])
    return out


def wall_findings(source: str, module: str) -> list[str]:
    """扫一个模块的源码，返回违规清单（空 = 干净）。

    ``ast`` 与 ``tokenize`` 两条路都走，任一发现就报。
    """
    findings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # **解析失败必须报出来**，不能当成「没有违规」。一个解析不了的模块
        # 恰恰是最需要被看一眼的那个。
        return [f"{module}: ast 解析失败（{exc.msg} @ 行 {exc.lineno}）—— 「没扫到」不等于「干净」"]
    for name in sorted(_called_names(tree) & set(FORBIDDEN_CALLS)):
        findings.append(f"{module}: 调用了训练入口 {name}() —— 流水线不做训练")
    for name in sorted(_imported_modules(tree) & set(FORBIDDEN_MODULES)):
        findings.append(f"{module}: 导入了 {name} —— 那是架构墙的另一侧")

    # 第二路：token 层面。它抓的是 ast 看不到的写法（拼字符串造名字、
    # 或者某个未来会加进去的调用形式）。这里**只查名字是否出现**，
    # 不查它是不是调用 —— 保守一侧更安全，而误报在下面被显式容忍。
    toks = scan_tokens(source)
    names = {s for t, s, _ in toks if t == tokenize.NAME}
    for name in sorted(names & set(FORBIDDEN_CALLS)):
        # ``backward`` 会在 ``net`` 的注释里出现，但注释不进 token 流；
        # 出现在流水线模块里的 ``backward`` 无论何种用法都值得看一眼。
        if name in ("backward", "zero_grad") and not any(
            name in f for f in findings
        ):
            findings.append(f"{module}: 源码里出现标识符 {name} —— 反向传播属于另一侧")
    return findings


def data_side_findings(source: str, module: str) -> list[str]:
    """列出这个模块**调用**了哪些「训练那一侧、但不更新权重」的东西。

    这些**不算违规**（见 :data:`DATA_SIDE_CALLS`），但必须被报出来：
    ``pipeline`` 读了训练集才拟合得出 ``majority`` 基线，读者有权在报告里
    看到这件事，而不是自己从源码里推。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    hit = sorted(_called_names(tree) & set(DATA_SIDE_CALLS))
    return [f"{module}: {n}()" for n in hit]


def producer_importers(src_dir: Path | str) -> dict[str, list[str]]:
    """扫**整个包**，找出哪些模块导入了 producer。

    这一条取代了「``cli`` 也当流水线模块扫」的做法。原来的设计里 ``cli``
    被列入 :data:`PIPELINE_MODULES`，于是它**必然**报违规 —— 因为它的职责
    就是提供 ``produce`` 这个子命令。一条永远要豁免的检查不是检查。

    真正该守的不变量是**控制点唯一**：producer 只能被一个模块够得着。
    这比「谁都不许碰」更强也更有用 —— 「谁都不许碰」的极端满足方式是
    把 producer 从图里摘出去（没人导入 = 没人能用 = 检查全绿），
    而这条不变量要求它**恰好**被 :data:`CONTROL_MODULE` 导入。
    """
    src = Path(src_dir)
    out: dict[str, list[str]] = {}
    for p in sorted(src.glob("*.py")):
        mod = p.stem
        if mod in FORBIDDEN_MODULES:
            continue  # producer 自己不可能是「导入了 producer 的模块」
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        hits = sorted(_imported_modules(tree) & set(FORBIDDEN_MODULES))
        if hits:
            out[mod] = hits
    return out


@dataclass(frozen=True)
class CheckResult:
    """一条核对的结论。**``detail`` 一定要说清「差多少」**，不能只给一个布尔。"""

    name: str
    ok: bool
    detail: str
    #: 这一项是不是**验收门槛**。``False`` 表示它只报告、不否决。
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "required": self.required}


def _vs(
    value: float, limit: float, *, lower_is_ok: bool,
    vfmt: str = "{:.4f}", lfmt: str | None = None,
) -> str:
    """``值 <比较符> 阈值`` —— **比较符由实际关系现算，不是写死的**。

    写死比较符会造出这个项目里最难查的一种输出：``ok=False`` 的那一行
    自己写着 ``同向率 0.286 ≥ 0.50``。两半都「看起来合理」，合起来自相矛盾；
    读者看到 ``≥ 0.50`` 会以为这项过了，看到 ``False`` 又不知道信哪个。

    它躲过所有测试 —— 因为**通过**的输出恰好是对的（通过时关系确实是
    ``≥``），只有失败的那一次才写出假话，而失败恰恰是唯一会去读它的时刻。
    第一次真实运行（276 条组合）就撞上了。所以符号与数值同源：
    ``lower_is_ok`` 说明哪一侧算过，符号由 ``value`` 与 ``limit`` 现比现定。
    """
    if lower_is_ok:
        glyph = "≤" if value <= limit else ">"
    else:
        glyph = "≥" if value >= limit else "<"
    return f"{vfmt.format(value)} {glyph} {(lfmt or vfmt).format(limit)}"


def checks(
    result: Any,
    *,
    src_dir: Path | str,
    cfg: Mapping[str, Any],
    base_acc: Mapping[str, float],
    baselines: Mapping[str, Mapping[str, float]],
    roundtrip: Mapping[str, Any] | None = None,
) -> list[CheckResult]:
    """九项核对（第 10 项 ``artifact_roundtrip`` 只在传了 ``roundtrip`` 时出现）。
    顺序即报告里的顺序 —— 从「墙还在不在」到「质量够不够」。

    前四项是**结构性**的，没有开关：架构墙、控制点唯一、网格完整、``cid`` 规范。
    它们在任何配置下都必须成立，所以给它们一个 ``require_*`` 开关只会让
    「关掉」成为一种可选的运行方式，而那正是它们不该有的性质。

    ``result`` 是 :class:`~adapterfactory.schema.PipelineResult`（鸭子类型：
    只要有 ``runs`` / ``conflicts`` / ``adapters`` / ``capabilities``）。
    """
    acc = C.acceptance_of(cfg)
    out: list[CheckResult] = []
    src = Path(src_dir)

    # ① 架构墙
    findings: list[str] = []
    data_side: list[str] = []
    seen_modules = 0
    for mod in PIPELINE_MODULES:
        p = src / f"{mod}.py"
        if not p.is_file():
            continue
        seen_modules += 1
        source = p.read_text(encoding="utf-8")
        findings.extend(wall_findings(source, mod))
        data_side.extend(data_side_findings(source, mod))
    note = (
        f"；另有 {len(data_side)} 处属训练侧的只读调用（不否决）：" + "、".join(data_side[:3])
        if data_side
        else ""
    )
    # ``required`` 写死 ``True``，**不读配置**。
    #
    # 这里曾经是 ``acc.get("require_no_training_in_pipeline", True)`` —— 而
    # :func:`~adapterfactory.config.DEFAULTS` 里那个键默认是 ``True``，
    # 于是「能关掉」这件事从不显形，`configs/default.json` 里也就跟着写了 ``true``。
    # 症状是：本函数的 docstring 与 ``config.acceptance`` 的注释都写着「结构性四项
    # 没有开关」，而代码里有开关 —— 文档在陈述一件代码没做的事。
    # 那条注释恰好还解释了为什么不该有：一个可以关掉的架构墙，在关掉的那一刻
    # 就不再是墙；而验收会绿。
    out.append(
        CheckResult(
            "no_training_in_pipeline",
            not findings,
            (
                f"扫了 {seen_modules} 个流水线模块，未发现训练入口调用" + note
                if not findings
                else f"{len(findings)} 处违规：" + "；".join(findings[:4])
            ),
            required=True,
        )
    )

    # ② 控制点唯一：producer 只能被 cli 够得着
    #
    # 与 ① 是**互补**的两条，不是重复：① 问「流水线有没有偷偷训练」，
    # ⑨ 问「producer 有没有被谁够到」。① 全绿而 ⑨ 空的组合是完全可能的
    # （把 produce.py 删了），而那种情况下项目已经不能产出 Adapter 了。
    importers = producer_importers(src)
    ok_control = sorted(importers) == [CONTROL_MODULE]
    out.append(
        CheckResult(
            "producer_contained",
            ok_control,
            (
                f"producer 仅被 {CONTROL_MODULE} 导入（控制点唯一）"
                if ok_control
                else (
                    f"producer 的导入者是 {sorted(importers) or '（无人）'}，"
                    f"期望恰好是 ['{CONTROL_MODULE}'] —— "
                    + (
                        "多出来的每一个都说明训练代码图里可达流水线"
                        if any(m != CONTROL_MODULE for m in importers)
                        else "没有人导入它，等于它不可达"
                    )
                )
            ),
        )
    )

    # ③ 组合网格完整
    labels = [r.label for r in result.adapters]
    sizes = C.sizes_of(cfg)
    methods = C.methods_of(cfg)
    n_lab = len(labels)
    # 期望值从公式算，**不写死 276**：写死的话改了 max_size 这条检查会
    # 一直通过，而它唯一的作用就是发现那种改动。
    n_sub = sum(_comb(n_lab, k) for k in sizes)
    expect = n_sub * len(methods)
    got = len(result.runs)
    out.append(
        CheckResult(
            "grid_complete",
            got == expect,
            f"实测 {got} 条组合；公式 Σ C({n_lab},k)·|方法| = {expect}（子集 {n_sub} × 方法 {len(methods)}）",
            # 同 ①：结构性检查，没有开关。网格少一条**从来不是**一个合理的运行方式,
            # 它只意味着 `plan_grid` 与落盘的 `runs` 对不上。
            required=True,
        )
    )

    # ④ ``cid`` 唯一 + 成员已排序
    cids = [r.cid for r in result.runs]
    dup = sorted({c for c in cids if cids.count(c) > 1})
    unsorted = [r.cid for r in result.runs if list(r.members) != sorted(r.members)]
    out.append(
        CheckResult(
            "cid_canonical",
            not dup and not unsorted,
            (
                f"{len(cids)} 个 cid 全部唯一且成员已排序"
                if not dup and not unsorted
                else f"重复 cid {dup[:3]}；成员未排序的 {unsorted[:3]}"
            ),
        )
    )

    # ⑤ 基座是空模型（**两侧不对称**）
    #
    # 这条检查问的是「基座知不知道自己什么都不知道」—— 因为每个 Adapter 的
    # ``gain`` 都是相对它算的。它以前写成 ``|base − chance| ≤ slack``，
    # 而**对称的那一半是错的**，第一次真实运行就撞上了：``sort`` 上基座
    # 0.1823、chance 0.125，偏离 +0.0573 > 0.05，检查红。
    #
    # 那 0.0573 不是缺陷，是**这道题的性质**。基座是一个能看输入的因果模型，
    # 它「什么都不学」时的行为是**照着输入抄**（对齐位置那个 token），不是
    # 均匀地在 8 个位置里猜。而 ``sort`` 的输出是输入的排列 —— 随机 8 字符里
    # 平均约 1.4 个位置是**不动点**（排完还在原位），于是「照抄」在 sort 上
    # 天然拿到 ≈1.4/8 = 0.175，实测 0.1823。同一件事在 ``upper`` / ``shift`` /
    # ``reverse`` 上都接近 0（输出整体变形/位移/倒序，对齐位置几乎不可能相等），
    # 所以只有 sort 这一列显形。
    #
    # 这不是新发现：:func:`~adapterfactory.data.baselines` 的 ``copy`` 那一项
    # 早就把「照抄能拿几分」量出来了，它的 docstring 甚至逐字写着「在 sort
    # 这类『排序后常常有些位置跟原串一样』的任务上它不是 0」。把基座和
    # ``chance`` 比，是拿一个**看不见输入**的空模型去要求一个看得见输入的空模型。
    #
    # 于是两侧的含义不同，容差不能对称：
    #
    # - **下界 ``chance − slack`` 是危险的那一侧。** 基座低于机会水平 ⇒ 每个
    #   Adapter 的 gain 都被**抬高**，于是「学得动」可以是基座太差造成的。
    # - **上界取 ``heuristic + slack``，不是 ``chance + slack``。** 基座高过
    #   机会水平只说明这道题可抄；真正的上界是「最强的平凡预测器能做到几分」——
    #   基座到了那个水平就不再是空模型，而是它已经会抄了。
    #
    # 上界**不会**因此放过一个真会做事的基座：``heuristic`` 在 sort 上 0.3021、
    # 在 upper 上只有 0.10 上下 —— 越容易被平凡猜测的能力，上界才越高。
    slack = float(acc.get("base_chance_tolerance", 0.05))
    under: list[str] = []
    over: list[str] = []
    off: list[str] = []
    for c in sorted(base_acc):
        got = float(base_acc[c])
        ch = float(baselines[c]["chance"])
        hi = float(baselines[c].get("heuristic", ch))
        off.append(f"{c}{got - ch:+.4f}/上限{hi:.4f}")
        if got < ch - slack:
            under.append(f"{c} {got:.4f} < chance {ch:.4f} − {slack:.2f}")
        if got > hi + slack:
            over.append(f"{c} {got:.4f} > heuristic {hi:.4f} + {slack:.2f}")
    bad = under + over
    out.append(
        CheckResult(
            # 名字里没有 ``chance``：上界是 heuristic，写 ``base_at_chance``
            # 会让名字陈述一件检查没做的事（这一整条就是因为那个才重写的）。
            "base_is_null",
            not bad,
            (
                f"越界：{'；'.join(bad)}"
                if bad
                else f"基座落在 [chance−{slack:.2f}, heuristic+{slack:.2f}] 内：" + "  ".join(off)
            ),
            required=bool(acc.get("require_base_is_null", True)),
        )
    )

    # ⑥ 合并恒等式（**实现检查，不是能力指标**）
    errs = [r.merge_error for r in result.runs if r.merge_error is not None]
    worst_me = max(errs, default=None)
    limit = float(acc["max_merge_error"])
    out.append(
        CheckResult(
            "merge_identity",
            worst_me is not None and worst_me <= limit,
            (
                f"{len(errs)} 条走权重空间的组合，最大合并误差 "
                f"{_vs(worst_me, limit, lower_is_ok=True, vfmt='{:.3e}', lfmt='{:.0e}')}"
                if worst_me is not None
                else "没有任何走权重空间的组合 —— 这条检查无从执行"
            ),
        )
    )

    # ⑦ 单 Adapter 增益门槛（相对**更强的**启发式基线）
    margin = float(acc["min_single_gain_margin"])
    singles = [r for r in result.runs if len(r.members) == 1 and r.method == "stack"]
    if not singles:
        out.append(
            CheckResult(
                "single_gain_margin",
                False,
                "没有 size-1 的 stack 组合，无法核对增益门槛",
                required=bool(acc.get("require_single_gain_margin", True)),
            )
        )
    else:
        rows = []
        bad = []
        for r in singles:
            cap = _capability_of(result, r.members[0])
            m = r.metrics.get(cap)
            if m is None:
                bad.append(f"{r.members[0]}: 缺 {cap} 的实测")
                continue
            need = baselines[cap]["heuristic"]
            gap = m.gain - need
            rows.append(f"{r.members[0]} {gap:+.3f}")
            if gap < margin:
                bad.append(f"{r.members[0]} 差 {gap:+.3f} < {margin}")
        out.append(
            CheckResult(
                "single_gain_margin",
                not bad,
                (
                    f"{len(rows)} 条都超过 heuristic + {margin:.2f}：" + "  ".join(rows)
                    if not bad
                    else f"{len(bad)} 条未过：" + "；".join(bad[:3])
                ),
                required=bool(acc.get("require_single_gain_margin", True)),
            )
        )

    # ⑧ 同能力对次可加
    worst_sub = None
    worst_pair = ""
    for r in result.runs:
        if r.method != "stack" or len(r.members) != 2:
            continue
        caps = {_capability_of(result, mem) for mem in r.members}
        if len(caps) != 1:
            continue
        cap = next(iter(caps))
        by = {mem: _run_of(result, mem, "stack") for mem in r.members}
        sub = _subadditivity(result, r, by, cap)
        if sub is None:
            continue
        if worst_sub is None or sub < worst_sub:
            worst_sub, worst_pair = sub, "+".join(r.members)
    floor = float(acc["min_same_cap_subadditivity"])
    out.append(
        CheckResult(
            "same_cap_subadditive",
            worst_sub is not None and worst_sub >= floor,
            (
                f"最差的一对 {worst_pair} 次可加量 "
                f"{_vs(worst_sub, floor, lower_is_ok=False, vfmt='{:+.4f}', lfmt='{:.2f}')}"
                if worst_sub is not None
                else "zoo 里没有同能力的 size-2 组合 —— 这条检查无从执行"
            ),
            required=bool(acc.get("require_same_cap_subadditivity", True)),
        )
    )

    # ⑨ 两路同向率
    need = float(acc["min_two_path_agreement"])
    rated = [c for c in result.conflicts if c.agree is not None]
    rate = (sum(1 for c in rated if c.agree) / len(rated)) if rated else None
    out.append(
        CheckResult(
            "two_path_agreement",
            rate is not None and rate >= need,
            (
                f"{len(rated)}/{len(result.conflicts)} 对可预测，同向 {sum(1 for c in rated if c.agree)} 条，"
                f"同向率 {_vs(rate, need, lower_is_ok=False, vfmt='{:.3f}', lfmt='{:.2f}')}"
                if rate is not None
                else "没有一对能给出参数空间预测 —— 这条检查无从执行"
            ),
            required=False,  # 见 run_checks 的说明：它是**发现**，不是门槛
        )
    )

    if roundtrip is not None:
        tol = float(acc["roundtrip_tolerance"])
        okval = bool(roundtrip.get("ok"))
        out.append(
            CheckResult(
                "artifact_roundtrip",
                okval,
                str(roundtrip.get("detail", "")),
                required=False,
            )
        )
        del tol
    return out


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def _capability_of(result: Any, label: str) -> str:
    for a in result.adapters:
        if a.label == label:
            return a.capability
    raise KeyError(f"zoo 里没有 {label!r}")


def _run_of(result: Any, label: str, method: str) -> Any:
    for r in result.runs:
        if r.method == method and list(r.members) == [label]:
            return r
    raise KeyError(f"找不到 {label} 的 {method} 单干组合")


def _subadditivity(result: Any, combo: Any, by: Mapping[str, Any], cap: str) -> float | None:
    vals = []
    for mem in combo.members:
        m = by[mem].metrics.get(cap)
        if m is None:
            return None
        vals.append(m.gain)
    c = combo.metrics.get(cap)
    if c is None:
        return None
    return sum(vals) - c.gain


def run_checks(**kwargs: Any) -> list[CheckResult]:
    """跑全部核对。**不过门槛就抛错**，而不是返回一个带 ``False`` 的列表。

    ``required=False`` 的条目（两路同向率、产物往返）**不否决**。理由对这两条
    各自不同，而且都必须写下来，否则它们看起来像「为了让验收通过而豁免的」：

    - **两路同向率**：它是本项目的一个**发现**，不是质量指标。参数空间预测
      被实测推翻（同向率 1/6）正是报告里最有价值的结论。把它设成门槛，
      等于宣布「除非预测器好用，否则这次运行不算数」—— 而一个不好用的
      预测器照样是结论。它仍然每次都算、每次都进报告。
    - **产物往返**：它依赖 ``compositions/`` 里的文件，而 ``--no-export``
      之类的只读路径不产出它们。缺文件时它返回 ``False`` 会污染验收。
    """
    got = checks(**kwargs)
    bad = [c for c in got if c.required and not c.ok]
    if bad:
        raise AssertionError(
            "运行时核对未通过：\n  - " + "\n  - ".join(f"{c.name}: {c.detail}" for c in bad)
        )
    return got


def acceptance_report(
    results: Sequence[CheckResult],
    *,
    n_runs: int,
    n_conflicts: int,
    conflict_edges_harmful: int,
) -> list[dict[str, Any]]:
    """把核对结果 + 两条验收标准拼成报告里的那张表。"""
    rows = [c.to_dict() for c in results]
    rows.append(
        {
            "name": "acceptance_matrix",
            "ok": n_runs > 0,
            "detail": f"验收①：组合矩阵 {n_runs} 行，全指标落盘（matrix.csv + runs.jsonl）",
            "required": True,
        }
    )
    rows.append(
        {
            "name": "acceptance_graph",
            "ok": n_conflicts > 0,
            "detail": (
                f"验收②：Adapter Graph 有 {n_conflicts} 条冲突边，"
                f"其中 {conflict_edges_harmful} 条实测有害（带双路证据）"
            ),
            "required": True,
        }
    )
    return rows
