"""端到端编排：**把一整套 adapterfactory 的结论跑出来并落盘。**

### 两条通路，必须分开写

- :func:`run_pipeline` 是**写命令**的路径（``main.py run``）：它跑前向、落产物。
- :func:`load_result` / :func:`load_full` 是**只读命令**的路径：从 ``out/`` 复原。
  只读命令**绝不重新训练、也绝不重新前向** —— 否则「只读不变性」这条验收
  就变成一句空话（每次都重算，产物当然"一样"，因为根本没用它）。

### ``--no-export`` 与 ``roundtrip`` 的关系

导出组合产物（``compositions/<cid>.bin|.json``）**不是**评测的必要条件：
组合在内存里已经算完了。它是为了回答另一个问题 ——「组合完的东西还是不是
一个合法的 LoRA」。这件事只能用**一次真实的写入 + 读回**来回答。

所以 ``export=False`` 时那条 :func:`~adapterfactory.audit.checks` 里的
``artifact_roundtrip`` 会**缺席而不是失败**（``run_checks`` 把它标成
``required=False``）。让它失败等于说「只读路径因为没写文件所以不合格」。

### ``wall_ms`` 为什么不进主前沿

``wall_ms`` 是真实测量，但它**不是可复现量**（坑 25/33）：同一份产物两次运行
会差出百分之几。于是「两次运行逐位相同」这条验收**永远过不了**，
而绕开它的常见做法是把它从比对里悄悄排除 —— 那等于承认产物不可复现。

本项目的做法是把主前沿的延迟轴换成 :func:`~adapterfactory.evaluate.macs_of`
（MACs，是**被测量那一侧**的函数，与机器无关），``wall_ms`` 照样落盘、
照样进报告，但**明确标注它带一条不可复现轴**。"""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import audit as A
from . import compose as P
from . import conflict as K
from . import config as C
from . import data as D
from . import evaluate as E
from . import graph as G
from . import lineage as L
from . import net as N
from . import registry as R
from . import zoo as Z
from .schema import (
    AdapterRecord,
    ArtifactRef,
    CapabilityMetric,
    CompositionRun,
    ConflictEdge,
    Finding,
    PipelineResult,
    Version,
)

__all__ = [
    "OUT_FILES",
    "SOURCE_FILES",
    "DERIVED_FILES",
    "build_shape",
    "prepare",
    "run_pipeline",
    "write_outputs",
    "load_result",
    "load_full",
    "read_runs",
    "read_conflict",
    "matrix_rows",
    "method_macs",
    "compositions_dir",
]


#: 一次 ``run`` 落下的**源**文件（相对于 ``out/``）：结论的来源。
#: **缺任一个都读不出东西**，所以 :func:`load_result` 按这张表判，
#: 而不是「读得到就好」。
SOURCE_FILES: tuple[str, ...] = (
    "result.json",
    "runs.jsonl",
    "matrix.csv",
    "conflict.json",
    "registry.json",
    "graph.json",
    "graph.dot",
    "lineage.json",
    "lineage.dot",
    "version.json",
)

#: 由源文件**派生**的两份。它们必须是纯函数 —— ``main.py report`` 会拿现生成的
#: 文本与磁盘上的那份逐字节比，这条断言成立的**全部理由**就是这里没有额外输入。
#:
#: ⚠️ **它们不参与「产物是否完整」的判定。** 曾经它们和源文件一起放在
#: ``OUT_FILES`` 里，于是删掉 ``out/report.html``（或者只把它写坏了）之后，
#: 每一条只读命令都开始说「还没有产物，先跑 ``run``」—— **包括 ``visualize``**，
#: 而它的全部工作就是重出 ``report.html``。补救手段被它要补救的那个条件挡住了，
#: 唯一的出路是重跑整条 276 个组合的流水线，只为了重画一张 HTML。
#:
#: 判据是「这份文件是不是别的东西的函数」：是，就不该拿它当读产物**的前提**。
#: 完整性仍然被查 —— ``run`` 之后一张 ``test_pipeline_e2e`` 的断言盯的是
#: ``OUT_FILES``（源 + 派生），只是那道门槛在写侧，不在读侧。
DERIVED_FILES: tuple[str, ...] = ("report.md", "report.html")

#: ``run`` 应当落下的全部文件。**写侧**的完整性按它查。
OUT_FILES: tuple[str, ...] = SOURCE_FILES + DERIVED_FILES


def out_dir_of(root: str | Path) -> Path:
    return Path(root) / "out"


def compositions_dir(out: str | Path) -> Path:
    """组合产物的目录 —— **参数就是产物目录本身**（``<root>/out``），不是项目根。

    这里刻意不做「给根也行、给 out 也行」的兼容：那种兼容的判据只能靠路径名字
    （``p.name == "out"``），而一个恰好叫 ``out`` 的项目根会让它静默地指向
    ``out/out/compositions``。多一层目录不会让任何代码报错，
    它只会让所有读取方都**读到空目录**。
    """
    return Path(out) / "compositions"


# ----------------------------------------------------------------------
# 准备
# ----------------------------------------------------------------------
def build_shape(cfg: Mapping[str, Any]) -> N.ModelShape:
    """从配置造基座形状。**``seq_len`` 由数据索引推出来，配置只是把它钉进指纹。**

    这里不重新推导一遍 ``5 + 2L``（那是 :func:`~adapterfactory.data.seq_len_of`
    的职责）：两处各推一次的话，改了编码布局只会让其中一处跟着变。
    """
    b = cfg["base"]
    return N.ModelShape(
        vocab=int(b["vocab"]),
        d_model=int(b["d_model"]),
        n_layers=int(b["n_layers"]),
        n_heads=int(b["n_heads"]),
        d_ff=int(b["d_ff"]),
        max_len=int(b["seq_len"]),
    )


def prepare(*, root: str | Path = ".", cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把「跑之前要知道的一切」一次性备齐。

    单独拆出来，是因为 **``task`` / ``baselines`` / ``registry`` 这些只读命令
    需要的东西是它的子集** —— 让它们各自重写一遍载入逻辑，等于让
    「基座是什么」有多个来源。
    """
    root = Path(root)
    conf = dict(cfg) if cfg is not None else C.load_config(root=root)
    idx = D.load_index(C.resolve_path(root, conf["data"]["index"]))
    vocab = D.build_vocab(idx)
    shape = build_shape(conf)
    seq_len = D.seq_len_of(idx)
    if int(conf["base"]["seq_len"]) != seq_len:
        # 配置里的 seq_len 与索引推出来的不一致：**报错而不是取其一**。
        # 取配置的话，改了 payload_len 会让基座 max_len 不动、编码却更长，
        # 症状是「多了一截被截掉的 PAD」——不报错、不影响任何数值。
        raise ValueError(
            f"配置 base.seq_len={conf['base']['seq_len']} 与索引推出的 {seq_len} 不符"
            f"（= 5 + 2×payload_len）：改 payload_len 时必须同时改配置"
        )
    caps = C.capabilities_of(conf)
    base_weights = N.init_base_weights(shape, int(conf["base"]["seed"]))
    eval_sets = D.build_eval_sets(
        idx,
        vocab,
        seq_len=seq_len,
        n_eval=int(conf["data"]["n_eval"]),
        seed=int(conf["data"]["seed"]),
        capabilities=caps,
    )
    train_sets = D.build_train_set(
        idx,
        vocab,
        seq_len=seq_len,
        n_train=int(conf["data"]["n_train"]),
        seed=int(conf["data"]["seed"]),
        capabilities=caps,
        # **流水线也要排除评测 payload。** 这里的训练集不为训练服务 ——
        # 它只用来拟合 ``majority`` 基线 —— 所以曾经看起来「排不排无所谓」。
        # 不是的：``majority`` 是逐位置取训练集里的众数，若训练集里混着评测
        # payload，那条基线就偷看了评测集，于是验收盯着的那个上界被抬高，
        # 而抬高的方向恰好是「让我们的 Adapter 更容易看起来达标」。
        exclude=D.excluded_payloads(eval_sets),
    )
    baselines = D.baselines(idx, train_sets, eval_sets)
    return {
        "root": root,
        "cfg": conf,
        "idx": idx,
        "vocab": vocab,
        "shape": shape,
        "seq_len": seq_len,
        "capabilities": caps,
        "base_weights": base_weights,
        "eval_sets": eval_sets,
        "train_sets": train_sets,
        "baselines": baselines,
        "base_key": shape.key,
        "base_seed": int(conf["base"]["seed"]),
    }


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def run_pipeline(
    *,
    root: str | Path = ".",
    cfg: Mapping[str, Any] | None = None,
    out: str | Path | None = None,
    export: bool = True,
    progress: Any = None,
) -> PipelineResult:
    """跑完整条流水线并**落盘**，返回 :class:`~adapterfactory.schema.PipelineResult`。

    ``progress`` 是一个可选的 ``callable(done, total, note)``：训练/评测要跑几分钟，
    一个不说话的脚本与一个卡死的脚本从外面看是一样的。

    ``out`` 默认 ``<root>/out``。它可覆盖不是为了方便，而是为了让
    **「两次运行逐位相同」这条验收可执行** —— 两次都写同一个目录的话，
    第二次覆盖第一次，比对的对象只剩下一份，那条断言就成了同义反复。
    """
    t0 = time.perf_counter()
    ctx = prepare(root=root, cfg=cfg)
    conf = ctx["cfg"]
    shape: N.ModelShape = ctx["shape"]
    caps: list[str] = ctx["capabilities"]
    weights = ctx["base_weights"]
    eval_sets = ctx["eval_sets"]
    base_weights = ctx["base_weights"]
    root = Path(ctx["root"])

    reg = R.load_registry(root, conf, shape)
    labels = reg.labels()
    methods = C.methods_of(conf)
    sizes = C.sizes_of(conf)
    plans = P.plan_grid(labels, methods, sizes)
    temperature = float(C.route_of(conf)["temperature"])

    # 门向量一次性派生：它只依赖 (shape, 标签, 种子)，**不依赖组合** ——
    # 于是「route 的门在两次运行里是不是同一个」这件事不需要任何额外约定。
    gates_all = N.derive_gates(shape, labels, seed=ctx["base_seed"])

    # 用一个固定输入量 MACs：全部能力的 payload 长度相同（都是 4），
    # 所以这一个输入在四个能力上是同一个 T。**换输入会换 MACs**，
    # 而 MACs 是用来横向比较方法的，取哪一条只要求「所有组合取同一条」。
    mac_ids = list(eval_sets[caps[0]][0].ids)

    def note(done: int, total: int, msg: str) -> None:
        if progress is not None:
            progress(done, total, msg)

    # ---- 基座 ----
    base_adj = E.measure(P.realize_base(weights, shape), eval_sets, shape, caps)
    base_macs = E.macs_of(P.realize_base(weights, shape), mac_ids, shape)

    # ---- 第一遍：实例化 + 评测，**全部存下来** ----
    # 两遍而不是一遍：``merge_loss`` 需要同一个子集的 ``stack`` 结果，
    # 而「plan_grid 恰好把 stack 排在 merge 前面」是一条**隐式约定** ——
    # 依赖它的话，改了方法顺序会让 merge_loss 静默变成 None。
    adj: dict[str, E.Adjudication] = {}
    realized_of: dict[str, P.Realized] = {}
    wall_ms: dict[str, float] = {}
    total = len(plans)
    for i, plan in enumerate(plans, 1):
        ads = reg.subset(plan.members)
        gates = {m: gates_all[m] for m in plan.members} if plan.method == "route" else None
        rz = P.realize(weights, ads, plan, shape, gates=gates, temperature=temperature)
        realized_of[plan.cid] = rz
        t1 = time.perf_counter()
        adj[plan.cid] = E.measure(rz, eval_sets, shape, caps)
        # **这里量的是「评测这一轮花了多久」，不是「部署时多慢」。**
        # 它落盘只为「这台机器上大概多贵」有个数；主前沿的延迟轴是 MACs。
        wall_ms[plan.cid] = (time.perf_counter() - t1) * 1000.0
        note(i, total, plan.cid)

    # ---- 第二遍：算每条的派生量、组装记录 ----
    base_shapes = N.param_shapes(shape)
    stack_adj: dict[tuple[str, ...], E.Adjudication] = {
        tuple(plan.members): adj[plan.cid] for plan in plans if plan.method == "stack"
    }

    runs: list[CompositionRun] = []
    artifacts: dict[str, ArtifactRef] = {}
    target = Path(out) if out is not None else out_dir_of(root)
    cdir = compositions_dir(target)
    if export:
        cdir.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        rz = realized_of[plan.cid]
        a = adj[plan.cid]
        coefs = P.coefficients(plan.method, len(plan.members))

        # 改动了多少个**基座元素**：以挂点并集算，不是以适配器参数量算。
        # 两者差得很远（qv 有 768 个自己的参数，却改动了 2304 个基座元素），
        # 而 parameter_cost 问的是「部署时要重算多少」。
        if plan.method == "route":
            n_changed = 0  # route 不动基座一个数 —— 这正是它多花 MACs 换来的
        else:
            touched = {t for m in plan.members for t in reg.record(m).targets}
            n_changed = sum(
                N._numel(base_shapes[full]) for full in N.expand_targets(base_shapes, sorted(touched))
            )

        m_loss: float | None = None
        if plan.method == "merge":
            st = stack_adj.get(tuple(plan.members))
            vals = [E.merge_loss(st, a, c) for c in caps]
            vals = [v for v in vals if v is not None]
            m_loss = sum(vals) / len(vals) if vals else None

        art: ArtifactRef | None = None
        if export and coefs is not None:
            ads = reg.subset(plan.members)
            targets = _union_targets(ads)
            tensors, rank_total = Z.compose_factors(ads, coefs, shape)
            stem = cdir / plan.cid
            rec = Z.write_artifact(
                stem,
                tensors,
                name=plan.cid,
                # 挂点取**并集**（按基座参数的落盘顺序），不是全部可挂模块：
                # 写全部 7 个会让一份只动了 q/v 的组合在旁证里声称自己动了 7 个模块 ——
                # 而旁证是下游唯一的身份来源。
                targets=targets,
                rank=rank_total,
                # 系数已经折进 ``B`` 了，所以 scale 恰好是 1.0（alpha == rank）。
                # 与 :func:`~adapterfactory.produce.produce_zoo` 的写法一致。
                alpha=float(rank_total),
                capability="",
                variant="",
                base_key=shape.key,
                base_seed=ctx["base_seed"],
                source="compose",
                composed_of=list(plan.members),
            )
            art = ArtifactRef(
                digest=rec.digest,
                n_bytes=rec.n_bytes,
                rank_total=int(rank_total),
                n_floats=int(rec.n_params),
                path=str(Path(rec.path_bin)),
            )
            artifacts[plan.cid] = art

        metrics: dict[str, CapabilityMetric] = {}
        base_acc = {c: base_adj.acc(c) for c in caps}
        gain_of = {c: (a.acc(c) - base_adj.acc(c)) for c in caps}
        for cap in caps:
            m = a.measures.get(cap)
            if m is None:
                continue
            metrics[cap] = CapabilityMetric(
                capability=cap,
                acc=m.acc,
                gain=gain_of[cap],
                exact=m.exact,
                eos_acc=m.eos_acc,
                n_positions=m.n_positions,
                # 干扰是「成员各自单干时的增益之和」减去「组合的增益」。
                # 这里存的是**被减数**（成员单干之和），而 ``size-1`` 时是 ``None``
                # ——存一个等于自身 gain 的数会让「这份组合的干扰」在 size-1 上
                # 看起来是 0，而不是「这个问题在 size-1 上不存在」。
                interference_from=(
                    None if len(plan.members) < 2 else _solo_gain_sum(adj, plan, cap, base_acc)
                ),
            )

        runs.append(
            CompositionRun(
                cid=plan.cid,
                method=plan.method,
                members=list(plan.members),
                metrics=metrics,
                # **route 上必须是 ``None``，不能填一个数。**
                #
                # 这里以前是无条件算的，于是 route 那一行会拿到一个**和 stack 行
                # 一模一样**的数 —— 因为 ``merge_identity_error`` 比的是「把成员
                # 的 delta 合进权重再前向」与「不合、逐个施加」，而这两件事
                # route **都没做**：它不动权重，逐层按门混合激活。
                #
                # 于是那个数在 route 行上不只是「没用」，它是**另一个方法的结论
                # 被复制到了这一行**。而它的取值（~1e-16）完全正常，所以
                # ``merge_identity`` 那道核对一路绿，下游任何对 ``merge_error``
                # 求平均/求相关的聚合都会静默地多算一遍 stack。
                #
                # ``None`` 在这个项目里是有定义的：**「这一项不适用」**（见
                # schema 模块 docstring 的三种「没有值」）。cli 与 audit 早就
                # 按这个定义写好了分支 —— 是这里没有兑现。
                merge_error=(
                    None if plan.method == "route"
                    else E.merge_identity_error(weights, reg.subset(plan.members), mac_ids, shape)
                ),
                merge_loss=m_loss,
                n_params_changed=n_changed,
                macs=E.macs_of(rz, mac_ids, shape),
                wall_ms=wall_ms[plan.cid],
                artifact=art,
                coefs=None if coefs is None else list(coefs),
            )
        )

    # ---- 冲突：两两，固定用 stack ----
    threshold = 1.0 / float(conf["data"]["n_eval"])
    solo = {lab: adj[P.cid_of("stack", (lab,))] for lab in labels}
    combos = {P.cid_of("stack", (a, b)): adj[P.cid_of("stack", (a, b))]
              for i, a in enumerate(labels) for b in labels[i + 1:]}
    conflicts, preds = K.all_pairs(
        reg.records,
        solo,
        combos,
        base_adj,
        weights,
        shape,
        reg.adapters,
        capabilities=caps,
        threshold=threshold,
    )

    # ---- 前沿 ----
    # 只拿**全部**组合里互不支配的那些。三个轴：平均增益 ↑、基座改动 ↓、MACs ↓。
    # ``merge`` 不额外排除：它确实更差（同子集、同 MACs、同基座改动），
    # 于是会被同子集的 ``stack`` 支配掉，不需要一条「把 merge 拿掉」的特例 ——
    # 特例会让「merge 有没有可能更好」这个**可证伪的问题**失去被回答的机会。
    points = {}
    for r in runs:
        m = r.mean_gain(caps)
        if m is None:
            # 有能力的实测缺失就不进前沿。用 0.0 补位等于把一个**缺失量**
            # 当成「平均增益为 0」参与支配比较 —— 而它多半会把一个真实点挤掉。
            continue
        points[r.cid] = [m, -float(r.n_params_changed), -float(r.macs)]
    frontier = G.pareto_frontier(points, maximize=[True, True, True])

    # ---- 血缘 ----
    version = _make_version(conf, ctx, reg, n_runs=len(runs), n_conflicts=len(conflicts), t0=t0)
    result = PipelineResult(
        version=version,
        base_key=ctx["base_key"],
        base_seed=ctx["base_seed"],
        seq_len=ctx["seq_len"],
        capabilities=list(caps),
        adapters=list(reg.records),
        runs=runs,
        conflicts=conflicts,
        frontier=frontier,
        counters={},
        findings=[],
        notes={},
    )

    # ---- 报告用的统计 ----
    graph = L.build_lineage(result.to_dict())
    entity = G.graph_summary(
        graph,
        conflicts,
        frontier,
        n_adapters=len(labels),
        threshold=threshold,
    )
    predictor_rows = K.predictor_table(preds, conflicts)
    stats = predictor_rows[-1]["_predictor_stats"]

    result.findings = _derive_findings(
        runs=runs, conflicts=conflicts, frontier=frontier, caps=caps,
        adj=adj, baselines=ctx["baselines"], labels=labels,
        base_macs=base_macs, stats=stats,
    )
    result.counters = {
        "n_adapters": len(labels),
        "n_methods": len(methods),
        "n_subsets": sum(_comb(len(labels), k) for k in sizes),
        "n_runs": len(runs),
        "n_conflicts": len(conflicts),
        "n_conflicts_harmful": len(G.harmful_edges(conflicts, threshold=threshold)),
        "n_frontier": len(frontier),
        "n_exported": len(artifacts),
        "base_macs": base_macs,
        "macs_stack": _mac_of(runs, "stack"),
        "macs_merge": _mac_of(runs, "merge"),
        "macs_route": _mac_of(runs, "route"),
        "conflict_resolution": threshold,
        "graph": entity,
        # 逐对明细与相关系数**都进 result.json**，于是 ``report`` 这个只读命令
        # 不必再去读 conflict.json —— 否则「报告里的数」与「conflict.json 里的数」
        # 就成了两份可以各自漂移的副本。conflict.json 仍然独立落盘（它是
        # 「一次运行的全部证据」那份产物，而 result.json 是摘要）。
        "predictor_stats": stats,
        "predictor_rows": [r for r in predictor_rows if "_predictor_stats" not in r],
        "runtime_s": round(time.perf_counter() - t0, 1),
        "checks": [],
    }
    result.notes = {
        "route_gate": (
            "route 的门是 softmax(⟨h, uᵢ⟩/T)，uᵢ 是**种子派生的固定随机向量，不训练** —— "
            "它是结构对照，不是学出来的路由。把它当成 MoE 那样的学出来的门会让 "
            "「路由落在 stack 与 merge 之间」这个结论失去半边含义。"
        ),
        "merge_dilution": (
            "merge 的系数 1/k 是**故意**的稀释，不是干扰。所以冲突边固定用 stack 算，"
            "merge 的能力损失由 merge_loss 那一列单独承担。"
        ),
        "latency": (
            "macs 是可复现的延迟代理（被测量那一侧的函数）；wall_ms 照样落盘，"
            "但它带一条不可复现轴，**只进副前沿**。"
        ),
        "frontier_axes": "平均增益 ↑（全能力）、基座改动元素数 ↓、MACs ↓；三者由 graph.pareto_frontier 统一方向。",
    }

    # ---- 架构墙 + 九项核对 ----
    roundtrip = _roundtrip_check(target, runs) if export else None
    checks = A.run_checks(
        result=result,
        src_dir=Path(__file__).resolve().parent,
        cfg=conf,
        base_acc={c: base_adj.acc(c) for c in caps},
        baselines=ctx["baselines"],
        roundtrip=roundtrip,
    )
    counters = dict(result.counters)
    counters["checks"] = [c.to_dict() for c in checks]
    counters["acceptance"] = A.acceptance_report(
        checks,
        n_runs=len(runs),
        n_conflicts=len(conflicts),
        conflict_edges_harmful=int(counters["n_conflicts_harmful"]),
    )
    result.counters = counters

    write_outputs(
        result,
        target,
        cfg=conf,
        ctx=ctx,
        graph=graph,
        conflicts=conflicts,
        predictor_rows=predictor_rows,
        threshold=threshold,
    )
    return result


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def _solo_gain_sum(
    adj: Mapping[str, E.Adjudication],
    plan: P.Plan,
    cap: str,
    base_acc: Mapping[str, float],
) -> float | None:
    """``Σ 成员单干时在该能力上的增益``。

    **成员单干的成绩直接取 size-1 的 ``stack`` 组合**，不另算一遍 ——
    size-1 的 ``stack`` 系数是 ``[1.0]``，它按定义就是那个 Adapter 本身。
    另算一遍的话，两条路径（组合矩阵里的那一行 vs 这里）会在
    某次改动之后给出不同的数，而报告里两个数字都出现了。

    ``gain`` 在这里**现算**（``acc − 基座 acc``），不读
    :class:`~adapterfactory.schema.CapabilityMetric` 的 ``gain``：
    那个字段是这一轮才填进 ``runs`` 的，去读它等于让
    「正在组装的这条记录」依赖「已经组装好的那条记录」的顺序。
    """
    vals = []
    for m in plan.members:
        s = adj.get(P.cid_of("stack", (m,)))
        if s is None or cap not in s.measures or cap not in base_acc:
            return None
        vals.append(s.measures[cap].acc - base_acc[cap])
    return sum(vals) if vals else None


def _union_targets(adapters: Sequence[N.Adapter]) -> list[str]:
    """成员挂点的并集，**按 :data:`~adapterfactory.net.LORA_ABLE_MODULES` 的顺序**。

    顺序即产物字节顺序（见 :func:`~adapterfactory.zoo.compose_factors`），
    用 ``set`` 迭代序会让同一份组合在不同进程里长出不同的字节 —— 而
    ``digest`` 是它的函数，于是「内容寻址」会变成「进程寻址」。
    """
    got = {t for a in adapters for t in a.targets}
    return [t for t in N.LORA_ABLE_MODULES if t in got]


def method_macs(runs: Sequence[CompositionRun], method: str) -> tuple[int, ...]:
    """某一方法下 size-1 组合的 MACs，**去重后升序**。

    取 size-1 而不是平均：平均会把「组合了 3 个」这个因素混进来，
    而这里要回答的是「这个方法本身贵多少」。

    **为什么返回一列而不是一个数。** stack / merge 确实只有一个取值
    （逐位等于基座 —— 往权重里加东西不改变前向的乘加次数）。但 **route 不是
    一个取值**：它要为每个成员多做一次前向，于是成本取决于成员挂了几个模块 ——
    qv 变体挂 2 个、all 变体挂 7 个。返回一个数，就必须在「第一个」「最后一个」
    「最小的」里挑一个，而挑出来的那个数**印出来仍然像是「方法的固定开销」**。

    实测就是这么裂开的：`counters.macs_route` 与 `report.md` 里那条
    `method_macs` finding 是同一个量的两次独立推导，差了 1.59 倍
    （347360 vs 218816）—— 一处取**首个**匹配的 size-1 组合，另一处被
    dict 推导式留下了**最后一个**。两个数都印在报告里，都叫「route 的开销」，
    而且各自都「看起来合理」。所以这里返回全部取值，由调用方决定怎么呈现；
    凡是「同一个量在两处各推一遍」的地方，最终一定会在某一处对不上。
    """
    return tuple(sorted({int(r.macs) for r in runs if r.method == method and len(r.members) == 1}))


def _mac_of(runs: Sequence[CompositionRun], method: str) -> int | None:
    """某一方法 size-1 开销的**最坏情况**（见 :func:`method_macs`）。

    只留一个数是因为 ``counters`` 要进 result.json，而那里每项是一个标量。
    取最大而不是随便挑一个：报告里那行 route 的代价是**上界**，
    与图里那条「多花 1.86 倍算力」的说法同向，不会低报。
    """
    vals = method_macs(runs, method)
    return vals[-1] if vals else None


def _make_version(
    conf: Mapping[str, Any],
    ctx: Mapping[str, Any],
    reg: R.Registry,
    *,
    n_runs: int,
    n_conflicts: int,
    t0: float,
) -> Version:
    """内容寻址版本。``inputs`` 是**被量的东西**：zoo 与数据索引。

    两次 ``run`` 之间，``created_at`` 之外的一切都必须逐位相同。
    把 zoo 摘要与索引摘要都并进去，于是「换了 Adapter」与「换了题目」
    都会改版本号，而它们正是仅有的两种能让结论变质的改动。
    """
    inputs = [
        ("zoo", {"labels": reg.labels(), "digest": reg.digest, "base": reg.base_key}),
        ("data", {"index": ctx["idx"].to_dict() if hasattr(ctx["idx"], "to_dict") else str(ctx["idx"])}),
        ("seq_len", ctx["seq_len"]),
    ]
    vdigest = C.version_digest(conf, inputs)
    del t0
    return Version(
        version=C.short_version(vdigest),
        protocol_digest=C.protocol_digest(conf),
        version_digest=vdigest,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_adapters=len(reg.records),
        n_runs=n_runs,
        n_conflicts=n_conflicts,
    )


def _roundtrip_check(out: Path, runs: Sequence[CompositionRun]) -> dict[str, Any]:
    """把导出的组合产物**读回来**，逐位比对。

    这是「组合完的东西还是一个合法 LoRA」这句话的**唯一**证据来源。
    ``stack``/``merge`` 各抽一条 —— 全量读回没有必要（写法是同一条），
    而两条一起覆盖了「系数全 1」与「系数 1/k」这两条折系数的路径。
    """
    from . import rng as RNG

    picked: dict[str, str] = {}
    for r in runs:
        if r.artifact is not None and r.method not in picked:
            picked[r.method] = r.cid
    if not picked:
        return {"ok": False, "detail": "没有可读回的导出产物"}
    for cid in picked.values():
        stem = compositions_dir(out) / cid
        payload = Path(str(stem) + ".bin").read_bytes()
        # ``verify=True`` 会顺带核对 sha256 —— 于是「读得回来」与
        # 「读回来的正是写下去的那一份」不是两件事。
        _tensors, rec = Z.read_artifact(stem, verify=True)
        if RNG.sha256_bytes(payload) != rec.digest:
            return {"ok": False, "detail": f"{cid}: 读回的 sha256 与旁证的 digest 不符"}
    return {
        "ok": True,
        "detail": "读回 " + "、".join(f"{m}:{c}" for m, c in sorted(picked.items()))
        + "，字节 sha256 与旁证 digest 逐条相符",
    }


# ----------------------------------------------------------------------
# 发现（Finding）
# ----------------------------------------------------------------------
def _derive_findings(
    *,
    runs: Sequence[CompositionRun],
    conflicts: Sequence[ConflictEdge],
    frontier: Sequence[str],
    caps: Sequence[str],
    adj: Mapping[str, E.Adjudication],
    baselines: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    base_macs: int,
    stats: Sequence[Mapping[str, Any]],
) -> list[Finding]:
    """把「设计期望 vs 实测」写成条目。**被推翻的那些才是报告的价值。**

    每一条都从**已经落盘的数**里算出来，不重新跑任何前向 —— 于是
    ``findings`` 与 ``runs.jsonl`` 不可能互相矛盾。
    """
    out: list[Finding] = []
    rated = [c for c in conflicts if c.agree is not None]
    n_agree = sum(1 for c in rated if c.agree)
    rate = (n_agree / len(rated)) if rated else None

    if rate is not None:
        out.append(
            Finding(
                key="two_path_disagreement",
                statement="参数空间预测（cos 与符号一致率）能不能预测实测干扰",
                expectation="设计期望：cos < 0 的那些对应当更受损，两路同向率应 ≳ 0.5",
                measured=f"{n_agree}/{len(rated)} 对同向（同向率 {rate:.3f}）",
                verdict="预测被推翻" if rate < 0.5 else "预测成立",
                note=(
                    "delta 之间几乎正交，符号一致率贴着 0.50（随机）；而实测干扰每一对都接近满值。"
                    "「夹角」不是决定干扰的量 —— 这正是不预设哪个预测器对、"
                    "把 cos / expansion / overlap 一起报出来的理由。"
                ),
            )
        )

    # 同能力一对：是次可加，还是掉到基座以下
    worst = None
    for r in runs:
        if r.method != "stack" or len(r.members) != 2:
            continue
        cap = _common_capability(r.members)
        if cap is None or cap not in r.metrics:
            continue
        g = r.metrics[cap].gain
        if worst is None or g < worst[1]:
            worst = ("+".join(r.members), g, cap)
    if worst is not None:
        name, g, cap = worst
        out.append(
            Finding(
                key="same_capability_pair",
                statement="同能力的两个 Adapter 直接相加，增益是相加、次可加、还是互相破坏",
                expectation="设计期望：次可加（第二个 Adapter 的边际收益递减，但总和仍为正）",
                measured=f"最差的一对 {name} 在 {cap} 上的增益 {g:+.4f}",
                verdict="低于基座" if g < 0 else ("次可加" if g > 0 else "恰好归零"),
                note=(
                    "同能力的两个 delta 都在解同一个任务，把它们原样相加等于在共用挂点上"
                    "做了两倍于训练时见过的更新 —— 幅度过冲，而不是方向冲突。"
                    "这个解释给出一个可证伪的预测：merge（系数 1/k）应当把这一对救回来。"
                ),
            )
        )

    # 三条方法的固定开销
    m1 = {meth: method_macs(runs, meth) for meth in ("stack", "merge", "route")}
    m1 = {k: v for k, v in m1.items() if v}
    if len(m1) >= 2:
        def _span(vals: tuple[int, ...]) -> str:
            return f"{vals[0]}" if len(vals) == 1 else f"{vals[0]}–{vals[-1]}"

        def _ratios(vals: tuple[int, ...]) -> str:
            lo, hi = vals[0] / base_macs, vals[-1] / base_macs
            return f"{lo:.3f}×" if len(vals) == 1 else f"{lo:.3f}–{hi:.3f}×"

        cheap = min(m1, key=lambda k: m1[k][0])
        route_rng = m1.get("route")
        out.append(
            Finding(
                key="method_macs",
                statement="「合并进权重」与「保持分离按门混合」的计算代价差多少",
                expectation="设计期望：stack/merge 与基座逐位同价；route 要为每个成员多做一次矩阵乘",
                measured=(
                    f"基座 {base_macs} MACs；"
                    + "，".join(f"{k} {_span(v)}（{_ratios(v)}）" for k, v in sorted(m1.items()))
                    + (
                        " —— route **不是一个定值**：qv 变体挂 2 个模块、all 变体挂 7 个，"
                        "所以它的开销随成员走"
                        if route_rng is not None and len(route_rng) > 1
                        else ""
                    )
                ),
                verdict="与期望一致" if cheap in ("stack", "merge") else "与期望不一致",
                note=(
                    "route 不动基座一个数（n_params_changed=0），代价记在 MACs 上；"
                    + (
                        f"实测 route 是基座的 {_ratios(route_rng)}。"
                        if route_rng is not None
                        else ""
                    )
                    + "这个交换是 route 那条边与其它两条边不可直接比较的原因。"
                ),
            )
        )

    # 前沿
    if frontier:
        best = max(runs, key=lambda r: r.mean_gain(caps) or float("-inf"))
        out.append(
            Finding(
                key="frontier",
                statement=f"{len(runs)} 个组合里，有几个是「值得部署」的",
                expectation="设计期望：非支配前沿应当明显小于设计空间",
                measured=f"前沿 {len(frontier)} 个；单点最高平均增益由 {best.cid} 给出（{best.mean_gain(caps):+.4f}）",
                verdict="前沿已收窄" if len(frontier) < len(runs) else "前沿 == 全集（无一被支配）",
                note="三个轴：平均增益 ↑、基座改动元素数 ↓、MACs ↓。方向由 graph.pareto_frontier 统一，不在调用点各自取负。",
            )
        )

    # 教训：单干增益相对更强的基线
    weak = []
    for lab in labels:
        s = adj.get(P.cid_of("stack", (lab,)))
        if s is None:
            continue
        cap = lab.rsplit("-", 1)[0]
        v = s.acc(cap)
        if v is None:
            continue
        gap = v - baselines[cap]["heuristic"]
        if gap < 0.10:
            weak.append(f"{lab} {gap:+.4f}")
    if weak:
        out.append(
            Finding(
                key="weak_adapters",
                statement="每个 Adapter 相对**更强的**那条基线（heuristic）是否都有足够增益",
                expectation="设计期望：全部 8 个都超过 heuristic + 0.10",
                measured=f"{len(weak)} 个未过：{'；'.join(weak)}",
                verdict="有 Adapter 未过门槛",
                note=(
                    "「loss 在降、不报错」完全可能什么都没学到 —— 这正是 FineTuneFactory "
                    "在 lr=3e-2 上记录过的现象。门槛盯的是 heuristic 而不是 chance："
                    "chance 太弱，一个只会照抄输入的策略就能超过它。"
                ),
            )
        )
    return out


def _common_capability(members: Sequence[str]) -> str | None:
    caps = {m.rsplit("-", 1)[0] for m in members}
    return next(iter(caps)) if len(caps) == 1 else None


# ----------------------------------------------------------------------
# 落盘
# ----------------------------------------------------------------------
def write_outputs(
    result: PipelineResult,
    out: str | Path,
    *,
    cfg: Mapping[str, Any],
    ctx: Mapping[str, Any],
    graph: L.Lineage,
    conflicts: Sequence[ConflictEdge],
    predictor_rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> None:
    """把一次运行的全部文件写进 ``out/``。**先写小文件，最后写 ``result.json``。**

    顺序有意义：``result.json`` 是「这次运行完整」的标记。它先落盘的话，
    中途失败会留下一份**看起来完整**的产物目录 —— 而 :func:`load_full`
    只会发现少了一个无关紧要的 csv。
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / "version.json").write_text(_json_text(result.version.to_dict()), encoding="utf-8")
    (out / "registry.json").write_text(
        _json_text({"base": result.base_key, "digest": _registry_digest(result.adapters),
                    "adapters": [a.to_dict() for a in result.adapters]}),
        encoding="utf-8",
    )
    (out / "runs.jsonl").write_text(
        "".join(json.dumps(r.to_dict(), ensure_ascii=False, sort_keys=False) + "\n" for r in result.runs),
        encoding="utf-8",
    )
    (out / "matrix.csv").write_text(
        _matrix_csv(result, capabilities=list(result.capabilities)), encoding="utf-8"
    )
    (out / "conflict.json").write_text(
        _json_text(
            {
                "resolution": threshold,
                "note": "interference 的判有害门槛取 1/n_eval：低于这个量级时符号由浮点噪声决定。",
                "edges": [c.to_dict() for c in conflicts],
                "predictors": list(predictor_rows),
            }
        ),
        encoding="utf-8",
    )
    L.dumps(graph, out / "lineage.json")
    (out / "lineage.dot").write_text(graph.to_dot(), encoding="utf-8")

    # Adapter Graph：冲突子图 + 前沿子图。两张分开的图，**不合并** ——
    # 合并之后「哪些对互相干扰」与「哪些组合值得部署」会共用一套节点，
    # 而它们回答的是两个不同的问题。
    csub = G.conflict_subgraph(graph, conflicts, method_edges=_method_edges(conflicts), threshold=threshold)
    fsub = G.frontier_nodes(graph, result.frontier)
    (out / "graph.json").write_text(
        _json_text(
            {
                "summary": result.counters.get("graph", {}),
                "views": [csub.to_dict(), fsub.to_dict()],
                "conflict_edges": [c.to_dict() for c in G.harmful_edges(conflicts, threshold=threshold)],
                "frontier": list(result.frontier),
            }
        ),
        encoding="utf-8",
    )
    (out / "graph.dot").write_text(_graph_dot(graph, csub, fsub), encoding="utf-8")

    # 报告与可视化在 ``result.json`` **之前**写。它们都只吃 ``result`` 一个输入，
    # 所以要么一起成功、要么谁都还没落地；而 ``result.json`` 是「这次运行完整」
    # 的标记，它必须最后出现（见本函数 docstring）。
    from . import report as RP
    from . import visualize as V

    (out / "report.md").write_text(RP.render_markdown(result), encoding="utf-8")
    (out / "report.html").write_text(V.render_html(result), encoding="utf-8")
    (out / "result.json").write_text(_json_text(result.to_dict()), encoding="utf-8")
    del cfg, ctx


def _method_edges(conflicts: Sequence[ConflictEdge]) -> dict[str, Sequence[ConflictEdge]]:
    """按方法分组的冲突边。**本项目的冲突边只在 ``stack`` 下测**，
    所以这里返回的键只有 ``stack`` —— 留着这个结构是为了让
    「想加 merge 的边必须显式加」是一个有位置可改的动作。"""
    return {"stack": list(conflicts)}


def _graph_dot(graph: L.Lineage, *views: G.GraphView) -> str:
    """**一张图叠两个视图**：冲突子图加粗，前沿子图虚线。

    叠而不是各画一张，是因为「同一个组合既在冲突边上、又在前沿上」这件事
    只有在同一张图里才看得见 —— 而那恰好是最值得看的那种组合。
    """
    keep = {n for v in views for n in v.node_ids}
    bold = {tuple(sorted(e)) for v in views for e in v.edge_keys}
    lines = ["digraph adaptergraph {", '  rankdir=LR;', '  node [shape=box, fontsize=10];']
    for n in graph.nodes():
        if n.id not in keep:
            continue
        style = ' style=dashed' if n.kind == "combo" else ""
        lines.append(f'  "{_dot(n.id)}" [label="{_dot(n.label or n.id)}"{style}];')
    for e in graph.edges():
        if e.src not in keep or e.dst not in keep:
            continue
        attrs = ' [penwidth=2, color="#c0392b"]' if tuple(sorted((e.src, e.dst))) in bold else ""
        lines.append(f'  "{_dot(e.src)}" -> "{_dot(e.dst)}"{attrs};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _dot(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _registry_digest(records: Iterable[AdapterRecord]) -> str:
    from . import rng as RNG

    return RNG.join_digest([f"{a.label}={a.digest}" for a in records])


def _matrix_csv(result: PipelineResult, *, capabilities: Sequence[str]) -> str:
    """组合矩阵。**一行一个组合，一列一个指标。**

    用 :mod:`csv` 而不是手拼逗号：能力名、cid 里都可能出现需要转义的字符，
    而手拼时那个 bug 只会在某个特定名字上出现。空值写**空字符串**而不是 0 ——
    0 是一个合法的成绩，两者在表里必须分得开。
    """
    buf = io.StringIO()
    caps = list(capabilities)
    head = (
        ["cid", "method", "size", "members"]
        + [f"gain_{c}" for c in caps]
        + [f"acc_{c}" for c in caps]
        + ["mean_gain", "own_gain", "merge_error", "merge_loss",
           "n_params_changed", "macs", "wall_ms", "artifact_digest"]
    )
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    for r in result.runs:
        row: list[Any] = [r.cid, r.method, r.size, "+".join(r.members)]
        row += [_f(r.metrics[c].gain) if c in r.metrics else "" for c in caps]
        row += [_f(r.metrics[c].acc) if c in r.metrics else "" for c in caps]
        row += [
            _f(r.mean_gain(caps)),
            _f(r.own_gain()),
            _f(r.merge_error),
            _f(r.merge_loss),
            r.n_params_changed,
            r.macs,
            _f(r.wall_ms),
            r.artifact.digest[:12] if r.artifact else "",
        ]
        w.writerow(row)
    return buf.getvalue()


def _f(v: float | None) -> str:
    """浮点进 csv：``None`` → 空串。**不用 ``or 0``**（坑 104）：``0.0`` 与「没测到」
    是两件事，前者是成绩、后者是缺失。"""
    return "" if v is None else f"{v:.6f}"


def matrix_rows(result: PipelineResult) -> list[dict[str, Any]]:
    """``matrix.csv`` 的**结构化**版本，给 ``main.py matrix`` 用。

    从 ``result``（内存或读回的）生成，而不是解析 csv —— 解析会
    把「csv 里的空串」变成 ``""``，而调用方要区分的是 ``None``。
    """
    caps = list(result.capabilities)
    out: list[dict[str, Any]] = []
    for r in result.runs:
        out.append(
            {
                "cid": r.cid,
                "method": r.method,
                "size": r.size,
                "members": list(r.members),
                "gains": {c: (r.metrics[c].gain if c in r.metrics else None) for c in caps},
                "mean_gain": r.mean_gain(caps),
                "own_gain": r.own_gain(),
                "merge_error": r.merge_error,
                "merge_loss": r.merge_loss,
                "n_params_changed": r.n_params_changed,
                "macs": r.macs,
                "wall_ms": r.wall_ms,
                "artifact": None if r.artifact is None else r.artifact.to_dict(),
            }
        )
    return out


# ----------------------------------------------------------------------
# 只读路径
# ----------------------------------------------------------------------
def read_runs(out: str | Path) -> list[CompositionRun]:
    """读 ``runs.jsonl``。**一行坏掉就报错并指出行号** —— 跳过坏行会让
    「276 行」变成「275 行」，而后面所有「总数对不对」的检查就都失去了基准。"""
    p = Path(out) / "runs.jsonl"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 {p}：先跑 `python main.py run`")
    runs: list[CompositionRun] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            runs.append(CompositionRun.from_dict(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 —— 要把行号带出去
            raise ValueError(f"{p} 第 {i} 行读不出来：{exc}") from exc
    return runs


def read_conflict(out: str | Path) -> dict[str, Any]:
    p = Path(out) / "conflict.json"
    if not p.is_file():
        raise FileNotFoundError(f"缺少 {p}：先跑 `python main.py run`")
    return json.loads(p.read_text(encoding="utf-8"))


def load_result(out: str | Path) -> PipelineResult | None:
    """只读路径的总入口。**缺任一源文件返回 ``None``**（而不是抛错）。

    返回 ``None`` 而不是抛异常，是因为调用方（``main.py`` 的只读子命令）
    要说的是一句完整的话：「还没有产物，先跑 ``run``」。让它去 ``except``
    一个异常类型，那句提示就会散在各处，且迟早有一处写错异常类名 ——
    于是用户看到的是一段 traceback。

    查的是 :data:`SOURCE_FILES` 而**不是** ``OUT_FILES`` —— 派生文件（报告两份）
    缺了照样读得出来，而读得出来才有机会把它重新生成出来。见 :data:`DERIVED_FILES`。
    """
    out = Path(out)
    if not (out / "result.json").is_file():
        return None
    if not all((out / f).is_file() for f in SOURCE_FILES):
        return None
    d = json.loads((out / "result.json").read_text(encoding="utf-8"))
    return PipelineResult.from_dict(d)


def load_full(out: str | Path) -> dict[str, Any] | None:
    """把 ``out/`` 整份读成结构化字典。缺任一必需文件返回 ``None``。

    ``runs`` / ``conflicts`` 走各自的 reader（不是从 ``result.json`` 里取）——
    它们是**独立落盘**的，从 ``result.json`` 取等于让那两个文件变成纯装饰，
    而「只读命令从产物重建」这条验收就查不出「runs.jsonl 写坏了」。
    """
    res = load_result(out)
    if res is None:
        return None
    out = Path(out)
    return {
        "result": res,
        "runs": read_runs(out),
        "conflict": read_conflict(out),
        "registry": json.loads((out / "registry.json").read_text(encoding="utf-8")),
        "graph": json.loads((out / "graph.json").read_text(encoding="utf-8")),
        "lineage": json.loads((out / "lineage.json").read_text(encoding="utf-8")),
        "version": json.loads((out / "version.json").read_text(encoding="utf-8")),
        "matrix_csv": (out / "matrix.csv").read_text(encoding="utf-8"),
    }
