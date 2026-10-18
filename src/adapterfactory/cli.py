"""命令行。**13 个只读 + 4 个写命令。**

### 只读 / 写 的分界是一条被测试守着的约定

**只读命令跑完之后 ``out/`` 必须逐位不变。** 测试对 ``out/`` 做整目录 md5
比对来执行这条约定。于是分界不能凭手感划：

- **写**：``produce``（训练并冻结 zoo）、``run``（端到端 + 落盘）、
  ``visualize``（只重出 ``report.html``）、``import``（读外部产物 + 落一份报告）。
- **只读**：其余全部。它们**只打印**，一个字节都不写。

一个命令被误划进只读侧的症状是：不变性测试失败，但失败信息只会说
「目录变了」—— 所以 :data:`READ_ONLY` 与 :data:`WRITE` 两张表在这里
**显式列出来**，测试直接照它跑，不靠人去数。

### 为什么只读命令要 ``load_result`` 返回 ``None`` 而不是抛异常

提示语是同一句：「还没有产物，先跑 ``python main.py run``」。让它去 ``except``
一个异常类，那句话会散在每个子命令里，迟早有一处写错类名 ——
于是用户看到的是 traceback，而 traceback 不说「你该做什么」。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import compose as P
from . import config as C
from . import data as D
from . import graph as G
from . import lineage as L
from . import net as N
from . import pipeline as PL
from . import registry as R
from . import report as RP
from . import visualize as V
from . import zoo as Z
from .schema import CompositionRun, PipelineResult

#: ⚠️ **本模块是全项目唯一允许导入 producer 的地方**，而且只在 ``_cmd_produce``
#: 里**局部**导入 —— 不在模块顶层。理由是可检查性：``audit.producer_contained``
#: 扫的是「哪些模块导入了 produce」，顶层导入与局部导入在 ``ast`` 层面都看得见，
#: 但局部导入多给出一条更强的性质：**``main.py summary`` 这条只读路径上，
#: 一个训练符号都没有被加载过**。这不是洁癖 —— 顶层导入的后果是
#: 「只读命令不碰训练代码」只能靠读代码确认，而局部导入让它成为 import 图上的事实。

__all__ = ["main", "READ_ONLY", "WRITE", "build_parser"]

#: 这些**只打印，一个字节都不写**。测试照这张表跑不变性。
READ_ONLY: tuple[str, ...] = (
    "task",
    "version",
    "registry",
    "summary",
    "matrix",
    "conflict",
    "graph",
    "frontier",
    "methods",
    "explain",
    "lineage",
    "acceptance",
    "report",
)

#: 这些**会写**（``run`` / ``produce`` 写 ``data/`` 与 ``out/``，另外两个只写 ``out/``）。
WRITE: tuple[str, ...] = ("produce", "run", "visualize", "import")


# ----------------------------------------------------------------------
#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adapterfactory",
        description=(
            "Adapter 生命周期工厂：登记 → 组合 → 合并 → 评测 → 部署。"
            "**本项目不训练** —— 训练入口属于架构墙的另一侧（produce）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "只读命令（不写任何文件）：\n  " + "、".join(READ_ONLY)
            + "\n写命令：\n  " + "、".join(WRITE)
            + "\n\n典型流程：\n"
            "  python main.py task            # 看题与基线（不需要产物）\n"
            "  python main.py produce         # 一次性：训练并冻结 8 个 Adapter\n"
            "  python main.py run             # 端到端：276 个组合落盘\n"
            "  python main.py summary         # 之后都是只读的\n"
        ),
    )
    p.add_argument("--root", default=".", help="项目根目录；相对路径按项目根解析")
    p.add_argument("--out", default=None, help="产物目录（默认 <root>/out）；相对路径按项目根解析")
    p.add_argument("--config", default=None, help="配置文件（默认 configs/default.json，缺则用 DEFAULTS）；相对路径按项目根解析")
    sub = p.add_subparsers(dest="cmd", metavar="<命令>")

    sub.add_parser("task", help="任务定义、两条基线与 8 个 Adapter 的规格（不需要产物）")
    sub.add_parser("version", help="版本与口径摘要")

    sub.add_parser("produce", help="【写】训练并冻结 zoo 到 data/adapters/")
    r = sub.add_parser("run", help="【写】端到端跑一遍并落盘")
    r.add_argument("--no-export", action="store_true", help="不导出 compositions/（组合产物）")

    sub.add_parser("registry", help="zoo 登记表")
    sub.add_parser("summary", help="运行摘要与结论")
    # **帮助文案里不写条数。** 它曾经写的是「9 项」—— 而 `report.py` 那边
    # 同一个数字刚被改成「数出来的」，理由是当时报告的正文写着「8 项」、
    # 下面那张表是 9 行，两处都「看着对」。帮助文案比报告更难核对：
    # 它在一个还没有任何产物的进程里被渲染出来，没有清单可数。
    # 于是这里只说有什么，不说有几项。
    sub.add_parser("acceptance", help="运行时核对 + README 的两条验收标准")
    sub.add_parser("report", help="打印 report.md（从产物现生成，再与磁盘上的比）")

    m = sub.add_parser("matrix", help="组合矩阵")
    m.add_argument("--by", default="cid", choices=("cid", "gain", "size", "method", "cost"), help="排序键")
    m.add_argument("--method", default=None, help="只看某一种方法")
    m.add_argument("--limit", type=int, default=40, help="最多打印几行（0 = 全部）")

    c = sub.add_parser("conflict", help="逐对干扰：实测（路径 B）vs 参数空间预测（路径 A）")
    c.add_argument("--limit", type=int, default=40, help="最多打印几行（0 = 全部）")

    g = sub.add_parser("graph", help="Adapter Graph")
    g.add_argument("--dot", action="store_true", help="输出 DOT")
    g.add_argument("--kinds", default=None, help="只画这些节点种类（逗号分隔）")

    sub.add_parser("frontier", help="非支配前沿与它支配掉的点")

    me = sub.add_parser("methods", help="stack / merge / route 在同一子集上的对照")
    me.add_argument("--size", type=int, default=2, help="看多大小的子集（默认 2）")

    e = sub.add_parser("explain", help="一个组合的完整来历")
    e.add_argument("cid")

    li = sub.add_parser("lineage", help="血缘图（子集格）摘要")
    li.add_argument("--dot", action="store_true", help="输出 DOT")

    sub.add_parser("visualize", help="【写】重出 out/report.html")

    im = sub.add_parser("import", help="【写】读外部产物（FineTuneFactory 格式）并落一份可读性报告")
    im.add_argument("--from", dest="src", required=True, help="外部 out/ 目录（含 artifacts/）")
    im.add_argument("--limit", type=int, default=20, help="最多列几行")
    return p


# ----------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _anchor_args(args)
    if not args.cmd:
        build_parser().print_help()
        return 0
    root = Path(args.root)
    out = Path(args.out) if args.out else PL.out_dir_of(root)
    cfg = C.load_config(C.resolve_path(root, args.config) if args.config else None, root=root)
    try:
        return _dispatch(args, root=root, out=out, cfg=cfg)
    except (FileNotFoundError, AssertionError, ValueError, KeyError) as exc:
        # 只把这些**预期内的**失败变成一句话。真正的 bug（TypeError 之类）
        # 必须原样抛出并带 traceback —— 把它们也吞成一句话，
        # 会让「我改坏了一个类型」看起来像「用户用错了命令」。
        print(f"错误：{exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, *, root: Path, out: Path, cfg: Mapping[str, Any]) -> int:
    cmd = args.cmd
    if cmd == "task":
        return _cmd_task(root, cfg)
    if cmd == "version":
        return _cmd_version(root, cfg)
    if cmd == "produce":
        return _cmd_produce(root, cfg)
    if cmd == "run":
        return _cmd_run(root, cfg, out=out, export=not args.no_export)
    if cmd == "visualize":
        return _cmd_visualize(root, out)
    if cmd == "import":
        return _cmd_import(root, out, cfg, args)

    full = PL.load_full(out)
    if full is None:
        print(_NEED_RUN.format(out=out), file=sys.stderr)
        return 2
    res: PipelineResult = full["result"]
    if cmd == "registry":
        return _cmd_registry(root, cfg)
    if cmd == "summary":
        return _cmd_summary(res)
    if cmd == "acceptance":
        return _cmd_acceptance(res)
    if cmd == "report":
        return _cmd_report(res, out)
    if cmd == "matrix":
        return _cmd_matrix(res, args)
    if cmd == "conflict":
        return _cmd_conflict(res, args)
    if cmd == "graph":
        return _cmd_graph(res, full, args)
    if cmd == "frontier":
        return _cmd_frontier(res)
    if cmd == "methods":
        return _cmd_methods(res, args)
    if cmd == "explain":
        return _cmd_explain(res, args.cid)
    if cmd == "lineage":
        return _cmd_lineage(res, full, args)
    raise SystemExit(f"未知命令 {cmd!r}")


_NEED_RUN = (
    "还没有可用的产物（{out} 里缺少必需文件）。\n"
    "  先跑：python main.py run\n"
    "  （若连 zoo 都没有，先跑 python main.py produce —— 它会真的训练 8 个 Adapter。）"
)


# ----------------------------------------------------------------------
# 不需要产物的两个
# ----------------------------------------------------------------------
def _cmd_task(root: Path, cfg: Mapping[str, Any]) -> int:
    ctx = PL.prepare(root=root, cfg=cfg)
    idx: D.DataIndex = ctx["idx"]
    print(D.describe(idx))
    print()
    print("两条基线（**验收盯的是更强的 heuristic**，不是 chance）：")
    bl = ctx["baselines"]
    rows = sorted(bl)
    print("  能力       " + "".join(f"{k:>11}" for k in ("chance", "majority", "copy", "heuristic")))
    for cap in rows:
        print(f"  {cap:<10}" + "".join(f"{bl[cap][k]:>11.4f}" for k in ("chance", "majority", "copy", "heuristic")))
    print()
    print(
        f"  chance 的口径是 **1/|payload_chars| = {D.chance_level(idx):.4f}**，"
        "不是 1/|vocab| ——\n"
        "  逐字符映射任务里，「猜中」只发生在 payload 字母表上，混进 specials 与任务字母\n"
        "  会把一个恰好蒙对的模型判成「远高于乱猜」。"
    )
    print()
    rank = int(cfg["zoo"]["rank"])
    alpha = float(cfg["zoo"]["alpha_mult"]) * rank
    specs = Z.default_zoo(C.capabilities_of(cfg), rank=rank, alpha=alpha)
    print(f"zoo 规格（{len(specs)} 个 = {len(C.capabilities_of(cfg))} 能力 × 变体）：")
    print("  标签          能力      变体   挂点")
    for s in specs:
        print(f"  {s.label:<13} {s.capability:<9} {s.variant:<5} {len(s.targets)} 个")
    print()
    print("同能力的两个变体（qv / all）是**「收益相加还是相消」这一问的最小实验**；")
    print("跨能力的组合是「干扰」这一问的最小实验。两者合起来让 README 那句")
    print("「收益是相加、相消还是互相破坏」有三个可判定的答案。")
    return 0


def _cmd_version(root: Path, cfg: Mapping[str, Any]) -> int:
    from . import rng as RNG

    ctx = PL.prepare(root=root, cfg=cfg)
    pd = C.protocol_digest(cfg)
    vd = C.version_digest(cfg, [("seq_len", ctx["seq_len"]), ("base", ctx["base_key"])])
    print(f"口径摘要 protocol_digest : {pd}")
    print(f"（仅口径的）version_digest : {vd}")
    print(f"版本号                    : {C.short_version(vd)}")
    print(f"基座                      : {ctx['base_key']}")
    print(f"seq_len                   : {ctx['seq_len']}（= 5 + 2×{ctx['idx'].payload_len}）")
    print(f"索引摘要                  : {RNG.sha256_text(json.dumps(ctx['idx'].to_dict(), ensure_ascii=False, sort_keys=True))[:16]}")
    print()
    print("口径摘要变了**只需重出报告**；版本摘要变了（换了 zoo 或换了题）**必须重跑组合**。")
    print("两者的区别就是这种命令存在的理由 —— 合成一个数会让「该重跑什么」无从判断。")
    return 0


# ----------------------------------------------------------------------
# 写命令
# ----------------------------------------------------------------------
def _cmd_produce(root: Path, cfg: Mapping[str, Any]) -> int:
    from . import produce as PR  # 局部导入，见模块顶部的说明

    ctx = PL.prepare(root=root, cfg=cfg)
    rank = int(cfg["zoo"]["rank"])
    alpha = float(cfg["zoo"]["alpha_mult"]) * rank
    # 配方**只有一处定义**（``config.zoo_specs_of`` / ``config.zoo_recipe_of``）：
    # ``registry.load_registry`` 拿同一对函数重建「这次该是什么样」，
    # 然后核对磁盘上的产物。两处各写一遍的话，配置里加一个字段而只改了其中
    # 一处，产物的来路就会在另一侧被静默放过。
    specs = C.zoo_specs_of(cfg)
    settings = PR.TrainSettings(**C.zoo_recipe_of(cfg))
    print(f"produce：训练并冻结 {len(specs)} 个 Adapter → {PR.zoo_dir(root)}")
    print(f"  设置 lr={settings.lr} steps={settings.steps} batch={settings.batch} "
          f"rank={rank} alpha={alpha} 优化器={settings.optimizer}")
    print("  **这一步真的在训练。它是架构墙的另一侧，流水线永远不碰它。**")
    print()
    rep = PR.produce_zoo(
        root=root,
        shape=ctx["shape"],
        base_seed=ctx["base_seed"],
        specs=specs,
        settings=settings,
        vocab=ctx["vocab"],
        idx=ctx["idx"],
        seq_len=ctx["seq_len"],
        eval_sets=ctx["eval_sets"],
        train_seed=int(cfg["data"]["seed"]),
        force=False,
    )
    print()
    print("冻结完成：")
    for lab, r in sorted(rep["reports"].items()):
        if r.get("reused"):
            print(f"  [沿用] {lab:<13} digest {r['digest'][:12]}  {r['n_params']} 参")
        else:
            accs = " ".join(f"{c}={v:.3f}" for c, v in sorted(r["eval_acc"].items()))
            print(f"  [新训] {lab:<13} loss {r['first_loss']:.4f}→{r['last_loss']:.4f}  {accs}")
    print()
    print("下一步：python main.py run")
    return 0


def _cmd_run(root: Path, cfg: Mapping[str, Any], *, out: Path, export: bool) -> int:
    print(f"run：端到端（{'含' if export else '不含'}导出组合产物）")
    print()

    def progress(done: int, total: int, note: str) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"  评测 {done}/{total}  {note}", flush=True)

    res = PL.run_pipeline(root=root, cfg=cfg, out=out, export=export, progress=progress)
    print()
    print(f"完成。版本 {res.version.version}，组合 {len(res.runs)} 条，"
          f"冲突边 {len(res.conflicts)} 条，前沿 {len(res.frontier)} 个。")
    c = res.counters
    print(f"  产物目录：{out}")
    print(f"  核对：{sum(1 for x in c['checks'] if x['ok'])}/{len(c['checks'])} 通过"
          f"（其中门槛项 {sum(1 for x in c['checks'] if x['required'])} 条）")
    for x in c["checks"]:
        if not x["ok"]:
            mark = "❌" if x["required"] else "⚠️"
            print(f"    {mark} {x['name']}：{x['detail']}")
    print()
    print("下一步：python main.py summary | matrix | conflict | graph | report")
    return 0


def _cmd_visualize(root: Path, out: Path) -> int:
    res = PL.load_result(out)
    if res is None:
        print(_NEED_RUN.format(out=out), file=sys.stderr)
        return 2
    path = out / "report.html"
    text = V.render_html(res)
    old = path.read_text(encoding="utf-8") if path.is_file() else None
    path.write_text(text, encoding="utf-8")
    print(f"已写出 {path}（{len(text)} 字节）")
    if old is not None:
        # 同一条只读不变性覆盖了 report.html，所以这里能顺带报出「是否逐位相同」。
        print("  与原有的逐字节" + ("相同（可视化是纯函数，这条是它的自检）。" if old == text else "**不同** —— 产物与代码不一致，请重跑 run。"))
    return 0


def _import_row_line(row: Mapping[str, Any]) -> str:
    """``main.py import`` 明细表里的一行。

    读的全是 :func:`registry._normalize_row` 保证存在的键，所以三个分支
    （读不通 / 适配器 / 整模型检查点）走同一个模板，不会各自缺列。

    这三个键以前是 ``row['kind']`` / ``row['name']`` / ``row['note']``：
    ``name`` 没有任何分支写过，``kind`` 与 ``note`` 又都只在部分分支里有 ——
    于是这条命令在自己的打印循环上炸掉，而统计行早已打完，看起来像报告写到一半坏了。
    """
    if not row["read"]:
        return f"[{'读不通':>8}] {row['file'][:40]:<40} {row['error']}"
    tag = str(row["kind"] or "未知")
    head = f"[{tag:>8}] {row['file'][:40]:<40} "
    if not row["is_adapter"]:
        return head + str(row["note"] or "整模型检查点")
    verdict = "形状相容" if row["compatible"] else f"形状不符（{row['n_problems']} 处）"
    # ``n_hooks`` 是**落在已知挂点上的张量数**（不是挂点对数）—— 按它的实际含义标注。
    return head + f"{row['label']}｜rank={row['rank']}｜张量 {row['n_hooks']}/{row['n_tensors']}｜{verdict}"


def _cmd_import(root: Path, out: Path, cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    ctx = PL.prepare(root=root, cfg=cfg)
    shape: N.ModelShape = ctx["shape"]

    def resolve() -> Path:
        p = Path(args.src)
        if p.is_absolute():
            return p
        q = root / p
        return q if q.exists() else p

    src = resolve()
    if not src.is_dir():
        print(f"错误：找不到外部产物目录 {src}", file=sys.stderr)
        return 2
    rep = R.import_report(src, shape, limit=int(args.limit))
    print(f"外部产物可读性报告：{src}")
    print(f"  文件 {rep['n_files']} 个，读通 {rep['n_read']}，读不通 {rep['n_unreadable']}")
    print(f"  其中自报 kind ∈ (lora, qlora) 的 **Adapter** {rep['n_adapters']} 个，"
          f"与当前基座形状兼容 {rep['n_adapter_compatible']} 个")
    print(f"  其余 {rep['n_checkpoints']} 个是整模型检查点（full/freeze），不是 Adapter")
    # 这两行曾经合成一行，标签写「外部产物声明的基座标识」而值取的是 ``rep['base_key']``
    # —— 那是**本项目自己的**基座键（``import_report`` 里 ``"base_key": shape.key``）。
    # 外部那侧的真实答案是 ``foreign_base_identities``，这里是空的。
    # 于是一行字把结论说反了：它让读者以为 FineTuneFactory 声明了基座标识，
    # 而下面紧跟的结语正在说它没有。**两个数各有各的名字，不要挤进一行。**
    foreign = rep["foreign_base_identities"]
    print(f"  外部产物自报的基座标识：{'、'.join(foreign) if foreign else '（一个都没有 —— 见下）'}")
    print(f"  【本地】基座（本次比对用的那根）：{rep['base_key']}")
    print()
    for row in rep["rows"]:
        print(f"  {_import_row_line(row)}")
    if rep["n_rows_hidden"]:
        print(f"  … 另有 {rep['n_rows_hidden']} 个未列出（--limit 控制）")
    print()
    print("**边界的功能性验证：读得通，不等于装得上。**")
    print(f"  本次比对用的基座是 {rep['base_key']}。")
    if foreign:
        print(
            f"  这批产物自报了 {len(foreign)} 种基座标识（见上），于是「装不装得上」\n"
            "  可以先看标识、再谈形状：标识与本地一致的那几份才谈得上组合。"
        )
    else:
        print(
            "  而它们**一个基座标识都没自报** —— 边车 JSON 里只有张量形状。\n"
            f"  于是 {rep['n_adapters']} 个 Adapter 里 {rep['n_adapter_compatible']} 个形状兼容，"
            "而形状兼容的那些也不保证来自同一根基座。\n"
            "  这正是 AdapterFactory 的 base_key 要补上的那个洞：跨基座的 Adapter 不能组合，\n"
            "  这不是形状问题，是语义问题（权重空间里的 Δ 是**相对某根 W** 定义的）。"
        )
    out.mkdir(parents=True, exist_ok=True)
    path = out / "import.json"
    path.write_text(
        json.dumps(
            {k: v for k, v in rep.items() if k != "rows"} | {"rows": rep["rows"], "source": str(src)},
            ensure_ascii=False, indent=2, sort_keys=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\n已写出 {path}")
    return 0


# ----------------------------------------------------------------------
# 只读命令
# ----------------------------------------------------------------------
def _cmd_registry(root: Path, cfg: Mapping[str, Any]) -> int:
    ctx = PL.prepare(root=root, cfg=cfg)
    reg = R.load_registry(root, cfg, ctx["shape"])
    print(f"zoo 登记表（{len(reg.records)} 个 Adapter，基座 {reg.base_key}）")
    print(f"  登记摘要 registry_digest = {reg.digest}")
    print()
    print("  标签           能力     变体  秩  挂点  自己的参数  改动基座元素  字节数   digest")
    for a in reg.records:
        print(
            f"  {a.label:<14} {a.capability:<8} {a.variant:<4} {a.rank:<3} "
            f"{len(a.targets):<4} {a.n_params:>10} {a.n_base_params_touched:>13} "
            f"{a.n_bytes:>7}  {a.digest[:12]}"
        )
    print()
    print("  「自己的参数」与「改动基座元素」是两个不同的量：qv 变体自己有几百个参数，")
    print("  却改动了上千个基座元素 —— 后者才是 parameter_cost 的分子。")
    return 0


def _cmd_summary(res: PipelineResult) -> int:
    c = res.counters
    print(f"运行摘要 · {res.version.version}")
    print(f"  基座 {res.base_key}（种子 {res.base_seed}，seq_len {res.seq_len}）")
    print(f"  zoo {c.get('n_adapters')} 个 · 子集 {c.get('n_subsets')} × 方法 "
          f"{c.get('n_methods')} = 组合 **{c.get('n_runs')}** 条")
    print(f"  冲突边 {c.get('n_conflicts')} 条（实测有害 {c.get('n_conflicts_harmful')} 条，"
          f"门槛 {c.get('conflict_resolution'):.4f}）")
    print(f"  非支配前沿 {c.get('n_frontier')} 个")
    print(f"  本次运行耗时 {c.get('runtime_s')}s")
    print()
    print("结论（设计期望 vs 实测）：")
    print(RP.render_findings(res.findings))
    print()
    print(f"核对：{sum(1 for x in c.get('checks', []) if x['ok'])}/{len(c.get('checks', []))} 通过")
    print("（详细：python main.py acceptance）")
    return 0


def _cmd_acceptance(res: PipelineResult) -> int:
    c = res.counters
    print("运行时核对")
    print(f"  {'条目':<26} {'性质':<7} 结果")
    for x in c.get("checks", []):
        mark = "✅" if x["ok"] else ("❌" if x["required"] else "⚠️")
        print(f"  {mark} {x['name']:<23} {'门槛' if x['required'] else '报告项':<7} {x['detail']}")
    print()
    print("README 的验收标准")
    for a in c.get("acceptance", []):
        mark = "✅" if a["ok"] else "❌"
        print(f"  {mark} {a['name']:<23} {a['detail']}")
    print()
    print("⚠️  标记的两项**不否决验收**，理由写在 audit.run_checks 的 docstring 里：")
    print("   它们是**发现**（两路同向率被推翻正是最有价值的结论），不是质量指标。")
    return 0


def _cmd_report(res: PipelineResult, out: Path) -> int:
    text = RP.render_markdown(res)
    disk = out / "report.md"
    if disk.is_file():
        old = disk.read_text(encoding="utf-8")
        if old == text:
            print(f"（与磁盘上的 report.md 逐字节相同 —— 报告完全由产物决定。"
                  f"共 {len(text)} 字节）\n", file=sys.stderr)
        else:
            print(f"⚠️  磁盘上的 report.md 与现生成的不同（{len(old)} vs {len(text)} 字节）："
                  "产物与代码不一致，请重跑 run。\n", file=sys.stderr)
    sys.stdout.write(text)
    return 0


def _sorted_runs(res: PipelineResult, by: str, method: str | None) -> list[CompositionRun]:
    runs = [r for r in res.runs if method is None or r.method == method]
    caps = list(res.capabilities)
    if by == "cid":
        return sorted(runs, key=lambda r: r.cid)
    if by == "size":
        return sorted(runs, key=lambda r: (r.size, r.cid))
    if by == "method":
        return sorted(runs, key=lambda r: (r.method, r.cid))
    if by == "gain":
        # 缺一个能力的实测就排到最后 —— 用 0 补位会让「没测」混进「测得差」的档位。
        return sorted(runs, key=lambda r: -(r.mean_gain(caps) if r.mean_gain(caps) is not None else float("-inf")))
    if by == "cost":
        return sorted(runs, key=lambda r: (r.n_params_changed, r.macs, r.cid))
    raise SystemExit(f"未知排序键 {by!r}")


def _cmd_matrix(res: PipelineResult, args: argparse.Namespace) -> int:
    caps = list(res.capabilities)
    runs = _sorted_runs(res, args.by, args.method)
    lim = int(args.limit)
    shown = runs if lim <= 0 else runs[:lim]
    print(f"组合矩阵（共 {len(res.runs)} 条"
          + (f"，按 {args.by} 排序" if args.by != "cid" else "")
          + (f"，只看 {args.method}" if args.method else "")
          + f"，打印 {len(shown)} 条）")
    print()
    head = f"  {'cid':<34} {'大':<3} " + "".join(f"{'g:' + c:>10}" for c in caps) + f"{'平均':>9}{'改基座':>8}{'MACs':>9}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in shown:
        gains = "".join(
            f"{RP.fmt(r.metrics[c].gain, 4, signed=True):>10}" if c in r.metrics else f"{'—':>10}"
            for c in caps
        )
        mg = r.mean_gain(caps)
        print(
            f"  {r.cid:<34} {r.size:<3} {gains}"
            f"{RP.fmt(mg, 4, signed=True):>9}{r.n_params_changed:>8}{r.macs:>9}"
        )
    print()
    print("  g:<能力> 是该组合在**那个能力**上的增益（相对基座，与基线同口径的逐 token 准确率）。")
    print("  平均增益要**四个能力都有实测**才算得出来；缺一个就显示 — 而不是拿部分能力的平均冒充。")
    return 0


def _cmd_conflict(res: PipelineResult, args: argparse.Namespace) -> int:
    lim = int(args.limit)
    edges = sorted(res.conflicts, key=lambda c: (-c.interference, c.a, c.b))
    shown = edges if lim <= 0 else edges[:lim]
    thr = float(res.counters.get("conflict_resolution") or 0.0)
    print(f"两两干扰（{len(edges)} 对；门槛 {thr:.4f} = 1/n_eval；打印 {len(shown)} 对）")
    print()
    print("  路径 B（实测，能力空间）  vs  路径 A（预测，参数空间）")
    print(f"  {'对':<30}{'I(对称)':>9}{'I_b→a':>9}{'I_a→b':>9}{'cos':>9}{'符号率':>8}{'同向':>6}")
    print("  " + "-" * 78)
    for c in shown:
        agree = "—" if c.agree is None else ("是" if c.agree else "否")
        print(
            f"  {c.a + ' ~ ' + c.b:<30}"
            f"{RP.fmt(c.interference, 4, signed=True):>9}"
            f"{RP.fmt(c.i_a_from_b, 4, signed=True):>9}"
            f"{RP.fmt(c.i_b_from_a, 4, signed=True):>9}"
            f"{RP.fmt(c.predicted_cos, 4, signed=True):>9}"
            f"{RP.fmt(c.predicted_sign, 4):>8}{agree:>6}"
        )
    print()
    rated = [c for c in res.conflicts if c.agree is not None]
    if rated:
        n = sum(1 for c in rated if c.agree)
        print(f"  两路同向率 {n}/{len(rated)} = {n / len(rated):.3f}")
    print()
    print("  判据：cos < 0 ⇒ 预测有害。**这个判据被实测推翻了**（见 report.md 第五节），")
    print("  而推翻它本身就是本项目最有信息量的输出 —— 所以这两列照样每次都算、每次都印。")
    print()
    print("  定向值的读法：I_b→a 是「挂上 b 之后，a 在自己能力上损失了多少」。")
    print("  只印对称值会让「谁被谁破坏」这个问题从产物里消失。")
    return 0


def _cmd_graph(res: PipelineResult, full: Mapping[str, Any], args: argparse.Namespace) -> int:
    graph = L.build_lineage(res.to_dict())
    if args.dot:
        kinds = None if not args.kinds else [k.strip() for k in args.kinds.split(",") if k.strip()]
        sys.stdout.write(graph.to_dot(kinds=kinds))
        return 0
    s = res.counters.get("graph") or {}
    print(f"Adapter Graph（节点 {s.get('n_nodes')}，边 {s.get('n_edges')}，"
          f"无环 {s.get('is_acyclic')}）")
    print(f"  节点种类：" + "、".join(f"{k} {v}" for k, v in (s.get("nodes_by_kind") or {}).items()))
    print()
    print("  子集格（这一层就是血缘图，不是另一个产物）——")
    print("    combo:<cid> 的父节点**恰好**是它的成员 Adapter。")
    print("    于是「谁和谁组合过」看一张图就知道，不必翻报告。")
    print()
    thr = float(res.counters.get("conflict_resolution") or 0.0)
    bad = G.harmful_edges(res.conflicts, threshold=thr)
    print(f"  冲突子图（实测干扰 > {thr:.4f} 的 {len(bad)} 对）——")
    print(f"    {'对':<30}{'对称干扰':>10}  受影响最大的能力")
    for c in bad:
        top = next(iter(c.affected.items()), None)
        cap = "—" if top is None else f"{top[0]} {top[1]:+.3f}"
        print(f"    {c.a + ' ~ ' + c.b:<30}{c.interference:>10.4f}  {cap}")
    print()
    print(f"  前沿子图：{len(res.frontier)} 个组合指向收拢点 {L.SELECT_COMPOSE}")
    print()
    print("  用 `graph --dot` 得到 DOT；`out/graph.dot` 把两个视图叠在一张图上 ——")
    print("  「同一个组合既在冲突边上、又在前沿上」只有在同一张图里才看得见。")
    del full
    return 0


def _cmd_frontier(res: PipelineResult) -> int:
    caps = list(res.capabilities)
    by = {r.cid: r for r in res.runs}
    print(f"非支配前沿（{len(res.frontier)} / {len(res.runs)} 个组合）")
    print("  三个轴：平均增益 ↑、改动基座元素数 ↓、MACs ↓（方向由 graph.pareto_frontier 统一）")
    print()
    print(f"  {'cid':<34}{'方法':<7}{'大小':<5}{'平均增益':>10}{'自身增益':>10}{'改基座':>8}{'MACs':>9}")
    for cid in res.frontier:
        r = by.get(cid)
        if r is None:
            continue
        print(
            f"  {cid:<34}{r.method:<7}{len(r.members):<5}"
            f"{RP.fmt(r.mean_gain(caps), 4, signed=True):>10}"
            f"{RP.fmt(r.own_gain(), 4, signed=True):>10}"
            f"{r.n_params_changed:>8}{r.macs:>9}"
        )
    print()
    # 谁把谁支配掉了 —— 这才是「前沿」这个词的内容
    print("  被支配的点各自被谁支配（每行给一个支配者）：")
    front = set(res.frontier)
    shown = 0
    for r in res.runs:
        if r.cid in front:
            continue
        mg = r.mean_gain(caps)
        if mg is None:
            continue
        pt = [mg, -float(r.n_params_changed), -float(r.macs)]
        for o in res.runs:
            if o.cid == r.cid or o.cid not in front:
                continue
            om = o.mean_gain(caps)
            if om is None:
                continue
            opt = [om, -float(o.n_params_changed), -float(o.macs)]
            if G.dominance(opt, pt):
                print(f"    {r.cid:<34} ← {o.cid}")
                shown += 1
                break
        if shown >= 12:
            print("    …")
            break
    print()
    print("  `merge` 不需要一条「被排除」的特例：同子集同 MACs 同基座代价而能力更低，")
    print("  它就是这样被同子集的 `stack` 支配掉的。")
    return 0


def _cmd_methods(res: PipelineResult, args: argparse.Namespace) -> int:
    caps = list(res.capabilities)
    size = int(args.size)
    by: dict[tuple[str, ...], dict[str, CompositionRun]] = {}
    for r in res.runs:
        if len(r.members) == size:
            by.setdefault(tuple(r.members), {})[r.method] = r
    print(f"三种方法在**同一组**成员上的对照（size={size}，{len(by)} 组）")
    print()
    print(f"  {'成员':<30}{'stack':>10}{'merge':>10}{'route':>10}{'MACs(s/m/r)':>22}{'基座改动':>10}")
    for key in sorted(by):
        row = by[key]
        cells = []
        for m in ("stack", "merge", "route"):
            r = row.get(m)
            cells.append(RP.fmt(None if r is None else r.mean_gain(caps), 4, signed=True))
        macs = "/".join(str(row[m].macs) if m in row else "—" for m in ("stack", "merge", "route"))
        ch = next(iter(row.values())).n_params_changed
        print(f"  {'+'.join(key):<30}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}{macs:>22}{ch:>10}")
    print()
    print("  **merge 与 stack 不在同一个量纲上**：merge 的系数 1/k 是**故意**的稀释，")
    print("  它的能力差由 merge_loss 那一列承担，不能当成「干扰」读。")
    print("  **route 不动基座一个数**（基座改动 0），代价记在 MACs 上 —— 这是它那条边")
    print("  与另外两条不可直接比较的原因，也是报告里必须写明的地方。")
    return 0


def _cmd_explain(res: PipelineResult, cid: str) -> int:
    caps = list(res.capabilities)
    by = {r.cid: r for r in res.runs}
    r = by.get(cid)
    if r is None:
        near = sorted(by)[:6]
        print(f"没有组合 {cid!r}。前几个是：{near}", file=sys.stderr)
        return 2
    print(f"组合 {r.cid}")
    print(f"  方法 {r.method}　成员 {len(r.members)} 个：{' + '.join(r.members)}")
    print(f"  权重系数 {r.coefs if r.coefs is not None else '（route 不合并权重）'}")
    print()
    print(f"  {'能力':<10}{'增益':>10}{'准确率':>10}{'整串全对':>10}{'停得住':>9}")
    for c in caps:
        m = r.metrics.get(c)
        if m is None:
            print(f"  {c:<10}{'—':>10}{'—':>10}{'—':>10}{'—':>9}")
            continue
        print(f"  {c:<10}{m.gain:>+10.4f}{m.acc:>10.4f}{RP.fmt(m.exact, 4):>10}{m.eos_acc:>9.4f}")
    print()
    print(f"  平均增益（四个能力）{RP.fmt(r.mean_gain(caps), 4, signed=True)}")
    print(f"  自身增益（成员自己的能力）{RP.fmt(r.own_gain(), 4, signed=True)}")
    print(f"  改动的基座元素 {r.n_params_changed}　MACs {r.macs}　评测耗时 {r.wall_ms:.1f}ms")
    print(f"  合并误差 {RP.fmt(r.merge_error, 3)}（实现检查：合并前向 vs 逐个施加）"
          if r.merge_error is not None else "  合并误差 —（route 不走权重空间）")
    if r.merge_loss is not None:
        print(f"  与同子集 stack 的能力差 {RP.fmt(r.merge_loss, 4, signed=True)}"
              "（= 平均化付出的代价，**不是干扰**）")
    if r.artifact is not None:
        print(f"  产物 {r.artifact.path}　{r.artifact.n_bytes} 字节　"
              f"总秩 {r.artifact.rank_total}　digest {r.artifact.digest[:12]}")
    else:
        print("  产物 —（route 不可导出为一份 LoRA：它不产生一个权重增量）" if r.method == "route" else "  产物 —（本次运行 --no-export）")
    print()

    # 单干的对照
    print("  成员单干的对照：")
    for m in r.members:
        s = by.get(P.cid_of("stack", (m,)))
        cap = m.rsplit("-", 1)[0]
        if s is None or cap not in s.metrics:
            print(f"    {m:<14}（找不到它的单干记录）")
            continue
        print(f"    {m:<14} 自身能力 {cap}：增益 {s.metrics[cap].gain:+.4f}"
              f"　平均增益 {RP.fmt(s.mean_gain(caps), 4, signed=True)}")
    print()

    # 位置：前沿？被谁支配？
    front = set(res.frontier)
    mg = r.mean_gain(caps)
    if r.cid in front:
        print("  **这个组合在非支配前沿上** —— 没有任何组合在三个轴（平均增益↑、基座改动↓、MACs↓）")
        print("  上同时不差于它并至少一项更好。")
    elif mg is not None:
        pt = [mg, -float(r.n_params_changed), -float(r.macs)]
        for o in res.runs:
            if o.cid == r.cid:
                continue
            om = o.mean_gain(caps)
            if om is None:
                continue
            if G.dominance([om, -float(o.n_params_changed), -float(o.macs)], pt):
                print(f"  被 `{o.cid}` **支配**：它的平均增益 {om:+.4f} ≥ {mg:+.4f}，"
                      f"基座改动 {o.n_params_changed} ≤ {r.n_params_changed}，"
                      f"MACs {o.macs} ≤ {r.macs}，且至少一项严格更好。")
                print("  支配者不必在前沿上 —— 前沿的定义是「不被任何人支配」，不是「支配了谁」。")
                break
        else:
            print("  既不在前沿上、也找不到支配者 —— 这只可能发生在平均增益缺失时。")
    else:
        print("  平均增益算不出来（有能力的实测缺失），因此无法判断它在前沿上的位置。")
    print()
    # 干扰的落点
    if len(r.members) > 1:
        print("  这一组的干扰落点（组合相对最好的单干差了多少）：")
        for c in caps:
            m = r.metrics.get(c)
            if m is None or m.interference_from is None:
                continue
            print(f"    {c:<10} 成员单干之和 {m.interference_from:+.4f}　组合 {m.gain:+.4f}"
                  f"　差 {m.interference_from - m.gain:+.4f}")
    return 0


def _cmd_lineage(res: PipelineResult, full: Mapping[str, Any], args: argparse.Namespace) -> int:
    graph = L.build_lineage(res.to_dict())
    if args.dot:
        sys.stdout.write(graph.to_dot())
        return 0
    print(f"血缘图：节点 {len(graph.nodes())}，边 {len(graph.edges())}，无环 {graph.is_acyclic()}")
    by_kind: dict[str, int] = {}
    for n in graph.nodes():
        by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
    print("  节点：" + "、".join(f"{k} {v}" for k, v in sorted(by_kind.items())))
    print(f"  根节点（入度 0）：{len(graph.roots())} 个")
    print(f"  收拢点：{L.SELECT_COMPOSE}")
    print()
    print("  **子集格这一层就是血缘图**：combo:<cid> 的父节点恰好是它的成员 Adapter。")
    print("  于是本项目的图不是「另外画了一张」，而是同一个 DAG 的一个子图 ——")
    print("  这也是「Adapter Graph 不是独立产物」这句话的字面意思。")
    print()
    print(
        f"  版本 {res.version.version}　产物文件 {len(PL.OUT_FILES)} 个"
        f"（源 {len(PL.SOURCE_FILES)} 个，缺任一源文件都算不完整；"
        f"另 {len(PL.DERIVED_FILES)} 个由源文件派生）"
    )
    del full
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
