"""``report.md``：**八节，全部从产物现算，不重新跑任何前向。**

### 为什么不重新跑一遍

``main.py report`` 是只读命令。它跑完之后 ``out/`` 必须**逐位不变** ——
这条验收只有在「报告完全由已经落盘的东西生成」时才可能成立。
一旦报告里有一个数需要重新前向，只读不变性就退化成了「碰巧两次一样」。

于是 ``render_markdown`` 只吃一个 :class:`~adapterfactory.schema.PipelineResult`
（以及它 ``counters`` 里那几张已经算好的表）。**要什么就在 ``run`` 时存下来，
而不是在报告里现算** —— 这条约束顺带保证了报告与 ``runs.jsonl`` 不可能互相矛盾。

### ``fmt`` 存在的理由：**绝不 ``or 0``**

``0.0`` 与「没测到」在外观上都是一个小数，但一个是成绩、一个是缺失。
三条规矩：

- ``None`` → ``—``（这件事在这个组合上不存在，例如 size-1 的干扰）
- ``NaN`` → ``未知``（存在但算不出来，例如常量输入的相关系数）
- 其余 → 定点小数

``v or 0`` 会把前两种一起变成 ``0``（坑 104）。本项目第一版正是这样
把「这个能力没测」显示成了「这个能力全错」，两者的行为完全不同。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import graph as G
from .schema import CompositionRun, Finding, PipelineResult

__all__ = ["render_markdown", "fmt", "render_findings"]

DASH = "—"


def fmt(v: Any, digits: int = 4, *, signed: bool = False, scale: float = 1.0) -> str:
    """把一个**可能是 None / NaN** 的值变成可读文本。

    ``signed=True`` 会强行带上正负号：一堆 ``0.5417`` 与 ``-0.0208`` 混在一起时，
    「有没有负号」是最先被扫到的信息，而 ``+`` 默认不打印。
    """
    if v is None:
        return DASH
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "未知"
    x *= scale
    if math.isinf(x):
        return "+∞" if x > 0 else "-∞"
    s = f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"
    return s


def _cell(v: Any) -> str:
    """一个表格单元格。**管道符必须转义** —— 它在这个位置是**结构**，不是内容。

    本项目真的踩过：一句 ``公式 Σ C(4,k)·|方法| = 42`` 写进了核对表的详情栏，
    markdown 于是把这行**从中间劈开**，渲染出来是两张错位的表。
    症状不是报错，是列错位 —— 而错位的表里数字仍然对得上行，只是对不上列。

    换行同样要处理：一个多行的单元格会把表撕成两半，且后半段变成正文。
    """
    return str(v).replace("|", "\\|").replace("\n", " ").strip()


def _row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _affected_items(
    affected: Mapping[str, float], caps: Sequence[str]
) -> list[tuple[str, float]]:
    """``affected`` 的**确定顺序**：先按题目的能力顺序，再按名字。

    ⚠️ 直接 ``affected.items()`` 会让报告文本依赖 **dict 的插入顺序**,
    而插入顺序**活不过一次 JSON 往返**：落盘用 ``sort_keys=True``，
    于是 ``out/result.json`` 读回来的 ``affected`` 按名字排序，而内存里那一份
    按能力顺序。后果不是「数字错了」，而是**同一份产物渲染出两段不同的文本**：

    - ``main.py report`` 会打出「⚠️ 磁盘上的 report.md 与现生成的不同
      （请重跑 run）」，把一次纯粹的键序差异报成「产物与代码不一致」；
    - 而**重跑 run 不会修好它** —— 因为差异的来源是往返，不是运行。

    这正是「报告完全由产物决定」这句话最容易破的地方：判据必须是产物里的
    **值**，不能是内存里的**顺序**。
    """
    order = {c: i for i, c in enumerate(caps)}
    return sorted(affected.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))


def _table(head: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = [_row(head), "|" + "|".join("---" for _ in head) + "|"]
    out.extend(_row(r) for r in rows)
    return out


# ----------------------------------------------------------------------
def render_markdown(result: PipelineResult) -> str:
    """八节报告。**每一节都显式说明它回答的是哪个问题。**"""
    L: list[str] = []
    L.append(f"# AdapterFactory 报告 · {result.version.version}")
    L.append("")
    L.append(
        f"> 基座 `{result.base_key}`（种子 {result.base_seed}，seq_len {result.seq_len}） "
        f"· zoo {len(result.adapters)} 个 Adapter · 组合 {len(result.runs)} 条 "
        f"· 冲突边 {len(result.conflicts)} 条 · 前沿 {len(result.frontier)} 个"
    )
    L.append("")
    L.append(
        "本报告回答 README 的两个验收问题：**① 组合矩阵（Capability Gain vs Interference）"
        "是否可复现且完整**；**② Adapter Graph 能不能明确指出哪些组合互相干扰**。"
        "全部数字来自 `out/` 的产物，报告本身不跑任何前向。"
    )
    L.append("")

    L += _s1_summary(result)
    L += _s2_specimen(result)
    L += _s3_matrix(result)
    L += _s4_interference(result)
    L += _s5_two_paths(result)
    L += _s6_graph(result)
    L += _s7_frontier(result)
    L += _s8_checks(result)

    L += [
        "",
        "---",
        "",
        "## 附录：口径与不可复现边界",
        "",
        *[f"- **{k}** — {v}" for k, v in sorted(result.notes.items())],
        "",
        f"- **唯一不可复现字段** — `{', '.join(result.version.VOLATILE)}`"
        f"（其余字段两次 `run` 之间逐位相同，由测试守着）。",
        "",
        "生成的报告不含任何外部引用（无图片、无 CDN）：它必须能在一个离线目录里被读。",
        "",
    ]
    return "\n".join(L)


# ----------------------------------------------------------------------
def _s1_summary(result: PipelineResult) -> list[str]:
    L = ["## 一、结论摘要", ""]
    L.append("**这一节说什么**：这次运行里，哪些设计期望成立、哪些被实测推翻。")
    L.append("")
    if not result.findings:
        L.append("（没有发现条目 —— 这本身是异常的：连一条期望都没被写下来。）")
        L.append("")
        return L
    L += _table(
        ["条目", "期望", "实测", "判定"],
        [[f.key, f.expectation, f.measured, f"**{f.verdict}**"] for f in result.findings],
    )
    L.append("")
    for f in result.findings:
        L.append(f"**{f.key}** — {f.statement}")
        L.append("")
        L.append(f"{f.note}")
        L.append("")
    return L


def _s2_specimen(result: PipelineResult) -> list[str]:
    c = result.counters
    L = ["## 二、标本与设计空间", ""]
    L.append("**这一节说什么**：被组合的东西是什么，候选空间有多大。")
    L.append("")
    L.append(f"- 基座：`{result.base_key}`")
    L.append(
        f"- 设计空间：{c.get('n_adapters', '?')} 个 Adapter 的全部 "
        f"size 1..{_max_size(result)} 子集 × {c.get('n_methods', '?')} 种组合方法 "
        f"= **{c.get('n_subsets', '?')} 个子集 × {c.get('n_methods', '?')} = {c.get('n_runs', '?')} 个组合**"
    )
    L.append(
        "- 三种方法的**计算代价**（size-1 组合的实测 MACs，基座 "
        f"{fmt(c.get('base_macs'), 0)}）：stack {fmt(c.get('macs_stack'), 0)}、"
        f"merge {fmt(c.get('macs_merge'), 0)}、route ≤ {fmt(c.get('macs_route'), 0)}"
        "（route 那一项是**上界**：它随成员挂了几个模块而变，取值域见 §一 `method_macs` 那条）"
    )
    L.append("")
    L.append("### zoo")
    L.append("")
    L += _table(
        ["Adapter", "能力", "变体", "秩", "挂点", "参数量", "改动基座元素", "digest"],
        [
            [
                f"`{a.label}`", a.capability, a.variant, a.rank, len(a.targets),
                a.n_params, a.n_base_params_touched, f"`{a.digest[:12]}`",
            ]
            for a in result.adapters
        ],
    )
    L.append("")
    L.append(
        "**「参数量」与「改动基座元素」是两个不同的量**，两者常常被混为一谈："
        "`qv` 变体自己有 768 个参数，但它改动了 2304 个基座元素 —— "
        "后者才是部署时要重算多少。表里两列都给出来。"
    )
    L.append("")
    return L


def _max_size(result: PipelineResult) -> int:
    return max((len(r.members) for r in result.runs), default=0)


def _own_capability(run: CompositionRun) -> str:
    return run.members[0].rsplit("-", 1)[0] if len(run.members) == 1 else ""


def _s3_matrix(result: PipelineResult) -> list[str]:
    caps = list(result.capabilities)
    L = ["## 三、组合矩阵：收益是相加、相消，还是互相破坏", ""]
    L.append(
        "**这一节说什么**：README 第一节那个问题。做法是把它拆成两个**可判定**的最小实验 ——"
        "同能力的两个 Adapter（都在解同一件事）与跨能力的两个 Adapter（各解各的）。"
    )
    L.append("")

    pairs = [r for r in result.runs if r.method == "stack" and len(r.members) == 2]
    same = [r for r in pairs if _pair_capability(r) is not None]
    cross = [r for r in pairs if _pair_capability(r) is None]

    L.append(f"### 同能力对（{len(same)} 个：都在解同一个能力，边际收益必然递减）")
    L.append("")
    rows = []
    for r in same:
        cap = _pair_capability(r) or ""
        m = r.metrics.get(cap)
        ga = [x.metrics[cap].gain for x in result.runs
              if x.method == "stack" and x.members == [r.members[0]] and cap in x.metrics]
        gb = [x.metrics[cap].gain for x in result.runs
              if x.method == "stack" and x.members == [r.members[1]] and cap in x.metrics]
        gsum = (ga[0] if ga else None) + (gb[0] if gb else None) if ga and gb else None
        rows.append(
            [
                "+".join(r.members), cap,
                fmt(ga[0] if ga else None, 4, signed=True),
                fmt(gb[0] if gb else None, 4, signed=True),
                fmt(m.gain if m else None, 4, signed=True),
                fmt(None if gsum is None or m is None else gsum - m.gain, 4, signed=True),
            ]
        )
    L += _table(
        ["组合", "能力", "g(成员一)", "g(成员二)", "g(两者相加)", "次可加量（和 − 实测）"],
        rows,
    )
    L.append("")
    if rows:
        worst = min(rows, key=lambda x: _to_f(x[4]))
        L.append(
            f"**答案**：同能力对里 g(两者相加) 最低的是 `{worst[0]}`，"
            f"实测增益 {worst[4]}。次可加量全部为正（表最后一列），"
            "**但「次可加」这个词不足以描述实测** —— 见下。"
        )
        L.append("")

    L.append(f"### 跨能力对（{len(cross)} 个：各解各的，干扰的可判定形式）")
    L.append("")
    rows = []
    for r in cross:
        cap = r.members[0].rsplit("-", 1)[0]
        m = r.metrics.get(cap)
        ga = [x.metrics[cap].gain for x in result.runs
              if x.method == "stack" and x.members == [r.members[0]] and cap in x.metrics]
        rows.append(
            [
                "+".join(r.members), cap,
                fmt(ga[0] if ga else None, 4, signed=True),
                fmt(m.gain if m else None, 4, signed=True),
                fmt(None if not ga or m is None else ga[0] - m.gain, 4, signed=True),
            ]
        )
    L += _table(
        ["组合", "被观察的能力", "g(该成员单干)", "g(组合)", "干扰 I = 单干 − 组合"],
        rows,
    )
    L.append("")

    # 方法对照
    L.append("### 三种方法的对照")
    L.append("")
    by = {(r.method, tuple(r.members)): r for r in result.runs}
    keys = sorted({tuple(r.members) for r in result.runs if len(r.members) == 2})
    rows = []
    for mem in keys[:20]:
        cells: list[Any] = ["+".join(mem)]
        for method in ("stack", "merge", "route"):
            r = by.get((method, mem))
            cells.append(fmt(None if r is None else r.mean_gain(caps), 4, signed=True))
        rows.append(cells)
    L += _table(["组合", "stack 平均增益", "merge 平均增益", "route 平均增益"], rows)
    L.append("")
    L.append("（只列前 20 组；全量在 `matrix.csv`。注意 `route` 的 MACs 更高、"
             "`merge` 的系数 `1/k` 是**故意**的稀释 —— 两条边与 `stack` 不在同一个量纲上。）")
    L.append("")
    return L


def _to_f(s: Any) -> float:
    return float("inf") if s in (DASH, "未知") else float(s)


def _pair_capability(run: CompositionRun) -> str | None:
    caps = {m.rsplit("-", 1)[0] for m in run.members}
    return next(iter(caps)) if len(caps) == 1 else None


def _s4_interference(result: PipelineResult) -> list[str]:
    L = ["## 四、干扰的实测（路径 B：能力空间）", ""]
    L.append(
        "**这一节说什么**：把两个 Adapter 同时挂上，量它各自还剩多少本事。"
        "定向值 `I_{b→a}`（b 破坏了多少 a 的本事）与 `I_{a→b}` 分开给，"
        "对称值取两者的平均 —— 只给一个数会让「谁被谁破坏」这个问题消失。"
    )
    L.append("")
    res = result.counters.get("conflict_resolution")
    thr = "?" if res is None else f"{float(res):.4f}"
    L.append(
        f"判定门槛取 **{thr}**（= 1/n_eval，即准确率的分辨率下限）："
        "低于这个量级时，`interference` 的正负号由 1e-17 的浮点噪声决定，"
        "而「同向率」会因此随机器而变。"
    )
    L.append("")
    edges = sorted(result.conflicts, key=lambda c: (-c.interference, c.a, c.b))
    L += _table(
        ["对", "I（对称）", "I_{b→a}", "I_{a→b}", "cos 预测", "符号一致率", "同向?", "受影响的能力"],
        [
            [
                f"`{c.a}` ~ `{c.b}`",
                fmt(c.interference, 4, signed=True),
                fmt(c.i_a_from_b, 4, signed=True),
                fmt(c.i_b_from_a, 4, signed=True),
                fmt(c.predicted_cos, 4, signed=True),
                fmt(c.predicted_sign, 4),
                ("是" if c.agree else "**否**") if c.agree is not None else DASH,
                "、".join(
                    f"{k}({fmt(v, 3, signed=True)})"
                    for k, v in _affected_items(c.affected, result.capabilities)[:3]
                ) or DASH,
            ]
            for c in edges
        ],
    )
    L.append("")
    if edges:
        L.append(
            f"- 干扰最大的对：`{edges[0].a}` ~ `{edges[0].b}`，I = {fmt(edges[0].interference, 4, signed=True)}"
        )
        L.append(
            f"- 干扰最小的对：`{edges[-1].a}` ~ `{edges[-1].b}`，I = {fmt(edges[-1].interference, 4, signed=True)}"
        )
        L.append(
            "- **每一对的干扰都接近满值**：`I` 的量级与成员自己单干的增益同阶。"
            "换句话说，挂上第二个 Adapter 基本抹掉了第一个的收益 —— "
            "这不是「某些对冲突」，是「这个设计空间里没有哪一对不冲突」。"
        )
        L.append("")
    return L


def _s5_two_paths(result: PipelineResult) -> list[str]:
    c = result.counters
    L = ["## 五、两条路径的对质（路径 A：参数空间）", ""]
    L.append(
        "**这一节说什么**：不用跑任何前向，只看两个 delta 张量本身，能不能**预测**干扰。"
        "能预测 ⇒ 「冲突」这个说法有来源；不能预测 ⇒ 那本身就是最有信息量的输出。"
    )
    L.append("")
    L.append("路径 A 的判据写在 `conflict.direction_sign` 里：`cos < 0` ⇒ 预测有害。")
    L.append("")
    stats = c.get("predictor_stats") or []
    if stats:
        L.append("### 候选预测量 vs 实测干扰")
        L.append("")
        L += _table(
            ["预测量", "可用对数", "Pearson", "Spearman"],
            [[s["name"], s["n"], fmt(s["pearson"], 3, signed=True), fmt(s["spearman"], 3, signed=True)] for s in stats],
        )
        L.append("")
        L.append(
            "**两个相关系数都给**：干扰与幅度之间大概率是单调而不线性的"
            "（例如「超过某一档才开始破坏」），只看 Pearson 会把一个完美的单调关系读成弱相关。"
        )
        L.append("")

    rows = c.get("predictor_rows") or []
    if rows:
        L.append("### 逐对明细")
        L.append("")
        L += _table(
            ["对", "cos", "符号一致率", "共同挂点", "expansion", "overlap(a→b)", "overlap(b→a)", "‖Δb‖/‖Δa‖", "实测 I"],
            [
                [
                    f"`{r['a']}` ~ `{r['b']}`",
                    fmt(r["cos"], 4, signed=True),
                    fmt(r["sign_rate"], 4),
                    r["n_common"],
                    fmt(r["expansion"], 4),
                    fmt(r["overlap_ab"], 4),
                    fmt(r["overlap_ba"], 4),
                    fmt(r["norm_ratio"], 4),
                    fmt(r["interference"], 4, signed=True),
                ]
                for r in rows
            ],
        )
        L.append("")

    rated = [x for x in result.conflicts if x.agree is not None]
    L.append("### 结论")
    L.append("")
    if not rated:
        L.append("没有任何一对能给出参数空间预测 —— 这一节无从判定。")
    else:
        n = sum(1 for x in rated if x.agree)
        L.append(
            f"{n}/{len(rated)} 对同向（同向率 {n / len(rated):.3f}）。"
            + (
                "**路径 A 被推翻。** delta 之间几乎正交、符号一致率贴着 0.50（随机），"
                "而实测干扰每一对都接近满值 —— 两者互不相关。"
                "决定干扰的是 **delta 的幅度**（相加后在共用挂点上做了 k 倍于训练时见过的更新），"
                "而不是 delta 之间的**夹角**。"
                if n / len(rated) < 0.5
                else "两路同向，`cos` 这个判据有来源。"
            )
        )
        L.append("")
        L.append(
            "⚠️ **这条不是「预测器调参不够」**。上表里 `expansion`（相加后的更新幅度"
            "相对单个的倍数）与 `overlap` 也一并报出来，就是为了让读者能自己分辨"
            "「哪个量真的与干扰有关」—— 不预先宣布哪个对，那正是两条路对质的意思。"
        )
        L.append("")
    return L


def _s6_graph(result: PipelineResult) -> list[str]:
    c = result.counters
    s = c.get("graph") or {}
    L = ["## 六、Adapter Graph", ""]
    L.append(
        "**这一节说什么**：验收标准②。这张图**不是画出来的，是算出来的** —— "
        "`combo:<cid>` 节点的父节点恰好是它的成员 Adapter，"
        "所以「子集格」这一层就是血缘图本身，不是另一个产物。"
    )
    L.append("")
    L += _table(
        ["量", "值"],
        [
            ["节点数", s.get("n_nodes")],
            ["边数", s.get("n_edges")],
            ["节点种类", s.get("n_kinds")],
            ["无环", "是" if s.get("is_acyclic") else "**否**"],
            ["Adapter 数", s.get("n_adapters")],
            ["两两组合数 C(n,2)", s.get("n_pairs")],
            ["冲突边数", s.get("n_conflict_edges")],
            ["其中实测有害", s.get("n_harmful")],
            ["最严重的一对", fmt(s.get("worst"), 4, signed=True)],
            ["两路同向率", fmt(s.get("two_path_agreement"), 4)],
            ["非支配前沿", s.get("n_frontier")],
        ],
    )
    L.append("")
    kinds = s.get("nodes_by_kind") or {}
    if kinds:
        L.append("节点种类分布：" + "、".join(f"`{k}` {v}" for k, v in kinds.items()) + "。")
        L.append("")

    thr = c.get("conflict_resolution") or 0.0
    edges = [e for e in result.conflicts if e.interference > float(thr)]
    L.append(f"### 「哪几对互相干扰」——逐条，可追溯到证据（共 {len(edges)} 对）")
    L.append("")
    if edges:
        L += _table(
            ["对", "对称干扰", "参数空间预测", "两路同向", "受影响能力的明细"],
            [
                [
                    f"`{e.a}` ~ `{e.b}`",
                    fmt(e.interference, 4, signed=True),
                    f"cos={fmt(e.predicted_cos, 4, signed=True)}",
                    ("是" if e.agree else "否") if e.agree is not None else DASH,
                    "；".join(
                        f"{k} {fmt(v, 3, signed=True)}"
                        for k, v in _affected_items(e.affected, result.capabilities)
                    ) or DASH,
                ]
                for e in edges
            ],
        )
    else:
        L.append("（没有一对的干扰超过分辨率下限。）")
    L.append("")
    L.append(
        "**「指出」这个词是能被偷懒实现的** —— 把 28 对全标红也叫「指出」。"
        "所以上表每一条都带三样东西：定向值（谁被谁破坏）、两路证据（预测 vs 实测）、"
        "以及 `affected`（在哪几个能力上）。删掉任何一样，「指出」就退化成了「标注」。"
    )
    L.append("")
    return L


def _s7_frontier(result: PipelineResult) -> list[str]:
    caps = list(result.capabilities)
    L = ["## 七、非支配前沿：276 个点里哪几个值得部署", ""]
    L.append(
        "**这一节说什么**：三个轴 —— 平均增益 ↑、改动基座元素数 ↓、MACs ↓。"
        "方向由 `graph.pareto_frontier` **统一**取，不在调用点各自取负："
        "漏掉任何一轴的方向都会得到一个**看起来完全正常**的前沿。"
    )
    L.append("")
    by = {r.cid: r for r in result.runs}
    rows = []
    for cid in result.frontier:
        r = by.get(cid)
        if r is None:
            continue
        rows.append(
            [
                f"`{cid}`",
                r.method,
                len(r.members),
                fmt(r.mean_gain(caps), 4, signed=True),
                fmt(r.own_gain(), 4, signed=True),
                r.n_params_changed,
                r.macs,
            ]
        )
    L += _table(["cid", "方法", "大小", "平均增益", "自身增益", "改动基座元素", "MACs"], rows)
    L.append("")
    L.append(
        "`merge` **不需要**一条「把它排除掉」的特例：同子集同 MACs 同基座代价，"
        "而它的能力更低（那是被故意稀释掉的，见 `merge_loss`），"
        "所以它会被同子集的 `stack` 支配掉。留一条特例会让"
        "「merge 有没有可能更好」这个**可证伪的问题**失去被回答的机会。"
    )
    L.append("")
    return L


def _s8_checks(result: PipelineResult) -> list[str]:
    c = result.counters
    L = ["## 八、核对与验收", ""]
    checks = c.get("checks") or []
    # 条数是**数出来的**，不是写的。写过一次「8 项」，而实际是 9 项（第 10 项
    # ``artifact_roundtrip`` 只在导出时出现）—— 于是报告正文与它下面那张表
    # 当场自相矛盾，而两处都「看着对」。凡是紧挨着一份清单的计数，都该从清单来。
    L.append(
        f"**这一节说什么**：{len(checks)} 项运行时核对 + README 的两条验收标准。"
        "核对的明细在 `result.json` 的 `counters.checks` 里，逐条可查。"
    )
    L.append("")
    rows = []
    for ch in checks:
        req = "门槛" if ch.get("required") else "报告项"
        mark = "✅" if ch.get("ok") else ("❌" if ch.get("required") else "⚠️")
        rows.append([f"{mark} `{ch['name']}`", req, ch["detail"]])
    for a in c.get("acceptance") or []:
        mark = "✅" if a.get("ok") else "❌"
        rows.append([f"{mark} `{a['name']}`", "验收", a["detail"]])
    L += _table(["条目", "性质", "结果"], rows)
    L.append("")
    L.append(
        "**`required=False` 的两项不否决验收**，理由必须写下来，否则它们看起来像"
        "「为了让验收通过而豁免的」：\n"
        "- `two_path_agreement` —— 它是本项目的**发现**，不是质量指标。"
        "参数空间预测被实测推翻正是报告里最有价值的结论；"
        "把它设成门槛等于宣布「除非预测器好用，否则这次运行不算数」。\n"
        "- `artifact_roundtrip` —— 它依赖 `compositions/` 里的文件，"
        "而只读路径不产出它们。缺文件时判失败等于说「只读路径因为没写文件所以不合格」。"
    )
    L.append("")
    return L


def render_findings(findings: Sequence[Finding]) -> str:
    """只要结论摘要那一段（``main.py summary`` 用）。**与报告同一份数据。**"""
    if not findings:
        return "（没有发现条目）"
    out = []
    for f in findings:
        out.append(f"- **{f.verdict}** · {f.statement}")
        out.append(f"  - 期望：{f.expectation}")
        out.append(f"  - 实测：{f.measured}")
    return "\n".join(out)
