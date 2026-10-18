"""``report.html``：**自包含**（内联 CSS + 内联 SVG，零外部引用）。

### 为什么是自包含的而不引一个图表库

报告要能在**一个离线目录**里被打开 —— 它经常是被拷出来单独看的。
引一个 CDN 的代价不是「慢」：在离线环境里它渲染成一片空白，
而那会被读成「这个项目没什么可看的」。``<img src=...>`` 同理。

### 画什么，由「哪些结论需要一个形状」决定

- **单位圆上的冲突图** —— README 要的 Adapter Graph。8 个 Adapter 排在圆上，
  两两之间的实测干扰画成一条弧，弧的粗细与颜色都随干扰量。
  之所以画成弧而不是让读者去看 28 行表：**「每一对都接近满值」这件事
  在一张图上是一眼可得的，在一张表上需要逐行比较。**
- **增益 × 代价散点** —— 前沿为什么是那几个点。把改动基座元素数当横轴、
  平均增益当纵轴，非支配前沿圈出来。
- **干扰热图** —— 8×8，行是「被观察的」，列是「被挂上来的」。

三张图都是**从产物现算**的，不引入任何随机布局：同一份产物两次生成的
HTML 逐位相同（只读不变性覆盖 ``report.html``）。"""

from __future__ import annotations

import html
import math
from typing import Any, Sequence

from . import report as RP
from .schema import CompositionRun, PipelineResult

__all__ = ["render_html"]

_METHOD_COLOR = {"stack": "#c0392b", "merge": "#2471a3", "route": "#1e8449"}


def render_html(result: PipelineResult) -> str:
    caps = list(result.capabilities)
    body = [
        _header(result),
        _cards(result),
        _section(
            "冲突图：哪些 Adapter 互相干扰",
            "圆上 8 个 Adapter，一条弧就是一对。**弧的粗细与颜色都随实测干扰量** —— "
            "「每一对都接近满值」在一张图上一眼可得，在一张表上要逐行比较。"
            "`graph.json` 里那些 `conflict:` 节点在这里被画成了边："
            "它们在图上没有别的邻居，单独画出来只会让画面变密。",
            _conflict_circle(result),
        ),
        _section(
            "增益 × 代价：前沿为什么是那几个点",
            "横轴是改动了多少个基座元素（越左越省），纵轴是四个能力的平均增益（越高越好）。"
            "圈出来的是非支配前沿。**merge 的点与同子集的 stack 完全重合在横轴上**，"
            "但纵轴更低 —— 它就是这样被支配掉的，不需要一条特例。",
            _scatter(result, caps),
        ),
        _section(
            "干扰热图（对称值）",
            "行 `i`、列 `j` 的格子是 `½(I_{j→i} + I_{i→j})`。对角线留空："
            "自己与自己的干扰不是一个问题，填 0 会让它看起来是一个「没有干扰」的实测。",
            _heatmap(result),
        ),
        _section(
            "三种方法在同一组成员上的对照",
            "每个子集一条柱：三根柱分别是 stack / merge / route 的平均增益。"
            "横轴过 0 的基线是**基座本身**，不是「最差的方法」—— 否则"
            "「掉到基座以下」这件事在图上就看不见了。",
            _method_bars(result, caps),
        ),
        _findings(result),
    ]
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN"><head><meta charset="utf-8">',
            f"<title>AdapterFactory · {html.escape(result.version.version)}</title>",
            f"<style>{_CSS}</style>",
            "</head><body>",
            *body,
            "</body></html>",
            "",
        ]
    )


_CSS = """
:root{--fg:#1b1f23;--dim:#5b6570;--line:#d8dee6;--bg:#fbfcfd;--card:#fff}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--bg);color:var(--fg);
 font:14px/1.65 ui-sans-serif,system-ui,"Segoe UI","Microsoft YaHei",sans-serif}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:17px;margin:0 0 6px}
.sub{color:var(--dim);margin:0 0 24px}
section{margin:0 0 28px;background:var(--card);border:1px solid var(--line);
 border-radius:10px;padding:18px 20px}
section p.lead{color:var(--dim);margin:0 0 14px;max-width:88ch}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:10px 14px;min-width:120px}
.card b{display:block;font-size:20px;font-weight:600}
.card span{color:var(--dim);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:8px}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left}
th{background:#f2f5f8}
td.n{text-align:right;font-variant-numeric:tabular-nums}
ul.f{list-style:none;padding:0;margin:0}
ul.f li{border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:0 0 10px}
ul.f li b{font-size:13px}
svg{display:block;max-width:100%;height:auto}
.legend{color:var(--dim);font-size:12px;margin-top:6px}
code{background:#f2f5f8;border:1px solid var(--line);border-radius:4px;padding:0 4px}
"""


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _section(title: str, lead: str, svg: str) -> str:
    return f"<section><h2>{_esc(title)}</h2><p class='lead'>{lead}</p>{svg}</section>"


def _header(result: PipelineResult) -> str:
    v = result.version
    return (
        f"<h1>AdapterFactory · <code>{_esc(v.version)}</code></h1>"
        f"<p class='sub'>基座 <code>{_esc(result.base_key)}</code>（种子 {result.base_seed}，"
        f"seq_len {result.seq_len}）· 口径摘要 <code>{_esc(v.protocol_digest[:12])}</code> · "
        f"生成于 {_esc(v.created_at)}</p>"
    )


def _cards(result: PipelineResult) -> str:
    c = result.counters
    items = [
        (c.get("n_runs"), "组合"),
        (c.get("n_adapters"), "Adapter"),
        (c.get("n_conflicts_harmful"), "实测有害的对"),
        (c.get("n_frontier"), "非支配前沿"),
        (f"{c.get('runtime_s')}s", "本次运行耗时"),
    ]
    out = ["<div class='cards'>"]
    for val, label in items:
        out.append(f"<div class='card'><b>{_esc(RP.fmt(val, 0) if isinstance(val, float) else val)}</b>"
                   f"<span>{_esc(label)}</span></div>")
    out.append("</div>")
    return "".join(out)


# ----------------------------------------------------------------------
def _conflict_circle(result: PipelineResult, *, size: int = 560) -> str:
    """8 个 Adapter 排在圆上，两两干扰画成弧。

    弧的**粗细**与**颜色深浅**都编码同一个量。用两个通道编码同一个量看起来
    是浪费，但它解决的是一个真实的问题：黑白打印时颜色没了，
    而粗细还在。只编码一次的图在打印件上是废的。
    """
    labels = [a.label for a in result.adapters]
    n = len(labels)
    if n < 2:
        return "<p>（少于两个 Adapter，无冲突图可画。）</p>"
    cx = cy = size / 2
    r = size / 2 - 66
    pos = {}
    for i, lab in enumerate(labels):
        # 从正上方起、顺时针 —— 与阅读顺序一致，且第 0 个在正上方（无歧义的位置）。
        a = -math.pi / 2 + 2 * math.pi * i / n
        pos[lab] = (cx + r * math.cos(a), cy + r * math.sin(a))

    by: dict[tuple[str, str], float] = {}
    for c in result.conflicts:
        lo, hi = sorted((c.a, c.b))
        by[(lo, hi)] = float(c.interference)
    worst = max(by.values(), default=0.0) or 1.0
    thr = float(result.counters.get("conflict_resolution") or 0.0)

    parts = [f"<svg viewBox='0 0 {size} {size}' width='{size}' height='{size}' "
             "role='img' aria-label='Adapter 两两干扰图'>"]
    # 弧：二次贝塞尔，控制点拉向圆心，于是每对之间的连线不穿过圆心
    for (a, b), val in sorted(by.items()):
        if val <= thr:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        k = 0.35  # 控制点从弦中点往圆心方向收 —— 让弧在圆心附近留出空隙
        qx, qy = mx + (cx - mx) * k, my + (cy - my) * k
        w = 0.6 + 5.4 * (val / worst)
        # 红→橙：干扰越大越红。用同一个色相区间，避免读者去比两种颜色「哪个更严重」
        t = val / worst
        col = f"rgb({int(200 + 39 * t)},{int(120 - 60 * t)},{int(60 - 20 * t)})"
        parts.append(
            f"<path d='M {x1:.1f} {y1:.1f} Q {qx:.1f} {qy:.1f} {x2:.1f} {y2:.1f}' "
            f"fill='none' stroke='{col}' stroke-width='{w:.2f}' opacity='0.72'/>"
        )
    for lab in labels:
        x, y = pos[lab]
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='#1b1f23'/>")
        # 标签沿径向朝外推，避免压在弧上
        ox, oy = x - cx, y - cy
        m = math.hypot(ox, oy) or 1.0
        tx, ty = x + ox / m * 26, y + oy / m * 20
        anchor = "middle" if abs(ox / m) < 0.4 else ("start" if ox > 0 else "end")
        parts.append(
            f"<text x='{tx:.1f}' y='{ty:.1f}' text-anchor='{anchor}' "
            "dominant-baseline='middle' font-size='12'>" + _esc(lab) + "</text>"
        )
    parts.append(
        f"<text x='{cx:.1f}' y='{cy:.1f}' text-anchor='middle' dominant-baseline='middle' "
        f"font-size='11' fill='#5b6570'>最粗 = {worst:.3f}</text>"
    )
    parts.append("</svg>")
    parts.append(
        f"<div class='legend'>只画干扰 &gt; {thr:.4f}（= 1/n_eval 分辨率下限）的 "
        f"{sum(1 for v in by.values() if v > thr)} 对。定向值（谁被谁破坏）在 "
        f"<code>conflict.json</code> 里，这张图只画无向连接。</div>"
    )
    return "".join(parts)


# ----------------------------------------------------------------------
def _scatter(result: PipelineResult, caps: Sequence[str], *, w: int = 700, h: int = 380) -> str:
    pts = []
    for r in result.runs:
        g = r.mean_gain(caps)
        if g is None:
            continue
        pts.append((r, g))
    if not pts:
        return "<p>（没有可画的组合。）</p>"
    pad_l, pad_r, pad_t, pad_b = 58, 18, 18, 42
    xs = [float(r.n_params_changed) for r, _ in pts]
    ys = [g for _, g in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys + [0.0]), max(ys + [0.0])  # 0 一定进范围：否则「负增益」看不见
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    sx = lambda v: pad_l + (v - x0) / (x1 - x0) * (w - pad_l - pad_r)
    sy = lambda v: h - pad_b - (v - y0) / (y1 - y0) * (h - pad_t - pad_b)

    front = set(result.frontier)
    parts = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' aria-label='增益代价散点'>"]
    parts.append(f"<rect x='{pad_l}' y='{pad_t}' width='{w - pad_l - pad_r}' height='{h - pad_t - pad_b}' "
                 "fill='#fff' stroke='#d8dee6'/>")
    # 0 线
    if y0 < 0 < y1:
        parts.append(f"<line x1='{pad_l}' y1='{sy(0):.1f}' x2='{w - pad_r}' y2='{sy(0):.1f}' "
                     "stroke='#1b1f23' stroke-dasharray='4 3' stroke-width='1'/>")
        parts.append(f"<text x='{pad_l - 6}' y='{sy(0) + 4:.1f}' text-anchor='end' font-size='11'>0</text>")
    for r, g in pts:
        col = _METHOD_COLOR.get(r.method, "#888")
        on = r.cid in front
        # 前沿点画在最上层且更大：否则它们会被后面画的点盖住
        parts.append(
            f"<circle cx='{sx(float(r.n_params_changed)):.1f}' cy='{sy(g):.1f}' "
            f"r='{4.2 if on else 2.6}' fill='{col}' fill-opacity='{0.95 if on else 0.45}'"
            + (" stroke='#1b1f23' stroke-width='1'" if on else "") + "/>"
        )
    parts.append(f"<text x='{w - pad_r}' y='{h - 8}' text-anchor='end' font-size='12'>"
                 "改动的基座元素数（越左越省）→</text>")
    parts.append(f"<text x='14' y='{pad_t + 12}' font-size='12'>平均增益 ↑</text>")
    parts.append("</svg>")
    legend = "　".join(
        f"<span style='color:{col}'>●</span> {m}" for m, col in _METHOD_COLOR.items()
    )
    return (f"{''.join(parts)}<div class='legend'>{legend}　○ 大圈并带描边 = 非支配前沿"
            f"（{len(front)} 个）</div>")


# ----------------------------------------------------------------------
def _heatmap(result: PipelineResult, *, cell: int = 46) -> str:
    labels = [a.label for a in result.adapters]
    n = len(labels)
    if n < 2:
        return "<p>（少于两个 Adapter，无热图可画。）</p>"
    by: dict[tuple[str, str], float] = {}
    for c in result.conflicts:
        lo, hi = sorted((c.a, c.b))
        by[(lo, hi)] = float(c.interference)
    worst = max(by.values(), default=0.0) or 1.0

    pad = 74
    w = pad + n * cell + 10
    h = pad + n * cell + 10
    parts = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' aria-label='干扰热图'>"]
    for i, a in enumerate(labels):
        parts.append(f"<text x='{pad - 8}' y='{pad + i * cell + cell * 0.62:.1f}' "
                     f"text-anchor='end' font-size='11'>{_esc(a)}</text>")
        parts.append(f"<text x='{pad + i * cell + cell / 2:.1f}' y='{pad - 8}' "
                     f"text-anchor='end' font-size='11' transform='rotate(-45 "
                     f"{pad + i * cell + cell / 2:.1f} {pad - 8})'>{_esc(a)}</text>")
        for j, b in enumerate(labels):
            x, y = pad + j * cell, pad + i * cell
            if i == j:
                # 对角线**留空**：自己与自己不是一次实测，填 0 会让「没有干扰」
                # 与「这个问题不存在」长得一样。
                parts.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                             "fill='#f2f5f8' stroke='#d8dee6'/>")
                continue
            v = by.get((a, b) if (a, b) <= (b, a) else (b, a), 0.0)
            t = max(0.0, min(1.0, v / worst))
            parts.append(
                f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' "
                f"fill='rgb({int(255 - 40 * t)},{int(255 - 165 * t)},{int(255 - 200 * t)})' "
                "stroke='#d8dee6'/>"
            )
            parts.append(f"<text x='{x + cell / 2:.1f}' y='{y + cell * 0.62:.1f}' "
                         f"text-anchor='middle' font-size='10' fill='#1b1f23'>{v:+.2f}</text>")
    parts.append("</svg>")
    return (f"{''.join(parts)}<div class='legend'>行 = 被观察的（它的本事还剩多少），"
            "列 = 挂上来的。颜色越深 = 干扰越大；对角线留空（自己与自己不是一次实测）。</div>")


# ----------------------------------------------------------------------
def _method_bars(result: PipelineResult, caps: Sequence[str], *, w: int = 700, h: int = 300) -> str:
    by: dict[tuple[str, ...], dict[str, CompositionRun]] = {}
    for r in result.runs:
        by.setdefault(tuple(r.members), {})[r.method] = r
    keys = sorted(by)[:24]
    if not keys:
        return "<p>（没有可画的组合。）</p>"
    methods = [m for m in ("stack", "merge", "route") if any(m in v for v in by.values())]

    pad_l, pad_b, pad_t = 52, 76, 16
    vals = [r.mean_gain(caps) for v in by.values() for r in v.values()]
    lo = min([0.0] + [v for v in vals if v is not None])
    hi = max([0.0] + [v for v in vals if v is not None])
    if hi == lo:
        hi = lo + 1
    sy = lambda v: pad_t + (hi - v) / (hi - lo) * (h - pad_t - pad_b)
    slot = (w - pad_l - 20) / len(keys)
    bw = max(1.5, slot / (len(methods) + 1.4))

    parts = [f"<svg viewBox='0 0 {w} {h}' width='{w}' height='{h}' role='img' aria-label='方法对照柱状图'>"]
    parts.append(f"<line x1='{pad_l}' y1='{sy(0):.1f}' x2='{w - 10}' y2='{sy(0):.1f}' stroke='#1b1f23'/>")
    parts.append(f"<text x='{pad_l - 6}' y='{sy(0) + 4:.1f}' text-anchor='end' font-size='11'>0</text>")
    for i, key in enumerate(keys):
        bx = pad_l + i * slot + slot * 0.15
        for j, m in enumerate(methods):
            r = by[key].get(m)
            if r is None:
                continue
            g = r.mean_gain(caps)
            if g is None:
                continue
            y0, y1 = sy(0), sy(g)
            top, hgt = min(y0, y1), abs(y1 - y0)
            parts.append(f"<rect x='{bx + j * bw:.1f}' y='{top:.1f}' width='{max(1.0, bw - 0.6):.1f}' "
                         f"height='{max(0.5, hgt):.1f}' fill='{_METHOD_COLOR[m]}' fill-opacity='0.9'/>")
        parts.append(f"<text x='{bx + len(methods) * bw / 2:.1f}' y='{h - pad_b + 14}' "
                     f"text-anchor='end' font-size='10' fill='#5b6570' "
                     f"transform='rotate(-55 {bx + len(methods) * bw / 2:.1f} {h - pad_b + 14})'>"
                     + _esc("+".join(key)) + "</text>")
    parts.append("</svg>")
    legend = "　".join(f"<span style='color:{c}'>■</span> {m}" for m, c in _METHOD_COLOR.items())
    return (f"{''.join(parts)}<div class='legend'>{legend}　（只列前 24 组，按成员字典序；"
            "全量在 <code>matrix.csv</code>）</div>")


def _findings(result: PipelineResult) -> str:
    if not result.findings:
        return ""
    items = []
    for f in result.findings:
        items.append(
            f"<li><b>{_esc(f.key)}</b> — <b>{_esc(f.verdict)}</b><br>"
            f"{_esc(f.statement)}<br>"
            f"<span style='color:#5b6570'>期望：{_esc(f.expectation)}<br>"
            f"实测：{_esc(f.measured)}<br>{_esc(f.note)}</span></li>"
        )
    return ("<section><h2>设计期望 vs 实测</h2>"
            "<p class='lead'>被推翻的条目才是报告的价值所在 —— 全部成立的期望"
            "说明这次运行没有测到任何新东西。</p>"
            f"<ul class='f'>{''.join(items)}</ul></section>")
