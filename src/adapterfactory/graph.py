"""Adapter Graph：**README 第二节要的那张图。**

### 它不是画出来的，是算出来的

这张图有三个层次，而**只有第三层是本项目独有的**：

1. **子集格** —— 组合节点的父节点恰好是它的成员 Adapter。这一层是
   :mod:`lineage` 里 ``combo:<cid>`` 那一组边，不是新东西。
2. **冲突边** —— 两个 Adapter 直接相连，边上挂**双路证据**
   （参数空间预测 + 能力空间实测）。这一层是本项目对「干扰」的回答。
3. **前沿标记** —— 哪几个组合在非支配前沿上。它让「设计空间里的 276 个点，
   有几个值得部署」在图上是一个可数的东西。

### 「找得到干扰」意味着什么

README 的验收标准是「Adapter Graph 能**明确指出**哪些组合会互相干扰」。
「指出」这个词是可以被偷懒实现的：把 28 对全部标红也叫「指出」。
所以这里做三件事：

- 冲突边**必须带方向**（``i_a_from_b`` 与 ``i_b_from_a`` 分开），
  于是「谁被谁破坏」是可回答的；
- 每条边带 ``affected``（在哪几个能力上产生干扰），于是「干扰了什么」可回答；
- :func:`harmful_edges` / :func:`conflict_subgraph` 提供**从结论到证据的路径** ——
  报告里那句「这几对互相干扰」背后的边集合，就是这两个函数返回的东西。

### 为什么会有 ``stack`` / ``merge`` / ``route`` 三条不同的边集

干扰的定义随方法而变，而**只有 ``stack`` 的干扰是「净」的**：

- ``stack``：把别的成员原样加进来 ⇒ 破坏量就是干扰量；
- ``merge``：系数 ``1/k`` 是**故意**的稀释，破坏量 = 干扰 + 稀释，
  两者混在一起，不能当干扰读；
- ``route``：不合并权重，没有「权重被推开」这件事。

所以 :func:`conflict_subgraph` 默认只看 ``stack``，并且**参数可改但不是默认**
—— 想看 ``merge`` 的边必须显式说，于是报告里不会出现两张看起来一样、
含义不同的图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import lineage as L
from .schema import ConflictEdge

__all__ = [
    "GraphView",
    "harmful_edges",
    "conflict_subgraph",
    "frontier_nodes",
    "pair_edges",
    "graph_summary",
    "iter_capability_edges",
    "dominance",
    "pareto_frontier",
]


@dataclass(frozen=True)
class GraphView:
    """一张子图 + 它的来历。**``note`` 不是装饰** —— 它写清这张图排除了什么。"""

    name: str
    node_ids: tuple[str, ...]
    edge_keys: tuple[tuple[str, str], ...]
    note: str

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edge_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "note": self.note,
        }


def harmful_edges(conflicts: Sequence[ConflictEdge], *, threshold: float = 0.0) -> list[ConflictEdge]:
    """实测**有害**的那些边（``interference > threshold``），按破坏量降序。

    ``threshold`` 默认 0，而**调用方应当传一个分辨率下限**：干扰是
    ``gain`` 的差值，而 ``gain`` 的分辨率是 ``1/n_positions``。
    传 0 时，两条实测都是 0.0000 的边会因为 1e-17 的浮点噪声被判成
    「有害的正干扰」或「无害」，而**判决方向由噪声决定**。
    """
    out = [c for c in conflicts if c.interference > threshold]
    return sorted(out, key=lambda c: (-c.interference, c.a, c.b))


def pair_edges(conflicts: Sequence[ConflictEdge]) -> dict[tuple[str, str], ConflictEdge]:
    """``(a, b)`` → 边。**键里的两个名字先排序**，与 :func:`lineage.pair_node` 一致。"""
    out: dict[tuple[str, str], ConflictEdge] = {}
    for c in conflicts:
        lo, hi = sorted((c.a, c.b))
        key = (lo, hi)
        if key in out:
            raise ValueError(f"冲突边重复：{key}")
        out[key] = c
    return out


def conflict_subgraph(
    graph: L.Lineage,
    conflicts: Sequence[ConflictEdge],
    *,
    method_edges: Mapping[str, Sequence[ConflictEdge]] | None = None,
    threshold: float = 0.0,
    include_combos: bool = False,
) -> GraphView:
    """冲突子图：**只留 Adapter 节点与它们之间的冲突边**。

    ``method_edges`` 非空时，边上还会带上各方法的补充证据 —— 这是
    「``stack`` 破坏、``merge`` 救回来」那句话在图上可读的位置。

    ``include_combos=True`` 会把相关组合节点也拉进来。默认不拉：
    **一张图里的节点越多，越大声的那类节点就越显眼**，而 276 个组合节点
    会把 8 个 Adapter 与 28 条冲突边压成背景。
    """
    edges = harmful_edges(conflicts, threshold=threshold)
    keys: list[tuple[str, str]] = []
    for c in edges:
        pid = L.pair_node(c.a, c.b)
        if not graph.has(pid):
            # 图里没有这一对节点时**跳过而不是补一个**：补出来的节点没有
            # 「它由哪两个 Adapter 组成」这条边，于是它在图上是孤儿，
            # 而孤儿节点在可视化里看起来像「一个还没连上的东西」。
            continue
        keys.append((L.adapter_node(c.a), pid))
        keys.append((L.adapter_node(c.b), pid))
        if include_combos:
            for node in graph.by_kind("combo"):
                if set(node.meta.get("members") or []) == {c.a, c.b}:
                    keys.append((pid, node.id))
    note = (
        f"只画实测干扰 > {threshold:g} 的 {len(edges)} 对；"
        f"边的方向性证据（I_a←b 与 I_b←a）在 conflict.json 里，图上只画无向连接。"
    )
    if method_edges:
        note += f" 另有方法对比证据：{'、'.join(sorted(method_edges))}。"
    ids = tuple(sorted({n for pair in keys for n in pair}))
    return GraphView(name="conflict", node_ids=ids, edge_keys=tuple(keys), note=note)


def frontier_nodes(graph: L.Lineage, frontier: Sequence[str]) -> GraphView:
    """前沿上的组合节点。**收拢点 ``sel:compose`` 也拉进来**，
    否则这张图看不出「它是从 276 个候选里选出来的」。"""
    ids = [L.combo_node(cid) for cid in frontier if graph.has(L.combo_node(cid))]
    keys = [(cid_node, L.SELECT_COMPOSE) for cid_node in ids]
    return GraphView(
        name="frontier",
        node_ids=tuple(sorted(set(ids) | {L.SELECT_COMPOSE})),
        edge_keys=tuple(keys),
        note=f"非支配前沿 {len(ids)} 个组合；边指向收拢点 sel:compose。",
    )


def dominance(a: Sequence[float], b: Sequence[float]) -> bool:
    """``a`` 是否**支配** ``b``：每个轴都不更差，且至少一个轴更好。

    两个轴的方向由**调用方**在把数值传进来之前统一（本项目里已统一成
    「越大越好」—— 参数代价与 MACs 会先取负号）。不在这里带一个
    ``directions`` 参数，是因为「方向错了」的后果是前沿整体反了，
    而它是**一眼看不出来**的：反过来的前沿照样是一组互不支配的点。
    统一方向的地方只有一个，就在 :func:`pareto_frontier` 里。
    """
    if len(a) != len(b):
        raise ValueError(f"支配比较的两个点维数不同：{len(a)} vs {len(b)}")
    better = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better = True
    return better


def pareto_frontier(
    points: Mapping[str, Sequence[float]],
    *,
    maximize: Sequence[bool] | None = None,
) -> list[str]:
    """非支配前沿的 ``cid`` 列表（按 key 排序，稳定）。

    :func:`dominance` 说方向由调用方统一，而**这里就是那个调用方**。
    ``maximize[i]`` 为 ``False`` 的轴会先取负，再交给支配比较。

    为什么要单独一个参数而不是让调用方自己取负：取负这一步**漏掉任何一轴
    都会得到一个看起来正常的前沿**。所以它只在这里做一次，
    而且每个轴的期望方向都必须由调用方显式给出 —— 没有默认值，
    因为没有默认值就没有「忘了说方向」这回事。
    """
    keys = sorted(points)
    if not keys:
        return []
    dim = len(points[keys[0]])
    if maximize is None:
        raise ValueError("maximize 必须显式给出：漏说一个轴的方向会得到一个看起来正常的前沿")
    if len(maximize) != dim:
        raise ValueError(f"maximize 有 {len(maximize)} 个轴，点是 {dim} 维")
    for k in keys:
        if len(points[k]) != dim:
            raise ValueError(f"{k}: {len(points[k])} 维 vs {dim} 维")
    vals = {
        k: [v if maximize[i] else -v for i, v in enumerate(points[k])] for k in keys
    }
    front: list[str] = []
    for k in keys:
        if not any(dominance(vals[o], vals[k]) for o in keys if o != k):
            front.append(k)
    return front


def graph_summary(
    graph: L.Lineage,
    conflicts: Sequence[ConflictEdge],
    frontier: Sequence[str],
    *,
    n_adapters: int,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """一张摘要：图上有什么、多少、冲突有多严重。

    ``n_pairs`` 用 ``C(n,2)`` 算而不是数边数 —— 数边数会把「一对都没冲突」
    与「一对都没测」报成同一个 0。
    """
    from math import comb

    n_pairs = comb(n_adapters, 2)
    rated = [c for c in conflicts if c.agree is not None]
    return {
        "n_nodes": len(graph.nodes()),
        "n_edges": len(graph.edges()),
        "n_kinds": len({n.kind for n in graph.nodes()}),
        "is_acyclic": graph.is_acyclic(),
        "n_adapters": n_adapters,
        "n_pairs": n_pairs,
        "n_conflict_edges": len(conflicts),
        "n_harmful": len(harmful_edges(conflicts, threshold=threshold)),
        "worst": None if not conflicts else max(c.interference for c in conflicts),
        "two_path_agreement": (
            None if not rated else sum(1 for c in rated if c.agree) / len(rated)
        ),
        "n_frontier": len(frontier),
        "nodes_by_kind": _count_by_kind(graph),
    }


def _count_by_kind(graph: L.Lineage) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in graph.nodes():
        out[n.kind] = out.get(n.kind, 0) + 1
    return dict(sorted(out.items()))


def iter_capability_edges(conflicts: Iterable[ConflictEdge]) -> list[tuple[str, float]]:
    """把全部 ``affected`` 摊平成 ``(能力, 破坏量)``，用于「哪个能力最脆」的统计。"""
    out: list[tuple[str, float]] = []
    for c in conflicts:
        for cap, v in (c.affected or {}).items():
            out.append((cap, float(v)))
    return out
