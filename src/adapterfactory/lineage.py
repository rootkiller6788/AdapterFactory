"""血缘图：**它在本项目里不只是「有一次运行过」的记录，它就是 Adapter Graph 的骨架。**

### 子集格与血缘图是同一张图

一个组合 ``combo:<cid>`` 的父节点**恰好**是它的成员 Adapter：

```
base:tiny:…──►adapter:upper-qv──┐
      │                          ├──►combo:<cid>──►eval:<cid>:upper
      └────────►adapter:shift-qv─┘        │
                                          └──►artifact:<cid>
```

于是 README 第五节要的那个「Adapter Graph」不是一个需要另画的独立产物 ——
它就是这张图去掉 ``baseline`` 那一支之后的子图。**这不是巧合，是设计**：
如果组合的父节点是「一个叫 compose 的黑盒」，那「谁和谁组合过」就只能靠读
``combo`` 节点的元数据去拼，而元数据是一个不会被任何机制校验的自由字段。

### 收拢点

``sel:compose`` 汇聚全部 ``combo:*``。它是扇出-收拢形状的标记：扇出的宽度
（276 个组合）与收拢的位置在这张图上都是**节点度数**，于是
「设计空间有多大、有多少进了报告」可以直接从图上读出来，不必去数 JSON 里的数组。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .rng import sha256_text

__all__ = [
    "Node",
    "Edge",
    "Lineage",
    "base_node",
    "dataset_node",
    "adapter_node",
    "combo_node",
    "eval_node",
    "artifact_node",
    "pair_node",
    "select_node",
    "SELECT_COMPOSE",
    "build_lineage",
    "dumps",
]


def base_node(name: str) -> str:
    return f"base:{name}"


def dataset_node(capability: str) -> str:
    return f"ds:{capability}"


def adapter_node(label: str) -> str:
    return f"adapter:{label}"


def combo_node(cid: str) -> str:
    return f"combo:{cid}"


def eval_node(cid: str, capability: str) -> str:
    return f"eval:{cid}:{capability}"


def artifact_node(cid: str) -> str:
    return f"artifact:{cid}"


def pair_node(a: str, b: str) -> str:
    """冲突对。**两个名字先排序再拼** —— 否则 ``pair(x,y)`` 与 ``pair(y,x)``
    会变成两个节点，于是同一对干扰在图上出现两次、各自只连一半的边。"""
    lo, hi = sorted((a, b))
    return f"pair:{lo}|{hi}"


def select_node(name: str) -> str:
    return f"sel:{name}"


#: 扇出的收拢点。全部 ``combo:*`` 汇到这里。
SELECT_COMPOSE = select_node("compose")


@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "meta": dict(self.meta)}


@dataclass
class Edge:
    src: str
    dst: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "label": self.label}


class Lineage:
    """一张有向无环图。**只增不减**，于是它天然是一张「发生过什么」的记录。"""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._out: dict[str, list[str]] = {}
        self._in: dict[str, list[str]] = {}

    # ---- 构建 --------------------------------------------------------
    def add_node(self, node_id: str, kind: str, label: str = "", **meta: Any) -> Node:
        """加节点。**同名节点重复加是幂等的**（但 kind 必须一致）。

        kind 不一致时抛错而不是覆盖：那意味着两个不同的东西抢了同一个 id，
        而这类冲突的正确处置是改 id，不是让后写的那个静默赢。
        """
        existing = self._nodes.get(node_id)
        if existing is not None:
            if existing.kind != kind:
                raise ValueError(f"节点 {node_id!r} 的 kind 不一致：{existing.kind!r} vs {kind!r}")
            existing.meta.update(meta)
            return existing
        node = Node(id=node_id, kind=kind, label=label or node_id, meta=dict(meta))
        self._nodes[node_id] = node
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])
        return node

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        if src not in self._nodes:
            raise ValueError(f"边的起点不存在：{src!r}")
        if dst not in self._nodes:
            raise ValueError(f"边的终点不存在：{dst!r}")
        edge = Edge(src=src, dst=dst, label=label)
        if edge in self._edges:
            return
        self._edges.append(edge)
        self._out[src].append(dst)
        self._in[dst].append(src)

    # ---- 查询 --------------------------------------------------------
    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"图里没有节点 {node_id!r}")
        return self._nodes[node_id]

    def nodes(self) -> tuple[Node, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda n: n.id))

    def edges(self) -> tuple[Edge, ...]:
        return tuple(sorted(self._edges, key=lambda e: (e.src, e.dst, e.label)))

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._in.get(node_id, ())))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._out.get(node_id, ())))

    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._nodes if not self._in.get(n)))

    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(n for n in self._nodes if not self._out.get(n)))

    def by_kind(self, kind: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes() if n.kind == kind)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        seen: set[str] = set()
        stack = list(self.parents(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return tuple(sorted(seen))

    def is_acyclic(self) -> bool:
        """拓扑排序能跑完 ⟺ 无环。**必须断言** —— 有环的「血缘」不是血缘。

        本项目里这张图有一处**天然可能成环**的地方：``compose`` 的候选既有
        size-1（就是一个 Adapter 自己）也有 size-2、size-3。若哪天支持
        「用组合出来的产物再组合」，就一定会产生 ``combo:A → combo:B → combo:A``
        这种环 —— 那时这个函数是唯一会说出来的机制。
        """
        indeg = {n: len(self._in.get(n, ())) for n in self._nodes}
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            cur = queue.pop()
            seen += 1
            for nxt in self._out.get(cur, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        return seen == len(self._nodes)

    # ---- 序列化 ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes()],
            "edges": [e.to_dict() for e in self.edges()],
            "digest": sha256_text(
                json.dumps([n.to_dict() for n in self.nodes()], sort_keys=True, ensure_ascii=False)
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Lineage":
        """**与 :meth:`to_dict` 严格互逆**（``digest`` 除外，它是派生的）。

        节点的 ``meta`` 原样传回 ``add_node(**meta)``：读回来的图必须能再加边、
        再查祖先。少还原一样东西的后果是「读回来的图看起来一样，但用不了」，
        而 :mod:`pipeline` 的只读命令**正是靠读回来的图**工作的。
        """
        g = cls()
        for n in d.get("nodes", ()):
            g.add_node(n["id"], n["kind"], n.get("label", ""), **(n.get("meta") or {}))
        for e in d.get("edges", ()):
            g.add_edge(e["src"], e["dst"], e.get("label", ""))
        return g

    def to_dot(self, *, kinds: Sequence[str] | None = None, nodes: Sequence[str] | None = None) -> str:
        """Graphviz DOT。两个过滤器**按名字区分，刻意不合并成一个 ``include``**。

        合并的代价很具体：一个按**种类**过滤的参数，被传入一组**节点 id**
        —— ``to_dot(kinds=graph.ancestors("sel:compose"))`` 是一句读起来完全
        正确的代码 —— 会一个都对不上，输出一张空图。而空图不是报错：
        ``digraph lineage {}`` 是一份合法 DOT，画出来就是白纸，
        调用方只会以为「图太稀疏了」。

        - ``kinds`` —— 只保留这些**种类**的节点；
        - ``nodes`` —— 只保留这些 **id** 的节点。

        两者同时给就是「且」。边在**两端都留下**时才画 —— 只留一端的边会让
        ``dot`` 替它补一个匿名节点出来，那张图看起来是对的。
        """
        keep_kinds = set(kinds) if kinds is not None else None
        keep_ids = set(nodes) if nodes is not None else None

        def keep(nid: str) -> bool:
            if keep_ids is not None and nid not in keep_ids:
                return False
            if keep_kinds is not None and self.node(nid).kind not in keep_kinds:
                return False
            return True

        lines = ["digraph lineage {", '  rankdir="LR";', '  node [shape=box, fontname="Helvetica"];']
        for n in self.nodes():
            if not keep(n.id):
                continue
            lines.append(f'  "{n.id}" [label="{_dot_label(n.label + " (" + n.kind + ")")}"];')
        for e in self.edges():
            if not (keep(e.src) and keep(e.dst)):
                continue
            label = f' [label="{_dot_label(e.label)}"]' if e.label else ""
            lines.append(f'  "{e.src}" -> "{e.dst}"{label};')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, *, limit: int = 60) -> str:
        """人看的概览：按种类数节点、按种类对数列边。"""
        kinds: dict[str, int] = {}
        for n in self.nodes():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        pairs: dict[tuple[str, str], int] = {}
        for e in self.edges():
            k = (self.node(e.src).kind, self.node(e.dst).kind)
            pairs[k] = pairs.get(k, 0) + 1
        out = [
            f"血缘图：{len(self._nodes)} 个节点 / {len(self._edges)} 条边"
            f"（无环={self.is_acyclic()}）",
            "  节点：" + "、".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        ]
        shown = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        out.append("  边：" + "、".join(f"{a}→{b}={n}" for (a, b), n in shown))
        if len(pairs) > limit:
            out.append(f"  （另有 {len(pairs) - limit} 种边未列出）")
        return "\n".join(out)


def _dot_label(s: str) -> str:
    """DOT 标签里最容易被忽略的两个字符：``"`` 与 ``\\``。

    组合的 ``cid`` 是十六进制摘要，不会出问题 —— 但 ``label`` 里有能力名与
    方法名，而它们来自于配置。一个带引号的能力名会让整份 DOT **语法错误**，
    而 ``dot`` 的报错会指向一个看不出与配置有关的位置。
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ----------------------------------------------------------------------
# 建图
# ----------------------------------------------------------------------
def _f(o: Any, *names: str, default: Any = None) -> Any:
    """按**属性或键**取一个字段，可以给多个候选名。

    :func:`build_lineage` 被三个调用点调用，**三处全都传** ``result.to_dict()``
    （一个 ``dict``）。而它原来的顶层读的是 ``getattr(result, "adapters", ())`` ——
    对一个 dict 取属性只会安静地拿到默认值。于是整张图退化成
    ``{'base:': …, 'sel:compose': …}`` 两个节点、零条边：**一个看起来很稀疏的图**，
    而稀疏在本项目里是完全可能的（只跑了一个组合就长这样）。

    症状是验收②「Adapter Graph 能明确指出哪些组合会互相干扰」会**空着通过** ——
    没有任何东西互相干扰，因为图上根本没有东西。而 ``out/graph.json``
    里那份 ``n_nodes: 2`` 也看不出错。

    **为什么要多个候选名**：序列化会改名字。``ArtifactRef.n_bytes`` 落 JSON 时
    叫 ``"bytes"``（``ArtifactRef.to_dict``），``PipelineResult.base_key`` 落 JSON 时
    嵌在 ``{"base": {"key": …}}`` 里。这不是谁写错了 —— dataclass 字段名与
    对外 JSON 键名本来就可以不同 —— 但它意味着「属性或键」这一层抽象不够，
    还得知道**两边各叫什么**。所以候选名逐个列出来，列不全就会在
    ``test_graph_lineage`` 里那条「两种形式必须给出同一张图」上炸掉。
    """
    for name in names:
        if isinstance(o, Mapping):
            if name in o:
                return o[name]
        elif hasattr(o, name):
            return getattr(o, name)
    return default


def _base_key(result: Any) -> str:
    """基座标识。**两种形式的名字不一样**，这正是 :func:`_f` 要多个候选名的原因。

    ``PipelineResult.base_key`` 在 JSON 里是 ``{"base": {"key": …}}`` —— 一层嵌套。
    照 ``_f(result, "base_key")`` 写会安静地拿到空串，于是全部 Adapter 挂在一个
    名叫 ``base:`` 的节点上：图建得出来、也不报错，只是**每个 Adapter 的基座
    都不见了**。而「这份 zoo 架在同一根基座上」正是组合可比较的前提。
    """
    if isinstance(result, Mapping):
        base = result.get("base")
        if isinstance(base, Mapping):
            return str(base.get("key") or "")
        return str(result.get("base_key") or "")
    return str(getattr(result, "base_key", "") or "")


def build_lineage(result: Any) -> Lineage:
    """从一次流水线结果建血缘图。

    ``result`` 可以是 :class:`~adapterfactory.schema.PipelineResult`，也可以是
    它的 ``to_dict()``（甚至一个半读出来的 ``result.json``）—— 两者必须给出
    **同一张图**，见 :func:`_f`。用鸭子类型而不是直接 import :mod:`schema`
    是为了让只读路径能从产物建图。
    """
    g = Lineage()
    base_key = _base_key(result)
    g.add_node(base_node(base_key), "base", f"基座 {base_key}", key=base_key)
    for cap in _f(result, "capabilities", ()) or ():
        g.add_node(dataset_node(cap), "dataset", f"能力 {cap}", capability=cap)

    # 收拢点**先建**，再建扇出。曾经它建在最后，于是「组合 → 收拢点」那条边
    # 的终点还不存在，``add_edge`` 直接抛 ``ValueError: 边的终点不存在``。
    # 这条也在空图 bug 的遮蔽之下 —— 零个组合时那条边根本不会被加。
    #
    # 顺序不是风格问题：``add_edge`` 拒绝悬空边是本项目要的性质（悬空边会被
    # ``dot`` 补出一个匿名节点，那张图看起来是对的），代价就是**建点的顺序
    # 成了契约**。所以这里把它写在注释里，并让它在空结果上也存在 ——
    # 否则 ``roots()`` / ``leaves()`` 会说「图是空的」而不是「扇出宽度为 0」。
    g.add_node(SELECT_COMPOSE, "select", "全部组合（扇出收拢点）")

    for ad in _f(result, "adapters", ()) or ():
        lab = _f(ad, "label")
        aid = adapter_node(lab)
        g.add_node(
            aid,
            "adapter",
            f"{lab}（r{_f(ad, 'rank')} → {len(_f(ad, 'targets', ()) or ())} 挂点）",
            # **不要再加 ``label=lab``。** ``add_node(node_id, kind, label, **meta)``
            # 的第三个形参就叫 ``label``，于是它进 ``meta`` 的唯一方式是**撞车**：
            # ``TypeError: got multiple values for argument 'label'``。
            # 这一行原本就在，只是在上面那个空图 bug 存在时**永远走不到** ——
            # 两个 bug 叠着，报出来的是后面那个，而根因是前面那个。
            # 裸标签本来也不需要：它已经是节点 id 里 ``adapter:`` 后面的那一段。
            capability=_f(ad, "capability"),
            variant=_f(ad, "variant"),
            digest=_f(ad, "digest"),
            n_params=_f(ad, "n_params"),
        )
        # **不连 base → adapter**：那会在图上给每个 Adapter 加一条边，而
        # 「它们共用同一份基座」这件事在 base 节点的 meta 里说一次就够。
        # 连上去的副作用是「adapter 的入度」不再等于 0，于是
        # `roots()` 会把 base 与所有 adapter 混在一起返回 —— 而
        # 「这份 zoo 是架在同一根基座上」正是组合可比较的前提，
        # 它值得一条自己名字的边，不值得被混进 roots()。

    for run in _f(result, "runs", ()) or ():
        cid = _f(run, "cid")
        members = list(_f(run, "members", ()) or ())
        cid_node = combo_node(cid)
        g.add_node(
            cid_node,
            "combo",
            f"{_f(run, 'method')} {'+'.join(members)}",
            cid=cid,
            method=_f(run, "method"),
            members=members,
            size=len(members),
        )
        # **这一组边就是子集格。** 组合的父节点恰好是它的成员。
        for m in members:
            g.add_edge(adapter_node(m), cid_node, label="成员")
        g.add_edge(cid_node, SELECT_COMPOSE, label="候选")

        art = _f(run, "artifact")
        if art is not None:
            art_id = artifact_node(cid)
            g.add_node(
                art_id, "artifact", f"{_f(art, 'n_bytes', 'bytes')} B",
                cid=cid, digest=_f(art, "digest"),
            )
            g.add_edge(cid_node, art_id, label="导出")

        for cap, m in (_f(run, "metrics") or {}).items():
            if m is None:
                continue
            acc = _f(m, "acc")
            ev = eval_node(cid, cap)
            g.add_node(ev, "eval", f"{cap} acc={acc:.4f}", capability=cap, acc=acc)
            g.add_edge(cid_node, ev, label=f"评测 {cap}")

    for c in _f(result, "conflicts", ()) or ():
        a, b = _f(c, "a"), _f(c, "b")
        pid = pair_node(a, b)
        g.add_node(
            pid,
            "conflict",
            f"{a} ~ {b}（实测 I={_f(c, 'interference', 0.0):+.3f}）",
            a=a,
            b=b,
            interference=_f(c, "interference"),
            predicted_cos=_f(c, "predicted_cos"),
            agree=_f(c, "agree"),
        )
        g.add_edge(adapter_node(a), pid)
        g.add_edge(adapter_node(b), pid)
    return g


def dumps(g: Lineage, path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
