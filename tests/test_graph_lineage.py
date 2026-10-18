"""``lineage`` 与 ``graph``：**血统图不是记录，它就是 Adapter Graph 的骨架。**

README 的验收标准之一是「Adapter Graph 能明确指出哪些组合会互相干扰」。
这个文件验的是那份「指出」有没有内容：

- 组合节点的父节点**恰好**是它的成员（子集格 = 血缘图，不是两张图）；
- 冲突边**带方向与受影响能力**，不是一句「它们不合」；
- 从结论到证据的路径（``harmful_edges`` / ``conflict_subgraph``）存在且可查。

前半是纯函数的精确算术（``dominance`` / ``pareto_frontier`` / ``harmful_edges``），
后半吃 :func:`result` 那份真跑出来的图。
"""

from __future__ import annotations

import re
from math import comb

import pytest

from adapterfactory import graph as G
from adapterfactory import lineage as L
from adapterfactory import schema as S


# ----------------------------------------------------------------------
# Lineage 本体
# ----------------------------------------------------------------------
class TestLineageConstruction:
    def _g(self) -> L.Lineage:
        g = L.Lineage()
        g.add_node("a", "adapter", "A")
        g.add_node("b", "combo", "B")
        return g

    def test_same_id_twice_is_idempotent(self):
        g = self._g()
        g.add_node("a", "adapter", "A")
        assert len(g.nodes()) == 2

    def test_same_id_different_kind_raises(self):
        """两个不同的东西抢同一个 id，正确处置是改 id，不是让后写的那个静默赢。"""
        g = self._g()
        with pytest.raises(ValueError, match="kind 不一致"):
            g.add_node("a", "combo", "A")

    def test_meta_is_merged_not_replaced(self):
        g = self._g()
        g.add_node("a", "adapter", "A", extra=1)
        assert g.node("a").meta["extra"] == 1

    def test_edge_needs_both_ends(self):
        g = self._g()
        with pytest.raises(ValueError, match="终点不存在"):
            g.add_edge("a", "ghost")
        with pytest.raises(ValueError, match="起点不存在"):
            g.add_edge("ghost", "a")

    def test_duplicate_edge_is_idempotent(self):
        g = self._g()
        g.add_edge("a", "b")
        g.add_edge("a", "b")
        assert len(g.edges()) == 1

    def test_label_defaults_to_the_id(self):
        g = L.Lineage()
        g.add_node("x", "adapter")
        assert g.node("x").label == "x"

    def test_parents_and_children_are_inverses(self):
        g = self._g()
        g.add_edge("a", "b")
        assert g.parents("b") == ("a",)
        assert g.children("a") == ("b",)

    def test_ancestors_walks_the_whole_chain(self):
        g = L.Lineage()
        for n in "abc":
            g.add_node(n, "x")
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        assert g.ancestors("c") == ("a", "b")

    def test_ancestors_terminates_on_a_cycle(self):
        """``ancestors`` 用一个 ``seen`` 集合 —— 没有它，一个环会让它挂死，
        而挂死与「图很大」在表现上完全一样。

        在有环的图上，``ancestors("a")`` **包含 ``a`` 自己** —— 这不是副作用，
        是「可达性」这个词在环上的正确含义：``a`` 从 ``a`` 出发确实可达。
        把它读成「边界没处理干净」会诱人去加一条 ``if cur != node_id``，
        而那条判断会让环上的可达性少一个节点。"""
        g = L.Lineage()
        for n in "ab":
            g.add_node(n, "x")
        g.add_edge("a", "b")
        g.add_edge("b", "a")
        assert g.ancestors("a") == ("a", "b")
        assert not g.is_acyclic()

    def test_empty_graph_is_acyclic(self):
        assert L.Lineage().is_acyclic()

    def test_node_lookup_misses_loudly(self):
        with pytest.raises(KeyError, match="图里没有节点"):
            self._g().node("ghost")


class TestRoundTrip:
    def test_dict_round_trip_is_strict(self, result):
        g = L.build_lineage(result.to_dict())
        back = L.Lineage.from_dict(g.to_dict())
        assert back.to_dict()["nodes"] == g.to_dict()["nodes"]
        assert back.to_dict()["edges"] == g.to_dict()["edges"]

    def test_the_loaded_graph_is_usable(self, result):
        """**「看起来一样」不够。** 只读命令正是靠读回来的图工作的 ——
        少还原一样东西的后果是读回来的图查不了祖先、加不了边。"""
        g = L.Lineage.from_dict(L.build_lineage(result.to_dict()).to_dict())
        assert g.parents(L.SELECT_COMPOSE)
        assert g.ancestors(L.SELECT_COMPOSE)
        g.add_node("new", "x")
        g.add_edge("new", L.SELECT_COMPOSE)

    def test_digest_depends_only_on_nodes(self, result):
        """``digest`` 是**派生**字段：它必须能由节点集重算出来，
        否则「两份 lineage.json 是否同源」就无法回答。"""
        import json

        from adapterfactory.rng import sha256_text

        g = L.build_lineage(result.to_dict())
        d = g.to_dict()
        assert d["digest"] == sha256_text(
            json.dumps([n.to_dict() for n in g.nodes()], sort_keys=True, ensure_ascii=False)
        )


class TestDot:
    def test_valid_looking_dot(self, result):
        dot = L.build_lineage(result.to_dict()).to_dot()
        assert dot.startswith("digraph")
        assert dot.rstrip().endswith("}")
        assert dot.count("{") == dot.count("}") == 1

    #: 节点 id 与边端点都是位置固定的（``"id" [label=`` / ``"a" -> "b"``），
    #: 用正则锚在行首两个引号串上，而不是 ``split("->")`` —— 后者会被
    #: **标签里**恰好出现的 ``->`` 切断，而标签里可以有任何字符（见
    #: ``test_quote_in_a_label_is_escaped``）。本项目里 id 是由十六进制
    #: 摘要与标签拼出来的，不含引号，所以 ``[^"]*`` 够用。
    _NODE = re.compile(r'^\s*"([^"]*)"\s*\[label=')
    _EDGE = re.compile(r'^\s*"([^"]*)"\s*->\s*"([^"]*)"')

    def test_every_drawn_edge_has_both_ends_drawn(self, result):
        """只留一端的边会让 ``dot`` 替它补一个匿名节点出来，那张图**看起来是对的**。"""
        g = L.build_lineage(result.to_dict())
        dot = g.to_dot(kinds=["adapter", "conflict"])
        nodes, edges = set(), []
        for line in dot.splitlines():
            if (m := self._NODE.match(line)):
                nodes.add(m.group(1))
            elif (m := self._EDGE.match(line)):
                edges.append((m.group(1), m.group(2)))
        assert nodes and edges, "这份过滤下应当同时有节点与边"
        for src, dst in edges:
            assert src in nodes, f"只画了一端的边：{src} -> {dst}"
            assert dst in nodes, f"只画了一端的边：{src} -> {dst}"

    def test_kinds_and_nodes_are_not_interchangeable(self, result):
        """**坑**：``to_dot(kinds=graph.ancestors("sel:compose"))`` 是一句读起来
        完全正确的代码 —— 它把一堆**节点 id** 传给了按**种类**过滤的参数。
        一个都对不上，输出 ``digraph lineage {}``：一份合法 DOT，画出来是白纸。
        两个参数按名字区分、刻意不合并，就是为了让这一句读起来别扭。"""
        g = L.build_lineage(result.to_dict())
        ids = g.ancestors(L.SELECT_COMPOSE)
        assert 'combo:' not in g.to_dot(kinds=list(ids))
        assert "combo:" in g.to_dot(nodes=list(ids))

    def test_quote_in_a_label_is_escaped(self):
        g = L.Lineage()
        g.add_node("a", "x", 'has "quotes" and \\backslash')
        dot = g.to_dot()
        assert '\\"' in dot
        assert "\\\\" in dot

    def test_ascii_overview_counts_by_kind(self, result):
        txt = L.build_lineage(result.to_dict()).to_ascii()
        assert "血缘图：" in txt and "无环=True" in txt


# ----------------------------------------------------------------------
# 子集格 = 血缘图
# ----------------------------------------------------------------------
class TestSubsetLattice:
    """**本项目最关键的结构断言。**"""

    def test_combo_parents_are_exactly_its_members(self, result):
        """组合的父节点**恰好**是它的成员 Adapter，一个不多一个不少。

        这条成立，README 第五节要的 Adapter Graph 就不必另画一张 ——
        它就是这张图去掉 ``baseline`` 那一支后的子图。
        不成立的话，「谁和谁组合过」只能靠读 ``combo`` 节点的 ``meta`` 去拼，
        而 ``meta`` 是一个**不会被任何机制校验的自由字段**。
        """
        g = L.build_lineage(result.to_dict())
        combos = g.by_kind("combo")
        assert len(combos) == len(result.runs)
        for node in combos:
            want = sorted(L.adapter_node(m) for m in node.meta["members"])
            assert list(g.parents(node.id)) == want

    def test_members_meta_agrees_with_the_edges(self, result):
        """``members`` 有两条独立来源：节点 ``meta`` 与**入边**。
        这里额外断言两者一致 —— 不然 ``conflict_subgraph(include_combos=True)``
        用 ``meta`` 找组合、读者用边找组合，两边会给出不同的答案。"""
        g = L.build_lineage(result.to_dict())
        for node in g.by_kind("combo"):
            from_edges = sorted(p.split(":", 1)[1] for p in g.parents(node.id))
            assert from_edges == sorted(node.meta["members"])

    def test_every_combo_feeds_the_single_select_node(self, result):
        g = L.build_lineage(result.to_dict())
        assert len(g.parents(L.SELECT_COMPOSE)) == len(result.runs)
        assert g.children(L.SELECT_COMPOSE) == ()

    def test_select_node_exists_even_on_an_empty_result(self):
        """收拢点必须存在，哪怕一个组合都没有 —— 否则 ``roots()`` / ``leaves()``
        在空结果上给出的是「图是空的」，而不是「扇出宽度为 0」。"""
        g = L.build_lineage(_bare().to_dict())
        assert g.has(L.SELECT_COMPOSE)

    def test_adapters_and_datasets_are_roots(self, result):
        """**``base → adapter`` 那条边是刻意不画的。** 画上去的副作用是
        ``roots()`` 把 base 与所有 adapter 混在一起返回，于是
        「这份 zoo 架在同一根基座上」这件事从一个可查的结构退化成一句口号。"""
        g = L.build_lineage(result.to_dict())
        roots = set(g.roots())
        assert L.base_node(result.base_key) in roots
        assert {L.dataset_node(c) for c in result.capabilities} <= roots
        assert {L.adapter_node(a.label) for a in result.adapters} <= roots

    def test_the_whole_graph_is_acyclic(self, result):
        assert L.build_lineage(result.to_dict()).is_acyclic()

    def test_artifacts_hang_off_their_combo(self, result):
        g = L.build_lineage(result.to_dict())
        exported = [r for r in result.runs if r.artifact is not None]
        assert exported, "这份夹具带 export=True，应当有产物节点"
        for run in exported:
            art = L.artifact_node(run.cid)
            assert g.parents(art) == (L.combo_node(run.cid),)

    def test_one_eval_node_per_combo_per_capability(self, result):
        g = L.build_lineage(result.to_dict())
        assert len(g.by_kind("eval")) == len(result.runs) * len(result.capabilities)

    def test_conflict_pairs_are_sorted_so_a_pair_has_one_node(self):
        assert L.pair_node("b", "a") == L.pair_node("a", "b")


class TestBothFormsGiveTheSameGraph:
    """**这一组是一次真实故障的回归测试。**

    :func:`lineage.build_lineage` 的顶层读 ``getattr(result, "adapters", ())``，
    元素读 ``ad["label"]``；而三个调用点全都传 ``result.to_dict()``。
    对 dict 取属性安静地返回默认值 ⇒ 整张图退化成 ``{base:, sel:compose}``
    两个节点、零条边。它**不报错**，而且看起来只是一个稀疏的图 ——
    于是在这个 bug 底下，「Adapter Graph 能明确指出哪些组合会互相干扰」
    这条验收会空着通过：没有东西互相干扰，因为图上没有东西。

    修法是两种形式都读（``_f``）。**两种形式都必须被测到** ——
    只测一种，另一条路会再一次悄悄烂掉。
    """

    def test_dataclass_and_dict_agree(self, result):
        a = L.build_lineage(result)
        b = L.build_lineage(result.to_dict())
        assert a.to_dict()["nodes"] == b.to_dict()["nodes"]
        assert a.to_dict()["edges"] == b.to_dict()["edges"]

    def test_a_real_result_gives_a_graph_with_content(self, result):
        """**空图会静默通过验收。** 所以这里断言的是「有东西」：
        组合节点数 = 组合数，冲突对节点数 = 对数，并且冲突边真的连上了 Adapter。
        只写 ``assert graph.n_nodes > 0`` 是不够的 —— 收拢点那一个节点
        就足够让它成立。"""
        g = L.build_lineage(result.to_dict())
        assert len(g.by_kind("combo")) == len(result.runs) > 0
        assert len(g.by_kind("conflict")) == len(result.conflicts) > 0
        assert len(g.by_kind("eval")) == len(result.runs) * len(result.capabilities)
        for node in g.by_kind("conflict"):
            parents = g.parents(node.id)
            assert len(parents) == 2
            assert all(p.startswith("adapter:") for p in parents)

    def test_a_half_read_result_json_still_builds(self, result):
        """只读路径从 ``result.json`` 建图 —— 那里是**纯 JSON**：
        没有 dataclass，没有属性。这条断言把「构造函数的三种调用点传的是同一种
        东西」写下来。"""
        import json

        d = json.loads(json.dumps(result.to_dict()))
        assert len(L.build_lineage(d).by_kind("combo")) == len(result.runs)


def _bare() -> S.PipelineResult:
    """一个只有骨架的 ``PipelineResult``：没有 adapter、没有 run、没有冲突。"""
    return S.PipelineResult(
        version=S.Version("v1", "p", "v", "2026-01-01", 0, 0, 0),
        base_key="tiny", base_seed=1, seq_len=13, capabilities=[],
        adapters=[], runs=[], conflicts=[], frontier=[], counters={}, findings=[], notes={},
    )


# ----------------------------------------------------------------------
# 前沿与支配
# ----------------------------------------------------------------------
class TestDominance:
    def test_equal_points_do_not_dominate_each_other(self):
        """**``>=``≠支配。** 两个一模一样的点互不支配 —— 于是它们**都**在前沿上。
        用 ``>=`` 写支配会让每一个并列点都被剔除，前沿凭空变短。"""
        assert not G.dominance([1.0, 1.0], [1.0, 1.0])

    def test_strictly_better_on_one_axis_dominates(self):
        assert G.dominance([2.0, 1.0], [1.0, 1.0])

    def test_worse_on_any_axis_does_not(self):
        assert not G.dominance([2.0, 0.5], [1.0, 1.0])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="维数不同"):
            G.dominance([1.0], [1.0, 2.0])

    def test_pareto_excludes_the_dominated_point(self):
        pts = {"a": [2.0, 2.0], "b": [1.0, 1.0], "c": [0.5, 3.0]}
        assert G.pareto_frontier(pts, maximize=[True, True]) == ["a", "c"]

    def test_maximize_is_required(self):
        """有人漏说一个轴的方向，就会得到一个**看起来正常**的前沿 ——
        反过来的前沿照样是一组互不支配的点。所以没有默认值。"""
        with pytest.raises(ValueError, match="maximize 必须显式给出"):
            G.pareto_frontier({"a": [1.0, 2.0]}, maximize=None)

    def test_minimized_axis_flips_the_result(self):
        """同一对点，只翻一个轴的方向，答案就从「两个都留」变成「只留一个」。

        这就是「漏说一个轴的方向会得到一个看起来正常的前沿」的具体形状：
        两个版本都是一组互不支配的点。
        """
        pts = {"cheap": [0.1, 1.0], "good": [0.9, 5.0]}
        assert G.pareto_frontier(pts, maximize=[True, False]) == ["cheap", "good"]
        assert G.pareto_frontier(pts, maximize=[True, True]) == ["good"]

    def test_duplicates_all_survive(self):
        pts = {"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [0.0, 0.0]}
        assert G.pareto_frontier(pts, maximize=[True, True]) == ["a", "b"]

    def test_empty_input(self):
        assert G.pareto_frontier({}, maximize=[]) == []

    def test_axis_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="maximize 有"):
            G.pareto_frontier({"a": [1.0, 2.0]}, maximize=[True])

    def test_ragged_input_raises(self):
        with pytest.raises(ValueError, match="维"):
            G.pareto_frontier({"a": [1.0, 2.0], "b": [1.0]}, maximize=[True, True])


# ----------------------------------------------------------------------
# 冲突子图
# ----------------------------------------------------------------------
def _edge(a: str, b: str, i: float, *, agree=None) -> S.ConflictEdge:
    return S.ConflictEdge(
        a=a, b=b, interference=i, i_a_from_b=i, i_b_from_a=i,
        predicted_cos=None, predicted_sign=None, agree=agree, affected={"upper": i},
    )


class TestHarmfulEdges:
    def test_threshold_is_a_resolution_floor_not_zero(self):
        """``gain`` 的分辨率是 ``1/n_positions``。两条实测都是 0.0000 的边
        会因为 1e-17 的浮点噪声被判成有害或无害 —— **判决方向由噪声决定**。"""
        noise = 1e-17
        edges = [_edge("a", "b", noise), _edge("a", "c", 0.4)]
        assert len(G.harmful_edges(edges, threshold=0.0)) == 2
        assert [c.b for c in G.harmful_edges(edges, threshold=0.1)] == ["c"]

    def test_sorted_by_damage_then_names(self):
        edges = [_edge("a", "b", 0.1), _edge("a", "c", 0.5), _edge("b", "c", 0.1)]
        got = [(c.a, c.b) for c in G.harmful_edges(edges)]
        assert got == [("a", "c"), ("a", "b"), ("b", "c")]

    def test_negative_interference_is_not_harmful(self):
        assert G.harmful_edges([_edge("a", "b", -0.3)]) == []


class TestPairEdges:
    def test_key_is_sorted(self):
        e = _edge("z", "a", 0.2)
        assert list(G.pair_edges([e])) == [("a", "z")]

    def test_duplicate_pairs_raise(self):
        with pytest.raises(ValueError, match="冲突边重复"):
            G.pair_edges([_edge("a", "b", 0.2), _edge("b", "a", 0.2)])


class TestConflictSubgraph:
    def test_skips_pairs_that_are_not_in_the_graph(self, result):
        """**跳过而不是补一个。** 补出来的 pair 节点没有「它由哪两个 Adapter
        组成」这条边，于是它在图上是孤儿 —— 而在可视化里，孤儿画出来像
        「一个还没连上的东西」，读者会去找它该连到哪里。"""
        g = L.build_lineage(result.to_dict())
        ghosts = [_edge("ghost-1", "ghost-2", 0.9)]
        view = G.conflict_subgraph(g, ghosts, threshold=0.0)
        assert view.node_ids == ()
        assert view.n_edges == 0

    def test_note_states_the_threshold_and_the_missing_directionality(self, result):
        g = L.build_lineage(result.to_dict())
        view = G.conflict_subgraph(g, result.conflicts, threshold=0.05)
        assert "0.05" in view.note
        assert "方向性证据" in view.note

    def test_include_combos_off_by_default_because_noise(self, result):
        """一张图里的节点越多，越大声的那类节点就越显眼 —— 42 个组合节点
        会把 4 个 Adapter 与它们之间的冲突压成背景。"""
        g = L.build_lineage(result.to_dict())
        small = G.conflict_subgraph(g, result.conflicts, threshold=0.0)
        big = G.conflict_subgraph(g, result.conflicts, threshold=0.0, include_combos=True)
        assert small.n_nodes < big.n_nodes

    def test_real_graph_keeps_every_adapter_pair(self, result):
        g = L.build_lineage(result.to_dict())
        view = G.conflict_subgraph(g, result.conflicts, threshold=0.0)
        labels = {a.label for a in result.adapters}
        assert {n.split(":", 1)[1] for n in view.node_ids if n.startswith("adapter:")} == labels
        assert view.n_edges == 2 * len(result.conflicts)


class TestGraphSummary:
    def test_n_pairs_comes_from_the_formula_not_the_edges(self, tmp_path, result):
        """数边数会把「一对都没冲突」与「一对都没测」报成同一个 0。"""
        g = L.build_lineage(result.to_dict())
        n = len(result.adapters)
        s = G.graph_summary(g, [], [], n_adapters=n, threshold=0.0)
        assert s["n_pairs"] == comb(n, 2) > 0
        assert s["n_conflict_edges"] == 0
        assert s["worst"] is None

    def test_agreement_is_none_when_nothing_is_rated(self, result):
        g = L.build_lineage(result.to_dict())
        s = G.graph_summary(g, [_edge("a", "b", 0.2, agree=None)], [], n_adapters=4)
        assert s["two_path_agreement"] is None

    def test_agreement_rate_counts_only_rated_edges(self, result):
        g = L.build_lineage(result.to_dict())
        cs = [_edge("a", "b", 0.2, agree=True), _edge("a", "c", 0.2, agree=False),
              _edge("b", "c", 0.2, agree=None)]
        s = G.graph_summary(g, cs, [], n_adapters=4)
        assert s["two_path_agreement"] == pytest.approx(0.5)

    def test_real_summary_matches_the_conflict_count(self, result):
        g = L.build_lineage(result.to_dict())
        s = G.graph_summary(g, result.conflicts, result.frontier,
                            n_adapters=len(result.adapters), threshold=0.0)
        assert s["n_conflict_edges"] == len(result.conflicts)
        assert s["n_adapters"] == len(result.adapters)
        assert s["n_frontier"] == len(result.frontier)
        assert s["is_acyclic"] is True
        assert s["n_nodes"] == len(g.nodes()) and s["n_edges"] == len(g.edges())
        assert sum(s["nodes_by_kind"].values()) == s["n_nodes"]

    def test_kind_counts_are_sorted_for_diffable_output(self, result):
        g = L.build_lineage(result.to_dict())
        kinds = list(G.graph_summary(g, [], [], n_adapters=4)["nodes_by_kind"])
        assert kinds == sorted(kinds)


class TestFrontierView:
    def test_points_at_the_select_node(self, result):
        g = L.build_lineage(result.to_dict())
        view = G.frontier_nodes(g, result.frontier)
        assert L.SELECT_COMPOSE in view.node_ids
        assert all(L.combo_node(cid) in view.node_ids for cid in result.frontier)

    def test_unknown_cids_are_dropped_not_invented(self, result):
        g = L.build_lineage(result.to_dict())
        assert G.frontier_nodes(g, ["nope"]).node_ids == (L.SELECT_COMPOSE,)


class TestCapabilityEdges:
    def test_flattens_affected_into_pairs(self):
        got = G.iter_capability_edges([_edge("a", "b", 0.4)])
        assert got == [("upper", 0.4)]

    def test_empty_affected_is_empty(self):
        e = S.ConflictEdge(a="a", b="b", interference=0.1, i_a_from_b=0.1,
                           i_b_from_a=0.1, predicted_cos=None, predicted_sign=None,
                           agree=None, affected={})
        assert G.iter_capability_edges([e]) == []
