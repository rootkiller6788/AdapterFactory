"""``compose``：设计空间（子集格）与三种实例化方式。

这个文件盯的是两类错，它们都不抛异常：

1. **设计空间多算或少算**。把子集当排列展开，276 就变成 400；行数看起来只是
   「更全」，但重复的两行只差浮点累加序，于是「取最大值」的前沿会挑中副本，
   前沿条数虚高。
2. **顺序悄悄换了一个来源**。``cid`` 由排序后的成员定，``runs.jsonl`` 的行序
   由枚举顺序定。两者必须都由同一份 ``labels`` 定，否则「两次运行逐位相同」
   这条验收只能靠读回 JSON 再比对。
"""

from __future__ import annotations

import pytest

from adapterfactory import compose as P
from adapterfactory import net as N


def _weight_diff(a: dict, b: dict) -> float:
    """两份权重表的最大逐元素差。

    ⚠️ 不能用 ``N.max_abs_diff(a, b)`` 直接比两张表 —— 它迭代的是**顶层可迭代对象**，
    喂 ``{名字: 张量}`` 进去会去比键（字符串），报 ``unsupported operand type(s)
    for -: 'str' and 'str'``。那个报错离「我在比权重」这个意图很远。
    """
    return max(N.max_abs_diff(a[k], b[k]) for k in a)


def _gates(shape: N.ModelShape, labels, *, seed: int = 11) -> dict:
    """route 用的门向量。**用真正的 :func:`net.derive_gates`，不手搓。**

    手搓过一版，两个地方各错一次：键写成挂点名 ``q_proj`` 而不是模块全名
    ``layers.0.q_proj``（``KeyError`` 指着的方向是对的），以及向量长度写成
    ``d_model`` 而实际要求是**那个模块的输入维**。手搓一遍等于把
    ``derive_gates`` 的契约抄一遍，而抄错的代价是每个 route 测试一起挂。
    """
    return N.derive_gates(shape, list(labels), seed=seed)


# ----------------------------------------------------------------------
# 设计空间
# ----------------------------------------------------------------------
class TestCid:
    def test_is_order_insensitive(self):
        """``{a,b}`` 与 ``{b,a}`` 是**同一条记录**，不是「碰巧算出一样」。
        不排序的话同一份组合会在 ``runs.jsonl`` 里出现两次，而
        ``run_by_cid`` 静默返回其中一条 —— 另一条上的指标永远读不到。"""
        assert P.cid_of("stack", ["a", "b"]) == P.cid_of("stack", ["b", "a"])

    def test_method_is_part_of_the_identity(self):
        """同一个子集的 stack 与 merge 是**两个**组合，不是同一个的两种算法。"""
        assert P.cid_of("stack", ["a", "b"]) != P.cid_of("merge", ["a", "b"])

    def test_members_are_part_of_the_identity(self):
        assert P.cid_of("stack", ["a", "b"]) != P.cid_of("stack", ["a", "c"])

    def test_duplicate_members_are_rejected(self):
        """同一个 Adapter 挂两次：``Σ`` 会把它算两遍。这不是「更宽的组合」，
        是一个本来就不该存在于设计空间里的点。"""
        with pytest.raises(ValueError, match="重复"):
            P.cid_of("stack", ["a", "a"])

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError):
            P.cid_of("stack", [])

    def test_is_short_and_stable(self):
        cid = P.cid_of("stack", ["a"])
        assert len(cid) == 12 and cid == P.cid_of("stack", ["a"])


class TestIterSubsets:
    def test_members_are_sorted(self):
        assert list(P.iter_subsets(["b", "a"], [2])) == [("a", "b")]

    def test_counts_match_the_binomials(self):
        got = list(P.iter_subsets(["a", "b", "c", "d"], [1, 2, 3]))
        assert len(got) == 4 + 6 + 4

    def test_no_size_zero_subsets(self):
        """空组合不是组合 —— 它是**基座**，由 :func:`realize_base` 单独表示。
        混进设计空间会让「基座」在矩阵里出现三次（每种方法一次），
        而它的 gain 恒为 0，于是均值被稀释。"""
        with pytest.raises(ValueError, match=">= 1"):
            list(P.iter_subsets(["a", "b"], [0]))

    def test_rejects_size_over_the_pool(self):
        with pytest.raises(ValueError, match="超过"):
            list(P.iter_subsets(["a", "b"], [3]))

    def test_enumeration_order_is_stable(self):
        labels = ["c", "a", "b"]
        assert list(P.iter_subsets(labels, [1, 2])) == list(P.iter_subsets(["a", "b", "c"], [1, 2]))


class TestGridSize:
    def test_the_readme_number(self):
        """8 个 Adapter 的 size 1..3 子集 × 3 方法 = ``92 × 3 = 276``。
        **这个数字从公式来，不写死** —— 写死之后 ``max_size`` 改了也没人发现，
        而 ``require_grid_complete`` 那条验收会永远通过。"""
        assert P.grid_size(8, [1, 2, 3], 1) == 8 + 28 + 56
        assert P.grid_size(8, [1, 2, 3], 3) == 276

    def test_scales_with_the_method_count(self):
        assert P.grid_size(4, [1, 2], 3) == (4 + 6) * 3

    def test_matches_the_actual_grid(self):
        """公式与实际展开**必须一致**。两个独立来源互相印证；
        只信公式的话，展开那边少一个子集也没人知道。"""
        labels = [f"a{i}" for i in range(8)]
        plans = P.plan_grid(labels, ("stack", "merge", "route"), [1, 2, 3])
        assert len(plans) == P.grid_size(len(labels), [1, 2, 3], 3)


class TestPlanGrid:
    def test_is_deterministic(self):
        labels = ["b", "a", "c"]
        first = [p.to_dict() for p in P.plan_grid(labels, ("stack", "merge"), [1, 2])]
        second = [p.to_dict() for p in P.plan_grid(labels, ("stack", "merge"), [1, 2])]
        assert first == second

    def test_cids_are_unique(self):
        from math import comb

        labels = [f"a{i}" for i in range(8)]
        plans = P.plan_grid(labels, ("stack", "merge", "route"), [1, 2, 3])
        assert len({p.cid for p in plans}) == sum(comb(8, k) for k in (1, 2, 3)) * 3

    def test_members_are_sorted_in_every_plan(self):
        labels = ["z", "m", "a"]
        for p in P.plan_grid(labels, ("stack",), [1, 2, 3]):
            assert list(p.members) == sorted(p.members)

    def test_all_methods_of_a_subset_appear_together(self):
        """行序是「先子集、后方法」。按方法外层循环会让同一子集的三行分散在
        文件的不同位置，而 ``runs.jsonl`` 存在的意义之一就是人能扫着读。"""
        plans = P.plan_grid(["a", "b"], ("stack", "merge", "route"), [1])
        assert [p.method for p in plans[:3]] == ["stack", "merge", "route"]

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="未知的组合方法"):
            P.plan_grid(["a"], ("nope",), [1])

    def test_size_is_a_property(self):
        p = P.plan_grid(["a", "b"], ("stack",), [2])[0]
        assert p.size == 2


class TestCoefficients:
    def test_stack_is_all_ones(self):
        assert P.coefficients("stack", 3) == [1.0, 1.0, 1.0]

    def test_merge_is_dilution(self):
        """``1/k`` 是**刻意的稀释**，不是干扰。这条区分是 ``merge_loss``
        存在的全部理由：它把稀释单独记一栏，而不是混进 ``interference``。"""
        assert P.coefficients("merge", 4) == [0.25] * 4

    def test_route_has_no_coefficients(self):
        """``route`` 返回 ``None`` 而不是全 1 —— 它根本不进权重空间。
        给它一组系数等于宣布「route 也是一种合并」，而报告的第三列
        就是为了反驳这句话。"""
        assert P.coefficients("route", 3) is None

    def test_unknown_method(self):
        with pytest.raises((ValueError, KeyError)):
            P.coefficients("nope", 2)


# ----------------------------------------------------------------------
# 实例化
# ----------------------------------------------------------------------
class TestRealize:
    def _plan(self, method: str, members=("a", "b")) -> P.Plan:
        return P.Plan(cid=P.cid_of(method, members), method=method, members=tuple(sorted(members)))

    def test_stack_produces_a_new_weight_table(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        plans = self._plan("stack")
        rz = P.realize(base_weights, ads, plans, shape)
        assert rz.adapters == ()
        assert rz.weights is not base_weights
        assert rz.merge_coefs == [1.0, 1.0]

    def test_merge_differs_from_stack_only_in_coefs(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        stack = P.realize(base_weights, ads, self._plan("stack"), shape)
        merge = P.realize(base_weights, ads, self._plan("merge"), shape)
        assert merge.merge_coefs == [0.5, 0.5]
        assert stack.weight_scheme != merge.weight_scheme

    def test_route_keeps_the_base_weights_object(self, shape, base_weights, adapter_factory):
        """``route`` 的 ``weights`` 就是**同一份**基座对象 —— 这是
        「不合并权重」这句话在内存里的样子。``==`` 不够，必须是 ``is``。"""
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        rz = P.realize(base_weights, ads, self._plan("route"), shape, gates=_gates(shape, ("a", "b")))
        assert rz.weights is base_weights
        assert len(rz.adapters) == 2
        assert rz.merge_coefs is None

    def test_route_without_gates_fails_at_forward(self, shape, base_weights, adapter_factory):
        """缺门向量要在**前向时**报错，而不是在实例化时 —— 实例化不跑任何东西。
        这条的性质比它看起来重要：它保证「route 不是 stack」不是靠约定。"""
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, self._plan("route", ("a",)), shape)
        with pytest.raises(ValueError, match="门向量"):
            rz.logits([1, 2, 3], shape)

    def test_label_mismatch_is_rejected(self, shape, base_weights, adapter_factory):
        """**这是本项目最容易发生、最难看出来的一类错**：``Σ Δᵢ`` 对两个成员
        可交换，所以顺序错了数值几乎相同，而 ``cid`` 指向的是另一个组合。"""
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        plan = P.Plan(cid="x", method="stack", members=("b", "a"))  # 顺序与 adapters 相反
        with pytest.raises(ValueError, match="不对应"):
            P.realize(base_weights, ads, plan, shape)

    def test_count_mismatch_is_rejected(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",))]
        with pytest.raises(ValueError, match="!= 成员"):
            P.realize(base_weights, ads, self._plan("stack"), shape)

    def test_n_delta_floats_counts_the_members(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj", "v_proj"))]
        rz = P.realize(base_weights, ads, self._plan("stack"), shape)
        assert rz.n_delta_floats == sum(a.n_params for a in ads)

    def test_realize_does_not_mutate_the_base(self, shape, base_weights, adapter_factory):
        """``stack``/``merge`` 走 :func:`~adapterfactory.net.merge_into`，
        而它**返回值、不就地改**。搞错的后果是静默的：第一个组合算完，
        基座已经被改过，后面每一个组合的 gain 都是在一个被污染的基础上算的。"""
        import copy

        snapshot = copy.deepcopy(base_weights)
        ads = [adapter_factory("a", ("q_proj",))]
        P.realize(base_weights, ads, self._plan("stack", ("a",)), shape)
        assert _weight_diff(base_weights, snapshot) == 0.0

    def test_base_weights_fixture_is_actually_zero_delta_free(
        self, shape, base_weights, adapter_factory
    ):
        """夹具的前提：``adapter_factory`` 造的是**非零** delta。
        否则上面「route 与 stack 不同」那几条会恒真地失败或恒真地通过。"""
        ad = adapter_factory("a", ("q_proj",))
        d = ad.delta("layers.0.q_proj", N.param_shapes(shape)["layers.0.q_proj"])
        assert any(abs(x) > 0.0 for row in d for x in row)


class TestRealizeBase:
    def test_cid_is_the_constant_base(self, shape, base_weights):
        rz = P.realize_base(base_weights, shape)
        assert rz.cid == "base" and rz.members == ()

    def test_equals_a_zero_delta_combination(self, shape, base_weights, adapter_factory):
        """基座 = 「系数全 0 的组合」。零初始化的适配器让这条性质可测
        （``init_adapter`` 只零初始化 B，所以 delta 恒为 0）。"""
        rz = P.realize_base(base_weights, shape)
        ids = list(range(shape.max_len))
        assert N.max_abs_diff(rz.logits(ids, shape), N.logits_of(base_weights, ids, shape)) == 0.0


class TestLogitsCache:
    def test_second_call_returns_the_same_object(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("c", "stack", ("a",)), shape)
        ids = tuple(range(shape.max_len))
        assert rz.logits(ids, shape) is rz.logits(ids, shape)

    def test_different_ids_are_cached_separately(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("c", "stack", ("a",)), shape)
        a = rz.logits(tuple(range(shape.max_len)), shape)
        b = rz.logits(tuple(range(1, shape.max_len + 1)), shape)
        assert a is not b

    def test_use_cache_false_bypasses_it(self, shape, base_weights, adapter_factory):
        """**计数必须绕开记忆化。** 命中的那次前向一个算子都没跑，于是它
        「消耗 0 MACs」—— 对「这次调用花了多少计算」是对的，对「这个组合
        一次前向要多少计算」是错的，而后者才是 latency 轴要的量。"""
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("c", "stack", ("a",)), shape)
        ids = tuple(range(shape.max_len))
        warm = rz.logits(ids, shape)
        cold = rz.logits(ids, shape, use_cache=False)
        assert cold is not warm
        assert N.max_abs_diff(cold, warm) == 0.0
        assert N.max_abs_diff(rz.logits(ids, shape), warm) == 0.0

    def test_list_ids_do_not_escape_the_cache(self, shape, base_weights, adapter_factory):
        """传 list 也能命中缓存 —— 键是 ``tuple(ids)``。这不是「偷偷 tuple 化」，
        是「缓存键由 ids 的**内容**决定，与容器类型无关」。"""
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("c", "stack", ("a",)), shape)
        ids = list(range(shape.max_len))
        assert rz.logits(ids, shape) is rz.logits(tuple(ids), shape)


class TestMethodDispatch:
    """三个方法**必须走同一条前向入口**，差别只在 :func:`realize` 里那一次分岔。"""

    def test_stack_and_merge_agree_when_the_members_agree(self, shape, base_weights, adapter_factory):
        """只有一个成员（k=1）时 ``1/k = 1``，merge 与 stack 必须是同一个东西。
        这条把「merge 的损失来自平均」这句话钉住：k=1 时没有平均，也就没有损失。"""
        ads = [adapter_factory("a", ("q_proj",))]
        ids = tuple(range(shape.max_len))
        stack = P.realize(base_weights, ads, P.Plan("s", "stack", ("a",)), shape)
        merge = P.realize(base_weights, ads, P.Plan("m", "merge", ("a",)), shape)
        assert N.max_abs_diff(stack.logits(ids, shape), merge.logits(ids, shape)) < 1e-15

    def test_route_differs_from_stack(self, shape, base_weights, adapter_factory):
        """用**非零** delta 的适配器，否则两者都等于基座，测试恒真。"""
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        ids = tuple(range(shape.max_len))
        stack = P.realize(base_weights, ads, P.Plan("s", "stack", ("a", "b")), shape)
        route = P.realize(
            base_weights, ads, P.Plan("r", "route", ("a", "b")), shape, gates=_gates(shape, ("a", "b"))
        )
        assert N.max_abs_diff(stack.logits(ids, shape), route.logits(ids, shape)) > 1e-6

    def test_route_degenerates_to_stack_with_one_member(self, shape, base_weights, adapter_factory):
        """**一个成员时 softmax 只有一个元素，恒等于 1 ⇒ route 退化成 stack。**

        这条比「route 与 stack 不同」有用得多：它把 route 的语义钉住
        ——「逐层在**该模块上的 k 个** delta 之间分配」—— 而不是只证明
        「那两个数不一样」。顺带说明为什么软最大必须逐层做：全局 softmax
        会让没挂的适配器也占一份权重，于是 route 在每一层都被稀释，
        而报告会把它读成「路由的代价」。
        """
        a = adapter_factory("a", ("q_proj",))
        ids = tuple(range(shape.max_len))
        route = P.realize(
            base_weights, [a], P.Plan("r", "route", ("a",)), shape, gates=_gates(shape, ("a",))
        )
        stack = P.realize(base_weights, [a], P.Plan("s", "stack", ("a",)), shape)
        assert N.max_abs_diff(route.logits(ids, shape), stack.logits(ids, shape)) < 1e-12

    def test_route_on_an_uncovered_module_needs_no_gate(self, shape, base_weights, adapter_factory):
        """适配器只挂 ``q_proj`` 时，``k_proj`` 上没有 delta 可混 ⇒ 那一层的
        门向量**根本不会被读**。这条保证「缺门向量」不会在无关模块上误报 ——
        否则 route 会强制要求为基座的每一个模块都给一份门，而其中大部分
        永远用不上。"""
        a = adapter_factory("a", ("q_proj",))
        dims = N.module_indims(shape)
        # 只给 ``q_proj`` 的那两层，别的模块一概不给 —— 这正是要验的东西。
        gates = {"a": {f: [0.0] * dims[f] for f in N.expand_targets(dims, ("q_proj",))}}
        rz = P.realize(base_weights, [a], P.Plan("r", "route", ("a",)), shape, gates=gates)
        # 能跑完就是结论：没有因为缺 k_proj 的门向量而报错。
        out = rz.logits(tuple(range(shape.max_len)), shape)
        assert len(out) == shape.max_len


# ----------------------------------------------------------------------
# 参数空间：冲突预测
# ----------------------------------------------------------------------
class TestCosineAndSign:
    def test_identical_deltas(self):
        d = {"m": [[1.0, 2.0], [3.0, 4.0]]}
        cos, rate, n = P.cosine_and_sign(d, d)
        assert cos == pytest.approx(1.0) and rate == pytest.approx(1.0) and n == 1

    def test_negated_deltas(self):
        a = {"m": [[1.0, 2.0]]}
        b = {"m": [[-1.0, -2.0]]}
        cos, rate, _ = P.cosine_and_sign(a, b)
        assert cos == pytest.approx(-1.0) and rate == pytest.approx(0.0)

    def test_disjoint_hooks_give_none(self):
        """没有共同挂点 ⇒ 无法预测，返回 ``None`` 而**不是 0**。
        ``0`` 会被下游读成「预测为无关联」，而这里的实情是「没有依据」。"""
        a = {"m1": [[1.0]]}
        b = {"m2": [[1.0]]}
        assert P.cosine_and_sign(a, b) == (None, None, 0)

    def test_orthogonal(self):
        a = {"m": [[1.0, 0.0]]}
        b = {"m": [[0.0, 1.0]]}
        cos, rate, _ = P.cosine_and_sign(a, b)
        assert cos == pytest.approx(0.0)
        # 两个坐标上都是 0 乘积 ⇒ 既不同号也不同号，不计入分母。
        assert rate is None

    def test_zero_deltas_give_none_cos(self):
        a = {"m": [[0.0, 0.0]]}
        b = {"m": [[1.0, 1.0]]}
        cos, rate, n = P.cosine_and_sign(a, b)
        assert cos is None and rate is None and n == 1

    def test_zero_coordinates_do_not_count_as_agreement(self):
        """**严格 ``> 0`` 才算同号。** 让 0 参与计数会把「一方恰好是 0」这种
        退化情形算成一致 —— 而 delta 恰好为 0 的坐标很常见
        （B 被裁剪到 0、未训练的挂点）。

        三个坐标里：第 0 个是 ``0·0``（不进分母）、第 1 个 ``+·+``（同号）、
        第 2 个 ``-·+``（异号）⇒ 分母 2、同号 1 ⇒ 0.5。
        余弦用的是**全部**坐标（包括那个 0）：``(0+1-1)/√(2·2) = 0``。
        两个量在同一次调用里对「零坐标」的处置不同，这是刻意的 ——
        余弦问方向，符号率问逐坐标的取向。
        """
        a = {"m": [[0.0, 1.0, -1.0]]}
        b = {"m": [[0.0, 1.0, 1.0]]}
        cos, rate, _ = P.cosine_and_sign(a, b)
        assert rate == pytest.approx(0.5)
        assert cos == pytest.approx(0.0)

    def test_only_common_hooks_are_compared(self):
        """**比较口径必须只在共同挂点上。** 把单边的挂点算进去，
        它那侧是「与零做余弦」，结果被**结构性地**拉向 0 ——
        于是「覆盖率差得越多的一对，预测的干扰越小」，
        而那恰好与「挂点不重叠所以不干扰」这个看起来合理的结论同向。"""
        a = {"m1": [[1.0, 1.0]], "m2": [[1.0, 1.0]]}
        b = {"m2": [[1.0, 1.0]]}
        cos, rate, n = P.cosine_and_sign(a, b)
        assert n == 1 and cos == pytest.approx(1.0) and rate == pytest.approx(1.0)

    def test_shape_mismatch_is_loud(self):
        a = {"m": [[1.0, 2.0]]}
        b = {"m": [[1.0]]}
        with pytest.raises(ValueError):
            P.cosine_and_sign(a, b)

    def test_reports_n_common(self):
        a = {"m1": [[1.0]], "m2": [[1.0]]}
        assert P.cosine_and_sign(a, a)[2] == 2


class TestDeltaOfPair:
    def test_covers_exactly_the_declared_targets(self, shape, base_weights, adapter_factory):
        ad = adapter_factory("a", ("q_proj", "v_proj"))
        d = P.delta_of_pair(ad, base_weights, shape)
        hooks = {k.rpartition(".")[2] for k in d}
        assert hooks == {"q_proj", "v_proj"}
        assert len(d) == 2 * shape.n_layers  # 每一层各一个

    def test_matches_adapter_delta(self, shape, base_weights, adapter_factory):
        ad = adapter_factory("a", ("q_proj",))
        d = P.delta_of_pair(ad, base_weights, shape)
        base = N.param_shapes(shape)
        full = "layers.0.q_proj"
        assert N.max_abs_diff(d[full], ad.delta(full, base[full])) == 0.0
