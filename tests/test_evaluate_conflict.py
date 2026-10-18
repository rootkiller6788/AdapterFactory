"""``evaluate`` 与 ``conflict``：**指标算术 + 两条路径的对质**。

这里的算术部分是**纯函数**，所以用手造的 :class:`Adjudication` 精确验 ——
不用真模型，也就不会把「算术对不对」与「模型学得好不好」混成一条断言。
真实模型只用来验两件它才有意义的事：基座贴着 chance，以及 route 的 MACs 更高。
"""

from __future__ import annotations

import pytest

from adapterfactory import compose as P
from adapterfactory import conflict as CF
from adapterfactory import evaluate as E
from adapterfactory import net as N


def _measure(cap: str, acc: float, *, exact: float | None = None, eos: float = 1.0) -> E.Measure:
    n = 8
    return E.Measure(
        capability=cap,
        n_eval=n,
        n_correct=int(round(acc * n)),
        n_positions=n,
        acc=acc,
        n_exact=int(round((acc if exact is None else exact) * n)),
        exact=acc if exact is None else exact,
        eos_acc=eos,
    )


def _adj(cid: str, method: str, members, **by_cap: float) -> E.Adjudication:
    return E.Adjudication(
        cid=cid,
        method=method,
        members=tuple(members),
        measures={cap: _measure(cap, v) for cap, v in by_cap.items()},
        n_eval_by_cap={cap: 8 for cap in by_cap},
    )


class TestAdjudicationAccessors:
    def test_missing_capability_is_none_not_zero(self):
        """**「没测」与「全错」必须是两个值。** 把 ``0.0`` 当成缺失会让
        没测过的组合看起来是最差的一种，于是它在 ``max(gain)`` 里永远
        排不上号 —— 而那是「静默丢掉一部分设计空间」最省事的写法。"""
        adj = _adj("c", "stack", ["a"], upper=0.5)
        assert adj.acc("upper") == pytest.approx(0.5)
        assert adj.acc("sort") is None
        assert adj.exact("sort") is None
        assert adj.hit("sort", 0) is None

    def test_capabilities_follows_insertion_order(self):
        adj = _adj("c", "stack", ["a"], upper=0.5, sort=0.25)
        assert adj.capabilities() == ["upper", "sort"]

    def test_to_dict_reports_mean_and_per_capability(self):
        adj = _adj("c", "stack", ["a"], upper=0.5, sort=0.25)
        d = adj.to_dict()
        assert d["mean_acc"] == pytest.approx(0.375)
        assert set(d["metrics"]) == {"sort", "upper"}
        assert d["metrics"]["upper"]["capability"] == "upper"

    def test_to_dict_of_nothing_has_no_mean(self):
        assert _adj("c", "stack", []).to_dict()["mean_acc"] is None


class TestGainArithmetic:
    def test_gain_is_relative_to_the_measured_base(self):
        """减数是**实测**的基座成绩，不是假设的 0.5 或 chance。
        用 chance 当减数会让 gain 里混进「这个能力本来就比乱猜容易多少」——
        那是任务的性质，不是适配器的贡献。"""
        base = _adj("base", "base", [], upper=0.125)
        run = _adj("c", "stack", ["a"], upper=0.5)
        assert E.gain(run, base, "upper") == pytest.approx(0.375)

    def test_gain_can_be_negative(self):
        base = _adj("base", "base", [], upper=0.5)
        run = _adj("c", "stack", ["a"], upper=0.25)
        assert E.gain(run, base, "upper") == pytest.approx(-0.25)

    def test_gain_propagates_none(self):
        base = _adj("base", "base", [], upper=0.125)
        run = _adj("c", "stack", ["a"], upper=0.5)
        assert E.gain(run, base, "sort") is None

    def test_gain_table_covers_every_requested_capability(self):
        base = _adj("base", "base", [], upper=0.125, shift=0.125)
        run = _adj("c", "stack", ["a"], upper=0.5, shift=0.25)
        t = E.gain_table(run, base, ["upper", "shift", "sort"])
        assert set(t) == {"upper", "shift", "sort"} and t["sort"] is None


class TestInterference:
    def _scene(self):
        base = _adj("base", "base", [], upper=0.125, sort=0.125)
        solo_a = _adj("a", "stack", ["a"], upper=0.5, sort=0.125)
        solo_b = _adj("b", "stack", ["b"], upper=0.125, sort=0.5)
        combo = _adj("ab", "stack", ["a", "b"], upper=0.2, sort=0.3)
        return base, solo_a, solo_b, combo

    def test_is_directed_and_asymmetric(self):
        """``I_{j→i}`` 与 ``I_{i→j}`` **一般不等** —— 一个强适配器可能几乎
        不受打扰，却让另一个弱的归零。「谁被谁破坏」正是要回答的问题，
        所以先算方向、再对称化，顺序反过来方向信息就丢了。"""
        base, solo_a, solo_b, combo = self._scene()
        a_from_b = E.interference(solo_a, combo, base, "upper")
        b_from_a = E.interference(solo_b, combo, base, "sort")
        assert a_from_b == pytest.approx((0.5 - 0.125) - (0.2 - 0.125))
        assert b_from_a == pytest.approx((0.5 - 0.125) - (0.3 - 0.125))
        assert a_from_b != pytest.approx(b_from_a)

    def test_symmetric_is_the_average(self):
        assert E.interference_sym(0.4, 0.2) == pytest.approx(0.3)

    def test_symmetric_is_none_if_either_side_is_missing(self):
        """**只有一个方向测过时不能拿单侧冒充对称值** —— 那会把一个方向的
        干扰读成双向的，而报告里每一行都要标方向。"""
        assert E.interference_sym(None, 0.2) is None
        assert E.interference_sym(0.4, None) is None

    def test_missing_base_gives_none(self):
        base = _adj("base", "base", [])
        solo = _adj("a", "stack", ["a"], upper=0.5)
        assert E.interference(solo, solo, base, "upper") is None


class TestSubadditivity:
    def _scene(self, combo_upper: float):
        base = _adj("base", "base", [], upper=0.0)
        solo_a = _adj("a", "stack", ["a"], upper=0.3)
        solo_b = _adj("b", "stack", ["b"], upper=0.2)
        combo = _adj("ab", "stack", ["a", "b"], upper=combo_upper)
        return base, solo_a, solo_b, combo

    def test_additive_when_it_is_zero(self):
        base, a, b, combo = self._scene(0.5)  # 0.3 + 0.2 = 0.5
        assert E.subadditivity(a, b, combo, base, "upper") == pytest.approx(0.0)

    def test_subadditive_is_positive(self):
        """同能力的一对必然落在这一档：它们在做同一件事，
        第二个的边际收益递减。"""
        base, a, b, combo = self._scene(0.4)
        assert E.subadditivity(a, b, combo, base, "upper") == pytest.approx(0.1)

    def test_superadditive_is_negative_and_allowed(self):
        """**接受负值。** 把超可加当成错误会让这一列只输出非负数，
        而「相加／相消／互相破坏」里第二个答案就永远给不出来。"""
        base, a, b, combo = self._scene(0.6)
        v = E.subadditivity(a, b, combo, base, "upper")
        assert v == pytest.approx(-0.1) and v < 0.0

    def test_none_if_any_of_the_three_is_missing(self):
        base = _adj("base", "base", [], upper=0.0)
        a = _adj("a", "stack", ["a"], upper=0.3)
        b = _adj("b", "stack", ["b"])
        combo = _adj("ab", "stack", ["a", "b"], upper=0.5)
        assert E.subadditivity(a, b, combo, base, "upper") is None


class TestMergeLoss:
    def test_measures_the_dilution(self):
        stack = _adj("s", "stack", ["a", "b"], upper=0.6)
        merge = _adj("m", "merge", ["a", "b"], upper=0.45)
        assert E.merge_loss(stack, merge, "upper") == pytest.approx(0.15)

    def test_refuses_to_compare_different_members(self):
        """不同成员相减会得到一个「差值」，而它没有任何解释。"""
        stack = _adj("s", "stack", ["a", "b"], upper=0.6)
        merge = _adj("m", "merge", ["a", "c"], upper=0.45)
        assert E.merge_loss(stack, merge, "upper") is None

    def test_none_input_stays_none(self):
        assert E.merge_loss(None, _adj("m", "merge", ["a"]), "upper") is None


class TestMergeIdentityError:
    """**实现检查，不是能力指标。** 恒等式在实数上成立，所以能查出的
    只有实现错误（漏一层、挂点展开不一致、系数用错），门槛才敢定在 1e-12。"""

    def test_holds_for_a_real_combination(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj", "v_proj")), adapter_factory("b", ("q_proj",))]
        ids = list(range(shape.max_len))
        assert E.merge_identity_error(base_weights, ads, ids, shape) < 1e-12

    def test_holds_for_a_single_adapter(self, shape, base_weights, adapter_factory):
        ads = [adapter_factory("a", ("q_proj",))]
        assert E.merge_identity_error(base_weights, ads, list(range(shape.max_len)), shape) < 1e-12

    def test_zero_adapters_is_also_the_identity(self, shape, base_weights):
        assert E.merge_identity_error(base_weights, [], list(range(shape.max_len)), shape) == 0.0

    def test_would_catch_a_wrong_coefficient(self, shape, base_weights, adapter_factory):
        """反证：换一组系数，这个量就真的会变大。没有这条，
        「恒等式成立」有可能只是因为 **两边都没算**。"""
        ads = [adapter_factory("a", ("q_proj",))]
        ids = list(range(shape.max_len))
        merged = N.merge_into(base_weights, ads, weights_by_adapter=[0.5])
        ref = N.logits_of(base_weights, ids, shape, adapters=ads)
        assert N.max_abs_diff(N.logits_of(merged, ids, shape), ref) > 1e-6


class TestMacs:
    """``latency`` 轴的全部证据都在这个函数里。"""

    def test_stack_costs_the_same_as_the_base(self, shape, base_weights, adapter_factory):
        """合并把矩阵加在一起，前向仍是一次更大的乘法 ⇒ MACs **逐位相同**。
        「合并省下的是计算」这句话在这个数字上成立。"""
        ids = list(range(shape.max_len))
        base = E.macs_of(P.realize_base(base_weights, shape), ids, shape)
        ads = [adapter_factory("a", ("q_proj",))]
        stack = E.macs_of(
            P.realize(base_weights, ads, P.Plan("s", "stack", ("a",)), shape), ids, shape
        )
        assert stack == base

    def test_route_costs_more(self, shape, base_weights, adapter_factory):
        """**这是 latency 轴存在的理由。** route 不合并，所以每个适配器
        都要多做一次 ``x @ Δᵢ``。若把它优化成「先合并再乘一次」，
        报告里 route 的高成本就凭空消失了。"""
        ids = list(range(shape.max_len))
        ads = [adapter_factory("a", ("q_proj",)), adapter_factory("b", ("q_proj",))]
        gates = N.derive_gates(shape, ["a", "b"], seed=3)
        route = E.macs_of(
            P.realize(base_weights, ads, P.Plan("r", "route", ("a", "b")), shape, gates=gates),
            ids,
            shape,
        )
        stack = E.macs_of(
            P.realize(base_weights, ads, P.Plan("s", "stack", ("a", "b")), shape), ids, shape
        )
        assert route > stack

    def test_bypasses_the_cache(self, shape, base_weights, adapter_factory):
        """**这是本项目最值钱的一条回归测试。**

        ``mac_ids`` 取的正是第一个评测样本的 ids，而 ``measure`` 先跑、
        已经把那些 ids 灌进了缓存 —— 于是 ``macs_of`` 次次命中，
        整条 latency 轴恒为 0。而 **0 看起来只是一个「很小但合理」的值**：
        它不抛错、不违反任何检查，只是让 ``pareto_frontier`` 的第三个轴
        退化成恒等，让 route「多做一次乘法」失去证据。

        所以这里**先**把同一份 ids 的前向跑热，**再**计数，并要求非零。
        """
        ids = tuple(range(shape.max_len))
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("s", "stack", ("a",)), shape)
        rz.logits(ids, shape)  # 灌热缓存
        assert E.macs_of(rz, list(ids), shape) > 0

    def test_counting_does_not_poison_the_cache(self, shape, base_weights, adapter_factory):
        """计数**不该**把结果写回缓存，也不该把它清掉 —— 它是只读的旁观者。"""
        ids = tuple(range(shape.max_len))
        ads = [adapter_factory("a", ("q_proj",))]
        rz = P.realize(base_weights, ads, P.Plan("s", "stack", ("a",)), shape)
        first = rz.logits(ids, shape)
        E.macs_of(rz, list(ids), shape)
        assert rz.logits(ids, shape) is first


class TestMeasureOnRealModel:
    def test_base_sits_at_chance(self, ctx, shape):
        """**S0 的门槛 ①。** 基座在四个能力上都必须贴着 chance ——
        不贴的话 ``gain`` 里就混进了「基座本来就有一点本事」那一份，
        而本项目全部结论都建立在 gain 上。"""
        caps = list(ctx["cfg"]["data"]["capabilities"])
        adj = E.measure(P.realize_base(ctx["base_weights"], shape), ctx["eval_sets"], shape, caps)
        for cap in caps:
            chance = ctx["baselines"][cap]["chance"]
            assert abs(adj.acc(cap) - chance) <= ctx["cfg"]["acceptance"]["base_chance_tolerance"]

    def test_records_n_eval_per_capability(self, ctx, shape):
        """每个能力的探针集大小**逐能力记**，不取一个全局值 ——
        全局值的风险是「某个能力的题少了几个」不会在任何地方显示出来。"""
        caps = list(ctx["cfg"]["data"]["capabilities"])
        adj = E.measure(P.realize_base(ctx["base_weights"], shape), ctx["eval_sets"], shape, caps)
        assert set(adj.n_eval_by_cap) == set(caps)

    def test_missing_eval_set_is_skipped_not_zeroed(self, ctx, shape):
        adj = E.measure(
            P.realize_base(ctx["base_weights"], shape),
            ctx["eval_sets"],
            shape,
            [*ctx["cfg"]["data"]["capabilities"], "not-a-capability"],
        )
        assert adj.acc("not-a-capability") is None

    def test_acc_is_within_the_unit_interval(self, ctx, shape):
        caps = list(ctx["cfg"]["data"]["capabilities"])
        adj = E.measure(P.realize_base(ctx["base_weights"], shape), ctx["eval_sets"], shape, caps)
        for cap in caps:
            assert 0.0 <= adj.acc(cap) <= 1.0

    def test_exact_is_never_above_acc(self, ctx, shape):
        """整串全对比逐 token 严格 —— 每一个全对的样本当然也逐 token 全对，
        反之不成立。这条不等式是「两把尺子」之间唯一必须成立的关系。"""
        caps = list(ctx["cfg"]["data"]["capabilities"])
        adj = E.measure(P.realize_base(ctx["base_weights"], shape), ctx["eval_sets"], shape, caps)
        for cap in caps:
            assert adj.exact(cap) <= adj.acc(cap) + 1e-12

    def test_eos_acc_is_separate(self, ctx, shape):
        """EOS 单独一格。合成一个数会让「答对了但不停」与「停下来但答错了」
        拿到同一个分数 —— 而它们在部署时的下场完全不同。"""
        from adapterfactory import data as D

        caps = list(ctx["cfg"]["data"]["capabilities"])
        adj = E.measure(P.realize_base(ctx["base_weights"], shape), ctx["eval_sets"], shape, caps)
        for cap in caps:
            m = adj.measures[cap]
            assert 0.0 <= m.eos_acc <= 1.0
            # EOS 不进 ``acc`` 的分母：位置数**精确等于**「样本数 × 每个样本的内容位数」，
            # 而回答位比内容位多一个（那个 EOS）。
            per_sample = len(D.content_positions(ctx["eval_sets"][cap][0]))
            assert m.n_positions == m.n_eval * per_sample
            assert ctx["eval_sets"][cap][0].response_len == per_sample + 1


# ----------------------------------------------------------------------
# conflict
# ----------------------------------------------------------------------
class TestPredictors:
    def test_identical_adapters(self, shape, base_weights, adapter_factory):
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("q_proj",), seed=7)
        # 让两份 delta 完全一样：把 b 的张量换成 a 的
        import copy

        b.tensors = copy.deepcopy(a.tensors)
        p = CF.predictors_of("a", a, "b", b, base_weights, shape)
        assert p.cos == pytest.approx(1.0)
        assert p.expansion == pytest.approx(2.0)
        assert p.n_common == shape.n_layers

    def test_opposite_adapters_shrink_to_zero(self, shape, base_weights, adapter_factory):
        """``Δb = −Δa`` ⇒ ``cos = −1``、``expansion = ‖Δa+Δb‖/max = 0``。

        ⚠️ 只能翻**一个**因子。把 ``A`` 与 ``B`` 一起取反得到的是同一个 delta
        （``(−B)(−A) = BA``）—— 我第一版就是这么写的，于是 ``cos`` 算出来是
        ``+1``，而 ``+1`` 看起来同样「像个答案」。
        ``delta`` 里 ``A`` 与 ``B`` 是相乘的，所以「取反一个 delta」这件事
        必须落在恰好一个因子上。
        """
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("q_proj",), seed=7)
        b.tensors = {k: [list(r) for r in v] for k, v in a.tensors.items()}
        for k in b.tensors:
            if k.endswith(".lora_A"):
                b.tensors[k] = [[-x for x in row] for row in b.tensors[k]]
        p = CF.predictors_of("a", a, "b", b, base_weights, shape)
        assert p.cos == pytest.approx(-1.0)
        assert p.expansion == pytest.approx(0.0, abs=1e-9)

    def test_disjoint_targets_predict_nothing(self, shape, base_weights, adapter_factory):
        """``qv`` 与只挂 ``gate_proj`` 的适配器没有共同挂点 ⇒ ``cos`` 与
        ``expansion`` 都是 ``None``（**不是 0**），而重叠度是 0.0 ——
        两者是不同的问题：「测不了」与「测了，是零接触面」。"""
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("gate_proj",))
        p = CF.predictors_of("a", a, "b", b, base_weights, shape)
        assert p.n_common == 0 and p.cos is None and p.expansion is None
        assert p.overlap_ab == pytest.approx(0.0)

    def test_overlap_is_the_touch_surface(self, shape, base_weights, adapter_factory):
        """``overlap_ab``：``a`` 的改动里有多少落在 ``b`` 也动的地方。"""
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("q_proj",))
        p = CF.predictors_of("a", a, "b", b, base_weights, shape)
        assert p.overlap_ab == pytest.approx(1.0)

    def test_overlap_is_directional(self, shape, base_weights, adapter_factory):
        """``qv`` 的改动**全部**落在 ``all`` 覆盖的 7 个模块里 ⇒ 从 ``qv``
        看去是 1.0；反过来 ``all`` 还有 5 个模块是 ``qv`` 不碰的 ⇒ 小于 1。

        **两个方向必须分开报** —— 一个数会把「接触面」这件事的方向丢掉。
        至于反过来的那个数**具体是多少**，取决于各模块的 ‖Δ‖ 之比，
        而那是个模型量、不是结构量，所以这里只断言方向。
        确切的算术由 :class:`TestNormAndOverlap` 用手造张量精确验。
        """
        qv = adapter_factory("qv", ("q_proj", "v_proj"))
        al = adapter_factory("all", tuple(N.LORA_ABLE_MODULES))
        p = CF.predictors_of("qv", qv, "all", al, base_weights, shape)
        assert p.overlap_ab == pytest.approx(1.0)
        assert 0.0 < p.overlap_ba < 1.0


class TestNormAndOverlap:
    """``_norm_and_overlap`` / ``_total_sq`` 是 ``overlap``/``expansion``/``norm_ratio``
    三个量共同的底。**用手造张量精确验**，因为到了真适配器上，这三个数
    都取决于各模块 ‖Δ‖ 之比 —— 那是个模型量，拿它当期望值等于把
    「算术对不对」和「这个适配器训得怎么样」混成一条断言。"""

    def test_norms_are_squared_and_summed(self):
        da = {"m": [[1.0, 2.0]]}
        db = {"m": [[3.0, 4.0]]}
        sa, sb, sab, ca, cb = CF._norm_and_overlap(da, db)
        assert sa == pytest.approx(5.0)
        assert sb == pytest.approx(25.0)
        assert sab == pytest.approx((4.0) ** 2 + (6.0) ** 2)
        assert (ca, cb) == (5.0, 25.0)

    def test_only_common_hooks_enter(self):
        """共同挂点是**唯一**的累加范围 —— 单边挂点上一方是「与零相加」，
        那会把重叠度结构性地拉向 0，而症状是「覆盖率差得越多的一对，
        重叠越小」，看起来完全合理。"""
        da = {"m1": [[1.0]], "m2": [[100.0]]}
        db = {"m1": [[1.0]], "m3": [[100.0]]}
        sa, sb, sab, ca, cb = CF._norm_and_overlap(da, db)
        assert (sa, sb) == (1.0, 1.0)
        assert (ca, cb) == (1.0, 1.0)

    def test_total_sq_covers_every_named_hook(self):
        d = {"m1": [[3.0]], "m2": [[4.0]]}
        assert CF._total_sq(d, ["m1", "m2"]) == pytest.approx(25.0)
        assert CF._total_sq(d, ["m1"]) == pytest.approx(9.0)

    def test_expansion_of_doubled_deltas(self):
        """两倍关系的两个 delta：``‖a+b‖/max = 3/2``。"""
        import math

        da, db = {"m": [[1.0]]}, {"m": [[2.0]]}
        sa, sb, sab, _, _ = CF._norm_and_overlap(da, db)
        assert math.sqrt(sab) / math.sqrt(max(sa, sb)) == pytest.approx(1.5)


    def test_norm_ratio_is_directional(self, shape, base_weights, adapter_factory):
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("q_proj",), seed=99)
        p = CF.predictors_of("a", a, "b", b, base_weights, shape)
        q = CF.predictors_of("b", b, "a", a, base_weights, shape)
        assert p.norm_ratio is not None and q.norm_ratio is not None
        assert p.norm_ratio * q.norm_ratio == pytest.approx(1.0)

    def test_mixed_targets_are_not_silently_excluded(self, shape, base_weights, adapter_factory):
        """``upper-qv`` 与 ``sort-all`` 的 ``overlap`` 是**特意不取 1.0** 的：
        它们确实共用 ``q,v``，但各自还动了别的模块。这一条防的是
        「只算共同挂点 ⇒ 重叠永远是 1」这种把两个量混成一个的实现。"""
        qv = adapter_factory("qv", ("q_proj", "v_proj"))
        al = adapter_factory("all", tuple(N.LORA_ABLE_MODULES))
        p = CF.predictors_of("qv", qv, "all", al, base_weights, shape)
        assert p.n_common == 2 * shape.n_layers
        assert 0.0 < p.overlap_ba < 1.0


class TestDirectionSign:
    def test_negative_cos_predicts_harm(self):
        assert CF.direction_sign(-0.1) is True

    def test_zero_cos_predicts_no_harm(self):
        """边界取 ``< 0`` 严格不等：恰好正交（0.0）意味着**没有可用的方向信息**，
        判成「有害」会让同向率在一个抛硬币的输入上涨上去。"""
        assert CF.direction_sign(0.0) is False

    def test_positive_cos_predicts_no_harm(self):
        assert CF.direction_sign(0.3) is False

    def test_none_stays_none(self):
        """**预测不了 ≠ 预测为无害。** 返回 ``False`` 会让「两路一致率」
        把那些本来预测不出来的对算成成功的预测。"""
        assert CF.direction_sign(None) is None


class TestPairConflict:
    def _scene(self, cos: float | None = -0.5):
        base = _adj("base", "base", [], upper=0.125, sort=0.125)
        solo_a = _adj("a", "stack", ["a"], upper=0.5, sort=0.125)
        solo_b = _adj("b", "stack", ["b"], upper=0.125, sort=0.5)
        combo = _adj("ab", "stack", ["a", "b"], upper=0.2, sort=0.3)
        pred = CF.Predictors(
            a="a", b="b", cos=cos, sign_rate=0.5, n_common=4,
            expansion=1.4, overlap_ab=1.0, overlap_ba=1.0, norm_ratio=1.0,
        )
        return pred, solo_a, solo_b, combo, base

    def test_edge_is_symmetric_in_interference_directed_in_direction(self):
        pred, sa, sb, combo, base = self._scene()
        e = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        assert e.interference == pytest.approx(
            0.5 * ((0.5 - 0.2) + (0.5 - 0.3))
        )
        assert e.i_a_from_b == pytest.approx(0.3)
        assert e.i_b_from_a == pytest.approx(0.2)

    def test_agree_uses_the_sign_of_cos(self):
        pred, sa, sb, combo, base = self._scene(cos=-0.4)
        e = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        assert e.predicted_cos == -0.4
        assert e.agree is True  # 预测有害，实测干扰为正

    def test_agree_is_none_when_cos_is_none(self):
        pred, sa, sb, combo, base = self._scene(cos=None)
        e = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        assert e.agree is None

    def test_affected_is_sorted_by_magnitude(self):
        """``affected`` 让「这一对互相干扰」有一个可追溯的落点，
        而不是一个孤零零的标量。按量级降序 —— 报告里第一行就是最该看的那个。"""
        pred, sa, sb, combo, base = self._scene()
        e = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        vals = list(e.affected.values())
        assert vals == sorted(vals, key=abs, reverse=True)

    def test_affected_compares_against_the_better_member(self):
        """用「更好的那个单干」当基准，而不是平均值 —— 用平均会让
        「一强一弱」的那一对看起来比实际平和。"""
        pred, sa, sb, combo, base = self._scene()
        e = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        assert e.affected["upper"] == pytest.approx(0.5 - 0.2)

    def test_a_missing_side_raises_instead_of_guessing(self):
        """有一侧测不出来时**不猜** —— 用一个单侧值冒充对称值，
        数值上完全合理，而「方向」这个信息被静默丢掉了。"""
        pred, sa, _sb, combo, base = self._scene()
        bare = _adj("b", "stack", ["b"])  # 一个能力都没有 ⇒ 定向干扰缺一侧
        with pytest.raises(ValueError, match="缺一侧"):
            CF.pair_conflict(pred, sa, bare, combo, base, cap_a="upper", cap_b="sort")

    def test_threshold_makes_the_verdict_resolution_aware(self):
        """正负号在浮点噪声下不稳定：两个实测都是 0.0000 的场合，
        符号由 1e-17 决定。调用方可以给一个分辨率下限（例如 1/n_positions）。"""
        pred, sa, sb, combo, base = self._scene(cos=-0.4)
        e0 = CF.pair_conflict(pred, sa, sb, combo, base, cap_a="upper", cap_b="sort")
        e1 = CF.pair_conflict(
            pred, sa, sb, combo, base, cap_a="upper", cap_b="sort", threshold=0.99
        )
        assert e0.agree is True and e1.agree is False


class TestAllPairs:
    def test_pairs_every_adapter_exactly_once(self, shape, base_weights, adapter_factory):
        labels = ["a", "b", "c"]
        ads = {lab: adapter_factory(lab, ("q_proj",)) for lab in labels}
        records = [_rec(lab, "upper") for lab in labels]
        base = _adj("base", "base", [], upper=0.125)
        solo = {lab: _adj(lab, "stack", [lab], upper=0.5) for lab in labels}
        combos = {}
        for i, a in enumerate(labels):
            for b in labels[i + 1 :]:
                combos[P.cid_of("stack", (a, b))] = _adj("x", "stack", [a, b], upper=0.2)
        edges, preds = CF.all_pairs(
            records, solo, combos, base, base_weights, shape, ads, capabilities=["upper"]
        )
        assert len(edges) == 3 and len(preds) == 3
        assert {(e.a, e.b) for e in edges} == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_missing_combo_is_loud(self, shape, base_weights, adapter_factory):
        """缺一个 stack 组合的评测 ⇒ **报错**，不是悄悄少一行。
        少一行会让「28 对」变成 27 对，而报告里的相关表照样算得出来。"""
        labels = ["a", "b"]
        ads = {lab: adapter_factory(lab, ("q_proj",)) for lab in labels}
        records = [_rec(lab, "upper") for lab in labels]
        solo = {lab: _adj(lab, "stack", [lab], upper=0.5) for lab in labels}
        with pytest.raises(KeyError, match="缺少组合"):
            CF.all_pairs(
                records, solo, {}, _adj("base", "base", [], upper=0.125),
                base_weights, shape, ads, capabilities=["upper"],
            )


def _rec(label: str, capability: str):
    from adapterfactory.schema import AdapterRecord

    return AdapterRecord(
        label=label, capability=capability, variant="qv", targets=["q_proj"],
        rank=4, alpha=8.0, scale=2.0, digest="d", n_bytes=0, n_params=0,
        base_key="", source="test", n_base_params_touched=0, train={},
    )


class TestPredictorTable:
    def _rows(self, xs, ys):
        preds = [
            CF.Predictors(
                a=f"a{i}", b=f"b{i}", cos=x, sign_rate=0.5, n_common=2,
                expansion=None, overlap_ab=1.0, overlap_ba=1.0, norm_ratio=None,
            )
            for i, x in enumerate(xs)
        ]
        edges = [_edge(y) for y in ys]
        return preds, edges

    def test_rows_pair_up_with_edges(self):
        preds, edges = self._rows([-0.5, -0.3, -0.1], [0.9, 0.5, 0.2])
        rows = CF.predictor_table(preds, edges)
        assert len(rows) == len(preds) + 1  # 末尾一行是统计
        assert rows[0]["interference"] == pytest.approx(0.9)

    def test_rejects_length_mismatch(self):
        preds, edges = self._rows([-0.5], [0.9])
        with pytest.raises(ValueError, match="不等"):
            CF.predictor_table(preds, edges[:0])

    def test_stats_include_every_candidate(self):
        """候选量**并列报出，不预选一个**：只报 cos 的相关系数，
        「幅度才是原因」这个结论就没有数字支撑。"""
        preds, edges = self._rows([-0.5, -0.3, -0.1], [0.9, 0.5, 0.2])
        stats = CF.predictor_table(preds, edges)[-1]["_predictor_stats"]
        assert {s["name"] for s in stats} == {
            "cos", "sign_rate", "expansion", "overlap_ab", "overlap_ba", "norm_ratio",
        }

    def test_constant_candidate_reports_none_not_zero(self):
        """某一侧完全没有变化（全 zoo 的 overlap 都是 1.0）时相关系数**无定义**。
        返回 ``0.0`` 会被读成「测过了，没有相关性」，而事实是
        「这个量在这里是常数，它无法参与任何排名」。"""
        preds, edges = self._rows([-0.5, -0.3, -0.1], [0.9, 0.5, 0.2])
        stats = {s["name"]: s for s in CF.predictor_table(preds, edges)[-1]["_predictor_stats"]}
        assert stats["overlap_ab"]["pearson"] is None
        assert stats["cos"]["pearson"] is not None

    def test_fewer_than_three_points_is_not_a_correlation(self):
        preds, edges = self._rows([-0.5, -0.3], [0.9, 0.5])
        stats = {s["name"]: s for s in CF.predictor_table(preds, edges)[-1]["_predictor_stats"]}
        assert stats["cos"]["n"] == 2 and stats["cos"]["pearson"] is None


class TestCorrelationMath:
    def test_pearson_perfect_monotone_linear(self):
        assert CF._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)

    def test_pearson_perfect_antitone(self):
        assert CF._pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)

    def test_pearson_of_a_constant_is_none(self):
        assert CF._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_ranks_give_ties_the_average(self):
        """并列直接按出现顺序给秩会**制造伪相关**：同样的点集换个输入顺序
        就会得到不同的相关系数。"""
        assert CF._ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_sees_monotone_nonlinear(self):
        """干扰与幅度之间大概率是单调而不线性的。只看 Pearson 会把一个
        完美的单调关系读成「相关性很弱」。"""
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 4.0, 9.0, 16.0]
        assert CF._pearson(xs, ys) == pytest.approx(0.984, abs=1e-3)
        assert CF._spearman(xs, ys) == pytest.approx(1.0)

    def test_spearman_of_a_constant_is_none(self):
        assert CF._spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def _edge(interference: float):
    from adapterfactory.schema import ConflictEdge

    return ConflictEdge(
        a="a", b="b", interference=interference, i_a_from_b=interference,
        i_b_from_a=interference, predicted_cos=None, predicted_sign=None,
        agree=None, affected={},
    )
