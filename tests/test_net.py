"""``net``：数值引擎。**这里每一条都在守一个「不写下来就会被违反」的性质。**

最重要的是两条：

- **LoRA 的 ``B`` 是零初始化的** ⇒ ``delta = 0`` ⇒ 「组合的起点就是基座」
  是一条**可证明的**性质，而不是「训练得好所以差不多」。测试直接量它。
- **``merge_into`` 系数全 1 时是代数恒等式**：``x@W + Σx@Δᵢ == x@(W+ΣΔᵢ)``。
  这是一条**实现检查**（两条算法路径必须给同一个数），不是能力指标 ——
  「合并没有损失能力」是 ``merge_loss`` 的事。
"""

from __future__ import annotations

import pytest

from adapterfactory import net as N

D_MODEL = 24
SHAPE = N.ModelShape(vocab=26, d_model=D_MODEL, n_layers=2, n_heads=4, d_ff=48, max_len=13)
IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)


@pytest.fixture(scope="module")
def weights() -> dict:
    return N.init_base_weights(SHAPE, 20260930)


class TestModelShape:
    def test_head_dim_divides(self):
        assert SHAPE.head_dim == D_MODEL // 4
        assert D_MODEL % 4 == 0

    def test_key_is_stable_and_distinguishing(self):
        assert SHAPE.key == SHAPE.key
        other = N.ModelShape(vocab=26, d_model=32, n_layers=2, n_heads=4, d_ff=48, max_len=13)
        assert SHAPE.key != other.key

    def test_rejects_heads_that_do_not_divide_d_model(self):
        """``d_model=25, n_heads=4`` 不能只是「结果有点怪」——
        ``head_dim`` 是整除除法，不平分会在 reshape 时报一个与真实原因无关的错。"""
        with pytest.raises((ValueError, AssertionError)):
            N.ModelShape(vocab=26, d_model=25, n_layers=2, n_heads=4, d_ff=48, max_len=13)

    def test_roundtrip(self):
        assert N.ModelShape.from_dict(SHAPE.to_dict()) == SHAPE


class TestParamShapes:
    def test_lora_targets_are_a_subset_of_known_modules(self):
        """挂点名字必须真的存在于模型里。写错一个字母（``q_pro``）的症状是
        ``expand_targets`` 什么都不匹配、Adapter 静默地一个模块都没挂上 ——
        而「没挂上」在下游表现为「这个 Adapter 没学到东西」。"""
        assert set(N.LORA_ABLE_MODULES) <= {k.rsplit(".", 1)[-1] for k in N.module_indims(SHAPE)}

    def test_rope_means_no_learnable_position_table(self):
        """**这是选 RoPE 的全部理由**：没有位置嵌入表，参数量就只是
        「张量形状的函数」，于是 ``n_params`` 这条公式能被独立核对。
        加一张位置表会让参数量多一项 ``max_len × d_model``，
        而那一项与 ``seq_len`` 耦合 —— 换一个 seq_len 就换一个参数量。"""
        keys = set(N.param_shapes(SHAPE))
        assert not any("pos" in k or "position" in k for k in keys)
        assert "emb" in keys, "词嵌入必须在"

    def test_lora_shapes_follow_A_rank_by_in_and_B_in_by_rank(self):
        shapes = N.adapter_shapes(SHAPE, ["q_proj"], rank=4)
        for layer in range(SHAPE.n_layers):
            assert shapes[f"layers.{layer}.q_proj.lora_A"] == (4, D_MODEL)
            assert shapes[f"layers.{layer}.q_proj.lora_B"] == (D_MODEL, 4)

    def test_lora_touches_every_layer(self):
        """挂点少的变体也必须覆盖**每一层**。只挂第 0 层的 Adapter
        在 qv 变体里看起来「参数更少更划算」，而它其实只改了一半模型。"""
        shapes = N.adapter_shapes(SHAPE, ["v_proj"], rank=4)
        layers = {k.split(".")[1] for k in shapes}
        assert layers == {str(i) for i in range(SHAPE.n_layers)}

    def test_rank_scales_lora_param_count(self):
        n4 = N.count_params(N.adapter_shapes(SHAPE, ["q_proj"], 4))
        n8 = N.count_params(N.adapter_shapes(SHAPE, ["q_proj"], 8))
        assert n8 == 2 * n4

    def test_more_targets_means_more_params(self):
        one = N.count_params(N.adapter_shapes(SHAPE, ["q_proj"], 4))
        two = N.count_params(N.adapter_shapes(SHAPE, ["q_proj", "v_proj"], 4))
        assert two == 2 * one

    def test_all_equals_seven_qv_variants(self):
        """``all`` 变体 = 全部 7 个模块，``qv`` = 2 个。这是「同能力、不同覆盖」
        那组对照的**全部内容**，所以它得有个能算出倍数的来源。"""
        assert len(N.LORA_ABLE_MODULES) == 7
        qv = N.count_params(N.adapter_shapes(SHAPE, ["q_proj", "v_proj"], 4))
        allm = N.count_params(N.adapter_shapes(SHAPE, list(N.LORA_ABLE_MODULES), 4))
        assert allm > qv

    def test_expand_targets_covers_all_layers(self):
        base = {k: 1 for k in N.param_shapes(SHAPE)}
        got = N.expand_targets(base, ["q_proj"])
        assert len(got) == SHAPE.n_layers
        assert all(k.endswith("q_proj") for k in got)


class TestBaseWeights:
    """权重表与 Adapter 张量都是**嵌套 list**，不是 :class:`Tensor`。

    这个区分是刻意的：:class:`Tensor` 带 ``prev``/``grad``，是**训练时**的
    概念；推理路径上的权重只是数。把两者混成一种类型，代价是每个推理张量
    都拖着一套永远为空的梯度结构 —— 而且在「前向是否纯」这类测试里，
    分不清「这个数变了」与「这个对象的梯度被重置了」。
    """

    def test_deterministic(self):
        a = N.init_base_weights(SHAPE, 7)
        b = N.init_base_weights(SHAPE, 7)
        assert sorted(a) == sorted(b)
        for k in a:
            assert a[k] == b[k]

    def test_different_seed_different_weights(self):
        a, b = N.init_base_weights(SHAPE, 7), N.init_base_weights(SHAPE, 8)
        assert any(a[k] != b[k] for k in a)

    def test_param_count_is_a_formula_not_a_measurement(self):
        """参数量必须能从形状**算出来**并与实际张量逐位相符。
        只「量一下」的后果是：哪天真多了一个张量，参数量跟着变，
        没有任何东西会说不该变。"""
        w = N.init_base_weights(SHAPE, 1)
        assert N.count_params(w) == N.count_params(N.param_shapes(SHAPE))

    def test_nested_list_shapes_match(self):
        def shape_of(v):
            out = []
            while isinstance(v, list):
                out.append(len(v))
                v = v[0] if v else 0
            return tuple(out)

        w = N.init_base_weights(SHAPE, 1)
        for name, v in w.items():
            assert shape_of(v) == tuple(N.param_shapes(SHAPE)[name]), name


class TestLoRAZeroInit:
    """**「组合的起点就是基座」的可证明来源。**"""

    def test_lora_B_is_zero_so_delta_is_zero(self, weights):
        ad = N.init_adapter(SHAPE, ["q_proj", "v_proj"], rank=4, seed=7)
        for name, t in ad.tensors.items():
            if name.endswith(".lora_B"):
                assert all(v == 0.0 for row in t for v in row), f"{name} 不是零初始化"
        for layer in range(SHAPE.n_layers):
            for full in ("layers.%d.q_proj" % layer, "layers.%d.v_proj" % layer):
                d = ad.delta(full, N.param_shapes(SHAPE)[full])
                assert all(v == 0.0 for row in d for v in row)

    def test_lora_B_is_zero_but_A_is_not(self):
        """``A`` 也零初始化的话整个 delta 恒为 0 —— 训练**永远不会**开始，
        而症状是 loss 一动不动，看起来像学习率太小。标准做法是
        ``A`` 随机、``B`` 归零，让初始 delta 恰好为 0 而梯度不为 0。"""
        ad = N.init_adapter(SHAPE, ["q_proj"], rank=4, seed=7)
        a = ad.tensors["layers.0.q_proj.lora_A"]
        assert any(v != 0.0 for row in a for v in row), "lora_A 全零 ⇒ delta 恒为 0 ⇒ 学不动"

    def test_a_fresh_adapter_changes_nothing(self, weights):
        """挂上一个刚初始化的 Adapter，logits 必须**逐位相同**。
        这是「零初始化 ⇒ delta=0 ⇒ 起点是基座」那句推论的可执行形式。"""
        ad = N.init_adapter(SHAPE, list(N.LORA_ABLE_MODULES), rank=4, seed=3)
        base = N.logits_of(weights, IDS, SHAPE)
        with_ad = N.logits_of(weights, IDS, SHAPE, adapters=[ad])
        assert N.max_abs_diff(base, with_ad) == 0.0

    def test_nonzero_B_changes_the_output(self, weights):
        """反向：``B`` 一旦非零，输出就必须变。
        少了这一条，上面那条「没变」也可能是因为 Adapter 根本没接上去。"""
        ad = N.init_adapter(SHAPE, ["q_proj"], rank=4, seed=3)
        for row in ad.tensors["layers.0.q_proj.lora_B"]:
            for i in range(len(row)):
                row[i] = 0.02
        base = N.logits_of(weights, IDS, SHAPE)
        with_ad = N.logits_of(weights, IDS, SHAPE, adapters=[ad])
        assert N.max_abs_diff(base, with_ad) > 0.0

    def test_scale_is_alpha_over_rank(self):
        ad = N.init_adapter(SHAPE, ["q_proj"], rank=4, seed=1)
        assert ad.scale > 0.0

    def test_adapter_delta_zero_for_untargeted_modules(self, weights):
        """没挂的模块**连张量都没有** —— 「qv 变体没动 k_proj」这句话
        必须有地方能验证，否则它只是一句注释。

        这里断言的是「张量不存在」而不是「delta 是 0」：``delta`` 对没挂的
        模块抛 ``KeyError``（这是对的 —— 它和 ``d['missing']`` 一样，
        问一个不存在的东西就该报出来）。若把它改成「返回零矩阵」，
        那么「忘记挂 k_proj」与「挂了但恰好没学到」就长得一模一样了。
        """
        ad = N.init_adapter(SHAPE, ["q_proj"], rank=4, seed=5)
        for name in ad.tensors:
            assert not name.startswith("layers.0.k_proj"), name
        with pytest.raises(KeyError):
            ad.delta("layers.0.k_proj", N.param_shapes(SHAPE)["layers.0.k_proj"])

    def test_adapter_tensors_cover_every_target_and_layer(self):
        ad = N.init_adapter(SHAPE, ["q_proj", "v_proj"], rank=4, seed=1)
        want = {
            f"layers.{i}.{t}.lora_{ab}"
            for i in range(SHAPE.n_layers)
            for t in ("q_proj", "v_proj")
            for ab in ("A", "B")
        }
        assert set(ad.tensors) == want


class TestForward:
    def test_logits_shape(self, weights):
        out = N.logits_of(weights, IDS, SHAPE)
        assert len(out) == len(IDS)
        assert all(len(row) == SHAPE.vocab for row in out)

    def test_causal_mask_blocks_the_future(self, weights):
        """改最后一个位置之后，**前面所有位置**的 logits 必须逐位不变。
        这一条是因果性的定义，也是「训练时能并行、推理时必须顺序」的来源。"""
        a = N.logits_of(weights, IDS, SHAPE)
        b = N.logits_of(weights, IDS[:-1] + (IDS[-1] + 1,), SHAPE)
        for i in range(len(IDS) - 1):
            assert N.max_abs_diff(a[i], b[i]) == 0.0

    def test_changing_prefix_changes_later_logits(self, weights):
        a = N.logits_of(weights, IDS, SHAPE)
        b = N.logits_of(weights, (IDS[0] + 1,) + IDS[1:], SHAPE)
        assert N.max_abs_diff(a[-1], b[-1]) > 0.0

    def test_forward_is_deterministic(self, weights):
        assert N.max_abs_diff(N.logits_of(weights, IDS, SHAPE), N.logits_of(weights, IDS, SHAPE)) == 0.0

    def test_forward_does_not_mutate_weights(self, weights):
        """**前向必须是纯的。** 一旦它就地改了权重，同一个基座在不同组合之间
        就不再是同一根基座 —— 而「所有组合共享一根基座」是整套对比的前提。"""
        before = {k: [list(r) if isinstance(r, list) else r for r in v] for k, v in weights.items()}
        N.logits_of(weights, IDS, SHAPE)
        for k, v in weights.items():
            assert v == before[k], f"{k} 被前向改了 —— 前向必须是纯的"


class TestMergeIdentity:
    """``merge_into`` 是**代数恒等式**，不是近似。"""

    def _adapters(self):
        ads = []
        for i, targets in enumerate((["q_proj"], ["v_proj"], ["q_proj", "v_proj"])):
            ad = N.init_adapter(SHAPE, targets, rank=4, seed=100 + i)
            for name, t in ad.tensors.items():
                if name.endswith(".lora_B"):
                    for r, row in enumerate(t):
                        for c in range(len(row)):
                            row[c] = 0.03 * ((r + c) % 5 - 2)
            ads.append(ad)
        return ads

    def test_coefs_one_is_identity(self, weights):
        """``x@W + Σ x@Δᵢ == x@(W + ΣΔᵢ)``。

        ⚠️ **``merge_into`` 返回值，不就地改。** 我最初写成
        ``merge_into(w, ads, ...)`` 然后拿 ``w`` 去比较，得到 0.145 的差 ——
        看起来像「恒等式不成立」，实际上只是没接返回值。
        这个错误的表现很坏：基座**一点没变**，而「合并没有任何效果」
        与「合并有损失」在数值上分不开。所以下面两条分别钉住
        「返回值是对的」和「入参没被动过」。
        """
        ads = self._adapters()
        merged = N.merge_into(weights, ads, weights_by_adapter=[1.0] * len(ads))
        a = N.logits_of(merged, IDS, SHAPE)
        b = N.logits_of(weights, IDS, SHAPE, adapters=ads)
        assert N.max_abs_diff(a, b) < 1e-12

    def test_merge_into_does_not_mutate_its_argument(self, weights):
        import copy

        ads = self._adapters()
        before = copy.deepcopy(weights)
        N.merge_into(weights, ads, weights_by_adapter=[1.0] * len(ads))
        assert weights == before, "merge_into 改了入参 —— 它会污染所有组合共用的基座"

    def test_merged_table_only_differs_on_targeted_modules(self, weights):
        """合并只该动被挂过的模块。动了别的，症状是「qv 变体也在改 MLP」——
        而参数量那一列照样是对的。"""
        ads = [ad for ad in self._adapters() if ad.targets == ("q_proj",)]
        merged = N.merge_into(weights, ads, weights_by_adapter=[1.0])
        touched = {k for k in weights if merged[k] != weights[k]}
        assert touched, "至少要有变动的模块，否则这条测试什么都没测"
        assert all(k.endswith(".q_proj") or k == "emb" for k in touched), sorted(touched)
        assert not any(k.endswith(".k_proj") for k in touched)

    def test_coefs_half_is_not_the_same(self, weights):
        """``1/k`` 的稀释**必须真的改变结果** —— 否则 ``merge`` 就是 ``stack``
        换了个名字，而报告里「merge 的能力损失」那一列会恒为 0。"""
        ads = self._adapters()
        one = N.merge_into(weights, ads, weights_by_adapter=[1.0] * len(ads))
        half = N.merge_into(weights, ads, weights_by_adapter=[0.5] * len(ads))
        assert N.max_abs_diff(N.logits_of(one, IDS, SHAPE), N.logits_of(half, IDS, SHAPE)) > 1e-6

    def test_coefs_zero_is_the_base(self, weights):
        """系数全 0 ⇒ 合并出来的表必须**逐位等于基座**。
        这条给「稀释到 0」一个锚点：``merge`` 的损失随 k 增大而增大时，
        它的上界就是这里。"""
        ads = self._adapters()
        merged = N.merge_into(weights, ads, weights_by_adapter=[0.0] * len(ads))
        assert N.max_abs_diff(N.logits_of(merged, IDS, SHAPE), N.logits_of(weights, IDS, SHAPE)) < 1e-15

    def test_merge_does_not_mutate_adapters(self, weights):
        """**按键要按 (适配器, 张量名) 而不是只按张量名。**
        我第一版只用张量名当键，而三个适配器的张量名是**同一套** ——
        字典于是只剩最后一个适配器的值，断言变成「第三个适配器没被改」，
        前两个改没改它根本看不见。这类错误我修一次就够了：
        它安静地通过，而它唯一的作用就是发现「适配器被就地改了」。
        """
        ads = self._adapters()
        before = [
            {n: [list(r) for r in t] for n, t in ad.tensors.items()} for ad in ads
        ]
        N.merge_into(weights, ads, weights_by_adapter=[0.5] * len(ads))
        for ad, snap in zip(ads, before):
            for n, t in ad.tensors.items():
                assert t == snap[n], f"{n} 被 merge_into 改了"

    def test_mismatched_coef_count_is_rejected(self, weights):
        """系数个数与适配器个数不符必须报错。静默截断到较短的那个，
        症状是「第三个适配器完全没被合并进去」—— 数值合理，结论错。"""
        ads = self._adapters()
        with pytest.raises((ValueError, AssertionError)):
            N.merge_into(weights, ads, weights_by_adapter=[1.0])

    def test_merging_nothing_is_identity(self, weights):
        merged = N.merge_into(weights, [], weights_by_adapter=[])
        assert N.max_abs_diff(N.logits_of(merged, IDS, SHAPE), N.logits_of(weights, IDS, SHAPE)) == 0.0



class TestMacCounter:
    def test_counts_nothing_outside_a_block(self, weights):
        """**计数器默认关着。** 若它全局常开，那么两次嵌套调用的计数会串，
        而且任何没打算计数的地方都会被算进去。"""
        N.logits_of(weights, IDS, SHAPE)  # 没有 mac_counter()

    def test_counts_a_forward(self, weights):
        with N.mac_counter() as mc:
            N.logits_of(weights, IDS, SHAPE)
        total = mc.to_dict()["total"]
        assert total > 0

    def test_counts_are_additive(self, weights):
        with N.mac_counter() as mc1:
            N.logits_of(weights, IDS, SHAPE)
        with N.mac_counter() as mc2:
            N.logits_of(weights, IDS, SHAPE)
            N.logits_of(weights, IDS, SHAPE)
        assert mc2.to_dict()["total"] == 2 * mc1.to_dict()["total"]

    def test_route_costs_more_than_the_base(self, weights):
        """**这是 latency 轴存在的理由。** ``route`` 不合并权重，于是每个
        Adapter 的 ``x @ Δᵢ`` 都要真的算一遍；``stack`` 把它们并进了同一张
        权重表，代价与基座相同。若这条不成立，「合并省下多少计算」
        就只是一段论述，而不是一个数字。"""
        ads = []
        for i, (targets, label) in enumerate(((["q_proj"], "a"), (["v_proj"], "b"))):
            ad = N.init_adapter(SHAPE, targets, rank=4, seed=200 + i)
            for name, t in ad.tensors.items():
                if name.endswith(".lora_B"):
                    for row in t:
                        for c in range(len(row)):
                            row[c] = 0.01
            # ``gates_by_module`` 按 ``ad.label`` 找门向量，标签为空会报
            # KeyError: "缺少适配器 '' 的门向量"。这不是多余的仪式：
            # route 的系数必须与 delta 一一对应，而标签就是那个对应关系。
            ad.label = label
            ads.append(ad)
        with N.mac_counter() as base_mc:
            N.logits_of(weights, IDS, SHAPE)
        gates = N.derive_gates(SHAPE, ["a", "b"], seed=1)
        with N.mac_counter() as route_mc:
            N.forward_routed(weights, ads, IDS, SHAPE, gates=gates, temperature=1.0)
        assert route_mc.to_dict()["total"] > base_mc.to_dict()["total"]

    def test_to_dict_is_json_safe(self, weights):
        import json

        with N.mac_counter() as mc:
            N.logits_of(weights, IDS, SHAPE)
        json.dumps(mc.to_dict(), ensure_ascii=False)


class TestGates:
    def test_derive_gates_is_deterministic(self):
        a = N.derive_gates(SHAPE, ["x", "y"], seed=5)
        b = N.derive_gates(SHAPE, ["x", "y"], seed=5)
        assert {k: [list(v) for v in vs.values()] for k, vs in a.items()} == {
            k: [list(v) for v in vs.values()] for k, vs in b.items()
        }

    def test_derive_gates_depends_on_label_order_consistently(self):
        """同样的标签集合必须派生出同样的门 —— 门是**种子派生的固定向量**，
        不是训练出来的；它的正当性在于「同一个组合每次都走同一扇门」。"""
        a = N.derive_gates(SHAPE, ["x", "y"], seed=5)
        b = N.derive_gates(SHAPE, ["y", "x"], seed=5)
        assert set(a) == set(b)

    def test_gate_coeffs_sum_to_one(self):
        """门是 softmax ⇒ 每个模块上各 Adapter 的系数和为 1。
        少了它，``route`` 的输出尺度会随成员个数变化，
        于是「route 的结果」里混进了「几个成员」这个因素。"""
        gates = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        out = N.gate_coeffs(N.constant([[0.1, 0.2]]), gates, temperature=1.0)
        assert len(out) == 1 and len(out[0]) == len(gates)
        assert sum(out[0]) == pytest.approx(1.0)

    def test_gate_coeffs_temperature_controls_sharpness(self):
        """**温度是「门有多信自己」的旋钮。** 低温 ⇒ 接近 argmax（一个 Adapter
        拿走全部权重），高温 ⇒ 接近均匀（等于把所有 Adapter 平均）。
        少了这一条，``route.temperature`` 就只是一个写进配置却没人验的数。

        这里需要**至少两个**门：只有一个门时 softmax 恒等于 1，
        温度再怎么变都看不出差别 —— 那种测试会永远通过。"""
        gates = [[1.0, 0.0], [0.0, 1.0]]
        x = N.constant([[3.0, 0.0]])
        cold = N.gate_coeffs(x, gates, temperature=0.1)
        hot = N.gate_coeffs(x, gates, temperature=100.0)
        assert max(cold[0]) > max(hot[0])
        assert max(hot[0]) < 0.6

    def test_gate_coeffs_all_positive(self):
        gates = [[1.0, -1.0], [-1.0, 1.0]]
        out = N.gate_coeffs(N.constant([[0.5, 0.5]]), gates, temperature=1.0)
        assert all(v > 0.0 for v in out[0])
