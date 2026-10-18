"""``zoo`` 与 ``registry``：**磁盘格式，以及与 FineTuneFactory 的边界**。

这一层里有两种失败，它们的代价完全不对称：

1. **读错了**（偏移错位、形状没核）—— 不会抛异常。切出来的仍然是一堆合法的
   浮点数，于是「同一个 Adapter」静默地变成另一个。所有断言都得是「逐位相同」
   这个量级，不能是「差不多」。
2. **登记漏了**（zoo 少一个）—— 组合网格是从 zoo 的大小推出来的，8 个变 7 个
   会让网格从 276 变成 189。报告里只会显示一张「稍微小一点」的表。

所以这里的测试密度高，而且几乎每一条都在验「坏输入被挡住」而不只是「好输入能过」。
"""

from __future__ import annotations

import json
import struct

import pytest

from adapterfactory import net as N
from adapterfactory import registry as R
from adapterfactory import zoo as Z


class TestShapeOf:
    def test_one_dimensional(self):
        assert Z.shape_of([1.0, 2.0, 3.0]) == [3]

    def test_two_dimensional(self):
        assert Z.shape_of([[1.0, 2.0], [3.0, 4.0]]) == [2, 2]

    def test_empty_is_shape_zero(self):
        assert Z.shape_of([]) == [0]

    def test_rejects_jagged(self):
        """锯齿张量按行主序拼接会**静默**丢掉错位那一段 ——
        而读回来的时候形状字段是自洽的，所以错误永远不会自己冒出来。"""
        with pytest.raises(ValueError):
            Z.shape_of([[1.0, 2.0], [3.0]])

    def test_rejects_scalars(self):
        with pytest.raises(TypeError):
            Z.shape_of(1.0)


class TestToF32:
    def test_roundtrip_is_idempotent(self):
        """``f32(f32(x)) == f32(x)`` —— 「一次往返有损、两次不是」那条性质的具名形式。
        没有它，「第二次写盘与第一次逐字节相同」只能靠运气。"""
        for x in (0.1, 1 / 3, 1e-7, -2.5, 12345.6789):
            once = Z.to_f32(x)
            assert Z.to_f32(once) == once

    def test_actually_quantizes(self):
        """它必须真的丢精度，否则上面那条恒等式是废话（恒等函数也满足它）。"""
        assert Z.to_f32(0.1) != 0.1
        assert abs(Z.to_f32(0.1) - 0.1) < 1e-8


class TestPackUnpack:
    def test_roundtrip_shapes_and_values(self):
        tensors = {"a": [[1.0, 2.0, 3.0]], "b": [4.0, 5.0]}
        payload, meta = Z.pack(tensors)
        back = Z.unpack(payload, meta)
        assert back["a"] == [[1.0, 2.0, 3.0]]
        assert back["b"] == [4.0, 5.0]

    def test_offsets_are_byte_offsets(self):
        """**这是整个格式最容易写错的一处。** 写成元素偏移在本项目的尺寸下
        只会错位一点点 —— 直到某个张量的长度恰好让它越界。

        两个 float32 张量 ⇒ 第二个偏移是 **8**，不是元素序号 2。
        这条与 FineTuneFactory 的约定一致，两边任一改都会让边界失效。
        """
        tensors = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
        payload, meta = Z.pack(tensors)
        assert meta["offsets"] == [0, 8]
        assert len(payload) == 16

    def test_payload_size_is_arithmetic(self):
        """字节数是一个**算术量**：``Σ 元素数 × 4``。加任何魔数/长度前缀都会让它差出
        固定的几字节，而那个差看起来完全像「正常的序列化开销」。"""
        tensors = {"a": [[1.0] * 5, [2.0] * 5], "b": [3.0] * 3}
        payload, meta = Z.pack(tensors)
        assert len(payload) == (10 + 3) * 4
        assert meta["n_floats"] == 13
        assert meta["dtype"] == "float32" and meta["endian"] == "little"

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError):
            Z.pack({})

    def test_key_order_is_the_concatenation_order(self):
        a, _ = Z.pack({"x": [1.0], "y": [2.0]})
        b, _ = Z.pack({"y": [2.0], "x": [1.0]})
        assert a != b


class TestUnpackRejectsBadMeta:
    """旁证元数据的每一道检查都对应一种真实发生过的错位。"""

    def _good(self):
        return Z.pack({"a": [[1.0, 2.0]], "b": [3.0]})

    def test_missing_field(self):
        payload, meta = self._good()
        del meta["offsets"]
        with pytest.raises(ValueError, match="缺字段"):
            Z.unpack(payload, meta)

    def test_wrong_dtype(self):
        payload, meta = self._good()
        meta["dtype"] = "float64"
        with pytest.raises(ValueError, match="float32"):
            Z.unpack(payload, meta)

    def test_wrong_endian(self):
        """大端写出来的字节全是合法的浮点数，只是每个都**恰好错一点**。"""
        payload, meta = self._good()
        meta["endian"] = "big"
        with pytest.raises(ValueError, match="endian"):
            Z.unpack(payload, meta)

    def test_n_floats_mismatch(self):
        payload, meta = self._good()
        meta["n_floats"] += 1
        with pytest.raises(ValueError, match="n_floats"):
            Z.unpack(payload, meta)

    def test_duplicate_names(self):
        payload, meta = self._good()
        meta["names"] = ["a", "a"]
        with pytest.raises(ValueError, match="重复"):
            Z.unpack(payload, meta)

    def test_offset_not_multiple_of_four(self):
        payload, meta = self._good()
        meta["offsets"] = [0, 3]
        with pytest.raises(ValueError, match="4 的倍数"):
            Z.unpack(payload, meta)

    def test_offsets_not_ascending(self):
        payload, meta = self._good()
        meta["offsets"] = [8, 0]
        with pytest.raises(ValueError, match="升序"):
            Z.unpack(payload, meta)

    def test_shape_disagrees_with_slice(self):
        """形状说「这个张量有 2 个数」，切片只给出 1 个 —— 也就是它其实
        吃掉了下一个张量的第一个数。这是偏移写错时**最不容易被发现**的形态：
        每个张量都还是合法的浮点数，只是边界错了。"""
        payload, meta = self._good()  # a: [[1,2]], b: [3]
        meta["shapes"] = [[1, 2], [2]]
        with pytest.raises(ValueError, match="形状"):
            Z.unpack(payload, meta)

    def test_payload_not_multiple_of_four(self):
        payload, meta = self._good()
        with pytest.raises(ValueError, match="4 的倍数"):
            Z.unpack(payload[:-1], meta)


class TestArtifactDisk:
    def _tensors(self):
        return {"layers.0.q_proj.lora_A": [[0.5, -0.25]], "layers.0.q_proj.lora_B": [[1.0]]}

    def test_digest_is_of_payload_only(self, tmp_path):
        """``digest = sha256(裸字节)``。**不是** sha256(整个文件)，也不是 sha256(JSON) ——
        否则同一份权重会因为 JSON 里多一个展示字段而换一个身份。"""
        from adapterfactory.rng import sha256_bytes

        rec = Z.write_artifact(
            tmp_path / "x", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0
        )
        assert rec.digest == sha256_bytes((tmp_path / "x.bin").read_bytes())
        assert rec.n_bytes == (tmp_path / "x.bin").stat().st_size

    def test_reader_verifies_digest(self, tmp_path):
        Z.write_artifact(tmp_path / "x", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0)
        (tmp_path / "x.bin").write_bytes(struct.pack("<3f", 9.0, 9.0, 9.0))
        with pytest.raises(ValueError, match="sha256"):
            Z.read_artifact(tmp_path / "x")

    def test_second_write_is_bit_identical(self, tmp_path):
        """「一次往返有损、两次不是」在磁盘上的样子：
        读回来的张量再写一遍，字节与第一次**逐位相同**。"""
        first = Z.write_artifact(
            tmp_path / "a", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0
        )
        tensors, _ = Z.read_artifact(tmp_path / "a")
        second = Z.write_artifact(
            tmp_path / "b", tensors, name="x", targets=["q_proj"], rank=1, alpha=2.0
        )
        assert first.digest == second.digest
        assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()

    def test_missing_sidecar_is_an_error(self, tmp_path):
        """裸字节单独存在时**没有任何意义** —— 缺旁证要比「猜一下形状」更响亮。"""
        (tmp_path / "x.bin").write_bytes(struct.pack("<3f", 1.0, 2.0, 3.0))
        with pytest.raises(FileNotFoundError, match="旁证"):
            Z.read_artifact(tmp_path / "x")

    def test_artifact_paths_accepts_either_extension(self, tmp_path):
        for given in ("foo", "foo.bin", "foo.json"):
            b, j = Z.artifact_paths(tmp_path / given)
            assert b.name == "foo.bin" and j.name == "foo.json"

    def test_record_roundtrip(self, tmp_path):
        rec = Z.write_artifact(
            tmp_path / "x", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0,
            capability="upper", variant="qv", base_key="tiny", train={"steps": 7},
        )
        again = Z.ZooRecord.from_dict(json.loads((tmp_path / "x.json").read_text(encoding="utf-8")))
        assert again == rec
        assert again.scale == 2.0

    def test_to_dict_has_derived_fields(self, tmp_path):
        """``scale`` 是派生字段，``(**d)`` 会 TypeError —— 而那条路径只有读盘时才走。"""
        rec = Z.write_artifact(
            tmp_path / "x", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0
        )
        d = rec.to_dict()
        assert "scale" in d
        assert Z.ZooRecord.from_dict(d) == rec

    def test_canonical_record_ignores_location_and_train(self, tmp_path):
        """身份问的是「这份权重是什么」，不是「它在哪、怎么来的」。"""
        base = dict(name="x", targets=["q_proj"], rank=1, alpha=2.0, capability="upper")
        a = Z.write_artifact(tmp_path / "a" / "x", self._tensors(), train={"steps": 1}, **base)
        b = Z.write_artifact(tmp_path / "b" / "x", self._tensors(), train={"steps": 999}, **base)
        assert a.digest == b.digest
        assert Z.canonical_record(a) == Z.canonical_record(b)

    def test_canonical_record_notices_a_different_weight(self, tmp_path):
        base = dict(name="x", targets=["q_proj"], rank=1, alpha=2.0)
        a = Z.write_artifact(tmp_path / "a", self._tensors(), **base)
        other = self._tensors()
        other["layers.0.q_proj.lora_A"] = [[0.5, -0.75]]
        b = Z.write_artifact(tmp_path / "b", other, **base)
        assert Z.canonical_record(a) != Z.canonical_record(b)

    def test_spec_does_not_join_the_identity(self, tmp_path):
        """配方与 ``train`` 一样是**过程的记录**：同一份权重可以是两份配方的输出
        （rank=4 与 rank=8 也可以训出逐位相同的权重 —— 罕见，但不是不可能），
        而身份的依据始终是权重本身。把配方算进身份会让同一份 Adapter 换个名字。"""
        base = dict(name="x", targets=["q_proj"], rank=1, alpha=2.0)
        a = Z.write_artifact(tmp_path / "a", self._tensors(), spec={"rank": 1}, **base)
        b = Z.write_artifact(tmp_path / "b", self._tensors(), spec={"rank": 8}, **base)
        assert a.digest == b.digest
        assert Z.canonical_record(a) == Z.canonical_record(b)

    def test_spec_survives_the_roundtrip(self, tmp_path):
        Z.write_artifact(
            tmp_path / "x", self._tensors(), name="x", targets=["q_proj"], rank=1, alpha=2.0,
            spec={"label": "upper-qv", "rank": 1, "alpha": 2.0},
        )
        again = Z.ZooRecord.from_dict(json.loads((tmp_path / "x.json").read_text(encoding="utf-8")))
        assert again.spec == {"label": "upper-qv", "rank": 1, "alpha": 2.0}


def _rec(**kw):
    """一份最小的旁证，只用来跑 :func:`Z.recipe_mismatch` 的判定。"""
    d = dict(
        name="upper-qv", path_bin="upper-qv.bin", digest="d", n_bytes=0, n_params=0,
        targets=["q_proj", "v_proj"], rank=4, alpha=8.0, capability="upper", variant="qv",
        base_key="tiny", base_seed=7, source="produce", payload={}, composed_of=[],
    )
    d.update(kw)
    return Z.ZooRecord(**d)


SPEC = {"label": "upper-qv", "capability": "upper", "variant": "qv",
        "targets": ["q_proj", "v_proj"], "rank": 4, "alpha": 8.0, "scale": 2.0}


class TestRecipeMismatch:
    """「这份产物是不是这份配方的输出」—— 以前这个问题是靠**文件在不在**回答的。

    那不是一个小疏漏。``zoo.rank`` 从 4 改成 8 之后，8 个产物被原样沿用，
    配置、报告、``registry`` 三处都写着 8，而权重是 4 训出来的：
    参数量、``MACs``、``rank_total`` 全跟着错，**而没有任何一处会自相矛盾**——
    每个数字单看都合理，只有拿来跟另一处比才不对。所以这一组测试的重心在
    「错的那种必须被点出来」，而不只是「对的那种能过」。
    """

    def test_a_matching_recipe_is_silent(self):
        assert Z.recipe_mismatch(_rec(spec=dict(SPEC)), spec=SPEC) == ""

    def test_names_the_field_that_drifted(self):
        """报错必须**点名**，还要把两个值都摆出来 —— 只说「不符」的话，
        改了三个字段时得靠猜是哪个。"""
        got = Z.recipe_mismatch(_rec(spec=dict(SPEC, rank=8)), spec=SPEC)
        assert "spec.rank" in got and "8" in got and "4" in got

    def test_the_first_drifted_field_in_key_order_is_the_one_reported(self):
        """同时漂了两个字段时只报一个。**顺序是确定的**（字典键序），
        不是「随便挑一个」—— 不然同一次漂移在两台机器上会给出不同的措辞。"""
        got = Z.recipe_mismatch(_rec(spec=dict(SPEC, rank=8, alpha=16.0)), spec=SPEC)
        assert "spec.alpha" in got  # alpha 排在 rank 前面

    def test_an_artifact_without_a_recipe_index_is_not_accepted(self):
        """「没记下配方」**不等于**「就是你要的那份」。
        缺索引的旧产物必须被判成要重训，否则这道检查对存量磁盘完全无效 ——
        而存量磁盘正是它要防的那个场景。"""
        got = Z.recipe_mismatch(_rec(spec={}), spec=SPEC)
        assert "spec" in got and "重训" in got

    def test_only_the_keys_it_was_handed_are_compared(self):
        """配置只说了 7 项训练超参，``betas`` / ``eps`` 是代码常量。
        把「配置没提的键」也算进来的话，每个产物都会被判成不符 ——
        那种红会逼着人把这项检查关掉，而它就是为此存在的。"""
        rec = _rec(spec=dict(SPEC), train={"steps": 800, "betas": [0.9, 0.95], "eps": 1e-8})
        assert Z.recipe_mismatch(rec, spec=SPEC, train={"steps": 800}) == ""
        assert "train.steps" in Z.recipe_mismatch(rec, spec=SPEC, train={"steps": 400})

    def test_a_key_the_disk_never_recorded_is_a_mismatch(self):
        """磁盘上没有这个键 ⇒ 不知道它当时是多少 ⇒ 不能当作「相等」。"""
        got = Z.recipe_mismatch(_rec(spec=dict(SPEC), train={}), spec=SPEC, train={"lr": 0.005})
        assert "train.lr" in got and "没记" in got

    def test_base_identity_is_optional(self):
        """默认**不查**基座：跨基座是语义错误，``load_registry`` 已经单独报过，
        两处都报只会让同一个原因在报错里出现两遍。传进来才查。"""
        rec = _rec(spec=dict(SPEC), base_key="other", base_seed=99)
        assert Z.recipe_mismatch(rec, spec=SPEC) == ""
        assert "base.key" in Z.recipe_mismatch(rec, spec=SPEC, base_key="tiny")
        assert "base.seed" in Z.recipe_mismatch(rec, spec=SPEC, base_seed=7)

    def test_target_order_is_part_of_the_recipe(self):
        """挂点顺序进了产物的张量拼接顺序 ⇒ 它是配方的一部分，不是展示字段。"""
        rec = _rec(spec=dict(SPEC, targets=["v_proj", "q_proj"]))
        assert "spec.targets" in Z.recipe_mismatch(rec, spec=SPEC)


def _matmul(a, b):
    """独立的矩阵乘法。**故意不复用 net 里的那份** —— 用被测代码验证被测代码
    等于什么都没验证。"""
    n, k, m = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


class TestComposeFactors:
    """``Σ sᵢBᵢAᵢ`` 必须能被写成一个**合法的低秩适配器**。"""

    def test_block_identity(self, shape, adapter_factory):
        """``[s₁B₁|s₂B₂] @ [A₁;A₂] == s₁B₁A₁ + s₂B₂A₂``。
        这是「导出的组合不是快照、而是可以再组合的产物」那条设计的全部内容。"""
        a = adapter_factory("a", ("q_proj",))
        b = adapter_factory("b", ("q_proj",))
        coefs = [1.0, 1.0]
        tensors, rank_total = Z.compose_factors([a, b], coefs, shape)
        assert rank_total == a.rank + b.rank
        full = "layers.0.q_proj"
        got = _matmul(tensors[full + ".lora_B"], tensors[full + ".lora_A"])
        want = _matmul(
            [[x * (a.scale * coefs[0]) for x in row] for row in a.tensors[full + ".lora_B"]],
            a.tensors[full + ".lora_A"],
        )
        want2 = _matmul(
            [[x * (b.scale * coefs[1]) for x in row] for row in b.tensors[full + ".lora_B"]],
            b.tensors[full + ".lora_A"],
        )
        for i in range(len(got)):
            for j in range(len(got[0])):
                assert got[i][j] == pytest.approx(want[i][j] + want2[i][j], abs=1e-12)

    def test_coefficient_is_folded_exactly_once(self, shape, adapter_factory):
        """系数折进 ``B`` 一边。折进两边会变成 ``cᵢ²`` ——
        而那个错误在 ``cᵢ = 1``（也就是 ``stack``）时**完全看不出来**。"""
        a = adapter_factory("a", ("q_proj",))
        full = "layers.0.q_proj"
        t1, _ = Z.compose_factors([a], [1.0], shape)
        t2, _ = Z.compose_factors([a], [0.5], shape)
        r1 = _matmul(t1[full + ".lora_B"], t1[full + ".lora_A"])
        r2 = _matmul(t2[full + ".lora_B"], t2[full + ".lora_A"])
        for i in range(len(r1)):
            for j in range(len(r1[0])):
                assert r2[i][j] == pytest.approx(0.5 * r1[i][j], abs=1e-12)

    def test_per_hook_rank_is_the_member_sum(self, shape, adapter_factory):
        """挂点取并集，但**每个挂点各自拼**：只被一个成员覆盖的挂点上，
        拼出来的秩是那个成员自己的秩。这一条不成立的话，``all`` 与 ``qv``
        组合时会在 ``q_proj`` 之外多出几行零 —— 数值上一样，产物却大一截。"""
        a = adapter_factory("a", ("q_proj", "v_proj"))
        b = adapter_factory("b", ("q_proj",))
        tensors, rank_total = Z.compose_factors([a, b], [1.0, 1.0], shape)
        assert rank_total == a.rank + b.rank
        assert len(tensors["layers.0.q_proj.lora_A"]) == a.rank + b.rank
        assert len(tensors["layers.0.v_proj.lora_A"]) == a.rank

    def test_exported_scale_is_one(self, shape, adapter_factory):
        """缩放折进 B，导出的 ``alpha`` 取 ``rank_total`` ⇒ ``scale = 1``。
        留成一个统一的 alpha 会让 ``stack`` 与 ``merge`` 的产物长得一模一样。"""
        a = adapter_factory("a", ("q_proj",))
        tensors, rank_total = Z.compose_factors([a], [0.5], shape)
        rec_ad = Z.adapter_from_record(
            tensors, Z.ZooRecord(
                name="c", path_bin="c.bin", digest="", n_bytes=0, n_params=0,
                targets=["q_proj"], rank=rank_total, alpha=float(rank_total),
                capability="", variant="", base_key="", base_seed=0, source="compose",
                payload={}, composed_of=["a"],
            ),
        )
        assert rec_ad.scale == 1.0

    def test_mismatched_coef_count(self, shape, adapter_factory):
        a = adapter_factory("a", ("q_proj",))
        with pytest.raises(ValueError, match="系数个数"):
            Z.compose_factors([a], [1.0, 1.0], shape)

    def test_empty_is_rejected(self, shape):
        with pytest.raises(ValueError):
            Z.compose_factors([], [], shape)

    def test_mixed_ranks_are_rejected(self, shape, adapter_factory):
        """混秩其实可支持，但**静默地当成同秩处理**会把成员拼错位置。"""
        a = adapter_factory("a", ("q_proj",), rank=4)
        b = adapter_factory("b", ("q_proj",), rank=2)
        with pytest.raises(ValueError, match="同秩"):
            Z.compose_factors([a, b], [1.0, 1.0], shape)

    def test_disjoint_targets_are_rejected(self, shape, adapter_factory):
        """两个成员没有任何共同挂点 ⇒ 组合在算术上就是「两个各干各的」，
        合成一份产物是空的。报错而不是产出空表。"""
        import dataclasses

        a = adapter_factory("a", ("q_proj",))
        empty = dataclasses.replace(a, targets=())
        with pytest.raises(ValueError):
            Z.compose_factors([empty], [1.0], shape)

    def test_disk_roundtrip_reproduces_the_composition(self, shape, adapter_factory, tmp_path):
        """组合产物读回来之后，``delta`` 必须等于「逐个施加再相加」的 delta。
        这条就是 audit 里 ``disk_roundtrip`` 在核的东西。"""
        a = adapter_factory("a", ("q_proj", "v_proj"))
        b = adapter_factory("b", ("q_proj",))
        coefs = [1.0, 0.5]
        tensors, rank_total = Z.compose_factors([a, b], coefs, shape)
        Z.write_artifact(
            tmp_path / "c", tensors, name="c", targets=sorted({*a.targets, *b.targets}),
            rank=rank_total, alpha=float(rank_total), source="compose", composed_of=["a", "b"],
        )
        back = Z.load_adapter(tmp_path / "c")
        base = N.param_shapes(shape)
        for full in N.expand_targets(base, sorted({*a.targets, *b.targets})):
            hook = full.rpartition(".")[2]
            want = [[0.0] * base[full][1] for _ in range(base[full][0])]
            for ad, c in ((a, coefs[0]), (b, coefs[1])):
                if hook not in ad.targets:
                    continue
                d = ad.delta(full, base[full])
                for i in range(len(want)):
                    for j in range(len(want[0])):
                        want[i][j] += c * d[i][j]
            got = back.delta(full, base[full])
            # float32 落盘一次 —— 误差在 1e-7 量级，用相对容差而不是逐位相等。
            for i in range(len(got)):
                for j in range(len(got[0])):
                    assert got[i][j] == pytest.approx(want[i][j], abs=1e-6, rel=1e-5)


class TestAdapterSpec:
    def test_label(self):
        assert Z.AdapterSpec("upper", Z.VARIANT_QV, 4, 8.0).label == "upper-qv"

    def test_scale_is_alpha_over_rank(self):
        assert Z.AdapterSpec("upper", Z.VARIANT_QV, 4, 8.0).scale == 2.0

    def test_variant_targets(self):
        assert Z.VARIANT_TARGETS[Z.VARIANT_QV] == ("q_proj", "v_proj")
        assert Z.VARIANT_TARGETS[Z.VARIANT_ALL] == N.LORA_ABLE_MODULES

    def test_default_zoo_order_follows_capabilities(self):
        """**顺序即报告里的顺序。** 用 ``sorted`` 会让 ``upper-all`` 排到
        ``upper-qv`` 前面，于是目录序与设计意图序不一致。"""
        caps = ["shift", "upper"]
        labels = [s.label for s in Z.default_zoo(caps, rank=4, alpha=8.0)]
        assert labels == ["shift-qv", "shift-all", "upper-qv", "upper-all"]

    def test_default_zoo_size(self):
        assert len(Z.default_zoo(["a", "b", "c", "d"], rank=4, alpha=8.0)) == 8

    def test_spec_roundtrip(self):
        s = Z.AdapterSpec("upper", Z.VARIANT_ALL, 4, 8.0)
        d = s.to_dict()
        assert d["targets"] == list(N.LORA_ABLE_MODULES)
        assert d["label"] == "upper-all"


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
class TestExpectedLabels:
    def test_order_is_capability_then_variant(self):
        assert R.expected_labels(["upper", "sort"]) == [
            "upper-qv", "upper-all", "sort-qv", "sort-all",
        ]


class TestNBaseParamsTouched:
    def test_qv_touches_two_modules_per_layer(self, shape):
        base = N.param_shapes(shape)
        want = 0
        for full in N.expand_targets(base, ("q_proj", "v_proj")):
            want += base[full][0] * base[full][1]
        assert R.n_base_params_touched(shape, ("q_proj", "v_proj")) == want

    def test_all_touches_more_than_qv(self, shape):
        """``all`` 改动的**基座元素**是 ``qv`` 的 3.5 倍 —— 而两者的
        *适配器参数量* 也是同样的比。这两个量常被混为一谈：
        ``parameter_cost`` 问的是前者（部署时要重算多少）。"""
        qv = R.n_base_params_touched(shape, Z.VARIANT_TARGETS[Z.VARIANT_QV])
        allm = R.n_base_params_touched(shape, Z.VARIANT_TARGETS[Z.VARIANT_ALL])
        assert allm > qv > 0

    def test_is_not_the_adapter_param_count(self, shape, adapter_factory):
        ad = adapter_factory("a", ("q_proj",))
        assert R.n_base_params_touched(shape, ("q_proj",)) != ad.n_params


class TestLoadRegistry:
    def test_missing_zoo_names_the_command(self, tmp_path, tiny_cfg, shape):
        """缺 Adapter 的后果不是「少一格」—— 组合网格是从 zoo 的大小推出来的。"""
        (tmp_path / "data" / "adapters").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="produce"):
            R.load_registry(tmp_path, tiny_cfg, shape)

    def test_a_zoo_from_another_recipe_is_refused(self, tiny_root, tiny_cfg, ctx, produced, tmp_path):
        """**这道题就是那个真 bug。** ``zoo.rank`` 从 4 改成 8 之后，磁盘上还是
        rank=4 的产物：配置、报告、registry 三处都写着 8，权重却是 4 训出来的。
        参数量、``MACs``、``rank_total`` 全跟着错，且**没有任何一处会自相矛盾**。
        只读路径是唯一每次都经过的地方，于是它是唯一挡得住这件事的地方。"""
        import json as _json
        import shutil

        shutil.copytree(tiny_root / "data" / "adapters", tmp_path / "data" / "adapters")
        sidecar = tmp_path / "data" / "adapters" / "upper-qv.json"
        doc = _json.loads(sidecar.read_text(encoding="utf-8"))
        doc["spec"]["rank"] = doc["spec"]["rank"] + 1
        sidecar.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError) as err:
            R.load_registry(tmp_path, tiny_cfg, ctx["shape"])
        msg = str(err.value)
        assert "upper-qv" in msg
        assert "spec.rank" in msg
        assert "produce" in msg  # 报错必须说出修法

    def test_a_zoo_without_a_recipe_index_is_refused(self, tiny_root, tiny_cfg, ctx, produced, tmp_path):
        """存量磁盘上那些**没有配方索引**的产物 —— 它们属于这道检查要防的
        那个场景本身，所以「没记下」必须判成「不是你要的那份」。"""
        import json as _json
        import shutil

        shutil.copytree(tiny_root / "data" / "adapters", tmp_path / "data" / "adapters")
        path = tmp_path / "data" / "adapters" / "shift-all.json"
        doc = _json.loads(path.read_text(encoding="utf-8"))
        doc.pop("spec")
        path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="配方索引"):
            R.load_registry(tmp_path, tiny_cfg, ctx["shape"])

    def test_loads_the_produced_zoo(self, tiny_root, tiny_cfg, ctx, produced):
        reg = R.load_registry(tiny_root, tiny_cfg, ctx["shape"])
        assert reg.labels() == R.expected_labels(tiny_cfg["data"]["capabilities"])
        assert reg.base_key == ctx["shape"].key

    def test_subset_keeps_the_requested_order(self, tiny_root, tiny_cfg, ctx, produced):
        reg = R.load_registry(tiny_root, tiny_cfg, ctx["shape"])
        labels = reg.labels()
        picked = reg.subset([labels[1], labels[0]])
        assert [a.label for a in picked] == [labels[1], labels[0]]

    def test_subset_rejects_unknown(self, tiny_root, tiny_cfg, ctx, produced):
        reg = R.load_registry(tiny_root, tiny_cfg, ctx["shape"])
        with pytest.raises(KeyError):
            reg.subset(["nope"])

    def test_record_rejects_unknown(self, tiny_root, tiny_cfg, ctx, produced):
        reg = R.load_registry(tiny_root, tiny_cfg, ctx["shape"])
        with pytest.raises(KeyError, match="没有 Adapter"):
            reg.record("nope")

    def test_by_capability_partitions_the_zoo(self, tiny_root, tiny_cfg, ctx, produced):
        reg = R.load_registry(tiny_root, tiny_cfg, ctx["shape"])
        caps = reg.capabilities()
        assert caps == list(tiny_cfg["data"]["capabilities"])
        total = sum(len(reg.by_capability(c)) for c in caps)
        assert total == len(reg.labels())

    def test_duplicate_labels_are_rejected(self, shape):
        """同一个标签指向两份权重 —— 检索会静默取到一个。"""
        from adapterfactory.schema import AdapterRecord

        def rec(label: str):
            return AdapterRecord(
                label=label, capability="upper", variant="qv", targets=["q_proj"],
                rank=1, alpha=2.0, scale=2.0, digest="d", n_bytes=0, n_params=0,
                base_key=shape.key, source="test", n_base_params_touched=0, train={},
            )

        with pytest.raises(ValueError, match="重复标签"):
            R.Registry(records=[rec("a"), rec("a")], adapters={}, base_key=shape.key, digest="")

    def test_base_key_mismatch_is_fatal(self, tiny_root, tiny_cfg, ctx, produced, tmp_path):
        """把别的基座上的适配器挂过来，若只检查形状可能全部通过，
        于是「组合」在算术上成立、在语义上毫无意义。"""
        import shutil

        shutil.copytree(tiny_root / "data" / "adapters", tmp_path / "data" / "adapters")
        other = N.ModelShape(
            vocab=ctx["shape"].vocab, d_model=ctx["shape"].d_model,
            n_layers=ctx["shape"].n_layers + 1, n_heads=ctx["shape"].n_heads,
            d_ff=ctx["shape"].d_ff, max_len=ctx["shape"].max_len, name="other",
        )
        with pytest.raises(ValueError, match="基座"):
            R.load_registry(tmp_path, tiny_cfg, other)


class TestRegistryDigest:
    def test_does_not_depend_on_train_metadata(self, shape):
        """换一次 ``steps`` 而权重逐位相同时，版本摘要**不该**变 ——
        否则整批产物会被判为过期。"""
        from adapterfactory.schema import AdapterRecord

        def mk(train):
            return AdapterRecord(
                label="upper-qv", capability="upper", variant="qv", targets=["q_proj"],
                rank=1, alpha=2.0, scale=2.0, digest="d", n_bytes=64, n_params=8,
                base_key=shape.key, source="produce", n_base_params_touched=4, train=train,
            )

        assert R.registry_digest([mk({"steps": 1})]) == R.registry_digest([mk({"steps": 400})])

    def test_changes_when_a_digest_changes(self, shape):
        from adapterfactory.schema import AdapterRecord

        def mk(digest):
            return AdapterRecord(
                label="upper-qv", capability="upper", variant="qv", targets=["q_proj"],
                rank=1, alpha=2.0, scale=2.0, digest=digest, n_bytes=64, n_params=8,
                base_key=shape.key, source="produce", n_base_params_touched=4, train={},
            )

        assert R.registry_digest([mk("a")]) != R.registry_digest([mk("b")])

    def test_is_order_insensitive(self, shape):
        """``join_digest`` 会排序 —— 成员本来就是规范化排序过的。"""
        from adapterfactory.schema import AdapterRecord

        def mk(label):
            return AdapterRecord(
                label=label, capability="upper", variant="qv", targets=["q_proj"],
                rank=1, alpha=2.0, scale=2.0, digest=label, n_bytes=64, n_params=8,
                base_key=shape.key, source="produce", n_base_params_touched=4, train={},
            )

        assert R.registry_digest([mk("a"), mk("b")]) == R.registry_digest([mk("b"), mk("a")])


class TestForeignBaseIdentity:
    def test_finetunefactory_sidecar_has_no_base_identity(self):
        """**这条是本项目最有意思的外部结论之一。**
        FineTuneFactory 的边车 JSON 只记张量形状，形状不足以确定基座：
        ``d_model=16/vocab=28`` 与 ``d_model=16/vocab=26`` 的所有线性层形状逐字相同。
        所以 :func:`base_identity_of` 在任何一份真实产物上都返回空串 ——
        这不是「没找到」，是「那里本来就没有」。"""
        doc = {"spec_id": "s", "meta": {"rank": 4, "payload": {"shapes": [[4, 16], [16, 4]]}}}
        assert R.base_identity_of(doc) == ""

    def test_reads_a_base_key_when_one_exists(self):
        assert R.base_identity_of({"base": "tiny-1"}) == "tiny-1"
        assert R.base_identity_of({"meta": {"base_key": "tiny-1"}}) == "tiny-1"

    def test_adapter_kinds_are_only_lora_variants(self):
        """``full`` / ``freeze`` 是整模型检查点 —— 把它们计入分母会得到
        「``0/64`` 相容」这种数字，而它看起来像「格式完全对不上」。"""
        assert set(R._ADAPTER_KINDS) == {"lora", "qlora"}

    def test_import_report_classifies_checkpoints_out_of_the_denominator(self, tmp_path, shape):
        """造一对产物：一份是策略上可读的 lora，一份是 full。
        报告必须把它们分开计数 —— 同一个 0 背后是两个不同的结论。"""
        import struct

        from adapterfactory.rng import sha256_bytes

        base = N.param_shapes(shape)
        hook = "layers.0.q_proj"
        i_dim, o_dim = base[hook]
        lora = {
            f"{hook}.lora_A": [[0.0] * o_dim for _ in range(4)],
            f"{hook}.lora_B": [[0.0] * 4 for _ in range(i_dim)],
        }
        for name, kind in (("good", "lora"), ("checkpoint", "full")):
            payload, meta = Z.pack(lora)
            (tmp_path / f"{name}.bin").write_bytes(payload)
            (tmp_path / f"{name}.json").write_text(
                json.dumps(
                    {
                        "spec_id": name, "kind": kind, "digest": sha256_bytes(payload),
                        "n_bytes": len(payload),
                        "meta": {"rank": 4, "alpha": 8.0, "payload": meta},
                    }
                ),
                encoding="utf-8",
            )
        rep = R.import_report(tmp_path, shape)
        assert rep["n_files"] == 2
        assert rep["n_read"] == 2
        assert rep["n_adapters"] == 1
        assert rep["n_checkpoints"] == 1
        assert rep["n_adapter_compatible"] == 1
        assert rep["foreign_base_identities"] == []

    def test_import_report_flags_shape_incompatibility(self, tmp_path, shape):
        """跨基座的适配器：读得进来，但 ``lora_A`` 的 ``out`` 维对不上。
        这两件事必须分成两个字段 —— 压成一个 ``bool`` 会把后者吞掉。"""
        from adapterfactory.rng import sha256_bytes

        base = N.param_shapes(shape)
        hook = "layers.0.q_proj"
        i_dim, o_dim = base[hook]
        wrong = {
            f"{hook}.lora_A": [[0.0] * (o_dim + 8) for _ in range(4)],  # out 维跟着 d_model 走
            f"{hook}.lora_B": [[0.0] * 4 for _ in range(i_dim)],
        }
        payload, meta = Z.pack(wrong)
        (tmp_path / "x.bin").write_bytes(payload)
        (tmp_path / "x.json").write_text(
            json.dumps(
                {
                    "spec_id": "x", "kind": "lora", "digest": sha256_bytes(payload),
                    "n_bytes": len(payload), "meta": {"rank": 4, "payload": meta},
                }
            ),
            encoding="utf-8",
        )
        rep = R.import_report(tmp_path, shape)
        assert rep["n_read"] == 1 and rep["n_adapters"] == 1
        assert rep["n_adapter_compatible"] == 0
        assert rep["rows"][0]["n_problems"] >= 1

    def test_import_report_rejects_a_missing_dir(self, tmp_path, shape):
        with pytest.raises(FileNotFoundError):
            R.import_report(tmp_path / "nope", shape)

    def test_a_same_base_adapter_is_reported_compatible(self, tmp_path, shape):
        """**这条是一条回归测试。** 曾经「张量成对」那道检查把键数当挂点对数用，
        于是一份形状完全正确的适配器也被判成不相容 —— 而这个 bug 藏在
        「跨基座产物本来就不相容」的一堆真实 ``False`` 后面，看不出来。"""
        from adapterfactory.rng import sha256_bytes

        base = N.param_shapes(shape)
        hook = "layers.0.q_proj"
        i_dim, o_dim = base[hook]
        good = {
            f"{hook}.lora_A": [[0.0] * o_dim for _ in range(4)],
            f"{hook}.lora_B": [[0.0] * 4 for _ in range(i_dim)],
        }
        payload, meta = Z.pack(good)
        (tmp_path / "x.bin").write_bytes(payload)
        (tmp_path / "x.json").write_text(
            json.dumps(
                {
                    "spec_id": "x", "kind": "lora", "digest": sha256_bytes(payload),
                    "n_bytes": len(payload), "meta": {"rank": 4, "payload": meta},
                }
            ),
            encoding="utf-8",
        )
        rep = R.import_report(tmp_path, shape)
        assert rep["n_adapters"] == 1
        assert rep["n_adapter_compatible"] == 1
        assert rep["rows"][0].get("problems") == []

    def test_half_a_factor_pair_is_flagged(self, tmp_path, shape):
        """只有 ``lora_A`` 没有 ``lora_B`` 的产物**读得进来**
        （键名合法、形状也对），但它是半个适配器。"""
        from adapterfactory.rng import sha256_bytes

        base = N.param_shapes(shape)
        hook = "layers.0.q_proj"
        _i_dim, o_dim = base[hook]
        half = {f"{hook}.lora_A": [[0.0] * o_dim for _ in range(4)]}
        payload, meta = Z.pack(half)
        (tmp_path / "x.bin").write_bytes(payload)
        (tmp_path / "x.json").write_text(
            json.dumps(
                {
                    "spec_id": "x", "kind": "lora", "digest": sha256_bytes(payload),
                    "n_bytes": len(payload), "meta": {"rank": 4, "payload": meta},
                }
            ),
            encoding="utf-8",
        )
        rep = R.import_report(tmp_path, shape)
        assert rep["n_read"] == 1
        assert rep["n_adapter_compatible"] == 0
        assert any("缺因子" in p for p in rep["rows"][0]["problems"])


class TestRowSchemaIsACommitment:
    """``import_report`` 的 ``rows`` 是一份**契约**，不是「凑出来的字典」。"""

    def test_every_branch_emits_the_same_key_set(self, foreign_dir, shape):
        """**这条是一条回归测试，它守的是渲染层。**

        ``main.py import`` 曾经打印 ``row['name']`` —— 而 ``name`` 在任何一个
        分支里都没有被写过；同一行的 ``row['kind']`` 在「读不通」那一支也不存在。
        于是那条命令在自己的打印循环上炸掉（``错误：'name'``，exit 2），
        而它前面几行统计早已打完，看上去像报告写到一半坏了。

        四个分支（相容 / 跨基座 / 检查点 / 读不通）都会出现在这个夹具里，
        所以「键集一致」这句话是被四种输入一起验的，不是被一种。
        """
        rep = R.import_report(foreign_dir, shape)
        assert len(rep["rows"]) == 4
        keysets = {frozenset(r) for r in rep["rows"]}
        assert len(keysets) == 1, f"行与行的键集不一致：{keysets}"
        assert keysets.pop() == frozenset(R._ROW_KEYS)

    def test_none_means_not_applicable_and_is_not_zero(self, foreign_dir, shape):
        """补 ``None`` 而不是补 ``0``/``[]`` —— 这一节已经区分过一次的区别，
        不该在序列化时又被抹平：``compatible=None``（检查点，不适用）
        vs ``compatible=False``（判过了，不相容）是两个结论。"""
        rep = {r["file"]: r for r in R.import_report(foreign_dir, shape)["rows"]}
        assert rep["good.bin"]["compatible"] is True
        assert rep["crossbase.bin"]["compatible"] is False
        assert rep["checkpoint.bin"]["compatible"] is None
        assert rep["broken.bin"]["compatible"] is None
        assert rep["broken.bin"]["read"] is False
        assert rep["broken.bin"]["problems"] is None
        assert rep["broken.bin"]["n_problems"] is None
        # 读不通的那一行照样有 ``file`` 与错误原因 —— 它只是「别的字段不适用」。
        assert "broken.json" in rep["broken.bin"]["error"]

    def test_a_field_nobody_registered_is_a_loud_error(self):
        """多出来的键**报错，不静默丢弃**：丢掉的话，以后有人在某个分支加了字段，
        它会无声消失在这份唯一的事实来源里。"""
        with pytest.raises(AssertionError, match="未登记"):
            R._normalize_row({"file": "x.bin", "read": True, "surprise": 1})

    def test_the_fixture_covers_all_four_branches(self, foreign_dir, shape):
        """**前提自检。** 上面几条只在「四种行真的都造出来了」时才有意义；
        夹具悄悄少造一种，那几条会照绿，而它们本来要守的那个 bug 会回来。"""
        rows = {r["file"]: r for r in R.import_report(foreign_dir, shape)["rows"]}
        assert set(rows) == {"good.bin", "crossbase.bin", "checkpoint.bin", "broken.bin"}
        assert rows["broken.bin"]["read"] is False
        assert rows["checkpoint.bin"]["is_adapter"] is False
        assert rows["crossbase.bin"]["is_adapter"] is True
        assert rows["good.bin"]["compatible"] is True


class TestEmbeddedGlobals:
    def test_constants_are_what_the_zero_dependency_claim_rests_on(self):
        """格式常量是**字面量**，不是从别处推出来的 —— 它们是与
        FineTuneFactory 约定的那部分，改一个就换一个格式。"""
        assert (Z.DTYPE, Z.ENDIAN) == ("float32", "little")
        assert "no header" in Z.LAYOUT

    def test_foreign_reader_does_not_reuse_zoo_record(self):
        """两条读取路径分开，任一条坏掉都能被另一条照出来。

        判据是**代码里没有对 ``ZooRecord`` 的引用**，不是「源码里没有这四个字母」——
        后者连解释「为什么不复用它」的那句 docstring 一起算违规，
        而那种检查最后总是被改成在注释里加个豁免标记。文档提及不算引用。
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(R.read_foreign))
        refs = [
            n for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id == "ZooRecord")
            or (isinstance(n, ast.Attribute) and n.attr == "ZooRecord")
            or (isinstance(n, ast.alias) and n.name.endswith("ZooRecord"))
        ]
        assert refs == []
