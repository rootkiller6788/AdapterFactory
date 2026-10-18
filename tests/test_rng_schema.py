"""``rng`` 与 ``schema``：**本项目所有可复现性的地基**。

两条性质决定了这两块怎么做：

1. **内容寻址**：`digest` 必须是「这份内容是什么」的函数，与对象的内存布局、
   字典插入顺序、浮点打印位数都无关。否则同一份产物会拿到两个名字。
2. **``from_dict`` 与 ``to_dict`` 严格互逆**。产物是给人读也给程序读的，
   而「读回来差了 1」这种错误不会让任何东西崩 —— 它只让结论悄悄偏一点。
   这一条对每个记录类都单独验，不靠抽样。
"""

from __future__ import annotations

import json

import pytest

from adapterfactory import rng as R
from adapterfactory import schema as S


class TestCanonicalJson:
    def test_key_order_does_not_matter(self):
        assert R.canonical_json({"b": 1, "a": 2}) == R.canonical_json({"a": 2, "b": 1})

    def test_nested_key_order_does_not_matter(self):
        x = {"o": {"y": 1, "x": 2}, "l": [{"q": 1, "p": 2}]}
        y = {"l": [{"p": 2, "q": 1}], "o": {"x": 2, "y": 1}}
        assert R.canonical_json(x) == R.canonical_json(y)

    def test_no_incidental_whitespace(self):
        assert " " not in R.canonical_json({"a": [1, 2], "b": {"c": 3}})

    def test_non_ascii_is_not_escaped(self):
        """``ensure_ascii=True`` 会把中文转成 ``\\uXXXX`` —— 那样产物的可读性
        就没了，而摘要是同一份。这里要的是「人也能读」。"""
        assert "能力" in R.canonical_json({"k": "能力"})

    def test_float_repr_is_shortest_roundtrip(self):
        # 0.1 + 0.2 的 repr 是 0.30000000000000004；写死 6 位会把两个
        # 不同的浮点变成同一个字符串，于是「内容变了、摘要没变」。
        assert "0.30000000000000004" in R.canonical_json({"v": 0.1 + 0.2})


class TestDigests:
    def test_digest_of_is_order_independent_for_mappings(self):
        assert R.digest_of({"a": 1, "b": 2}) == R.digest_of({"b": 2, "a": 1})

    def test_digest_of_order_dependent_for_lists(self):
        """**列表的顺序是内容的一部分。** 成员顺序变了就是另一个组合。"""
        assert R.digest_of([1, 2]) != R.digest_of([2, 1])

    def test_digest_of_changes_when_content_changes(self):
        assert R.digest_of({"a": 1}) != R.digest_of({"a": 2})

    def test_sha256_text_matches_known_vector(self):
        assert R.sha256_text("abc") == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_sha256_bytes_of_utf8_matches_sha256_text(self):
        assert R.sha256_bytes("能力".encode("utf-8")) == R.sha256_text("能力")

    def test_short_digest_length(self):
        full = R.sha256_text("x")
        assert len(R.short_digest(full, 12)) == 12
        assert full.startswith(R.short_digest(full, 12))

    def test_short_digest_detects_a_prefix_collision(self):
        """前缀相同、后面不同 ⇒ 短摘要不同。走的是「截断」而不是「取模」，
        所以这一条不需要构造真正的 sha256 碰撞。"""
        a, b = "a" * 12 + "1", "a" * 12 + "2"
        assert R.short_digest(a, 12) == R.short_digest(b, 12)
        assert R.short_digest(a, 13) != R.short_digest(b, 13)

    def test_join_digest_is_order_insensitive(self):
        """**先排序**，于是拼接顺序不是隐藏输入。

        我最初在这里断言的是「顺序敏感」，理由是「列表的顺序是内容的一部分」。
        那是错的，而错得值得记下来：``join_digest`` 合成的是一**组**摘要，
        组合的成员本来就是规范化排序过的 —— 若 ``join_digest`` 不排序，
        那么「调用方已经排过序」就成了一条它无法验证、也无法在违约时报错的前提。
        它自己排掉，前提就不需要了。
        """
        assert R.join_digest(["a", "b"]) == R.join_digest(["b", "a"])

    def test_join_digest_is_multiset_sensitive(self):
        """顺序不敏感，但**个数**敏感 —— 否则 ``[a, b]`` 与 ``[a, a, b]`` 同摘要，
        而那正是「同一个 Adapter 被挂两次」这种错误唯一会被发现的地方。"""
        assert R.join_digest(["a", "b"]) != R.join_digest(["a", "a", "b"])

    def test_join_digest_handles_empty(self):
        assert isinstance(R.join_digest([]), str)


class TestSeeds:
    def test_derive_seed_is_deterministic(self):
        assert R.derive_seed("lora", "upper-qv", 7) == R.derive_seed("lora", "upper-qv", 7)

    def test_derive_seed_separates_roles(self):
        """角色名进种子。少了它，``derive_seed("a","b")`` 与 ``derive_seed("ab")``
        会撞 —— 而种子相撞的症状是两组本该独立的实验变成同一组。"""
        assert R.derive_seed("a", "b") != R.derive_seed("ab")

    def test_derive_seed_separates_parts(self):
        assert R.derive_seed("x", 1, 2) != R.derive_seed("x", 12)

    def test_derive_seed_is_in_int_range(self):
        """上界是 ``2**64`` 而不是 ``2**63`` —— ``stable_hash64`` 取的是
        sha256 十六进制的**前 16 位**，那正好是完整的 64 位无符号数。

        这一条值得留着：种子只喂给 ``random.Random``（它接受任意整数），
        但若哪天有人把种子塞进一个要求 **有符号** 64 位的地方
        （NumPy 的 ``default_rng``、某些序列化格式），越界的那一半会报错
        或静默截断 —— 而那种错误离这里很远，查起来很贵。
        """
        for parts in (("a",), ("a", "b"), (1, 2, 3), ("",)):
            s = R.derive_seed(*parts)
            assert isinstance(s, int) and 0 <= s < 2**64

    def test_derive_seed_distributes(self):
        """32 个标签派生出的种子必须两两不同。撞了就要看得见 ——
        一个「种子相撞」的 bug 在所有下游都表现为「结果有点怪」。"""
        seeds = {R.derive_seed("zoo", f"cap-{i}") for i in range(32)}
        assert len(seeds) == 32

    def test_stable_hash64_is_stable_and_sensitive(self):
        assert R.stable_hash64("a", "b") == R.stable_hash64("a", "b")
        assert R.stable_hash64("a", "b") != R.stable_hash64("b", "a")


class TestQuantizeForHash:
    def test_collapses_nearby_floats(self):
        assert R.quantize_for_hash(1.0000001) == R.quantize_for_hash(1.0000002)

    def test_separates_far_floats(self):
        assert R.quantize_for_hash(1.0) != R.quantize_for_hash(1.001)

    def test_digits_controls_resolution(self):
        assert R.quantize_for_hash(1.0001, 3) == R.quantize_for_hash(1.0004, 3)
        assert R.quantize_for_hash(1.0001, 6) != R.quantize_for_hash(1.0004, 6)

    def test_rejects_non_finite(self):
        """NaN/Inf 进摘要会让**任何**两个含 NaN 的对象互相相等
        （NaN != NaN，但它们的 repr 相同）。要么拒绝，要么明确定义，不能含糊。"""
        with pytest.raises((ValueError, AssertionError)):
            R.quantize_for_hash(float("nan"))


# ----------------------------------------------------------------------
# schema
# ----------------------------------------------------------------------
def _metric(**kw) -> S.CapabilityMetric:
    base = dict(
        capability="upper", acc=0.5, gain=0.2, exact=0.25, eos_acc=1.0,
        n_positions=48.0, interference_from=None,
    )
    base.update(kw)
    return S.CapabilityMetric(**base)


def _run(**kw) -> S.CompositionRun:
    base = dict(
        cid="cid1", method="stack", members=["a"], metrics={"upper": _metric()},
        merge_error=1e-16, n_params_changed=100, macs=1000, wall_ms=1.5,
        artifact=None, coefs=[1.0], merge_loss=None,
    )
    base.update(kw)
    return S.CompositionRun(**base)


def _edge(**kw) -> S.ConflictEdge:
    base = dict(
        a="a", b="b", interference=0.3, i_a_from_b=0.2, i_b_from_a=0.4,
        predicted_cos=-0.1, predicted_sign=-1.0, agree=True, affected={"upper": 0.2},
    )
    base.update(kw)
    return S.ConflictEdge(**base)


class TestSchemaRoundTrip:
    """``from_dict(to_dict(x)) == x``。**每个记录类都要过**（坑 41/42/43）。"""

    @pytest.mark.parametrize(
        "obj",
        [
            _metric(),
            _metric(interference_from=0.3),
            _run(),
            _run(coefs=None, artifact=S.ArtifactRef("d1", 64, 8, 16, "out/x.bin")),
            _edge(),
            _edge(predicted_cos=None, predicted_sign=None, agree=None),
            S.AdapterRecord(
                label="upper-qv", capability="upper", variant="qv",
                targets=["q_proj", "v_proj"], rank=4, alpha=8.0, scale=2.0,
                digest="abc", n_bytes=128, n_params=64, base_key="tiny-1",
                source="produce", n_base_params_touched=576, train={"steps": 400},
            ),
            S.Finding(key="k", statement="s", expectation="e", measured="m",
                      verdict="推翻", note="n"),
        ],
        ids=lambda o: type(o).__name__,
    )
    def test_roundtrip(self, obj):
        cls = type(obj)
        again = cls.from_dict(obj.to_dict())
        assert again == obj, f"{cls.__name__} 往返不相等"

    def test_roundtrip_survives_json(self):
        """经一遍真的 JSON 文本 —— 内存里的 dict 往返会漏掉
        「tuple 变成 list」这类只在落盘后才出现的问题。"""
        for obj in (_metric(), _run(), _edge()):
            text = json.dumps(obj.to_dict(), ensure_ascii=False)
            again = type(obj).from_dict(json.loads(text))
            assert again == obj

    def test_none_is_not_conflated_with_zero(self):
        """``None``（没测）与 ``0.0``（测了，是 0）在产物里必须是两个值。"""
        m = _metric(interference_from=None)
        assert m.to_dict()["interference_from"] is None
        assert _metric(interference_from=0.0).to_dict()["interference_from"] == 0.0
        assert _metric(interference_from=None) != _metric(interference_from=0.0)

    def test_edge_roundtrip_keeps_none_cos(self):
        """``cos=None``（无法预测）与 ``cos=0.0``（预测为无关联）不是一回事。
        ``or 0.0`` 会把前者变成后者（坑 104）。"""
        e = _edge(predicted_cos=None)
        assert S.ConflictEdge.from_dict(e.to_dict()).predicted_cos is None


class TestCompositionRunHelpers:
    def test_size_is_member_count(self):
        """``size`` 是 **property**，不是方法 —— 写成 ``.size()`` 会得到
        ``'int' object is not callable``，一眼能看出是拼错了；
        真正危险的是反过来：把一个方法当属性用，拿到的是绑定方法对象，
        它到处都能比较、也永远不等于任何数。"""
        assert _run(members=["a", "b", "c"]).size == 3
        assert _run(members=[]).size == 0

    def test_mean_gain_requires_every_capability(self):
        """缺一个能力就返回 ``None``，**不拿已有能力的平均冒充**。"""
        r = _run(metrics={"upper": _metric(gain=0.4)})
        assert r.mean_gain(["upper"]) == pytest.approx(0.4)
        assert r.mean_gain(["upper", "shift"]) is None

    def test_mean_gain_averages(self):
        r = _run(metrics={"upper": _metric(gain=0.4), "shift": _metric(gain=0.2)})
        assert r.mean_gain(["upper", "shift"]) == pytest.approx(0.3)

    def test_mean_gain_of_nothing_is_none(self):
        assert _run().mean_gain([]) is None

    def test_own_gain_uses_member_capability(self):
        """``own_gain`` 问的是「成员自己那个能力上表现如何」，
        它靠 ``members[0]`` 的**后缀**推能力，所以标签格式是它的隐含前提。"""
        r = _run(members=["upper-qv"], metrics={"upper": _metric(gain=0.3)})
        assert r.own_gain() == pytest.approx(0.3)

    def test_own_gain_none_when_capability_missing(self):
        r = _run(members=["upper-qv"], metrics={"shift": _metric(gain=0.3)})
        assert r.own_gain() is None

    def test_own_gain_none_when_no_members(self):
        assert _run(members=[]).own_gain() is None


class TestConflictEdgeHelpers:
    def test_harmful_needs_positive_interference(self):
        """``harmful`` 是 **property**。边界取 ``> 0``：干扰恰好为 0
        与「干扰是负的（组合比单干还好）」都不是「有害」，
        把 0 也算进去会让「有害边数」在噪声上虚增。"""
        assert _edge(interference=0.3).harmful
        assert not _edge(interference=0.0).harmful
        assert not _edge(interference=-0.3).harmful


class TestVersion:
    def test_only_created_at_is_volatile(self):
        """``VOLATILE`` 是「两次跑会变的字段」的**显式清单**。
        它变长就等于「可复现性悄悄弱了一点」而没人发现。"""
        assert S.Version.VOLATILE == ("created_at",)


class TestPipelineResult:
    def test_run_by_cid_raises_on_missing(self):
        res = S.PipelineResult(
            version=S.Version("v1", "p", "v", "2026-01-01", 0, 0, 0),
            base_key="b", base_seed=1, seq_len=13, capabilities=["upper"],
            adapters=[], runs=[_run()], conflicts=[], frontier=[],
            counters={}, findings=[], notes={},
        )
        assert res.run_by_cid("cid1").cid == "cid1"
        with pytest.raises((KeyError, ValueError)):
            res.run_by_cid("nope")

    def test_stable_view_excludes_volatile_fields(self):
        res = S.PipelineResult(
            version=S.Version("v1", "p", "v", "2026-01-01", 0, 0, 0),
            base_key="b", base_seed=1, seq_len=13, capabilities=["upper"],
            adapters=[], runs=[_run()], conflicts=[], frontier=[],
            counters={}, findings=[], notes={},
        )
        a = res.stable_view()
        res.version = S.Version("v1", "p", "v", "2099-12-31", 0, 0, 0)
        assert res.stable_view() == a, "created_at 不该进稳定视图"


class TestVolatilePaths:
    """``VOLATILE_PATHS`` 这张**豁免单**。

    上面那条只改了 ``created_at`` 一个字段 —— 一张豁免单有四项，只验一项的
    断言会让另外三项处于「声明了但没人核对」的状态，而漏掉一项的症状是
    **随机失败**（真实运行里耗时差到 0.05 秒以上才会显形），随机失败的断言
    最后的归宿通常是加一个 ``pytest.mark.flaky`` 而不是被修对。

    所以这里**逐项**验：每一项都必须真的被 ``stable_view`` 剔掉。
    """

    def test_the_declared_exceptions_are_exactly_these(self):
        """这张表是**结论**，不是借口。放宽它（比如说把整条 ``runs`` 加进去）
        能让「两次运行逐位相同」继续绿 —— 而那时它已经什么都不查了。"""
        assert S.PipelineResult.VOLATILE_PATHS == (
            "version.created_at",
            "counters.runtime_s",
            "runs[].wall_ms",
            "runs[].artifact.path",
        )

    @pytest.mark.parametrize(
        "path,ok",
        [
            ("version.created_at", True),
            ("counters.runtime_s", True),
            ("runs[0].wall_ms", True),
            ("runs[12].artifact.path", True),
            # ``[]`` 只匹配**一个下标**，不是「任意多段」：写成通配的话
            # ``runs[].metrics`` 这种模板会顺带赦免整棵子树。
            ("runs[].wall_ms", False),
            ("runs[0].metrics.upper.acc", False),
            ("runs[0].macs", False),
            ("counters.runtime", False),
            ("counters.runtime_s.x", False),
            ("version.created_at ", False),
        ],
    )
    def test_is_volatile_matches_one_index_at_a_time(self, path, ok):
        assert S.PipelineResult.is_volatile(path) is ok

    def _result(self) -> S.PipelineResult:
        """一份**四项豁免都活着**的最小结果：有 ``created_at``、有 ``runtime_s``、
        有 ``wall_ms``、也有 ``artifact.path``。少一项，该项的「剔掉了没」就
        变成了空真。"""
        return S.PipelineResult(
            version=S.Version("v1", "p", "v", "2026-01-01", 0, 0, 0),
            base_key="b", base_seed=1, seq_len=13, capabilities=["upper"],
            adapters=[],
            runs=[_run(artifact=S.ArtifactRef("abc", 64, 4, 16, "out/compositions/cid1.bin"))],
            conflicts=[], frontier=[], counters={"runtime_s": 1.0}, findings=[], notes={},
        )

    def test_every_declared_path_is_implemented(self):
        """**声明与实现必须是一回事。** 逐项改掉一个被声明的字段，
        稳定视图不该动；改掉一个没被声明的字段，稳定视图必须动。

        第二半是防「剔多了」：``stable_view`` 里多 pop 一个键（比如顺手把
        ``counters`` 整个删掉）会让判据比声明更宽，而正向比较照样绿。
        """
        res = self._result()
        baseline = res.stable_view()

        # 键**故意用模板本身**：这样 ``set(mutants) == VOLATILE_PATHS`` 才是
        # 一条真有内容的相等断言（用 ``runs[0].wall_ms`` 当键的话，两边永远不等）。
        # ``Version`` 与 ``ArtifactRef`` 是 frozen 的 —— 改它们要**换掉整个对象**，
        # 不能 setattr。这不是绕路：frozen 正是为了让「产物读完之后还能不能变」
        # 这个问题有一个「不能」的答案。
        mutants = {
            "version.created_at": lambda r: setattr(
                r, "version", S.Version("v1", "p", "v", "2099-12-31", 0, 0, 0)),
            "counters.runtime_s": lambda r: r.counters.__setitem__("runtime_s", 99.0),
            "runs[].wall_ms": lambda r: setattr(r.runs[0], "wall_ms", 12345.0),
            "runs[].artifact.path": lambda r: setattr(
                r.runs[0], "artifact", S.ArtifactRef("abc", 64, 4, 16, "/elsewhere/cid1.bin")),
        }
        assert set(mutants) == set(S.PipelineResult.VOLATILE_PATHS)
        for path, mutate in mutants.items():
            res = self._result()
            mutate(res)
            assert res.stable_view() == baseline, f"{path} 声明为易变却没被剔掉"

        # 反向：没被声明的字段必须**留**在视图里
        for path, mutate in {
            "runs[0].macs": lambda r: setattr(r.runs[0], "macs", 1),
            "runs[0].artifact.digest": lambda r: setattr(
                r.runs[0], "artifact", S.ArtifactRef("zzz", 64, 4, 16, r.runs[0].artifact.path)),
            "counters.n_runs": lambda r: r.counters.__setitem__("n_runs", 7),
            "frontier": lambda r: r.frontier.append("cid1"),
        }.items():
            res = self._result()
            mutate(res)
            assert res.stable_view() != baseline, f"{path} 不该被剔掉"
            assert not S.PipelineResult.is_volatile(path), path
