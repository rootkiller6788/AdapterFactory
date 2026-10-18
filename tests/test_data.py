"""``data``：题、词表、两条基线。**两条口径在这里定下来，整个项目都得跟着走。**

1. **``chance`` 是 ``1/|payload_chars|``，不是 ``1/|vocab|``。**
   逐字符映射任务里「猜中」只可能发生在 payload 字母表上；把 specials 与
   任务字母也算进去，会把一个恰好蒙对的模型报成「远高于乱猜」。
2. **``mean_accuracy`` 是按位置微平均、且不含 EOS。** 与 ``heuristic``
   基线、与 ``gain`` 必须同口径 —— 两套口径下的两个数相减，
   是一个两边都「算对了」的错数。
"""

from __future__ import annotations

import pytest

from adapterfactory import data as D


class TestIndex:
    def test_payload_alphabet_is_smaller_than_vocab(self, index, vocab):
        """这不是巧合，是这条测试的前提。若 payload 字母表等于整个词表，
        下面两条关于 chance 的断言就会一起失去意义。"""
        assert len(index.payload_chars) < vocab.size

    def test_seq_len_is_derived_from_payload_len(self, index):
        assert D.seq_len_of(index) == 5 + 2 * index.payload_len

    def test_roundtrip_is_exact_except_for_declared_volatile_fields(self, index):
        """``from_dict∘to_dict`` 互逆 —— **除了 :attr:`DataIndex.VOLATILE`**。

        ``source`` 记的是「从哪个路径读的」，它进指纹会让同一份索引在两个
        挂载点下算出两个版本摘要。所以它被排除是对的；但它被排除这件事
        必须**声明出来**，否则「严格互逆」这条性质就悄悄变成
        「除了 source 以外互逆」，而且没有任何东西会说出这个例外。
        """
        import dataclasses

        again = D.DataIndex.from_dict(index.to_dict())
        diffs = {
            f.name
            for f in dataclasses.fields(D.DataIndex)
            if getattr(again, f.name) != getattr(index, f.name)
        }
        assert diffs == set(D.DataIndex.VOLATILE), f"往返差异 {diffs} 与声明的 {D.DataIndex.VOLATILE} 不符"

    def test_roundtrip_is_exact_when_source_is_empty(self, index):
        """把易变字段归零之后，往返必须**逐字段相等**。"""
        import dataclasses

        clean = dataclasses.replace(index, source="")
        assert D.DataIndex.from_dict(clean.to_dict()) == clean

    def test_volatile_is_exactly_the_expected_field(self):
        """这条挡的是「为了让某条测试通过而往 VOLATILE 里加一项」。"""
        assert D.DataIndex.VOLATILE == ("source",)

    def test_capabilities_follows_the_index(self, index):
        assert index.capabilities == ("upper", "shift", "reverse", "sort")

    def test_iter_capabilities_keeps_the_requested_order(self, index):
        """它是个**过滤器**：按调用方给的顺序返回，而不是按索引里的顺序。
        报告里 zoo 的顺序由配置里的能力列表决定 —— 若这里擅自按索引排序，
        那么「报告的顺序」就有两个来源，而它们只在有人改了配置时才不一致。
        """
        assert list(D.iter_capabilities(index, ("sort", "upper"))) == ["sort", "upper"]

    def test_iter_capabilities_dedupes_keeping_the_first_occurrence(self, index):
        """重复的能力名必须去重。不去重的后果是**同一个能力被算两遍**：
        zoo 里出现两个 ``upper-qv``，而网格公式照样自洽（它按标签个数算），
        于是「276 个组合」里有一半是重复的而没人发现。"""
        caps = ["upper", "shift", "upper"]
        assert list(D.iter_capabilities(index, caps)) == ["upper", "shift"]

    def test_unknown_capability_is_rejected(self, index):
        """索引只能**选**能力，不能发明能力 —— 一个拼错的能力名必须在
        这里就炸，而不是等到「这个能力没有探针集所以没测」。"""
        with pytest.raises((KeyError, ValueError)):
            D.iter_capabilities(index, ("nope",))


class TestVocab:
    def test_tokens_is_a_tuple_not_a_method(self, vocab):
        """``tokens`` 是 **property**。写成 ``vocab.tokens()`` 会立刻报
        ``TypeError: 'tuple' object is not callable``，所以这个错误是安全的；
        但把它读成「一个方法」还会让人以为它在每次调用时重新拼一遍。"""
        assert isinstance(vocab.tokens, tuple)

    def test_ids_are_dense_from_zero(self, vocab):
        ids = sorted(vocab.id(t) for t in vocab.tokens)
        assert ids == list(range(vocab.size))

    def test_specials_are_present(self, vocab, index):
        for s in index.specials:
            assert vocab.has(s)

    def test_unknown_token_raises(self, vocab):
        with pytest.raises((KeyError, ValueError)):
            vocab.id("ZZZ")

    def test_pad_bos_eos_are_distinct(self, vocab):
        assert len({vocab.pad, vocab.bos, vocab.eos}) == 3

    def test_payload_chars_are_a_subset_of_chars(self, index):
        assert set(index.payload_chars) <= set(index.chars)


class TestChanceLevel:
    def test_is_over_payload_alphabet_not_vocab(self, index, vocab):
        """**本项目最容易被写错的一个数。** 1/26 = 0.0385 与 1/8 = 0.125
        差了三倍多，而两者看起来都「像 chance」。
        """
        assert D.chance_level(index) == pytest.approx(1.0 / len(index.payload_chars))
        assert D.chance_level(index) != pytest.approx(1.0 / vocab.size)

    def test_is_in_a_sane_range(self, index):
        assert 0.0 < D.chance_level(index) < 1.0


class TestExamples:
    def test_deterministic(self, vocab, index):
        a = D.build_examples(vocab, index, "upper", seq_len=13, n=8, seed=5)
        b = D.build_examples(vocab, index, "upper", seq_len=13, n=8, seed=5)
        assert [e.ids for e in a] == [e.ids for e in b]

    def test_different_seeds_differ(self, vocab, index):
        a = D.build_examples(vocab, index, "upper", seq_len=13, n=16, seed=5)
        b = D.build_examples(vocab, index, "upper", seq_len=13, n=16, seed=6)
        assert [e.ids for e in a] != [e.ids for e in b]

    def test_ids_have_the_configured_length(self, vocab, index):
        for e in D.build_examples(vocab, index, "upper", seq_len=13, n=4, seed=1):
            assert len(e.ids) == 13

    def test_targets_are_the_ids_shifted_by_one(self, vocab, index):
        """**老师的定义。** ``targets[t]`` 是「看到前 t+1 个 token 之后该预测的
        下一个」。错位的症状是 loss 学得好好的而准确率不动。"""
        for e in D.build_examples(vocab, index, "upper", seq_len=13, n=3, seed=1):
            assert e.targets[:-1] == e.ids[1:]
            assert e.targets[-1] == -1, "最后一个位置没有下一个 token"

    def test_mask_marks_the_prediction_positions_not_the_response_tokens(self, vocab, index):
        """**``mask`` 标的是「从哪个位置预测」，不是「哪个位置是回答」。**

        两者差一格：回答占据 ``ids[response_start : response_start+response_len]``，
        而预测它的是 ``response_start-1`` 起的那些位置（因为 ``targets[t] = ids[t+1]``）。
        我第一版把这两者当成同一个东西，断言 ``marked == range(response_start, ...)``，
        得到 ``[7,8,9,10,11]`` 对 ``[8,9,10,11,12]`` —— 差 1 的错位是这类 bug 的
        标准长相，而它的后果是**第一个回答 token 永远没有梯度**。
        """
        for e in D.build_examples(vocab, index, "upper", seq_len=13, n=3, seed=1):
            assert len(e.mask) == len(e.ids)
            marked = [i for i, m in enumerate(e.mask) if m]
            assert marked == list(D.response_positions(e))
            assert marked[0] == e.response_start - 1
            assert marked[-1] == e.response_start + e.response_len - 2

    def test_content_positions_exclude_eos(self, vocab, index):
        """**这是本项目两把尺子的分界。** ``content_positions`` 是逐 token
        准确率的分母（不含 EOS）；``response_positions`` 多一个 EOS 位置。
        把 EOS 算进 ``acc``，会让「停得住」这件事混进「内容对不对」。"""
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        assert len(D.content_positions(e)) == e.response_len - 1
        assert len(D.response_positions(e)) == e.response_len
        assert set(D.content_positions(e)) < set(D.response_positions(e))
        assert D.response_positions(e)[-1] not in D.content_positions(e)

    def test_target_at_eos_position_is_eos(self, vocab, index):
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        assert e.ids[e.response_start + e.response_len - 1] == vocab.eos
        # 而且预测 EOS 的那个位置的 target 就是 EOS
        assert e.targets[D.response_positions(e)[-1]] == vocab.eos

    def test_response_region_is_inside_the_sequence(self, vocab, index):
        for e in D.build_examples(vocab, index, "upper", seq_len=13, n=3, seed=1):
            assert 0 <= e.response_start
            assert e.response_start + e.response_len <= len(e.ids)

    def test_payload_has_the_configured_length(self, vocab, index):
        for e in D.build_examples(vocab, index, "upper", seq_len=13, n=4, seed=1):
            assert len(e.payload) == index.payload_len

    def test_exclude_removes_those_payloads(self, vocab, index):
        """**训练集与评测集必须显式互斥。** 「种子不同所以大概不会撞」不够：
        payload 空间只有 8⁴ = 4096 种，而 n_eval + n_train = 240 ——
        靠概率排除重叠，就是把一个必然事件当成偶然事件。"""
        first = D.build_examples(vocab, index, "upper", seq_len=13, n=32, seed=1)
        taken = frozenset(e.payload for e in first[:4])
        rest = D.build_examples(
            vocab, index, "upper", seq_len=13, n=32, seed=1, exclude=taken
        )
        assert not (set(e.payload for e in rest) & taken)

    def test_every_task_actually_transforms_the_payload(self, vocab, index):
        """四个任务都必须真的在改东西。若某个任务的实现是恒等，
        「这个能力学到了」与「什么都没做」在产物里就分不开 ——
        而 ``copy`` 基线会突然变成 1.0，把 heuristic 顶到天花板上。

        用**至少 8 条**样本：单看一条，``sort`` 在已经有序的 payload 上
        就是恒等，那会让这条测试随机地失败。"""
        toks = vocab.tokens
        for cap in index.capabilities:
            exs = D.build_examples(vocab, index, cap, seq_len=13, n=8, seed=3)
            changed = sum(
                1
                for e in exs
                if e.payload != "".join(toks[e.targets[i]] for i in D.content_positions(e))
            )
            assert changed > 0, f"{cap} 的任务函数是恒等映射"


class TestAccuracy:
    def test_perfect_prediction_is_one(self, vocab, index):
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        # 造一份「每个内容位置都答对」的 logits
        T, V = len(e.ids), vocab.size
        logits = [[0.0] * V for _ in range(T)]
        for t in D.content_positions(e):
            logits[t][e.targets[t]] = 10.0
        acc, eos_acc, n = D.token_accuracy(logits, e)
        assert acc == pytest.approx(1.0)
        assert n == len(D.content_positions(e))

    def test_returns_a_tuple_not_a_dict(self, vocab, index):
        """``token_accuracy`` 返回 ``(内容准确率, EOS 准确率, 内容位置数)``。

        与 :func:`mean_accuracy` 返回 dict 不一致 —— 这是刻意的：
        前者是**一个样本**的三个数，写成一个三元组让调用方必须当场拆开；
        后者是一组样本的聚合，用 dict 于是加字段不会破坏调用点。
        """
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        flat = [[0.0] * vocab.size for _ in range(len(e.ids))]
        out = D.token_accuracy(flat, e)
        assert isinstance(out, tuple) and len(out) == 3

    def test_wrong_answer_scores_zero_on_that_position(self, vocab, index):
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        T, V = len(e.ids), vocab.size
        logits = [[0.0] * V for _ in range(T)]
        for t in D.content_positions(e):
            wrong = (e.targets[t] + 1) % V
            logits[t][wrong] = 10.0
        acc, _, _ = D.token_accuracy(logits, e)
        assert acc == pytest.approx(0.0)

    def test_eos_position_does_not_enter_acc(self, vocab, index):
        """答对全部内容、**答错 EOS** ⇒ 内容准确率仍然是 1.0。
        这正是「两把尺子」的意思：EOS 由 ``eos_acc`` 单独承担。

        合成一个数（比如 ``0.8·acc + 0.2·eos``）的代价是：
        一个内容全对但停不住的模型，与一个内容全错但停得住的模型，
        会拿到同一个分数 —— 而它们在部署时的下场完全不同。
        """
        e = D.build_examples(vocab, index, "upper", seq_len=13, n=1, seed=1)[0]
        T, V = len(e.ids), vocab.size
        logits = [[0.0] * V for _ in range(T)]
        for t in D.content_positions(e):
            logits[t][e.targets[t]] = 10.0
        eos_t = D.response_positions(e)[-1]
        logits[eos_t][(e.targets[eos_t] + 1) % V] = 10.0
        acc, eos_acc, _ = D.token_accuracy(logits, e)
        assert acc == pytest.approx(1.0)
        assert eos_acc == pytest.approx(0.0)

    def test_mean_accuracy_is_micro_averaged(self, vocab, index):
        """**微平均（按位置）而不是宏平均（按样本）。** 两个样本一个 4 位、
        一个 4 位时两者相同；一旦长度不同，宏平均会让短样本权重过大。
        这里四个位置全对 ⇒ 1.0。"""
        exs = D.build_examples(vocab, index, "upper", seq_len=13, n=3, seed=1)
        T, V = len(exs[0].ids), vocab.size
        logits = []
        for e in exs:
            lg = [[0.0] * V for _ in range(T)]
            for t in D.content_positions(e):
                lg[t][e.targets[t]] = 10.0
            logits.append(lg)
        agg = D.mean_accuracy(logits, exs)
        assert agg["acc"] == pytest.approx(1.0)
        assert agg["n_positions"] == sum(len(D.content_positions(e)) for e in exs)

    def test_mean_accuracy_counts_positions_not_samples(self, vocab, index):
        exs = D.build_examples(vocab, index, "upper", seq_len=13, n=4, seed=1)
        agg = D.mean_accuracy([[[0.0] * vocab.size] * 13] * len(exs), exs)
        assert agg["n_positions"] == len(exs) * index.payload_len


class TestBaselines:
    """**两条基线，验收盯更强的那个**（坑 47）。"""

    def test_chance_matches_chance_level(self, idx_sets):
        idx, bl = idx_sets
        for cap, row in bl.items():
            assert row["chance"] == pytest.approx(D.chance_level(idx)), cap

    def test_heuristic_is_the_better_of_majority_and_copy(self, idx_sets):
        """``heuristic = max(majority, copy)``。取 max 而不是平均：
        参照实现的意义是「一个不做任何学习的人能达到什么」，
        而他一定会挑两者里对自己更有利的那个。"""
        _, bl = idx_sets
        for cap, row in bl.items():
            assert row["heuristic"] == pytest.approx(max(row["majority"], row["copy"])), cap

    def test_copy_scores_zero_for_tasks_that_always_move_the_token(self, idx_sets):
        """``upper`` 与 ``shift`` 下，输出字符**永远**不等于输入字符
        （大写字母 vs 小写、每个字符移位一位）⇒ 照抄得 0 分。

        我最初把 ``reverse`` 也写进这条，它 0.1458 ≠ 0。那**不是** bug：
        回文串的对称位置上「抄」恰好等于「反转」。这恰好说明
        ``copy`` 的分不是一个「实现质量」指标，而是一个任务相关的下界 ——
        所以下面那条按能力分别记下它的**实际**档位。
        """
        _, bl = idx_sets
        assert bl["upper"]["copy"] == pytest.approx(0.0)
        assert bl["shift"]["copy"] == pytest.approx(0.0)

    def test_copy_can_be_nonzero_for_position_symmetric_tasks(self, idx_sets):
        """``reverse`` 的 ``copy`` 不该是 0 —— 它是**结构性**的，不是噪声。
        钉住它，是为了让「某次改动让 reverse 的 copy 变成 0」这件事被看见。"""
        _, bl = idx_sets
        assert bl["reverse"]["copy"] > 0.0

    def test_sort_has_the_strongest_copy_baseline(self, idx_sets):
        """``sort`` 的 ``copy`` 是四个能力里最高的（实测约 0.30）——
        因为随机 payload 有一定概率本来就接近有序。

        这是「``min_single_gain_margin`` 定 0.10 而不是 0.15」那条注释的
        实测依据：要求 sort 超过 heuristic 0.15 分，等于要求它远超
        「照抄输入」，那是在测任务天花板而不是测 Adapter。
        """
        _, bl = idx_sets
        assert bl["sort"]["copy"] == max(bl[c]["copy"] for c in bl)
        assert bl["sort"]["heuristic"] == pytest.approx(bl["sort"]["copy"])

    def test_majority_is_fitted_on_train_not_eval(self, index, vocab):
        """**在评测集上拟合众数就是泄漏。**

        这里能观察到的性质有两个，且都不足以单独证明「没泄漏」，
        合起来才有力：

        - ``majority`` 随**训练集**变化 ⇒ 它确实读了训练集；
        - ``chance`` 与两个集合都无关 ⇒ 这一条是对照，防止
          「两者都变」被误读成「训练集被读了」。

        「没在评测集上拟合」这件事更强的证据是**结构性的**：
        :func:`D.fit_majority` 的参数表里根本没有评测集（见下一条）。
        """
        tr_a = D.build_train_set(index, vocab, seq_len=13, n_train=16, seed=11)
        tr_b = D.build_train_set(index, vocab, seq_len=13, n_train=16, seed=99)
        ev = D.build_eval_sets(index, vocab, seq_len=13, n_eval=32, seed=22)
        a = D.baselines(index, tr_a, ev)
        b = D.baselines(index, tr_b, ev)
        assert a != b, "换了训练集而 majority 一点没变 ⇒ 它没读训练集"
        for cap in a:
            assert a[cap]["chance"] == pytest.approx(b[cap]["chance"]), "chance 不该依赖任何集合"

    def test_fit_majority_cannot_see_the_eval_set(self, index, vocab):
        """**泄漏在这里被结构性地堵住**：``fit_majority`` 的签名里
        只有 ``(idx, train_sets)``。它想泄漏都拿不到评测集。

        这一条是有意去查签名的 —— 参数表是接口的一部分，
        而「哪些数据进得来」正是这里唯一要紧的事。"""
        import inspect

        params = list(inspect.signature(D.fit_majority).parameters)
        assert not any("eval" in p for p in params), params
        assert any("train" in p for p in params), params

    def test_fit_majority_is_deterministic(self, index, vocab):
        tr = D.build_train_set(index, vocab, seq_len=13, n_train=16, seed=11)
        assert D.fit_majority(index, tr) == D.fit_majority(index, tr)

    def test_baselines_are_json_safe(self, idx_sets):
        import json

        _, bl = idx_sets
        json.dumps(bl, ensure_ascii=False)


class TestSetDisjointness:
    def test_train_and_eval_payloads_are_disjoint(self, index, vocab):
        """**验收的前提之一。** 训练集与评测集若有重叠，「记住了」
        会被读成「学会了」—— 而这个项目里所有 gain 都建立在这上面。

        注意 ``exclude`` 是**显式**传进去的：种子不同并不保证不重叠
        （payload 空间只有 8⁴ = 4096 种），把「大概不会撞」当成保证，
        是在赌一个必然会输的赌局。
        """
        ev = D.build_eval_sets(index, vocab, seq_len=13, n_eval=48, seed=20260930)
        tr = D.build_train_set(
            index, vocab, seq_len=13, n_train=192, seed=20260930,
            exclude={c: frozenset(e.payload for e in ev[c]) for c in ev},
        )
        for cap in index.capabilities:
            overlap = {e.payload for e in tr[cap]} & {e.payload for e in ev[cap]}
            assert not overlap, f"{cap}: 训练与评测重叠 {len(overlap)} 条"

    def test_without_exclude_they_can_overlap(self, index, vocab):
        """反向：**不传 ``exclude`` 时重叠是可能发生的**。
        这一条存在的意义是证明上一条的 ``exclude`` 不是装饰 ——
        若 4096 的 payload 空间大到天然不重叠，上一条就什么都没测到。"""
        ev = D.build_eval_sets(index, vocab, seq_len=13, n_eval=192, seed=20260930)
        tr = D.build_train_set(index, vocab, seq_len=13, n_train=192, seed=20260930)
        total = sum(
            len({e.payload for e in tr[c]} & {e.payload for e in ev[c]})
            for c in index.capabilities
        )
        assert total > 0, "参数变了：现在的 n_eval/n_train 已经天然不重叠，这条测试该重新设计"

    def test_eval_sets_are_deterministic(self, index, vocab):
        a = D.build_eval_sets(index, vocab, seq_len=13, n_eval=8, seed=7)
        b = D.build_eval_sets(index, vocab, seq_len=13, n_eval=8, seed=7)
        for cap in index.capabilities:
            assert [e.ids for e in a[cap]] == [e.ids for e in b[cap]]


@pytest.fixture(scope="module")
def idx_sets():
    """``(index, baselines)`` —— 基线要给四个能力都算出来，代价是几百次
    查表，但它要真跑一遍才作数。"""
    index = D.load_index("data/datasets/index.json")
    vocab = D.build_vocab(index)
    ev = D.build_eval_sets(index, vocab, seq_len=D.seq_len_of(index), n_eval=48, seed=20260930)
    tr = D.build_train_set(
        index, vocab, seq_len=D.seq_len_of(index), n_train=192, seed=20260930,
        exclude={c: frozenset(e.payload for e in ev[c]) for c in ev},
    )
    return index, D.baselines(index, tr, ev)
