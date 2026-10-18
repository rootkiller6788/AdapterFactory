"""``audit``：**架构墙**与九项核对。

「AdapterFactory 不训练」是本项目的核心命题。这个文件验的是**它可执行** ——
而不是它写在 README 里。两类断言：

1. 墙**真的会响**（`train_adapter(...)`、`backward()`、`import produce`）；
2. 墙**不会被合法代码触发**（docstring、字段名、``build_train_set`` 这种
   「训练侧但不更新权重」的调用）。

第 2 类与第 1 类同等重要，甚至更重要：一条会把合法代码判成违规的检查，
最终一定会被人关掉 —— 而关掉之后第 1 类也不再成立。
"""

from __future__ import annotations

import re

import pytest

from adapterfactory import audit as A
from adapterfactory import compose as P
from adapterfactory import config as C
from adapterfactory import schema as S


# ----------------------------------------------------------------------
# 墙本身
# ----------------------------------------------------------------------
class TestWallCatchesRealViolations:
    def test_catches_a_training_call(self):
        src = "def f():\n    train_adapter(1, 2)\n"
        assert A.wall_findings(src, "compose")

    def test_catches_a_method_call(self):
        """``optimizer.backward()`` —— ``ast`` 的 ``Attribute`` 分支要认得。"""
        assert A.wall_findings("def f(o):\n    o.backward()\n", "evaluate")

    def test_catches_an_optimizer_construction(self):
        assert A.wall_findings("def f():\n    return AdamW(1)\n", "pipeline")

    def test_catches_importing_the_producer(self):
        assert A.wall_findings("from . import produce\n", "compose")

    def test_catches_a_plain_import(self):
        assert A.wall_findings("import adapterfactory.produce\n", "compose")

    def test_catches_a_syntax_error_instead_of_passing_it(self):
        """**「没扫到」不等于「干净」。** 一个解析不了的模块恰恰是最需要
        被看一眼的那个；静默返回空列表会把它伪装成通过。"""
        found = A.wall_findings("def f(:\n", "compose")
        assert found and "解析失败" in found[0]

    def test_catches_the_bare_identifier_backward(self):
        """保守一侧：出现在流水线模块里的 ``backward`` **无论何种用法**
        都值得看一眼 —— 这里是「当值用」而不是「当函数调」，
        ``ast`` 那条路只看得见 ``Call``，看不见这种写法。"""
        found = A.wall_findings("def f(g, backward):\n    return g(backward)\n", "report")
        assert any("backward" in f for f in found)

    def test_does_not_pretend_to_catch_a_concatenated_name(self):
        """**墙的范围要写清楚。** 拼字符串造名字（``"back" + "ward"``）两条路
        都看不见 —— tokenize 只把 ``"back"`` 与 ``"ward"`` 看成两个字符串
        字面量。把它当成「也能抓到」是给自己一个错误的安全感：
        真有人这么做的时候，墙会安静地放行。
        本项目的实际处境不同（流水线模块是我们自己写的，且 ① 每次 ``run``
        都在扫），所以这里如实记录边界，而不是加一条永远不触发的检查。"""
        found = A.wall_findings('def f(m):\n    getattr(m, "back" + "ward")()\n', "report")
        assert not any("backward" in f for f in found)


class TestWallDoesNotFireOnLegalCode:
    """**这一组是这个文件里最重要的部分。**"""

    def test_docstrings_do_not_count(self):
        """第一版查的是标识符 ``train``，扫出 15 处 —— 全是**名词**
        （``config["train"]``、``ZooRecord.train``、``AdapterRecord(train=...)``），
        没有一处是训练动作。名词不该被禁。"""
        src = (
            '"""这个模块谈 train_adapter 与 backward，但一次都不调用。"""\n'
            "def f(cfg):\n"
            '    steps = cfg["train"]["steps"]\n'
            "    return steps\n"
        )
        assert A.wall_findings(src, "compose") == []

    def test_comments_do_not_count(self):
        """``tokenize`` 而非文本扫描的全部理由：文本扫描会把注释与
        本文件自己的 docstring 当成违规。"""
        src = "# 这里曾经调用过 train_adapter，现在没有了\ndef f():\n    return 1\n"
        assert A.wall_findings(src, "report") == []

    def test_field_names_do_not_count(self):
        src = (
            "def f(rec):\n"
            "    return {'train': rec.train, 'steps': 1}\n"
        )
        assert A.wall_findings(src, "zoo") == []

    def test_data_side_calls_are_not_violations(self):
        """``build_train_set`` 只是把样本 materialize 出来，不更新任何权重 ——
        而它还是 ``majority`` 基线**必须**的输入（众数只能在训练集上拟合，
        在评测集上拟合就是泄漏）。禁掉它的后果不是「更安全」，
        而是让流水线要么算不出基线，要么去找一个语义不同的替代实现。"""
        src = "def f(idx):\n    return build_train_set(idx)\n"
        assert A.wall_findings(src, "pipeline") == []

    def test_but_data_side_calls_are_still_reported(self):
        """它们照报，只是不否决 —— 「流水线确实读了训练集」这件事
        应当留在记录里，而不是靠读者从源码里推。"""
        src = "def f(idx):\n    return build_train_set(idx)\n"
        assert A.data_side_findings(src, "pipeline") == ["pipeline: build_train_set()"]

    def test_the_two_lists_are_disjoint(self):
        """一个名字不可能既是「训练动作」又是「训练侧只读调用」。
        重叠会让 ``required`` 那个布尔变得含糊。"""
        assert not (set(A.FORBIDDEN_CALLS) & set(A.DATA_SIDE_CALLS))
        assert "build_train_set" not in A.FORBIDDEN_CALLS
        assert "batch_order" not in A.FORBIDDEN_CALLS


class TestProtectedModuleList:
    def test_the_engine_is_not_protected(self):
        """``net`` 是共享的数值引擎（它的 ``backward`` 只服务 ``produce``）；
        ``data`` 是 ``build_train_set`` 的**定义处** —— 把定义处也纳入扫描，
        只会让 ``def build_train_set`` 那一行报自己的违规；
        ``produce`` 就是墙的另一侧；``cli`` 是控制点。"""
        for mod in ("net", "data", "produce", "cli"):
            assert mod not in A.PIPELINE_MODULES, mod

    def test_the_control_point_is_stated_separately(self):
        assert A.CONTROL_MODULE == "cli"
        assert A.CONTROL_MODULE not in A.PIPELINE_MODULES

    def test_producer_importers_on_the_real_package(self, repo):
        """**控制点唯一**：producer 只能被 ``cli`` 够得着。

        「谁都不许碰」的极端满足方式是把 producer 从图里摘出去
        （没人导入 = 没人能用 = 检查全绿）。这条不变量要求它**恰好**
        被控制点导入，于是「可达」与「唯一」两件事一起被守住。
        """
        importers = A.producer_importers(repo / "src" / "adapterfactory")
        assert sorted(importers) == ["cli"], importers

    def test_producer_importers_ignores_the_producer_itself(self, tmp_path):
        (tmp_path / "produce.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "compose.py").write_text("from . import produce\n", encoding="utf-8")
        assert A.producer_importers(tmp_path) == {"compose": ["produce"]}

    def test_producer_importers_reports_nobody_as_empty(self, tmp_path):
        (tmp_path / "compose.py").write_text("x = 1\n", encoding="utf-8")
        assert A.producer_importers(tmp_path) == {}

    def test_real_package_is_clean(self, repo):
        """真实源码上两条路都必须干净 —— 这是 ① 在每次 ``run`` 上做的事。"""
        src = repo / "src" / "adapterfactory"
        findings = []
        for mod in A.PIPELINE_MODULES:
            p = src / f"{mod}.py"
            assert p.is_file(), f"受保护清单里的 {mod}.py 不存在"
            findings.extend(A.wall_findings(p.read_text(encoding="utf-8"), mod))
        assert findings == []

    def test_config_offers_no_structural_switch(self):
        """**可执行的判据**：``DEFAULTS.acceptance`` 里不许出现结构性的
        ``require_*`` 键。文档说「没有开关」是一句声明；这条断言让它变成
        一件代码能查的事 —— 否则那个键会再一次以「默认 True，所以无害」
        的理由长回来。"""
        acc = C.DEFAULTS["acceptance"]
        for name in ("no_training_in_pipeline", "producer_contained", "grid_complete", "cid_canonical"):
            assert f"require_{name}" not in acc, f"{name} 不该有开关"

    def test_the_shipped_config_matches_defaults(self, repo):
        """``configs/default.json`` 是**另一份**写下来的配置。曾经它比
        ``DEFAULTS`` 多两个键（就是上面那两个开关）—— 两份配置漂移时，
        真正生效的是哪一份取决于加载路径，而这类分歧不会有任何报错。

        比**整份**文件而不是只比 ``acceptance`` 一节：那一节之所以是被发现
        漂移的地方，是因为有人正好在看它。整份比才是这条断言的完整形态，
        而 ``configs/default.json`` 的注释一直声称的就是「逐键相同」——
        在只比一节的时候，那句话是**过宽**的。``_comment`` 不比：
        它是给人读的说明，两份文件里本来就各写各的。
        """
        import json

        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k != "_comment"}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o

        shipped = json.loads((repo / "configs" / "default.json").read_text(encoding="utf-8"))
        assert strip(shipped) == strip(dict(C.DEFAULTS))


class TestScanTokens:
    def test_keeps_names_and_operators(self):
        toks = A.scan_tokens("a = b + 1\n")
        names = [s for t, s, _ in toks]
        assert "a" in names and "=" in names and "+" in names

    def test_drops_comments_and_strings(self):
        names = [s for t, s, _ in A.scan_tokens('# train_adapter\nx = "backward"\n')]
        assert "train_adapter" not in names
        assert "backward" not in names

    def test_reports_line_numbers(self):
        toks = A.scan_tokens("a = 1\nb = 2\n")
        assert toks[-1][2] == 2

    def test_token_error_propagates(self):
        """源码读不成 token 时**不能**静默返回空列表 —— 那会把
        「tokenize 失败了」伪装成「干净」。"""
        import tokenize

        with pytest.raises((tokenize.TokenError, IndentationError, SyntaxError)):
            A.scan_tokens("def f(:\n    pass\n")


# ----------------------------------------------------------------------
# 九项核对
# ----------------------------------------------------------------------
def _metric(cap: str, gain: float) -> S.CapabilityMetric:
    return S.CapabilityMetric(
        capability=cap, acc=0.5 + gain, gain=gain, exact=0.1, eos_acc=1.0,
        n_positions=8.0, interference_from=None,
    )


def _baselines(caps, *, heuristic: float = 0.2) -> dict:
    return {c: {"chance": 0.125, "majority": 0.1, "copy": 0.0, "heuristic": heuristic} for c in caps}


def _fake(
    tmp_path,
    *,
    labels=("upper-qv", "upper-all", "shift-qv", "shift-all"),
    caps=("upper", "shift"),
    sizes=(1, 2, 3),
    methods=("stack", "merge", "route"),
    gain: float = 0.5,
    merge_error: float = 1e-16,
    heuristic: float = 0.2,
    agreements=(True, False),
    drop: int = 0,
    dup_cid: bool = False,
    unsorted: bool = False,
):
    """拼一个能被 :func:`audit.checks` 吃下的最小 :class:`PipelineResult`。

    **不跑真流水线** —— 这些检查验的是「公式与实测对得上吗」，
    用手造的 ``runs`` 才能精确制造「差一条」「cid 重复」这类输入。
    """
    adapters = [
        S.AdapterRecord(
            label=lab, capability=lab.split("-")[0], variant=lab.split("-")[1],
            targets=["q_proj"], rank=4, alpha=8.0, scale=2.0, digest=f"d-{lab}",
            n_bytes=64, n_params=8, base_key="tiny", source="test",
            n_base_params_touched=4, train={},
        )
        for lab in labels
    ]
    cap_of = {a.label: a.capability for a in adapters}
    runs = []
    for plan in P.plan_grid(list(labels), methods, sizes):
        members = plan.members
        # 同能力对：次可加量为正（两个都在做同一件事）
        sub = 0.1 if len({cap_of[m] for m in members}) == 1 else 0.0
        metrics = {}
        for c in caps:
            g = gain
            if len(members) == 1:
                g = gain
            elif len({cap_of[m] for m in members}) == 1:
                g = sum(gain for _ in members) - sub
            else:
                g = 0.1
            metrics[c] = _metric(c, g)
        runs.append(
            S.CompositionRun(
                cid=plan.cid, method=plan.method,
                members=list(reversed(members)) if unsorted else list(members),
                metrics=metrics, merge_error=None if plan.method == "route" else merge_error,
                n_params_changed=8, macs=100, wall_ms=1.0, artifact=None,
                coefs=P.coefficients(plan.method, len(members)), merge_loss=None,
            )
        )
    if dup_cid and runs:
        import dataclasses

        runs.append(dataclasses.replace(runs[0]))
    for _ in range(drop):
        runs.pop()
    conflicts = [
        S.ConflictEdge(
            a="upper-qv", b="shift-qv", interference=0.5, i_a_from_b=0.5,
            i_b_from_a=0.5, predicted_cos=(-0.2 if ok else 0.2), predicted_sign=0.5,
            agree=ok, affected={"upper": 0.5},
        )
        for ok in agreements
    ]
    res = S.PipelineResult(
        version=S.Version("v1", "p", "v", "2026-01-01", 0, 0, 0),
        base_key="tiny", base_seed=1, seq_len=13, capabilities=list(caps),
        adapters=adapters, runs=runs, conflicts=conflicts, frontier=[],
        counters={}, findings=[], notes={},
    )
    cfg = C.deep_merge(C.DEFAULTS, {"data": {"capabilities": list(caps)}})
    base_acc = {c: 0.13 for c in caps}
    return res, cfg, base_acc, _baselines(caps, heuristic=heuristic)


def _by_name(results) -> dict:
    return {c.name: c for c in results}


class TestCheckInventory:
    def test_the_structural_four_are_always_present(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        for name in ("no_training_in_pipeline", "producer_contained", "grid_complete", "cid_canonical"):
            assert name in got

    def test_roundtrip_is_conditional(self, tmp_path, repo):
        """产物往返依赖 ``compositions/`` 里的文件，而 ``--no-export``
        之类的只读路径不产出它们。缺文件时它返回 ``False`` 会污染验收。"""
        kw = dict(src_dir=repo / "src" / "adapterfactory", cfg=_fake(tmp_path)[1],
                  base_acc=_fake(tmp_path)[2], baselines=_fake(tmp_path)[3])
        res = _fake(tmp_path)[0]
        names = {c.name for c in A.checks(res, **kw)}
        assert "artifact_roundtrip" not in names
        names2 = {c.name for c in A.checks(res, roundtrip={"ok": True, "detail": "d"}, **kw)}
        assert "artifact_roundtrip" in names2

    def test_structural_checks_have_no_switch(self, tmp_path, repo):
        """前四项是**结构性**的 —— 给它们一个 ``require_*`` 开关只会让
        「关掉」成为一种可选的运行方式，而那正是它们不该有的性质。

        **这里故意把两个开关塞回配置里。** 曾经它们真的存在（默认 ``True``），
        于是「可以关掉」这件事从不显形 —— 直到有人读代码。现在即使有人
        再把它们写回配置，也是**惰性**的：检查根本不读。
        这条断言因此查的是「不再读」，而不是「默认值恰好是 True」。
        """
        res, cfg, base_acc, bl = _fake(tmp_path)
        cfg = C.deep_merge(cfg, {"acceptance": {
            "require_no_training_in_pipeline": False, "require_grid_complete": False,
        }})
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["no_training_in_pipeline"].required is True
        assert got["grid_complete"].required is True
        assert got["producer_contained"].required is True
        assert got["cid_canonical"].required is True


class TestGridComplete:
    def test_passes_on_a_full_grid(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["grid_complete"].ok

    def test_missing_a_row_fails(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path, drop=1)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert not got["grid_complete"].ok

    def test_expected_value_comes_from_the_formula(self, tmp_path, repo):
        """**期望值从公式算，不写死。** 写死的话改了 ``max_size``
        这条检查会一直通过，而它唯一的作用就是发现那种改动。"""
        res, cfg, base_acc, bl = _fake(tmp_path)
        cfg = C.deep_merge(cfg, {"compose": {"max_size": 2}})
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert not got["grid_complete"].ok
        assert "C(4,k)" in got["grid_complete"].detail


class TestCidCanonical:
    def test_passes_on_canonical_cids(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["cid_canonical"].ok

    def test_unsorted_members_fail(self, tmp_path, repo):
        """成员未排序 ⇒ 同一个集合会有两个 cid，于是 ``runs.jsonl`` 里
        同一份组合出现两次，而 ``run_by_cid`` 静默返回其中一条。"""
        res, cfg, base_acc, bl = _fake(tmp_path, unsorted=True)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert not got["cid_canonical"].ok


class TestBaseIsNull:
    """⑤ **两侧不对称**：下界是 ``chance − tol``，上界是 ``heuristic + tol``。

    这条检查的整个形状来自一次真实的红：``sort`` 上基座 0.1823、chance 0.125，
    对称容差 0.05 判它越界。而 0.1823 是**这道题的性质** —— 基座能看输入，
    「什么都不学」时是照抄，sort 的输出是输入的排列，平均约 1.4/8 个位置是
    不动点。所以下面第一条测试是**回归测试**：那种「高于 chance 但低于
    heuristic」的基座必须**通过**。
    """

    def _checks(self, repo, tmp_path, base_acc, tol=None, heuristic=0.2):
        res, cfg, _default_base, bl = _fake(tmp_path, heuristic=heuristic)
        if tol is not None:
            cfg = C.deep_merge(cfg, {"acceptance": {"base_chance_tolerance": tol}})
        return _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                 base_acc=base_acc, baselines=bl))

    def test_a_base_above_chance_but_under_the_trivial_ceiling_passes(self, tmp_path, repo):
        """**回归**：``|base − chance|`` 那个对称写法把这道题判成坏的。
        基座 0.18 在 sort 上就是「照抄」的水平（copy 基线 0.3021），
        它不是缺陷，而是空模型在可抄任务上的正常表现。"""
        got = self._checks(repo, tmp_path, {"upper": 0.1198, "shift": 0.1823},
                           heuristic=0.3021)
        assert got["base_is_null"].ok, got["base_is_null"].detail

    def test_a_base_above_the_trivial_ceiling_fails(self, tmp_path, repo):
        """上界**不是**放行一切：基座要是超过了最强的平凡预测器，
        那它已经不是空模型了 —— 此时 gain 的含义（「Adapter 贡献了多少」）
        就不再成立。"""
        got = self._checks(repo, tmp_path, {"upper": 0.40}, heuristic=0.3021)
        assert not got["base_is_null"].ok
        assert "heuristic" in got["base_is_null"].detail

    def test_a_base_below_chance_fails(self, tmp_path, repo):
        """下界是**危险的那一侧**：基座低于机会水平会**抬高**每一个 gain，
        于是「学得动」可以是基座太差造成的。"""
        got = self._checks(repo, tmp_path, {"upper": 0.05})
        assert not got["base_is_null"].ok
        assert "chance" in got["base_is_null"].detail

    def test_uses_the_configured_tolerance_on_both_sides(self, tmp_path, repo):
        """容差进配置而不是写死：它是这条检查唯一的自由参数（**两侧共用**），
        而「基座该多贴机会水平」取决于任务难度与探针集大小。
        这里走**低于 chance** 的那一侧 —— 它是会被容差救回来的那一侧。"""
        acc = {"upper": 0.05}  # chance 0.125
        assert not self._checks(repo, tmp_path, acc, tol=0.001)["base_is_null"].ok
        assert self._checks(repo, tmp_path, acc, tol=0.15)["base_is_null"].ok

    def test_reports_the_offsets_not_just_a_boolean(self, tmp_path, repo):
        got = self._checks(repo, tmp_path, {"upper": 0.13})
        assert "upper" in got["base_is_null"].detail


class TestSingleGainMargin:
    def test_compares_against_the_stronger_baseline(self, tmp_path, repo):
        """**验收盯的是更强的 ``heuristic``。** 用更弱的 ``chance`` 当门槛，
        一个「学会了输出某个固定字符」的适配器就能通过 —— 而它什么也没学会。"""
        res, cfg, base_acc, bl = _fake(tmp_path, gain=0.5, heuristic=0.2)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["single_gain_margin"].ok
        assert "超过 heuristic" in got["single_gain_margin"].detail

    def test_fails_when_the_margin_is_not_met(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path, gain=0.05, heuristic=0.2)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert not got["single_gain_margin"].ok

    def test_can_be_downgraded_to_advisory(self, tmp_path, repo):
        """tiny 配置里它必然达不到（60 步、32 条样本 —— 门槛问的是
        「这个 Adapter 学得好不好」，而 tiny 只够验「链路通不通」）。
        **关掉而不是把门槛数值调小**：调小会连门槛的含义一起改掉。"""
        res, cfg, base_acc, bl = _fake(tmp_path, gain=0.05)
        cfg = C.deep_merge(cfg, {"acceptance": {
            "require_single_gain_margin": False, "min_single_gain_margin": 0.0,
        }})
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["single_gain_margin"].required is False


class TestTwoPathAgreement:
    def test_computes_the_rate_from_the_edges(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path, agreements=(True, True, False, False))
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert "同向率 0.500" in got["two_path_agreement"].detail

    def test_is_never_a_gate(self, tmp_path, repo):
        """**它是发现，不是门槛。** 参数空间预测被实测推翻（同向率 1/6）
        正是报告里最有价值的结论。设成门槛等于宣布「除非预测器好用，
        否则这次运行不算数」—— 而一个不好用的预测器照样是结论。"""
        res, cfg, base_acc, bl = _fake(tmp_path, agreements=(False, False, False))
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert got["two_path_agreement"].required is False
        assert not got["two_path_agreement"].ok

    def test_none_when_nothing_is_predictable(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path, agreements=())
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        assert "无从执行" in got["two_path_agreement"].detail
        assert not got["two_path_agreement"].ok


class TestDetailSaysWhatItMeasured:
    """``detail`` 里的``值 <比较符> 阈值``**每一句都必须是真的**。

    真 bug（第一次 276 条真实运行撞上）：``two_path_agreement`` 的
    ``detail`` 把比较符写死成 ``≥``，于是失败的那一行印出
    ``同向率 0.286 ≥ 0.50`` —— 判定说 ``False``，正文说 ``≥``。
    两半都「看起来合理」，合起来没有任何一处不一致到能被别的检查发现。

    它躲过了所有测试，因为**通过的输出恰好是对的**（通过时关系确实是
    ``≥``）；只有失败的那一次才写出假话，而失败恰恰是唯一会读它的时刻。
    """

    def test_the_helper_picks_the_glyph_from_the_relation(self):
        """四个方向 × 通过/失败，逐个钉住。"""
        # 「越大越好」：过 ⇒ ≥，不过 ⇒ <
        assert A._vs(0.70, 0.50, lower_is_ok=False, vfmt="{:.3f}", lfmt="{:.2f}") == "0.700 ≥ 0.50"
        assert A._vs(0.286, 0.50, lower_is_ok=False, vfmt="{:.3f}", lfmt="{:.2f}") == "0.286 < 0.50"
        # 「越小越好」：过 ⇒ ≤，不过 ⇒ >
        assert A._vs(5.4e-14, 1e-12, lower_is_ok=True, vfmt="{:.3e}", lfmt="{:.0e}") == "5.400e-14 ≤ 1e-12"
        assert A._vs(3.1e-10, 1e-12, lower_is_ok=True, vfmt="{:.3e}", lfmt="{:.0e}") == "3.100e-10 > 1e-12"
        # 相等算过 —— 两侧都是闭区间，符号也得跟着闭
        assert A._vs(0.50, 0.50, lower_is_ok=False, vfmt="{:.3f}", lfmt="{:.2f}") == "0.500 ≥ 0.50"
        assert A._vs(1e-12, 1e-12, lower_is_ok=True, vfmt="{:.3e}", lfmt="{:.0e}") == "1.000e-12 ≤ 1e-12"

    def test_the_observed_failure_reads_as_a_failure(self, tmp_path, repo):
        """把实测拿到过的那一行钉死：0.286 不能再印成 ``≥ 0.50``。"""
        res, cfg, base_acc, bl = _fake(tmp_path, agreements=(True,) + (False,) * 6)
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))["two_path_agreement"]
        assert not got.ok
        assert "0.143 < 0.50" in got.detail, got.detail

    def test_every_check_detail_relation_holds(self, tmp_path, repo):
        """**扫全部十项、含四个失败分支**，逐句解析 ``a OP b`` 并验算。

        只扫通过的那一版是没用的 —— 那个版本原样放过这个 bug。
        所以这里同时把它们弄红：增益差不够、合并误差超限、
        次可加门槛抬高、同向率不足。
        """
        res, cfg, base_acc, bl = _fake(
            tmp_path, gain=0.05, merge_error=1e-6, agreements=(False,) * 3,
        )
        cfg = C.deep_merge(cfg, {"acceptance": {"min_same_cap_subadditivity": 0.5}})
        got = _by_name(A.checks(res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                                base_acc=base_acc, baselines=bl))
        # 前提：四个分支真的都红了，否则这条测试什么都没验到
        assert not got["merge_identity"].ok
        assert not got["single_gain_margin"].ok
        assert not got["same_cap_subadditive"].ok
        assert not got["two_path_agreement"].ok

        pat = re.compile(
            r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*(≥|≤|>|<)\s*"
            r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)"
        )
        seen = 0
        for name, chk in got.items():
            for a, op, b in pat.findall(chk.detail):
                seen += 1
                x, y = float(a), float(b)
                assert {"≥": x >= y, "≤": x <= y, ">": x > y, "<": x < y}[op], (
                    f"{name} 的 detail 里写着 {a} {op} {b}，但这是假的：{chk.detail}"
                )
        assert seen >= 4, f"只解析到 {seen} 处关系，扫描没覆盖到四个分支"


class TestRunChecks:
    def test_raises_on_a_failed_required_check(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path, drop=3)
        with pytest.raises(AssertionError, match="grid_complete"):
            A.run_checks(result=res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                         base_acc=base_acc, baselines=bl)

    def test_does_not_raise_on_a_failed_advisory_check(self, tmp_path, repo):
        """``required=False`` 的条目不否决 —— 但它们照样每次都算、都进报告。"""
        res, cfg, base_acc, bl = _fake(tmp_path, agreements=(False, False))
        got = A.run_checks(result=res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                           base_acc=base_acc, baselines=bl)
        assert any(c.name == "two_path_agreement" and not c.ok for c in got)

    def test_returns_every_check_it_computed(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path)
        got = A.run_checks(result=res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                           base_acc=base_acc, baselines=bl)
        assert {c.name for c in got} == {
            "no_training_in_pipeline", "producer_contained", "grid_complete",
            "cid_canonical", "base_is_null", "merge_identity", "single_gain_margin",
            "same_cap_subadditive", "two_path_agreement",
        }


class TestAcceptanceReport:
    def test_appends_the_two_readme_criteria(self, tmp_path, repo):
        res, cfg, base_acc, bl = _fake(tmp_path)
        got = A.run_checks(result=res, src_dir=repo / "src" / "adapterfactory", cfg=cfg,
                           base_acc=base_acc, baselines=bl)
        rows = A.acceptance_report(got, n_runs=len(res.runs), n_conflicts=1, conflict_edges_harmful=1)
        names = {r["name"] for r in rows}
        assert {"acceptance_matrix", "acceptance_graph"} <= names
        assert len(rows) == len(got) + 2

    def test_graph_criterion_needs_at_least_one_conflict_edge(self, tmp_path, repo):
        rows = A.acceptance_report([], n_runs=5, n_conflicts=0, conflict_edges_harmful=0)
        by = {r["name"]: r for r in rows}
        assert by["acceptance_matrix"]["ok"] is True
        assert by["acceptance_graph"]["ok"] is False

    def test_check_result_to_dict_has_required(self):
        d = A.CheckResult("x", True, "d", required=False).to_dict()
        assert d == {"name": "x", "ok": True, "detail": "d", "required": False}
