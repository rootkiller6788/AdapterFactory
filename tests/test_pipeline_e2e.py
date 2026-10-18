"""端到端：**产物、可复现性、以及「哪些字段不可复现」。**

这个文件里最贵也最重要的两条是：

1. **两次运行之间，可复现的那个投影逐位相同。** 这是「可复现的组合矩阵」
   （README 验收①）这句话的可执行形式。一条只写 ``len(runs) == 42`` 的断言
   不查这个 —— 它查的是「行数对不对」，而行数从公式来，跟复现没关系。
2. **``out/`` 的必需文件表本身要被查**，两个方向都查：删掉任一**源**文件
   ⇒ 读不出来；删掉任一**派生**文件 ⇒ 照样读得出来。第二个方向是
   「``visualize`` 必须能自救」那条性质。

「可复现的投影」由 :data:`~adapterfactory.schema.PipelineResult.VOLATILE_PATHS`
一处定义，本文件**不另抄一份**：抄一份就等于给自己发一张可以过期的豁免单，
而漏掉一项的症状是随机失败，随机失败的断言通常被 mark flaky 而不是被修对。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import shutil
from math import comb
from pathlib import Path

import pytest

from adapterfactory import audit as A
from adapterfactory import compose as P
from adapterfactory import config as C
from adapterfactory import pipeline as PL
from adapterfactory import produce as PR
from adapterfactory import registry as R
from adapterfactory import report as RP
from adapterfactory import schema as S
from adapterfactory import zoo as Z


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


#: 一条组合里，**与导不导出无关**的那几列。用来跨「导出 / 不导出」两次运行比对。
#: ``artifact`` 不在里面 —— 它就是导出与否的差别本身；``wall_ms`` 不在里面 ——
#: 它是实测耗时（坑 25/33）。
_CORE_RUN_KEYS = ("cid", "method", "members", "size", "metrics", "merge_error",
                  "merge_loss", "n_params_changed", "macs", "coefs")


def _core(runs) -> list[dict]:
    return [{k: r.to_dict()[k] for k in _CORE_RUN_KEYS} for r in runs]


# 两个额外的完整运行，**写在模块级**：类作用域的夹具写成实例方法在 pytest 10
# 会被移除，而且那种写法下「夹具里设的实例属性测试看不到」——正好是这两个夹具
# 不需要（它们只返回 dict）却很容易踩的形状。
@pytest.fixture(scope="module")
def second(tmp_path_factory: pytest.TempPathFactory, tiny_root: Path, tiny_cfg: dict) -> dict:
    """第二次运行，**导出**，写到另一个目录。

    ``out=`` 正是为这条验收存在的：两次都写同一个目录的话，第二次覆盖第一次，
    比对的对象只剩一份，那条断言就成了同义反复。
    """
    out = tmp_path_factory.mktemp("second-run") / "out"
    return {"out": out, "result": PL.run_pipeline(root=tiny_root, cfg=tiny_cfg, out=out, export=True)}


@pytest.fixture(scope="module")
def dry(tmp_path_factory: pytest.TempPathFactory, tiny_root: Path, tiny_cfg: dict) -> dict:
    """第三次运行，**不导出**。只读路径走的是同一条流水线，只是不留组合产物。"""
    out = tmp_path_factory.mktemp("dry-run") / "out"
    return {"out": out, "result": PL.run_pipeline(root=tiny_root, cfg=tiny_cfg, out=out, export=False)}

class TestRunArtifacts:
    # ⚠️ 下面每一条都**显式依赖 ``result``**，虽然它们只用 ``out`` 这个路径。
    # 原因是 ``out`` 夹具只给出**路径**（``<root>/out``），写它的是 ``result``
    # 夹具 —— 只请求 ``out`` 的话，这两条会跑在「目录还不存在」的时刻，
    # 失败信息是「缺 12 个产物」和 ``FileNotFoundError``，看起来像流水线坏了。
    # 顺带：这正是「接受一个路径参数」与「确保那个路径上有东西」的区别。
    def test_every_declared_file_exists(self, out, result):
        """``OUT_FILES`` 是**声明**，这里是**核实**。声明与目录内容分头维护时，
        漂移的方向总是「声明里有、磁盘上没有」，而那条命令的症状是
        「产物不完整」四个字，不说是缺了哪个。"""
        missing = [f for f in PL.OUT_FILES if not (out / f).is_file()]
        assert not missing, missing

    def test_the_directory_holds_nothing_undeclared(self, out, result):
        """反向：目录里不该有没写进 ``OUT_FILES`` 的顶层文件。
        一个「多出来的产物」意味着有人写了文件却没说 —— 下一个人会以为它是
        必需的，而它其实没被任何断言覆盖。"""
        top = {p.name for p in out.iterdir() if p.is_file()}
        assert top == set(PL.OUT_FILES), top ^ set(PL.OUT_FILES)

    def test_runs_jsonl_has_one_line_per_composition(self, out, result):
        rows = _read_jsonl(out / "runs.jsonl")
        assert len(rows) == len(result.runs)
        assert [r["cid"] for r in rows] == [r.cid for r in result.runs]

    def test_result_json_round_trips(self, out, result):
        d = json.loads((out / "result.json").read_text(encoding="utf-8"))
        assert S.PipelineResult.from_dict(d).to_dict() == result.to_dict()

    def test_matrix_csv_is_rectangular(self, out, result):
        lines = (out / "matrix.csv").read_text(encoding="utf-8").splitlines()
        width = lines[0].count(",")
        assert len(lines) == len(result.runs) + 1
        for line in lines[1:]:
            assert line.count(",") == width, line

    def test_matrix_csv_covers_every_run(self, out, result):
        """一行一条组合，且 ``cid`` 是**主键**。少了哪条、重了哪条，都从
        ``result.json`` 那边对出来 —— 两张表各存一份难免漂移，所以要能对上。"""
        import csv
        import io

        rows = list(csv.DictReader(io.StringIO((out / "matrix.csv").read_text(encoding="utf-8"))))
        assert [r["cid"] for r in rows] == [r.cid for r in result.runs]

    def test_dot_files_are_parseable_as_dot(self, out):
        for name in ("graph.dot", "lineage.dot"):
            text = (out / name).read_text(encoding="utf-8")
            assert text.startswith(("digraph", "strict digraph")), name
            assert text.rstrip().endswith("}"), name
            assert text.count("{") == text.count("}") == 1, name

    def test_version_is_content_addressed(self, out):
        """版本号是**内容**的摘要，不是时间戳。是时间戳的话，
        「两份产物是不是同一次配置跑出来的」就只能靠肉眼比报告。"""
        v = json.loads((out / "version.json").read_text(encoding="utf-8"))
        assert v["version"].startswith("v1-")
        assert len(v["version"].split("-", 1)[1]) == 12
        assert v["protocol_digest"] and v["version_digest"]

    def test_registry_json_matches_the_frozen_zoo(self, tiny_root, result):
        """``out/registry.json`` 是**登记表**；``data/adapters/`` 是**标本**。
        两者不一致时，下游拿到的是登记表里的元数据与磁盘上的张量各说各话。"""
        doc = json.loads((tiny_root / "out" / "registry.json").read_text(encoding="utf-8"))
        assert [r["label"] for r in doc["adapters"]] == [a.label for a in result.adapters]
        for rec in doc["adapters"]:
            blob = (tiny_root / "data" / "adapters" / f"{rec['label']}.bin").read_bytes()
            assert hashlib.sha256(blob).hexdigest() == rec["digest"], rec["label"]


class TestReadmeCriteria:
    """README §七 的两条验收标准，就地可查。"""

    def test_criterion_one_the_grid_is_complete(self, result, tiny_cfg):
        n = len(result.adapters)
        assert len(result.runs) == P.grid_size(n, C.sizes_of(tiny_cfg), len(C.methods_of(tiny_cfg)))
        assert len({r.cid for r in result.runs}) == len(result.runs)

    def test_criterion_two_the_graph_points_at_real_pairs(self, result):
        from adapterfactory import lineage as L

        g = L.build_lineage(result.to_dict())
        assert len(result.conflicts) == comb(len(result.adapters), 2)
        for c in result.conflicts:
            node = L.pair_node(c.a, c.b)
            assert g.has(node), node
            assert set(g.parents(node)) == {L.adapter_node(c.a), L.adapter_node(c.b)}

    def test_every_conflict_edge_carries_both_paths(self, result):
        """「带双路证据」是可检查的：每条边既有实测干扰，也有参数空间预测
        （或明确记为「这一对没有共同挂点，预测不出来」）。两种状态各自自洽 ——
        只写一半（有预测没实测，或反之）才是那个该炸的情形。"""
        for c in result.conflicts:
            assert c.interference is not None
            assert c.affected
            assert (c.predicted_cos is None) == (c.agree is None)


class TestMethodCostsAgreeEverywhere:
    """``macs_route`` 这一个量被渲染在**三处**：``counters``、``report.md`` §一
    那条 ``method_macs`` finding、``report.md`` §二那行「三种方法的计算代价」。

    真 bug：前两处**各推了一遍**，一处取首个 size-1 组合、另一处被 dict 推导式
    留下最后一个 —— 于是同一个量印出两个值（347360 与 218816，差 1.59 倍），
    而**两个都叫「route 的开销」**，各自都「看起来合理」。
    根因不是取错了哪一个，是 route 的开销**本来就不是一个定值**
    （qv 挂 2 个模块、all 挂 7 个），所以「挑一个」这个动作本身没有正确答案；
    挑出来的数还会继续伪装成「方法的固定开销」。
    这里钉住：三处都必须是 :func:`pipeline.method_macs` 那个取值域里的成员。
    """

    def test_the_counters_are_members_of_the_measured_set(self, result):
        for meth in ("stack", "merge", "route"):
            vals = PL.method_macs(result.runs, meth)
            assert vals, meth
            assert result.counters[f"macs_{meth}"] in vals, (
                meth, result.counters[f"macs_{meth}"], vals,
            )

    def test_route_is_exactly_where_a_single_number_would_have_lied(self, result):
        """tiny 的 zoo 两种覆盖都有 ⇒ 在这里 route 就该**不是一个定值**。
        哪天它变成定值，上面那条就退化成「随便挑一个都对」—— 这条会先响。"""
        assert len(PL.method_macs(result.runs, "route")) > 1
        assert len(PL.method_macs(result.runs, "stack")) == 1

    def test_every_number_the_finding_quotes_was_measured(self, result):
        f = next(x for x in result.findings if x.key == "method_macs")
        known = {result.counters["base_macs"]}
        for meth in ("stack", "merge", "route"):
            known |= set(PL.method_macs(result.runs, meth))
        quoted = {int(x) for x in re.findall(r"\d{5,}", f.measured)}
        assert quoted, f.measured
        assert quoted <= known, f"这条 finding 引用了没测过的数：{sorted(quoted - known)}"

    def test_the_report_says_route_is_a_range_and_a_bound(self, result):
        text = RP.render_markdown(result)
        lo, hi = PL.method_macs(result.runs, "route")
        assert f"{lo}–{hi}" in text, "§一 那条 finding 要印出取值域，不是一个数"
        # §二 那行印的是上界，必须带 `≤`，且就是 counters 里那个数
        assert f"route ≤ {result.counters['macs_route']}" in text


class TestAudit:
    def test_the_run_carried_a_full_check_list(self, result):
        names = [c["name"] for c in result.counters["checks"]]
        assert names[:4] == [
            "no_training_in_pipeline", "producer_contained", "grid_complete", "cid_canonical",
        ]
        assert "artifact_roundtrip" in names  # 这一次是 export=True

    def test_every_required_check_passed(self, result):
        bad = [c for c in result.counters["checks"] if c["required"] and not c["ok"]]
        assert not bad, bad

    def test_the_two_acceptance_rows_are_there(self, result):
        names = {a["name"] for a in result.counters["acceptance"]}
        assert {"acceptance_matrix", "acceptance_graph"} <= names

    def test_the_architecture_wall_reports_what_it_scanned(self, result):
        """一条只说「通过」的检查无法区分「扫了 14 个模块全干净」与
        「一个模块都没扫到」。所以详情里必须有**扫了几个**。"""
        detail = next(c for c in result.counters["checks"]
                      if c["name"] == "no_training_in_pipeline")["detail"]
        assert str(len(A.PIPELINE_MODULES)) in detail

    def test_merge_error_is_at_implementation_noise_scale(self, result):
        """合并与逐个施加是同一件事的两条算法路径，差别只有累加顺序。
        定在 1e-12 意味着它查的是**实现错误**，不是浮点噪声。"""
        merged = [r for r in result.runs if r.merge_error is not None]
        assert merged
        assert max(r.merge_error for r in merged) < 1e-12

    def test_route_has_no_merge_error_because_it_never_merges(self, result):
        every_route = [r for r in result.runs if r.method == "route"]
        assert every_route
        assert all(r.merge_error is None for r in every_route)


class TestDeterminism:
    """**README 验收①「可复现的组合矩阵」的可执行形式。**

    第二次运行写到**另一个目录**（见模块级的 ``second`` 夹具）。
    """

    def test_the_second_run_wrote_its_own_products(self, second):
        assert all((second["out"] / f).is_file() for f in PL.OUT_FILES)

    def test_the_stable_views_are_identical(self, result, second):
        a, b = result.stable_view(), second["result"].stable_view()
        d = S.dict_diff_paths(a, b)
        assert not d, f"稳定视图里还有 {len(d)} 处差异：{d[:8]}"

    def test_the_raw_differences_are_exactly_the_declared_ones(self, result, second):
        """**反向那一半。** 上一条只说「剔掉声明之后相同」；把声明写得宽一点
        （比如干脆整条 ``runs[]``）它照样绿，而那时它什么都不查了。
        所以这里从**未剔**的两棵树上取差异，要求每一处都落在声明里。"""
        d = S.dict_diff_paths(result.to_dict(), second["result"].to_dict())
        assert d, "两次运行一模一样 —— 那么实测轴根本没被记录，或者第二次其实复用了第一次的产物"
        stray = [p for p in d if not S.PipelineResult.is_volatile(p)]
        assert not stray, stray
        assert any(p.endswith("wall_ms") for p in d), d

    def test_exported_artifacts_have_identical_content(self, result, second):
        """``artifact.path`` 在豁免单上（两次写到不同目录），但
        ``digest`` / ``bytes`` / ``rank_total`` **不在** —— 「两次运行导出了
        同样的东西」这句话必须仍然有人查。"""
        def digests(rs):
            return {r.cid: (r.artifact.digest, r.artifact.n_bytes, r.artifact.rank_total)
                    for r in rs if r.artifact is not None}

        assert digests(result.runs) == digests(second["result"].runs)

    def test_the_version_is_the_same(self, result, second):
        """版本摘要覆盖配置与网格，**不覆盖计时**。两次运行的版本号必须一样 ——
        一样才说明「这份产物是这套配置跑出来的」这句话可判定。"""
        assert result.version.version == second["result"].version.version
        assert result.version.version_digest == second["result"].version.version_digest

    def test_the_frontier_is_the_same(self, result, second):
        """前沿是**结论**，不是**记录**。同一次配置必须给出同一个前沿 ——
        只写「前沿非空」的话，一个每次都随机挑几条的实现在这条断言下也是绿的。"""
        assert result.frontier == second["result"].frontier
        assert result.frontier, "前沿为空时上面那条是空的真"


class TestNoExportRun:
    """只读路径（``export=False``）与导出路径的**唯一**差别应该是产物本身。

    第三次完整运行（见模块级的 ``dry`` 夹具）—— 值得，因为它一次回答四件事：
    不导出时不写 ``compositions/``、``artifact_roundtrip`` **缺席而不是失败**、
    必需的核对仍然全过、以及「导不导出不影响任何测量值」。
    """

    def test_no_compositions_are_written(self, dry):
        cdir = PL.compositions_dir(dry["out"])
        assert not cdir.exists() or not any(cdir.iterdir())

    def test_the_roundtrip_check_is_absent_not_failed(self, dry):
        """**缺席而不是失败。** 只读路径不产出 ``compositions/``，
        让它返回 ``False`` 会把一次正常的只读运行报成核对未通过 ——
        而那条报告的读者是「来查为什么验收红了的」人，指错方向比不说更糟。"""
        names = [c["name"] for c in dry["result"].counters["checks"]]
        assert "artifact_roundtrip" not in names
        assert dry["result"].counters["n_exported"] == 0

    def test_every_required_check_still_passes(self, dry):
        bad = [c for c in dry["result"].counters["checks"] if c["required"] and not c["ok"]]
        assert not bad, bad

    def test_measurements_do_not_depend_on_exporting(self, result, dry):
        """导出是一次**写**。写不该改变任何被测量的东西 —— 不改变的话，
        「先跑一遍看看」与「跑一遍留下产物」就是同一次实验的两种收尾。"""
        assert _core(result.runs) == _core(dry["result"].runs)

    def test_the_verdicts_do_not_depend_on_exporting(self, result, dry):
        assert [f.to_dict() for f in result.findings] == [f.to_dict() for f in dry["result"].findings]
        assert result.frontier == dry["result"].frontier


class TestRequiredFileTable:
    """``SOURCE_FILES`` / ``DERIVED_FILES`` 这张表**两个方向都查**。"""

    @pytest.fixture
    def out_copy(self, tmp_path: Path, tiny_root: Path) -> Path:
        dst = tmp_path / "out"
        shutil.copytree(Path(tiny_root) / "out", dst)
        return dst

    @pytest.mark.parametrize("name", PL.SOURCE_FILES)
    def test_a_missing_source_file_means_no_products(self, out_copy, name):
        (out_copy / name).unlink()
        assert PL.load_result(out_copy) is None

    @pytest.mark.parametrize("name", PL.DERIVED_FILES)
    def test_a_missing_derived_file_still_reads(self, out_copy, name):
        """**派生文件不参与「产物是否完整」的判定。** 它们由源文件唯一决定，
        拿它们当读产物的前提，等于让「重画一张 HTML」必须先重跑整条流水线。"""
        (out_copy / name).unlink()
        assert PL.load_result(out_copy) is not None

    def test_the_two_tables_are_disjoint_and_complete(self):
        assert set(PL.SOURCE_FILES) & set(PL.DERIVED_FILES) == set()
        assert PL.OUT_FILES == PL.SOURCE_FILES + PL.DERIVED_FILES
        assert len(PL.OUT_FILES) == len(set(PL.OUT_FILES))


def _produce(root, cfg, ctx, *, specs, verbose=False):
    """按配置里的配方跑一次 ``produce_zoo``。测试里只此一份，避免每处各抄一遍。"""
    return PR.produce_zoo(
        root=root, shape=ctx["shape"], base_seed=ctx["base_seed"], specs=specs,
        settings=PR.TrainSettings(**C.zoo_recipe_of(cfg)),
        vocab=ctx["vocab"], idx=ctx["idx"], seq_len=ctx["seq_len"],
        eval_sets=ctx["eval_sets"], train_seed=int(cfg["data"]["seed"]),
        force=False, verbose=verbose,
    )


class TestProduceIdempotency:
    def test_a_second_produce_reuses_everything(self, tiny_root, tiny_cfg, produced):
        """``force=False`` 的第二次 produce 必须**沿用**已有产物，且 digest 不变。
        这条成立，「一次标定、多次分析」才不用每次都重训 ——
        而重训会在数值上给出略微不同的 zoo，让「两次运行的组合矩阵逐位相同」
        这条验收变得**不可能**成立（问题不在流水线，在标本换了）。"""
        zoo = tiny_root / "data" / "adapters"
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in zoo.glob("*.bin")}
        assert before
        ctx = PL.prepare(root=tiny_root, cfg=tiny_cfg)
        _produce(tiny_root, tiny_cfg, ctx, specs=C.zoo_specs_of(tiny_cfg))
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in zoo.glob("*.bin")}
        assert after == before

    def test_a_changed_recipe_forces_a_retrain(self, tmp_path, tiny_cfg, tiny_root, ctx):
        """**反向那一半，也是那个真 bug。** 上一条只说「配方没变就沿用」；
        把沿用条件写成「文件在不在」它照样绿 —— 而那时它什么都不查了。

        真实的经过：``zoo.rank`` 从 4 改成 8，``produce`` 打印 ``rank=8``，
        8 个产物全部 ``[沿用]``，而磁盘上是 rank=4 的权重。
        参数量、``MACs``、``rank_total`` 全跟着错，**每个数字单看都合理**。
        """
        import shutil

        shutil.copytree(tiny_root / "data" / "adapters", tmp_path / "data" / "adapters")
        zoo = tmp_path / "data" / "adapters"
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in zoo.glob("*.bin")}
        # 只动一个 Adapter 的配方，且只动一个字段：别的标签必须原样沿用，
        # 否则这条测试分不清「配方比对生效了」与「每次都重训」。
        specs = C.zoo_specs_of(tiny_cfg)
        target = specs[0]
        bumped = dataclasses.replace(target, rank=target.rank + 1)
        other = [s for s in specs if s.label != target.label]
        rep = _produce(tmp_path, tiny_cfg, ctx, specs=[bumped, *other], verbose=True)
        assert rep["reports"][target.label]["reused"] is False
        assert all(rep["reports"][s.label]["reused"] for s in other)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in zoo.glob("*.bin")}
        assert after[target.label + ".bin"] != before[target.label + ".bin"]
        assert {k: v for k, v in after.items() if k != target.label + ".bin"} == \
            {k: v for k, v in before.items() if k != target.label + ".bin"}

    def test_the_retrain_says_which_field_moved(self, tmp_path, tiny_cfg, tiny_root, ctx, capsys):
        """**这一条防的是「静默」本身。** 上面的 bug 之所以能活下来，不是因为
        沿用规则错，而是因为它错得**没有声音**：打印的是 ``[沿用]``，
        一个看起来完全正常的词。所以重训必须**说出来是哪个字段变了**。"""
        import shutil

        shutil.copytree(tiny_root / "data" / "adapters", tmp_path / "data" / "adapters")
        specs = C.zoo_specs_of(tiny_cfg)
        target = specs[0]
        _produce(tmp_path, tiny_cfg, ctx,
                 specs=[dataclasses.replace(target, rank=target.rank + 1)], verbose=True)
        out = capsys.readouterr().out
        assert "重训" in out
        assert "spec.rank" in out


class TestExportedCompositionsAreRealLoras:
    def test_each_exported_artifact_reads_back_bit_identical(self, out, result):
        """导出的是**一份合法的 LoRA**，不是本项目私有的中间格式 ——
        所以它能被写它的那套 reader 逐位读回来，也能被别的项目读（格式一致）。"""
        exported = [r for r in result.runs if r.artifact is not None]
        assert exported
        for r in exported[:6]:
            tensors, rec = Z.read_artifact(PL.compositions_dir(out) / r.cid, verify=True)
            assert rec.digest == r.artifact.digest
            assert rec.n_bytes == r.artifact.n_bytes
            assert tensors

    def test_only_weight_space_methods_are_exported(self, result):
        """``route`` 不产生一个权重增量，所以它**无法**被导出成一份 LoRA。
        导出一个全零的 delta 会让它看起来「也能部署」—— 而部署时的行为与它
        评测时的行为完全不同（评测时它走逐层混合）。"""
        for r in result.runs:
            if r.method == "route":
                assert r.artifact is None, r.cid
            else:
                assert r.artifact is not None, r.cid

    def test_exported_targets_are_the_union_of_the_members(self, out, result):
        """旁证里的挂点必须取**并集**。写全部 7 个会让一份只动了 q/v 的组合
        在旁证里声称自己动了 7 个模块 —— 而旁证是下游唯一的身份来源。"""
        recs = {a.label: a for a in result.adapters}
        checked = 0
        for r in result.runs:
            if r.artifact is None or len(r.members) != 2:
                continue
            _t, rec = Z.read_artifact(PL.compositions_dir(out) / r.cid, verify=True)
            want = sorted({t for m in r.members for t in recs[m].targets})
            assert sorted(rec.targets) == want, f"{r.cid}: {rec.targets} != {want}"
            checked += 1
        assert checked, "一条两成员的组合都没查到，这条断言其实没跑"


class TestBaseIdentity:
    """**``base_key`` 是本项目相对 FineTuneFactory 补上的那个字段。**

    FF 的旁证里只有张量形状，没有基座身份 —— 于是 48/48 个 LoRA 都能读、
    却全都装不上去，而**在读的那一刻看不出来**。这里查的是那道闸真的会落。
    """

    def test_every_adapter_declares_the_base_it_hangs_on(self, registry, result):
        for rec in registry.records:
            assert rec.base_key == result.base_key == registry.base_key
            assert rec.base_key

    def test_a_foreign_base_refuses_to_load(self, tmp_path, tiny_root, tiny_cfg, ctx):
        """把一个 Adapter 的基座标识改掉，``load_registry`` 必须**抛**。

        判成「兼容」是最贵的那种错：装上去之后静默地算出垃圾，而报告里
        每一项看起来都正常。所以这里查的是异常，不是某个布尔。"""
        src = tiny_root / "data" / "adapters"
        dst = tmp_path / "data" / "adapters"
        shutil.copytree(src, dst)
        sidecars = sorted(dst.glob("*.json"))
        assert sidecars
        doc = json.loads(sidecars[0].read_text(encoding="utf-8"))
        doc["base"]["key"] = "some-other-base"
        sidecars[0].write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        with pytest.raises(ValueError) as err:
            R.load_registry(tmp_path, tiny_cfg, ctx["shape"])
        msg = str(err.value)
        assert sidecars[0].stem in msg
        assert "some-other-base" in msg

    def test_a_same_base_artifact_has_no_shape_problems(self, tiny_root, ctx):
        """这道闸要能**放行**。只会拦不会放的检查等于把全部产物判成坏，
        而那时它的症状是「什么都不兼容」—— 与真的全都不兼容长得一样。
        （形状核对那条检查曾经就是这样：它拿挂点**计数**乘 2 当张量数，
        于是对每一份形状完好的产物都成立。）"""
        labels = R.expected_labels(ctx["capabilities"])
        assert labels
        tensors, rec = Z.read_artifact(tiny_root / "data" / "adapters" / labels[0])
        problems, seen = R._shape_compat(tensors, ctx["shape"])
        assert problems == []
        assert seen == len(tensors)
        assert rec.base_key == ctx["shape"].key
        assert labels[0] == "upper-qv"  # 顺序即设计顺序，不是字母序
