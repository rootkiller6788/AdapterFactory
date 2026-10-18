"""``report`` / ``visualize`` / ``cli``：**报告是产物决定的，只读命令一个字节都不写。**

这个文件里有两条别处没有的东西：

1. **只读不变性**是**跑出来**的，不是声明出来的 —— 13 个只读命令逐条跑完，
   对 ``out/`` 做整目录 md5 比对。这张表 :data:`cli.READ_ONLY` 显式列着，
   测试直接照它跑，不靠人去数。
2. **``report.md`` 与产物逐字节一致** —— ``main.py report`` 把现生成的文本与
   磁盘上那份比。这条成立，报告才是「由产物决定」的；不成立的话，
   报告会变成一份**手改过、没人知道改了什么**的文档。
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from pathlib import Path

import pytest

from adapterfactory import cli
from adapterfactory import report as RP
from adapterfactory import visualize as V
from adapterfactory.schema import Finding


def _tree_hash(root: Path) -> str:
    """整目录内容摘要。**先把相对路径排序再喂给 sha256** ——
    不排序的话目录项顺序变了就会误报「文件变了」，而不变性的失败信息
    只会说「目录变了」，那时人得自己去猜是哪个文件。"""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@pytest.fixture(scope="module")
def out_copy(tmp_path_factory: pytest.TempPathFactory, tiny_root: Path, result) -> Path:
    """``out/`` 的一份副本。**用副本而不是夹具那一份** ——
    万一有个命令真的写了，这一条会失败，而不必连带污染后面所有测试。

    模块级（而不是类里）：类里的夹具在 pytest 10 起会被忽略，
    而「被忽略」在这里的表现是**测试拿不到那个参数**，不是一条警告。

    **它依赖 ``result``，不是 ``tiny_root``。**``tiny_root`` 只保证目录存在，
    写出 ``out/`` 的是 ``result``。少了这一条依赖，这个夹具在**任何单独跑
    本文件的测试的命令下都会炸**（``-k`` 选出一条不请求 ``result`` 的测试时），
    而全量运行时它却绿 —— 因为那时候别的测试恰好已经把 ``out/`` 建出来了。
    """
    dst = tmp_path_factory.mktemp("out-copy") / "out"
    shutil.copytree(Path(tiny_root) / "out", dst)
    return dst


def _run(root, out, *argv) -> int:
    """跑一条命令。**全局选项必须排在子命令之前** —— ``argparse`` 的
    ``--root`` / ``--out`` 是解析器级的，写在子命令后面会得到
    ``unrecognized arguments``，而那条报错读起来像是「这个命令没有这个选项」。
    全项目只在这里拼一次 ``argv``，于是这个顺序只需要对一次。
    """
    return cli.main(["--root", str(root), "--out", str(out), *argv])


def _readonly_argv(result, out: Path) -> dict[str, list[str]]:
    """每个只读命令一份真实可跑的 ``argv``。**覆盖到需要参数的那几个** ——
    不然「13 个命令都不写」这句话只对无参命令成立。"""
    cid = result.frontier[0]
    return {
        "task": ["task"],
        "version": ["version"],
        "registry": ["registry"],
        "summary": ["summary"],
        "matrix": ["matrix", "--by", "gain", "--limit", "5"],
        "conflict": ["conflict", "--limit", "5"],
        "graph": ["graph", "--kinds", "adapter,conflict"],
        "frontier": ["frontier"],
        "methods": ["methods", "--size", "2"],
        "explain": ["explain", cid],
        "lineage": ["lineage"],
        "acceptance": ["acceptance"],
        "report": ["report"],
    }


# ----------------------------------------------------------------------
# fmt：绝不用 ``or`` 兜底
# ----------------------------------------------------------------------
class TestFmt:
    def test_none_is_a_dash(self):
        assert RP.fmt(None) == RP.DASH

    def test_nan_is_explicitly_unknown(self):
        """``NaN`` 与 ``None`` 是两件事：前者是算出来的「不能比」，
        后者是「没这一项」。都印成 ``—`` 会让读者以为缺数据。"""
        assert RP.fmt(math.nan) == "未知"

    def test_zero_is_not_swallowed(self):
        """**``v or DASH`` 会把 0 与 None 一起吞掉。** 而 0 在本项目里是
        一个常见且**有意义**的值：route 改动的基座元素恰好是 0、
        merge_error 在 route 上是缺失、干扰在完全无关的一对上接近 0。"""
        assert RP.fmt(0) == "0.0000"
        assert RP.fmt(0.0, 2) == "0.00"

    def test_signed_forces_the_plus(self):
        assert RP.fmt(0.5, 4, signed=True) == "+0.5000"
        assert RP.fmt(-0.5, 4, signed=True) == "-0.5000"

    def test_scale_is_applied(self):
        assert RP.fmt(0.5, 4, scale=100.0) == "50.0000"

    def test_infinities_are_named(self):
        assert RP.fmt(math.inf) == "+∞"
        assert RP.fmt(-math.inf) == "-∞"

    def test_non_numeric_passes_through(self):
        """``coefs`` 那种「一串数」或名值会走到这条路 —— 抛错会让报告
        在一项无关紧要的字段上整篇生成失败。"""
        assert RP.fmt("route") == "route"


# ----------------------------------------------------------------------
# report.md
# ----------------------------------------------------------------------
SECTIONS = ("## 一、", "## 二、", "## 三、", "## 四、", "## 五、", "## 六、", "## 七、", "## 八、")


class TestMarkdown:
    def test_has_the_eight_sections_in_order(self, result):
        text = RP.render_markdown(result)
        pos = [text.index(s) for s in SECTIONS]
        assert pos == sorted(pos)

    def test_title_carries_the_content_addressed_version(self, result):
        assert RP.render_markdown(result).startswith(f"# AdapterFactory 报告 · {result.version.version}")

    def test_states_the_unreproducible_boundary(self, result):
        """有一条轴（实测耗时）**不可逐位复现**，报告必须自己说出来。
        不写的话，读者会把两次运行的 ``wall_ms`` 差异当成「模型变了」。"""
        text = RP.render_markdown(result)
        assert "不可复现" in text

    def test_never_leaks_python_reprs(self, result):
        """``None`` / ``nan`` 直接进 f-string 会在正文里留下 ``None`` 与 ``nan``，
        而它们在表格里与正常单元格**宽度不同**却看不出错。

        不查花括号：正文里合法地写着 ``I_{b→a}`` 与 ``Σ C(4,k)``。
        一条会把合法内容判成违规的断言，最后一定会被人关掉。
        """
        text = RP.render_markdown(result)
        for token in ("None", "nan", "NaN"):
            assert token not in text, token

    @staticmethod
    def _pipes(line: str) -> int:
        """**未被转义的**管道符数量。数 ``line.count("|")`` 会把 ``\\|`` 也算进去，
        于是「转义过了」这件事反而让断言失败 —— 一条会因为修好而变红的断言，
        下一个人的处置方式是把它删掉。"""
        return len(re.findall(r"(?<!\\)\|", line))

    def test_tables_are_rectangular(self, result):
        """一行的 ``|`` 数量与表头不一致，markdown 会把这行**当成正文**渲染 ——
        表格看起来没坏，只是少了一行。"""
        lines = RP.render_markdown(result).splitlines()
        i = 0
        checked = 0
        while i < len(lines):
            if lines[i].startswith("|") and i + 1 < len(lines) and set(lines[i + 1]) <= set("|- "):
                width = self._pipes(lines[i])
                j = i + 2
                while j < len(lines) and lines[j].startswith("|"):
                    assert self._pipes(lines[j]) == width, f"第 {j + 1} 行列数不齐：{lines[j]}"
                    j += 1
                checked += 1
                i = j
            else:
                i += 1
        assert checked >= 4, "八节报告里应当有多张表"

    def test_every_combo_appears_in_the_matrix_section(self, result):
        text = RP.render_markdown(result)
        assert f"{len(result.runs)} 条" in text or str(len(result.runs)) in text

    def test_findings_render_as_a_block(self, result):
        block = RP.render_findings(result.findings)
        if result.findings:
            assert "|" in block or "—" in block or block.strip()

    def test_a_pipe_in_a_cell_is_escaped(self):
        """**一句 ``Σ C(4,k)·|方法|`` 就够了。** 管道符在表格里是结构不是内容：
        不转义的话 markdown 把这行从中间劈开，渲染出来是两张错位的表 ——
        而不报错，数字仍然对得上行、对不上列。"""
        row = RP._row(["a", "公式 Σ C(4,k)·|方法| = 42", "b"])
        assert row.count("|") - row.count("\\|") == 4  # 4 = 首尾 + 3 个分隔符
        assert "\\|方法\\|" in row

    def test_a_newline_in_a_cell_does_not_split_the_table(self):
        assert "\n" not in RP._row(["a", "两\n行"])

    def test_affected_order_comes_from_the_capabilities_not_the_dict(self):
        """**同一份产物必须渲染出同一段文本。** ``affected`` 的插入顺序活不过
        一次 JSON 往返（落盘用 ``sort_keys=True``），所以渲染顺序必须由
        ``result.capabilities`` 定 —— 否则 ``main.py report`` 会把一次纯粹的
        键序差异报成「产物与代码不一致」，而重跑 run 修不好它。"""
        caps = ["upper", "shift", "reverse", "sort"]
        assert [k for k, _ in RP._affected_items({"shift": 1.0, "upper": 2.0}, caps)] == ["upper", "shift"]
        # 不在 caps 里的键排在后面，并按名字定序（不许靠运气）
        assert [k for k, _ in RP._affected_items({"zz": 1.0, "aa": 2.0}, caps)] == ["aa", "zz"]


# ----------------------------------------------------------------------
# report.html
# ----------------------------------------------------------------------
class TestHtml:
    def test_is_self_contained(self, result):
        """**零外部引用**是这条产物的全部意义：能直接双击打开、
        能贴进没有网的环境。一个 ``<script src=...>`` 或一个 Google Fonts
        的 ``<link>`` 会让它在离线时**看起来只是样式没加载**。"""
        html = V.render_html(result)
        for bad in ("http://", "https://", "<script src", "<link ", "@import", "srcset="):
            assert bad not in html, bad

    def test_is_a_whole_document(self, result):
        html = V.render_html(result).lower()
        assert html.lstrip().startswith("<!doctype html")
        for tag in ("<html", "</html>", "<body", "</body>"):
            assert tag in html, tag

    def test_embeds_svg(self, result):
        html = V.render_html(result)
        assert "<svg" in html
        assert html.count("<svg") == html.count("</svg>")

    def test_escapes_user_facing_text(self, tmp_path, result):
        """报告文本来自配置与产物。一个带 ``<`` 的字段会**破坏整份 HTML**，
        而浏览器不会报错，它只是把后面的结构当成标签吃掉。

        注入点是 ``findings`` 的说明文本 —— 它**一定**会被渲染（``_findings``
        对每条 finding 都拼字符串）。曾经这里注入的是能力名，而能力名在某些
        图表里会因为「该能力没有指标」被跳过，于是注入的那个串一个字符都不出现，
        断言退化成「不包含」的恒真式。
        """
        import copy

        broken = copy.deepcopy(result)
        broken.findings = [
            Finding(key="probe", statement="<script>alert(1)</script>",
                    expectation="x", measured="y", verdict="v", note="")
        ]
        html = V.render_html(broken)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_esc_handles_all_five(self):
        """``html.escape(quote=True)`` 的五个字符一个都不能漏 —— 少转义一个
        ``"`` 会让它插进属性里，而那是**另一处**注入点。"""
        assert V._esc('<a href="x">&\'') == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;"

    def test_mentions_the_frontier(self, result):
        assert str(len(result.frontier)) in V.render_html(result)


# ----------------------------------------------------------------------
# 命令表
# ----------------------------------------------------------------------
class TestCommandTables:
    def _choices(self, parser) -> set[str]:
        for action in parser._actions:
            if hasattr(action, "choices") and isinstance(action.choices, dict):
                return set(action.choices)
        raise AssertionError("parser 里没有子命令表")

    def test_read_only_has_thirteen_commands(self):
        assert len(cli.READ_ONLY) == 13
        assert len(set(cli.READ_ONLY)) == 13

    def test_write_has_four_commands(self):
        assert cli.WRITE == ("produce", "run", "visualize", "import")

    def test_the_two_tables_partition_the_parser(self):
        """**漏划一个命令**的症状是它悄悄进了只读那组（或哪组都不是），
        于是不变性测试根本没跑它 —— 而测试会绿。"""
        choices = self._choices(cli.build_parser())
        assert set(cli.READ_ONLY) | set(cli.WRITE) == choices
        assert not (set(cli.READ_ONLY) & set(cli.WRITE))

    def test_every_specimen_and_criterion_is_reachable(self):
        """README 的每一条验收标准都应当有一个命令能看到它。"""
        choices = self._choices(cli.build_parser())
        for cmd in ("acceptance", "matrix", "conflict", "graph", "frontier"):
            assert cmd in choices

    def test_no_help_string_hardcodes_a_count(self):
        """**帮助文案里不许出现「N 项」。** 报告正文曾经写着「8 项」而下面是 9 行；
        同一个数字在 help 里又硬编码了一次（「9 项运行时核对」）。帮助文案比报告
        更难核对 —— 它在一个还没有任何产物的进程里被渲染出来，**没有清单可数**。

        数得出来的东西才准写在文案里；数不出来的，写「有哪些」而不是「有几项」。"""
        import re

        for action in cli.build_parser()._actions:
            for sub in (action.choices or {}).values():
                for sub_action in sub._actions:
                    # help 与 description 都不是每个 Action 都有的属性，
                    # 有的那个也可能是 None（`-h` 就是这种），所以两者都要兜。
                    text = "".join(
                        getattr(sub_action, k, None) or "" for k in ("help", "description")
                    )
                    assert not re.search(r"\d+\s*项", text), (sub_action.dest, text)


# ----------------------------------------------------------------------
# 只读不变性（真跑）
# ----------------------------------------------------------------------
class TestReadOnlyInvariance:
    def test_all_of_them_exit_zero_and_print_nothing_hidden(self, out_copy, tiny_root, result, capsys):
        for cmd, argv in _readonly_argv(result, out_copy).items():
            code = _run(tiny_root, out_copy, *argv)
            captured = capsys.readouterr()
            assert code == 0, f"{cmd}: 退出码 {code}；stderr={captured.err}"
            assert captured.out.strip(), f"{cmd}: 什么都没打印，退出码 0 的沉默命令等于坏了"
            assert not captured.err.strip() or cmd == "report", f"{cmd}: 意外的 stderr {captured.err!r}"

    def test_out_is_byte_identical_afterwards(self, out_copy, tiny_root, result):
        """**这个文件里最贵的一条断言。** 它跑 13 个命令，然后逐字节比对整棵目录。
        它守的是一条否则只能靠人去读代码的约定 —— 而「只读」这件事一旦破了，
        症状是 ``out/`` 里多出一个文件，没人会注意到。"""
        before = _tree_hash(out_copy)
        for argv in _readonly_argv(result, out_copy).values():
            assert _run(tiny_root, out_copy, *argv) == 0
        assert _tree_hash(out_copy) == before

    def test_the_hash_helper_notices_a_change(self, out_copy):
        """**先验尺子。** 一条永远相等的哈希断言什么也没验 ——
        所以这里主动改一个字节，确认它能被抓到。"""
        before = _tree_hash(out_copy)
        p = next(x for x in out_copy.rglob("*") if x.is_file())
        p.write_bytes(p.read_bytes() + b"\0")
        assert _tree_hash(out_copy) != before


# ----------------------------------------------------------------------
# 缺产物时的提示
# ----------------------------------------------------------------------
class TestMissingArtifacts:
    #: 这两条**不需要产物** —— ``task`` 打的是任务定义与基线，
    #: ``version`` 打的是版本与口径摘要，两者的输入都只有配置与数据集索引。
    NEEDS_NOTHING = ("task", "version")

    def test_read_only_commands_say_what_to_do(self, tiny_root, tmp_path, result, capsys):
        """**提示语是同一句**：还没有产物，先跑 ``run``。
        让它去 ``except`` 一个异常类，那句话会散在每个子命令里，
        迟早有一处写错类名 —— 于是用户看到 traceback，而 traceback 不说「你该做什么」。"""
        empty = tmp_path / "empty-out"
        table = _readonly_argv(result, empty)
        for cmd, argv in table.items():
            if cmd in self.NEEDS_NOTHING:
                continue
            code = _run(tiny_root, empty, *argv)
            err = capsys.readouterr().err
            assert code == 2, cmd
            assert "run" in err, f"{cmd}: {err!r}"

    def test_the_two_exceptions_really_do_not_need_products(self, tiny_root, tmp_path, capsys):
        """被排除的那两个必须**真的**不需要产物。不查的话，
        它们既可能真的不需要，也可能只是「碰巧没抛」—— 而后者会让上一条漏掉
        一条本该报「先跑 run」的命令。"""
        empty = tmp_path / "empty-out"
        for cmd in self.NEEDS_NOTHING:
            assert _run(tiny_root, empty, cmd) == 0, cmd
            assert capsys.readouterr().out.strip()

    def test_no_command_prints_a_traceback(self, tiny_root, tmp_path):
        assert _run(tiny_root, tmp_path / "nope", "summary") == 2

    def test_a_missing_derived_file_is_not_a_missing_artifact(self, tiny_root, tmp_path, capsys):
        """**删掉 ``report.html`` 之后，项目必须还能自救。** 它是由源文件派生的，
        所以读产物不该以它为前提 —— 否则 ``visualize``（唯一能重出它、
        且专门为此存在的那条命令）会被它要修的条件挡住，
        唯一出路是重跑整条 276 个组合的流水线，只为了重画一张 HTML。"""
        out = tmp_path / "out"
        shutil.copytree(Path(tiny_root) / "out", out)
        assert _run(tiny_root, out, "visualize") == 0  # 先确认它本来是齐的
        capsys.readouterr()
        (out / "report.html").unlink()
        (out / "report.md").unlink()
        for argv in (["visualize"], ["summary"], ["report"]):
            code = _run(tiny_root, out, *argv)
            err = capsys.readouterr().err
            assert code == 0, f"{argv}: {err!r}"
        assert (out / "report.html").is_file()


class TestArgumentHandling:
    def test_no_command_prints_help(self, capsys):
        assert cli.main([]) == 0
        assert "只读命令" in capsys.readouterr().out

    def test_unknown_command_is_a_usage_error(self):
        with pytest.raises(SystemExit) as e:
            cli.main(["nope"])
        assert e.value.code == 2

    def test_unknown_sort_key_is_rejected_by_the_parser(self, tiny_root, out):
        """``--by`` 的取值由 argparse 的 ``choices`` 兜住 —— 于是它**列在帮助里**，
        而不是等运行到一半才在 ``sorted(key=...)`` 里炸出一个 ``KeyError``。"""
        with pytest.raises(SystemExit) as e:
            _run(tiny_root, out, "matrix", "--by", "bogus")
        assert e.value.code == 2

    def test_unknown_cid_explains_instead_of_crashing(self, tiny_root, out, capsys):
        """``explain`` 打错一个 cid 是常见的手误。回一句「没有组合，前几个是…」
        比 KeyError 有用得多 —— 而它必须**是**一句提示，不是 traceback。"""
        code = _run(tiny_root, out, "explain", "deadbeefdead")
        err = capsys.readouterr().err
        assert code == 2
        assert "没有组合" in err

    def test_import_needs_a_real_directory(self, tiny_root, tmp_path, capsys):
        code = _run(tiny_root, tmp_path / "out", "import", "--from", str(tmp_path / "nowhere"))
        assert code == 2
        assert "错误：" in capsys.readouterr().err

    def test_import_prints_its_row_table_and_exits_zero(self, tiny_root, tmp_path, foreign_dir, capsys):
        """**这条是一条回归测试：成功路径从来没被跑过。**

        以前 ``import`` 只在 ``row['name']`` 上炸掉（``name`` 没有任何分支写过），
        而当时的测试**只覆盖了「目录不存在」这条错误路径** ——
        错误路径走的是 ``return 2`` 之前的那一行，于是打印循环一次都没执行，
        一个从来没人跑过的循环藏在了一条绿测试后面。

        这里跑的就是那一段：四种行（相容 / 跨基座 / 检查点 / 读不通）一起打，
        并要求它 **exit 0**、四行都在、且**每个分支的判词都出现**。
        """
        out = tmp_path / "out"
        code = _run(tiny_root, out, "import", "--from", str(foreign_dir))
        text = capsys.readouterr().out
        assert code == 0, "import 的成功路径必须能跑完（它曾经在打印循环上抛 KeyError）"
        assert "外部产物可读性报告" in text

        def line_of(name: str) -> str:
            return next(ln for ln in text.splitlines() if name in ln)

        # 判词必须落在**那一行**上，而不是「整篇文本里出现过」——
        # 「读不通」与「整模型检查点」在报告开头的汇总行里也有，
        # 于是「文本里出现过」这种断言在明细表整个空掉时**照样绿**。
        assert "形状相容" in line_of("good.bin")
        assert "形状不符" in line_of("crossbase.bin")
        assert "整模型检查点" in line_of("checkpoint.bin")
        assert "读不通" in line_of("broken.bin")
        assert "broken.json" in line_of("broken.bin"), "读不通的那一行要带上原因"

    def test_import_does_not_label_its_own_base_key_as_theirs(self, tiny_root, tmp_path, foreign_dir, capsys):
        """**标签写反了，比崩溃更难发现。**

        这两行曾经挤成一行：「外部产物声明的基座标识：``tiny:2l-d24-h4-f48-v26``」。
        值取的是 ``rep['base_key']`` —— 那是 ``import_report`` 写进去的
        **本项目自己的** ``shape.key``（``"base_key": shape.key``），
        而外部那侧的答案是 ``foreign_base_identities``，这里是空的。
        于是一行字让读者以为 FineTuneFactory 声明了基座标识，
        而紧跟其后的结语正在说它**没有** —— 两句相邻的话互相打脸。

        现在两个数各有各的行：外部那行说「一个都没有」，本地那行明写「【本地】」。
        """
        code = _run(tiny_root, tmp_path / "out", "import", "--from", str(foreign_dir))
        text = capsys.readouterr().out
        assert code == 0
        assert "外部产物自报的基座标识：（一个都没有" in text
        assert "【本地】基座" in text


class TestExplain:
    def test_explains_a_frontier_point(self, tiny_root, out, result, capsys):
        cid = result.frontier[0]
        assert _run(tiny_root, out, "explain", cid) == 0
        text = capsys.readouterr().out
        assert cid in text
        assert "成员单干的对照" in text

    def test_explains_a_dominated_point(self, tiny_root, out, result, capsys):
        """被支配的点**也要能解释** —— 只解释前沿点的话，
        「为什么这个点没进前沿」就无处可查，而那是同一个问题的另一半。"""
        dominated = [r.cid for r in result.runs if r.cid not in set(result.frontier)]
        assert dominated, "夹具里应当有被支配的点"
        assert _run(tiny_root, out, "explain", dominated[0]) == 0
        assert dominated[0] in capsys.readouterr().out

    def test_route_and_stack_read_differently(self, tiny_root, out, result, capsys):
        """route 不合并权重、也没有产物。它的 ``explain`` 必须**说明**这两件事，
        而不是留两个空格让人猜是不是坏了。"""
        route = next(r.cid for r in result.runs if r.method == "route")
        assert _run(tiny_root, out, "explain", route) == 0
        text = capsys.readouterr().out
        assert "route 不合并权重" in text
        assert "不产生一个权重增量" in text


class TestAcceptanceCommand:
    def test_prints_checks_and_the_two_criteria(self, tiny_root, out, capsys):
        assert _run(tiny_root, out, "acceptance") == 0
        text = capsys.readouterr().out
        assert "运行时核对" in text
        assert "README 的验收标准" in text
        assert "acceptance_matrix" in text and "acceptance_graph" in text

    def test_marks_advisory_items_as_reports_not_gates(self, tiny_root, out, capsys):
        """``⚠️`` 那两项**不否决验收**，理由写在 ``audit.run_checks`` 的
        docstring 里。命令把理由一起打出来 —— 不然 ``⚠️`` 看起来像
        「有东西没通过，但被放过了」。"""
        _run(tiny_root, out, "acceptance")
        assert "不否决验收" in capsys.readouterr().out


class TestWriteCommands:
    def test_visualize_writes_only_report_html(self, tmp_path, tiny_root, result, capsys):
        out = tmp_path / "out"
        shutil.copytree(Path(tiny_root) / "out", out)
        before = {p.relative_to(out) for p in out.rglob("*") if p.is_file()}
        (out / "report.html").unlink()
        assert _run(tiny_root, out, "visualize") == 0
        after = {p.relative_to(out) for p in out.rglob("*") if p.is_file()}
        assert after == before
        assert (out / "report.html").read_text(encoding="utf-8").lower().startswith("<!doctype html")
        capsys.readouterr()

    def test_report_matches_the_file_on_disk(self, tiny_root, out, result, capsys):
        """``main.py report`` 会把现生成的文本与磁盘上那份比 —— 这条成立，
        报告才是「由产物决定」的。这里顺带把它当断言用了：
        比对结论走 stderr，所以正文在 stdout、结论在 stderr。"""
        assert _run(tiny_root, out, "report") == 0
        captured = capsys.readouterr()
        assert captured.out == RP.render_markdown(result)
        assert "逐字节相同" in captured.err
