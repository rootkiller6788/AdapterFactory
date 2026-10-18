"""测试夹具。**分层：``tiny`` / ``fast`` / ``result`` —— 每一层的代价写在名字旁边。**

### 为什么端到端测试要自己造一个「变小」的项目根

真实配置是 8 个 Adapter × 92 个子集 × 3 方法 = 276 个组合，
训练一次 400 步要十几分钟。把它放进测试意味着**每次跑测试都要等十几分钟** ——
而那样做的直接后果是没人跑测试。

所以这里造一个**同构但变小**的根：能力从 4 个降到 2 个（``upper`` / ``shift``），
zoo 就从 8 个变成 4 个，网格从 276 降到 ``C(4,1)+C(4,2)+C(4,3)=14`` ×3 = **42**。
结构完全一样（子集格、三条方法、两条路径、九项核对都在），
代价降到几秒。

### 这里的 zoo 是**真的训练出来的**

不是 ``init_adapter`` 加扰动。理由是那条铁律：**必须实际跑一遍**。
用随机 delta 造出来的 zoo 会让「组合之后还有没有能力」这个问题无意义
（随机 delta 的 gain 是噪声），于是测试通过也说明不了任何事。

代价是 4 个 Adapter × 60 步 × batch 8 —— 实测几秒。用一个 session 级夹具
摊掉，全测试套件只训一次。

### 两条质量门槛在 tiny 配置里是 ``required=False``

``min_single_gain_margin`` 与 ``min_same_cap_subadditivity`` 在 60 步、
32 条训练样本的条件下**必然达不到**（它们问的是「这个 Adapter 学得好不好」，
而 tiny 只够验证「整条链路通不通」）。所以 tiny 配置里把这两项关掉，
而它们的**机制**由 ``test_audit.py`` 直接对着 ``audit.checks`` 验 ——
那里可以精确构造「过」与「不过」两种输入，不必依赖模型质量。

关掉而不是把门槛数值调小：调小会连**门槛的含义**一起改掉，
而那正是最不该在测试里被动过的东西。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapterfactory import config as C  # noqa: E402
from adapterfactory import data as D  # noqa: E402
from adapterfactory import net as N  # noqa: E402
from adapterfactory import pipeline as PL  # noqa: E402
from adapterfactory import produce as PR  # noqa: E402
from adapterfactory import zoo as Z  # noqa: E402


#: tiny 配置的覆写。**每一项后面都要能说出理由**（见模块 docstring）。
TINY_OVERRIDE: dict = {
    "data": {"capabilities": ["upper", "shift"], "n_eval": 8, "n_train": 16},
    # 24 步 × batch 8 = 192 次「前向 + 反向」每个 Adapter。实测这已经是
    # 每个 Adapter 几秒的量级，四个加起来让 session 夹具停在半分钟出头 ——
    # 再往上加，测试就开始变得没人愿意跑，而那是所有测试问题的根源。
    "train": {"steps": 24, "batch": 8},
    "acceptance": {
        # 60 步 + 32 条样本达不到质量门槛，也**不该**达到 —— 见模块 docstring。
        "min_single_gain_margin": 0.0,
        "require_single_gain_margin": False,
        "require_same_cap_subadditivity": False,
        # 8 个样本 × 4 个位置 = 32 个位置，分辨率 1/32 = 0.031。0.05 的容差
        # 在这张探针集上不到两个分辨率单位 —— 那查的就不再是「基座贴不贴 chance」，
        # 而是「采样噪声有多大」。0.15 才让这条检查在这张探针集上有意义。
        "base_chance_tolerance": 0.15,
    },
    "report": {"max_matrix_rows": 42},
}


@pytest.fixture(scope="session")
def repo() -> Path:
    """真实项目根（只读用：拿 index.json 与源码）。"""
    return REPO


@pytest.fixture(scope="session")
def tiny_cfg() -> dict:
    """变小但同构的配置。**从 DEFAULTS 深合并**，于是它自动跟随默认值的变化。"""
    return C.deep_merge(json.loads(C.canonical_json(C.DEFAULTS)), TINY_OVERRIDE)


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory: pytest.TempPathFactory, tiny_cfg: dict, repo: Path) -> Path:
    """一个完整的小项目根：``data/datasets/index.json`` + ``data/adapters/``。

    索引是从真实项目**复制**过来的，不是另写一份 —— 另写一份会让
    「测试里的题」与「真实运行时的题」差一个字符而没人发现。
    """
    root = tmp_path_factory.mktemp("tiny-project")
    (root / "data" / "datasets").mkdir(parents=True)
    shutil.copyfile(repo / "data" / "datasets" / "index.json", root / "data" / "datasets" / "index.json")
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "default.json").write_text(
        json.dumps(tiny_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root


@pytest.fixture(scope="session")
def produced(tiny_root: Path, tiny_cfg: dict) -> dict:
    """**真的训练**一个 zoo 并冻结进 ``tiny_root/data/adapters/``。

    session 级：整轮测试只训这一次。返回 ``produce_zoo`` 的报告。
    """
    ctx = PL.prepare(root=tiny_root, cfg=tiny_cfg)
    # 配方走**配置模块的那一对函数**，与 ``cli._cmd_produce`` 逐字同源。
    # 这里以前把 7 个字段抄了一遍 —— 抄的那一份与生产路径那一份是两份代码，
    # 而 ``load_registry`` 现在会核对「磁盘上的产物是不是这次配方的输出」，
    # 两份代码一旦漂移，测试与真实运行就会在**同一份产物**上得出不同结论。
    specs = C.zoo_specs_of(tiny_cfg)
    settings = PR.TrainSettings(**C.zoo_recipe_of(tiny_cfg))
    return PR.produce_zoo(
        root=tiny_root,
        shape=ctx["shape"],
        base_seed=ctx["base_seed"],
        specs=specs,
        settings=settings,
        vocab=ctx["vocab"],
        idx=ctx["idx"],
        seq_len=ctx["seq_len"],
        eval_sets=ctx["eval_sets"],
        train_seed=int(tiny_cfg["data"]["seed"]),
        force=False,
        verbose=False,
    )


@pytest.fixture(scope="session")
def ctx(tiny_root: Path, tiny_cfg: dict) -> dict:
    """``pipeline.prepare`` 的全部中间物（基座、探针集、基线…）。"""
    return PL.prepare(root=tiny_root, cfg=tiny_cfg)


@pytest.fixture(scope="session")
def result(tiny_root: Path, tiny_cfg: dict, produced: dict) -> "object":
    """**整条流水线跑一次**（含导出组合产物）。session 级。

    之后全部只读断言都吃这一份结果 —— 每个测试各跑一次 42 个组合
    会让「跑测试」变成一件没人愿意做的事。
    """
    return PL.run_pipeline(root=tiny_root, cfg=tiny_cfg, export=True)


@pytest.fixture(scope="session")
def out(tiny_root: Path) -> Path:
    return PL.out_dir_of(tiny_root)


@pytest.fixture(scope="session")
def shape(tiny_cfg: dict) -> N.ModelShape:
    """基座形状。session 级 —— 它只由配置决定，且被 :func:`base_weights`
    这类 session 级夹具依赖；函数级会带来 ``ScopeMismatch``。"""
    return PL.build_shape(tiny_cfg)


@pytest.fixture(scope="session")
def base_weights(shape: N.ModelShape, tiny_cfg: dict) -> dict:
    """基座权重，**按配置里的种子初始化**。

    session 级且与 :func:`ctx` 里那一份**同源同种**：``init_base_weights``
    是纯函数，所以这里可以直接再造一份，而不必把 ``ctx`` 拖进每一个只想
    要一张权重表的测试。两边都从 ``tiny_cfg["base"]["seed"]`` 来 ——
    若哪天初始化不再是纯函数，这条会立刻炸。
    """
    return N.init_base_weights(shape, int(tiny_cfg["base"]["seed"]))


@pytest.fixture
def registry(tiny_root: Path, tiny_cfg: dict, ctx: dict, produced: dict) -> "object":
    from adapterfactory import registry as R

    return R.load_registry(tiny_root, tiny_cfg, ctx["shape"])


@pytest.fixture(scope="session")
def probe_ids(ctx: dict) -> list[int]:
    caps = list(ctx["cfg"]["data"]["capabilities"])
    return list(ctx["eval_sets"][caps[0]][0].ids)


@pytest.fixture
def index(repo: Path) -> D.DataIndex:
    return D.load_index(repo / "data" / "datasets" / "index.json")


@pytest.fixture
def vocab(index: D.DataIndex) -> D.Vocab:
    return D.build_vocab(index)


@pytest.fixture
def adapter_factory(shape: N.ModelShape):
    """造一个形状合法、delta 非零的 Adapter（**不训练**）。

    只给「只关心结构」的测试用 —— 例如「读回来的是不是同一份张量」。
    任何涉及「学到了什么」的断言都必须吃 :func:`result`，
    因为随机 delta 的 gain 是噪声，对它下结论没有意义。
    """
    from adapterfactory import rng as RNG

    def make(label: str, targets: tuple[str, ...], *, rank: int = 4, seed: int = 7) -> N.Adapter:
        ad = N.init_adapter(shape, list(targets), rank, RNG.derive_seed("test-lora", label, seed))
        rng = __import__("random").Random(RNG.derive_seed("test-fill", label, seed))
        scales = N.param_shapes(shape)
        for full in N.expand_targets(scales, list(targets)):
            # ``lora_B`` 是零初始化的（标准做法），所以 delta 初始恒为 0。
            # 要造出「有 delta」的对象就必须显式填 B —— 这也顺带证明了
            # 那条「组合的起点就是基座」的性质是从初始化来的，不是巧合。
            b = ad.tensors[full + ".lora_B"]
            for r in range(len(b)):
                for i in range(len(b[r])):
                    b[r][i] = rng.gauss(0.0, 0.02)
        ad.label = label
        return ad

    return make


@pytest.fixture
def zoo_tmp(tmp_path: Path, tiny_cfg: dict, repo: Path):
    """在 ``tmp_path`` 里造一个可写的小项目根（写命令的测试用）。"""
    (tmp_path / "data" / "datasets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / "data" / "datasets" / "index.json", tmp_path / "data" / "datasets" / "index.json")
    (tmp_path / "configs" / "default.json").write_text(
        json.dumps(tiny_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def foreign_dir(tmp_path: Path, shape: N.ModelShape, adapter_factory):
    """造一个「外部产物目录」，**每种行都凑一份**（FineTuneFactory 的磁盘格式）。

    四种行对应 ``registry.import_report`` 的四个分支，而它们的键集曾经各不相同：

    ==========================  ==========================================
    ``good``                    同基座、形状相容的适配器（``compatible=True``）
    ``crossbase``               ``lora_A`` 的 out 维比基座大 —— 跨基座，不相容
    ``checkpoint``              ``kind=freeze``，整模型检查点，**不参与**相容性判定
    ``broken``                  边车 JSON 没有 ``payload`` —— 读不通这一支
    ==========================  ==========================================

    放在 ``conftest`` 而不是某个测试文件里，是因为**两份测试都要它**
    （``test_zoo_registry`` 验键集，``test_report_cli`` 验渲染）。
    边车 JSON 的字段表只写一遍：写两遍的话，它就成了又一处「两处各推一次、
    改一处忘一处」—— 那正是这个夹具要照出来的 bug。
    """
    from adapterfactory import zoo as Z
    from adapterfactory.rng import sha256_bytes

    base = N.param_shapes(shape)
    hook = "layers.0.q_proj"
    i_dim, o_dim = base[hook]

    def write(name: str, kind: str, tensors: dict) -> None:
        payload, meta = Z.pack(tensors)
        (tmp_path / f"{name}.bin").write_bytes(payload)
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "spec_id": name,
                    "kind": kind,
                    "digest": sha256_bytes(payload),
                    "n_bytes": len(payload),
                    # ``rank``/``alpha``/``label`` 在 FF 格式里挂在 ``meta`` 下，
                    # 而 payload 的字节布局与 :func:`zoo.pack` 写出来的逐字同构。
                    "meta": {"rank": 4, "alpha": 8.0, "label": f"{name}/fake/qv", "payload": meta},
                }
            ),
            encoding="utf-8",
        )

    write("good", "lora", adapter_factory("upper-qv", ("q_proj",)).tensors)
    write(
        "crossbase",
        "lora",
        {
            f"{hook}.lora_A": [[0.0] * (o_dim + 8) for _ in range(4)],
            f"{hook}.lora_B": [[0.0] * 4 for _ in range(i_dim)],
        },
    )
    write("checkpoint", "freeze", {hook: [[0.0] * o_dim for _ in range(i_dim)]})
    # 读不通：有 .bin 有 .json，但 JSON 里没有 payload 元数据 ——
    # 于是 raw 字节无从解释，`read_foreign` 抛 ValueError（**不是**解出垃圾数字）。
    (tmp_path / "broken.bin").write_bytes(b"\x00\x01\x02\x03")
    (tmp_path / "broken.json").write_text(
        json.dumps({"spec_id": "broken", "kind": "lora"}), encoding="utf-8"
    )
    return tmp_path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 需要真跑一遍（训练或整条流水线）")


__all__ = ["TINY_OVERRIDE", "REPO", "SRC"]
