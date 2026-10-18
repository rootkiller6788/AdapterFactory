"""配置：口径与门槛的唯一来源。

### 为什么阈值不能在代码里

本工作区的约定是「**代码里不硬编码阈值**」。理由不是洁癖，而是本项目有一条
特定的风险：组合空间的规模、评测集大小、判定「干扰」的门槛，三者的乘积决定了
报告结论的强度。如果 ``n_eval`` 藏在代码里，那么「缩小评测集让结果好看」
这件事在 diff 里是看不见的 —— 而它恰好是会改变结论的那一类改动。

配置进指纹（:func:`version_digest`），于是「换了配置」在产物上留下痕迹。

### 两种摘要，不能合用一个

- **口径摘要**（:func:`protocol_digest`）：只与**怎么量**有关 —— 评测集大小、
  门槛、组合方法集合。它变了，只需要重出报告。
- **版本摘要**（:func:`version_digest`）：与**被量的东西**有关 —— 基座结构、
  数据索引、zoo 里那 8 份 Adapter 的摘要、组合网格。它变了，必须整体重跑。

合成一个的后果是：改了一句报告措辞会让整份产物作废，而换了一份 Adapter
可能看起来只是「报告该重出了」。两种代价完全不同。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .rng import canonical_json, digest_of, join_digest, sha256_text

__all__ = [
    "METHODS",
    "VARIANTS",
    "DEFAULTS",
    "deep_merge",
    "resolve_path",
    "load_config",
    "validate_config",
    "protocol_digest",
    "version_digest",
    "short_version",
    "weight_scheme",
    "methods_of",
    "sizes_of",
    "capabilities_of",
    "zoo_recipe_of",
    "zoo_specs_of",
    "route_of",
    "acceptance_of",
]


#: 三种组合方法。**顺序即报告里的顺序**，也是 ``matrix.csv`` 的列序。
#: 顺序本身有含义：``stack`` 与 ``merge`` 只差一个系数（求和 vs 平均），
#: 把它们排在一起，读者一眼能看到「这一对是同一条轴的两端点」。
METHODS: tuple[str, ...] = ("stack", "merge", "route")

#: 两个覆盖变体。``all`` 覆盖全部 7 个可挂载模块，``qv`` 只覆盖 q/v。
VARIANTS: tuple[str, ...] = ("qv", "all")

#: 组合方法 → 权重系数口径。``stack`` 全 1（加性假设），``merge`` 全 1/k（稀释共识）。
#: ``route`` 不走权重空间，所以它的系数是 ``None`` —— **不是全 1**：
#: 写成全 1 会让「route 也是一种合并」这个错误看起来是被支持的。
_WSUM = "sum"
_WAVG = "avg"


DEFAULTS: dict[str, Any] = {
    "base": {
        "name": "tiny",
        "vocab": 26,
        "d_model": 24,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 48,
        "seq_len": 13,
        "seed": 20260930,
        "_comment": (
            "基座是 (结构, 种子) 的纯函数，不落盘 —— 落盘只会多出一个可以漂移的"
            "自由度。seq_len=13 由 data.seq_len_of 推出（5 + 2×payload_len），"
            "这里写它是为了让它进指纹：改 payload 长度会让它变。"
            "head_dim=6 为偶数，RoPE 才成立；RoPE 不带参数，所以"
            "「参数量公式 == 实测参数量」这条互校才守得住（可学习的位置表会破坏它）。"
        ),
    },
    "data": {
        "index": "data/datasets/index.json",
        "capabilities": ["upper", "shift", "reverse", "sort"],
        "n_eval": 48,
        "n_train": 192,
        "seed": 20260930,
        "_comment": (
            "评测 payload 与训练 payload 用不同种子并**显式互相排除** —— 8 选 4 "
            "的空间只有 4096，靠「种子不同所以大概不会撞」是不够的。"
            "n_eval 是全部指标的分辨率下限：组合之间的差异只要小于 1/n_eval "
            "就不可能被看见，所以它进口径摘要。"
        ),
    },
    "zoo": {
        "rank": 8,
        "alpha_mult": 2.0,
        "_comment": (
            "alpha = alpha_mult × rank。倍数进配置而不是写死在代码里，"
            "否则改 rank 会同时改有效适配强度，rank 那一列就不是单变量了。"
            "8 个 Adapter = 4 能力 × 2 变体，**共用同一份基座** —— 这是"
            "「组合」这件事有意义的前提。"
            "rank=8 是**实测标定**出来的，而且是被 rank 单独判定的：rank=4/steps=400 时"
            "两个需要**位置移动**的 qv 变体（reverse-qv、sort-qv）清不过 heuristic+0.10；"
            "rank=4 只把步数加到 1200 能救回 reverse（+0.3906）却救不回 sort（+0.0781）；"
            "rank=8/steps=800 两个同时过关（+0.4114 / +0.1250）。"
            "**「多训一会儿」不是这个瓶颈的答案，容量才是。** 标定表见 README 附录 B。"
        ),
    },
    "train": {
        "lr": 0.005,
        "steps": 800,
        "batch": 8,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "_comment": (
            "lr=5e-3 是**实测标定**出来的，不是抄来的。同一份代码在 lr=5e-2 时"
            "八个 Adapter 里有一半学不动（-all 变体全部贴在乱猜上、|g|max 命中裁剪），"
            "而 lr=5e-3 时全部过关。"
            "这条与 FineTuneFactory 记录的现象逐字同构：它在 lr=3e-2 时 LoRA 恰好等于乱猜。"
            "**「跑通了、损失在降、不报错」完全可能什么都没学到** —— 所以这个值"
            "只被 adopt 过一次，理由与数字都留在 README 附录 B 的标定表里。"
            "steps=800 与 zoo.rank=8 是同一次标定的另一半：**单加步数不够** —— "
            "rank=4 时 sort-qv 一路加到 1200 步仍然是红的。"
        ),
    },
    "compose": {
        "methods": ["stack", "merge", "route"],
        "max_size": 3,
        "_comment": (
            "设计空间 = 8 个 Adapter 的全部 size 1..max_size 子集 × 3 方法。"
            "max_size=3 是**算得动的上界**：C(8,1)+C(8,2)+C(8,3)=8+28+56=92 个子集，"
            "×3=276 个组合；再往上每加一档，子集数就翻一倍而「多于三个 Adapter "
            "同时挂上」这个问题的答案并不会更清楚。"
        ),
    },
    "route": {
        "temperature": 1.0,
        "_comment": (
            "route 的门是 :func:`softmax(⟨h, uᵢ⟩/T)`，其中 **uᵢ 是种子派生的固定随机"
            "向量，不训练**。它是**结构对照**，不是学出来的路由 —— 这一点必须在"
            "报告里逐字写出来。温度 T 进配置是为了让「门的尖锐程度」是一个可调的"
            "受控变量，而不是藏在算式里的一个 1.0。"
        ),
    },
    "acceptance": {
        "max_merge_error": 1e-12,
        "min_single_gain_margin": 0.10,
        "min_same_cap_subadditivity": 0.0,
        "min_two_path_agreement": 0.5,
        "base_chance_tolerance": 0.05,
        "require_base_is_null": True,
        "require_single_gain_margin": True,
        "require_same_cap_subadditivity": True,
        "roundtrip_tolerance": 1e-06,
        "_comment": (
            "min_single_gain_margin=0.10：单 Adapter 的增益必须超过**更强的**那条基线"
            "（heuristic）至少这么多。定 0.10 而不是 0.15 的理由是参照实现的上限 —— "
            "sort 的 heuristic（照抄输入）是 0.3021，要它超过 0.15 等于要求远超参照实现，"
            "那是在测「这个任务的天花板够不够高」，不是在测 Adapter。"
            "max_merge_error=1e-12：合并与逐个施加是同一件事的两条算法路径，"
            "差别只有累加顺序。184 条权重空间组合实测：中位数 5.3e-16、"
            "最大 5.4e-14（三个成员、挂点多的那几条）—— 门槛 1e-12 对**最坏**那条"
            "也只留了约 20 倍，不是「远在天边」。这个数随 rank 与成员个数增长，"
            "所以改 rank 之后要重看这一项，别把 1e-12 当成永远安全的。"
            "**它不查「合并没有损失能力」** —— 那是 merge_loss。"
            "base_chance_tolerance=0.05 是 base_is_null 那一项**两侧共用**的容差，"
            "而那一项的两侧不对称：下界是 chance − tol（基座低于机会水平会**抬高**"
            "每一个 gain，那才是危险的方向），上界是 heuristic + tol（基座能看输入，"
            "「什么都不学」时是照抄，所以它的天然水平本来就在 chance 之上；sort 的"
            "不动点让它拿到 0.18，而 copy 基线是 0.3021）。"
            "require_* 这一组：**每一项都必须能被单独关掉**。质量门槛"
            "（single_gain_margin / same_cap_subadditivity）在「用随机初始化的 zoo 只验"
            "结构」的测试里必然不过，而那种测试要验的是产物完整性不是模型质量 —— "
            "不给它开关，它就只能去改门槛数值，而那会把门槛的真实含义一起改掉。"
            "结构性的四项（no_training_in_pipeline / producer_contained / grid_complete /"
            " cid_canonical）没有开关：它们在任何配置下都必须成立。**它们连键都没有** ——"
            " 曾经有过 ``require_grid_complete`` 与 ``require_no_training_in_pipeline`` 两个键，"
            " 默认 ``True``，于是「能关掉」这件事从不显形，直到有人去读代码。"
            " 这里留一条可执行的判据而不是一句声明：本字典里出现的 ``require_*`` 只有"
            " 质量三项 + ``base_is_null``，任何一条结构性检查都不在其中。"
        ),
    },
    "report": {
        "headline_combos": 12,
        "max_matrix_rows": 276,
        "max_conflict_edges": 40,
        "_comment": "max_matrix_rows=276 就是整张矩阵，不截断 —— 组合矩阵是本项目的**主产物**，截断它等于截断结论。",
    },
}


# ----------------------------------------------------------------------
# 读写
# ----------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并。``override`` 里的 ``None`` **不是**「删掉这个键」，是「显式设成 None」。

    把 ``None`` 当成「删除」是一个很常见的约定，但它让「配置里写了 null」
    变成一个无法表达的状态，而本项目里 ``route`` 的权重系数恰恰用 ``None``
    表达「不走权重空间」。当成删除的话那个键会消失，读的时候 ``KeyError``
    会指向一个完全无关的地方。
    """
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_path(root: Path | str, value: str | Path) -> Path:
    """相对路径一律相对 ``root`` 解析。**绝对路径原样返回**。"""
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: str | Path | None = None, *, root: str | Path = ".") -> dict[str, Any]:
    """读配置并与 :data:`DEFAULTS` 深合并。

    ``path=None`` 时读 ``configs/default.json``（不存在就用 DEFAULTS 本身）。
    合并方向是「默认值 ← 文件」，于是**文件只需要写它真正改了的那些键** ——
    这和「配置里有的东西一定是被试过的」那条约定是同一条：写进文件的每一项
    都是一次显式的表态。
    """
    cfg = json.loads(canonical_json(DEFAULTS))  # 深拷贝，且顺带验证 DEFAULTS 可序列化
    p = resolve_path(root, path) if path is not None else Path(root) / "configs" / "default.json"
    if Path(p).is_file():
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError(f"配置文件顶层必须是映射，得到 {type(raw).__name__}")
        cfg = deep_merge(cfg, raw)
    validate_config(cfg)
    return cfg


def _require(cond: bool, msg: str, errs: list[str]) -> None:
    if not cond:
        errs.append(msg)


def validate_config(cfg: Mapping[str, Any]) -> None:
    """**一次性把所有问题都找出来**，不 fail-fast。

    fail-fast 的代价是「修一个、跑一次、又发现一个」—— 而这里每跑一次都要
    重新加载整份配置。错误信息里的键路径要写全（``compose.max_size`` 而不是
    ``max_size``），因为深层键在配置文件里是要靠搜索才能找到的。
    """
    errs: list[str] = []

    base = cfg.get("base") or {}
    for k in ("vocab", "d_model", "n_layers", "n_heads", "d_ff", "seq_len"):
        v = base.get(k)
        _require(isinstance(v, int) and v > 0, f"base.{k} 必须是正整数，得到 {v!r}", errs)
    if isinstance(base.get("d_model"), int) and isinstance(base.get("n_heads"), int) and base["n_heads"] > 0:
        _require(base["d_model"] % base["n_heads"] == 0, f"base.d_model={base['d_model']} 必须能被 base.n_heads={base['n_heads']} 整除", errs)
        hd = base["d_model"] // base["n_heads"]
        _require(hd % 2 == 0, f"head_dim={hd} 必须是偶数（RoPE 按相邻通道配对）", errs)

    data = cfg.get("data") or {}
    caps = data.get("capabilities")
    _require(isinstance(caps, list) and len(caps) >= 2, "data.capabilities 至少要两个能力（干扰定义在别的能力上）", errs)
    if isinstance(caps, list):
        _require(len(set(caps)) == len(caps), f"data.capabilities 有重复：{caps}", errs)
        _require(all(isinstance(c, str) and c for c in caps), "data.capabilities 必须是非空字符串", errs)
    for k in ("n_eval", "n_train"):
        v = data.get(k)
        _require(isinstance(v, int) and v >= 8, f"data.{k} 必须 >= 8，得到 {v!r}", errs)

    train = cfg.get("train") or {}
    for k in ("lr", "steps", "batch", "grad_clip"):
        v = train.get(k)
        _require(isinstance(v, (int, float)) and v > 0, f"train.{k} 必须为正，得到 {v!r}", errs)
    _require(train.get("optimizer") in ("adamw", "sgd"), f"train.optimizer 必须是 adamw 或 sgd，得到 {train.get('optimizer')!r}", errs)

    zoo = cfg.get("zoo") or {}
    _require(isinstance(zoo.get("rank"), int) and zoo.get("rank", 0) >= 1, f"zoo.rank 必须 >= 1，得到 {zoo.get('rank')!r}", errs)
    _require(isinstance(zoo.get("alpha_mult"), (int, float)) and zoo.get("alpha_mult", 0) > 0, f"zoo.alpha_mult 必须为正，得到 {zoo.get('alpha_mult')!r}", errs)

    comp = cfg.get("compose") or {}
    methods = comp.get("methods")
    _require(isinstance(methods, list) and len(methods) >= 1, "compose.methods 至少要一种方法", errs)
    if isinstance(methods, list):
        unknown = [m for m in methods if m not in METHODS]
        _require(not unknown, f"compose.methods 里有未知方法 {unknown}（可选：{list(METHODS)}）", errs)
        _require(len(set(methods)) == len(methods), f"compose.methods 有重复：{methods}", errs)
    ms = comp.get("max_size")
    _require(isinstance(ms, int) and ms >= 1, f"compose.max_size 必须 >= 1，得到 {ms!r}", errs)
    if isinstance(caps, list) and isinstance(ms, int):
        n_adapters = len(caps) * len(VARIANTS)
        _require(ms <= n_adapters, f"compose.max_size={ms} 超过 Adapter 总数 {n_adapters}", errs)

    rt = cfg.get("route") or {}
    _require(isinstance(rt.get("temperature"), (int, float)) and rt.get("temperature", 0) > 0, f"route.temperature 必须为正，得到 {rt.get('temperature')!r}", errs)

    acc = cfg.get("acceptance") or {}
    for k in ("max_merge_error", "min_single_gain_margin", "roundtrip_tolerance"):
        v = acc.get(k)
        _require(isinstance(v, (int, float)) and v >= 0, f"acceptance.{k} 必须是非负数，得到 {v!r}", errs)
    _require(0.0 <= acc.get("min_two_path_agreement", -1) <= 1.0, f"acceptance.min_two_path_agreement 必须在 [0,1]，得到 {acc.get('min_two_path_agreement')!r}", errs)

    if errs:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errs))


# ----------------------------------------------------------------------
# 取用
# ----------------------------------------------------------------------
def capabilities_of(cfg: Mapping[str, Any]) -> list[str]:
    return [str(c) for c in cfg["data"]["capabilities"]]


def zoo_recipe_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """配置里的**训练配方**，键就是 :meth:`produce.TrainSettings.to_dict` 的那套。

    这一项存在的理由是「配方只有一个来源」：``main.py produce`` 拿它去构造
    ``TrainSettings``，``registry.load_registry`` 拿它去核对磁盘上那份产物
    是不是这份配方的输出。两处各写一遍的话，某天配置里加一个字段而只改了
    其中一处，产物的来路就会在一侧被**静默放过**。

    ``betas`` / ``eps`` **不在这里** —— 它们是代码常量，配置里从来没有过。
    配方比对只核对「配置说了的那些键」，多比一个配置没说话的键，就会把
    「配置没提」判成「不符」。

    调用方会写 ``TrainSettings(**zoo_recipe_of(cfg))``。这里允许 ``**``
    （代码里别处一律显式取字段）是因为这个字典的**键集是封闭的、由本函数
    产生**：``TrainSettings`` 那边多一个键会 ``TypeError``，少一个也会 ——
    两条都当场炸，而不是静默取默认值。
    """
    t = cfg["train"]
    return {
        "lr": float(t["lr"]),
        "steps": int(t["steps"]),
        "batch": int(t["batch"]),
        "weight_decay": float(t["weight_decay"]),
        "grad_clip": float(t["grad_clip"]),
        "optimizer": str(t["optimizer"]),
        "n_train": int(cfg["data"]["n_train"]),
    }


def zoo_specs_of(cfg: Mapping[str, Any]):
    """配置里的 zoo 目录：8 份 :class:`~adapterfactory.zoo.AdapterSpec`。

    放在这里而不是 ``cli`` 里，是为了让 ``registry`` 能**不碰到训练模块**
    就重建出「这次的配方应该长什么样」—— ``produce`` 是被架构墙挡在
    流水线外面的，只读路径更不能 import 它。
    """
    from . import zoo as Z

    rank = int(cfg["zoo"]["rank"])
    return Z.default_zoo(capabilities_of(cfg), rank=rank, alpha=float(cfg["zoo"]["alpha_mult"]) * rank)


def methods_of(cfg: Mapping[str, Any]) -> list[str]:
    m = [str(x) for x in cfg["compose"]["methods"]]
    unknown = [x for x in m if x not in METHODS]
    if unknown:
        raise ValueError(f"未知的组合方法 {unknown}（可选：{list(METHODS)}）")
    return m


def sizes_of(cfg: Mapping[str, Any]) -> list[int]:
    ms = int(cfg["compose"]["max_size"])
    return list(range(1, ms + 1))


def route_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {"temperature": float(cfg["route"]["temperature"])}


def acceptance_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg["acceptance"])


def weight_scheme(method: str) -> str | None:
    """组合方法 → 权重系数口径。``route`` 返回 ``None``（它不走权重空间）。

    **绝不静默退回 ``sum``。** 退回的后果是 ``route`` 被当成 ``stack`` 去算，
    于是报告里「route 是结构对照」这句话与它实际被算成的东西不一致，
    而两个方法都会给出数值，看不出异常。
    """
    return {"stack": _WSUM, "merge": _WAVG, "route": None}.get(method)


# ----------------------------------------------------------------------
# 摘要
# ----------------------------------------------------------------------
#: 口径摘要覆盖的顶层小节。**只有这些** —— 加一个小节进这里等于宣布
#: 「它有资格让全部产物作废」，而 `report` 那一节显然没有。
_PROTOCOL_KEYS: tuple[str, ...] = ("data", "acceptance", "compose", "route")


def _strip_comments(obj: Any) -> Any:
    """剥掉 ``_comment``。**注释不该进任何指纹** —— 改一句说明文字让产物作废
    会让「内容寻址」从一个可信机制变成一个烦人的机制。"""
    if isinstance(obj, Mapping):
        return {k: _strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """口径摘要：**怎么量**。它变了只需重出报告，不必重跑组合。"""
    sub = {k: _strip_comments(cfg[k]) for k in _PROTOCOL_KEYS if k in cfg}
    return digest_of(sub)


def version_digest(cfg: Mapping[str, Any], inputs: Iterable[tuple[str, Any]]) -> str:
    """版本摘要：**被量的东西**。

    ``inputs`` 是 ``(名字, 可摘要对象)`` 的序列（对象要有 ``to_dict``，或者是
    可 JSON 化的值）。名字进摘要，所以「把 zoo 的摘要换到 data 的位置」会被发现。

    口径摘要也并进来：它变了但版本摘要不变的话，两份结论会被判成同一次运行 ——
    而那正是「内容寻址」最不该出的错。
    """
    parts: list[str] = [protocol_digest(cfg)]
    for name, obj in sorted(inputs, key=lambda kv: kv[0]):
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        parts.append(f"{name}={sha256_text(canonical_json(obj))}")
    return join_digest(parts)


def short_version(vdigest: str) -> str:
    """``v1-<digest 前 12 位>``。"""
    return f"v1-{vdigest[:12]}"
