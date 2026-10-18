"""登记（Registry）：**zoo 里有什么、它们各自是什么、能不能被组合。**

### 「登记」在本项目里是一个动词，不是一张表

README 把 Registry 放在组合与评测**之前**。它实质上要回答三个问题，
每个问题都有一种具体的、静默的失败方式：

1. **这份 Adapter 是不是我以为的那一份？** —— ``digest``（内容寻址）。
   只按文件名认人，会在文件被覆盖之后继续用旧标签解释新权重。
2. **它架在哪根基座上？** —— ``base_key``。**这一条最容易被跳过，也最致命**：
   把 d_model=16 的适配器挂到 d_model=24 的基座上，若只检查形状，
   一个 ``[32,16]`` 的 A 与 ``[24,32]`` 的 B 会在矩阵乘法里报错；
   但若两个模型的维度恰好相同而**层数或词表**不同，张量形状可能全部通过，
   于是「组合」这件事在算术上成立、在语义上毫无意义。
   所以 :func:`load_registry` 核的是 ``base_key``（含层数、词表、头数），不是形状。
3. **它到底是干什么用的？** —— ``capability`` 标签。干扰（interference）的
   定义完全建立在它上面：``I_{j→i}`` 里的 ``c_i`` 是 ``i`` 的能力标签。
   标签错了，整张干扰矩阵的**行名**就错了，而矩阵本身依然是漂亮的数字。

### 边界的功能性验证

:func:`read_foreign` 读的是**真正的 FineTuneFactory 产物**（``out/artifacts/*.bin``），
不是为本项目造的一份仿制品。这是「不重复 FineTuneFactory」这条命题唯一能
被外部检验的地方：

- 该读不进来 —— 说明两侧的格式已经漂移，边界是假的；
- 读进来了但形状对不上 —— 说明**跨项目的 Adapter 搬运需要基座同一性**，
  这是一条关于 Adapter 生命周期的实质结论，报告里必须写。

两种结果都比「自证兼容」有价值。:func:`import_report` 把两者都算出来，
并且**先把「适配器」与「整模型检查点」分开** —— 否则一个
``0/64`` 的数字会同时藏住两条相反的结论。实测跑下来是第二种：
48 份 LoRA 逐份读得进来、元数据字段逐一同构，但 ``lora_A`` 的
``out`` 维跟着基座的 ``d_model`` 走，两侧一个 16 一个 24。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import config as C
from . import net as N
from . import zoo as Z
from .config import VARIANTS
from .rng import digest_of, join_digest, sha256_text

__all__ = [
    "Registry",
    "load_registry",
    "expected_labels",
    "registry_digest",
    "read_foreign",
    "import_report",
    "base_identity_of",
    "n_base_params_touched",
]


def expected_labels(capabilities: Sequence[str]) -> list[str]:
    """配置声明的 zoo 应有的标签，**顺序即报告里的顺序**（能力序 × 变体序）。"""
    return [f"{c}-{v}" for c in capabilities for v in VARIANTS]


def n_base_params_touched(shape: N.ModelShape, targets: Sequence[str]) -> int:
    """一个 Adapter 改动了基座里多少个参数元素。

    这是 ``parameter_cost`` 的分子。**它与 Adapter 自己的参数量是两个不同的量**，
    而两者常常被混为一谈：``qv`` 的适配器有 768 个参数，但它改动了
    ``2 层 × 2 模块 × 24×24 = 2304`` 个基座元素 —— 后者才是「部署时要重算多少」。
    """
    base = N.param_shapes(shape)
    total = 0
    for full in N.expand_targets(base, targets):
        s = base[full]
        total += math.prod(s)
    return total


@dataclass
class Registry:
    """载入内存的 zoo。**``adapters`` 与 ``records`` 一一对应且顺序相同。**"""

    records: list[Any]
    adapters: dict[str, N.Adapter]
    base_key: str
    digest: str
    _by_label: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_label = {r.label: r for r in self.records}
        if len(self._by_label) != len(self.records):
            dupes = sorted({r.label for r in self.records if sum(1 for x in self.records if x.label == r.label) > 1})
            raise ValueError(f"zoo 里有重复标签 {dupes}：同一个标签指向两份权重，检索会静默取到一个")

    # ---- 检索 --------------------------------------------------------
    def labels(self) -> list[str]:
        return [r.label for r in self.records]

    def record(self, label: str) -> Any:
        if label not in self._by_label:
            raise KeyError(f"zoo 里没有 Adapter {label!r}（有：{sorted(self._by_label)}）")
        return self._by_label[label]

    def capabilities(self) -> list[str]:
        seen: list[str] = []
        for r in self.records:
            if r.capability not in seen:
                seen.append(r.capability)
        return seen

    def by_capability(self, capability: str) -> list[Any]:
        return [r for r in self.records if r.capability == capability]

    def subset(self, labels: Sequence[str]) -> list[N.Adapter]:
        """按给定的标签取出适配器，**顺序照 ``labels`` 给**。

        调用方（:mod:`compose`）传进来的一定是 ``plan.members``，而它**已排序**
        —— 所以这里的顺序是确定的、也是规范序。:func:`~adapterfactory.compose.realize`
        会断言取出的适配器标签与 ``plan.members`` 逐个对应，于是「顺序错了」
        会在那里变成一条报错，而不是一个数值上几乎相同、``cid`` 却指向别处的组合。

        用 ``set`` 迭代去取会让顺序随进程变化 —— 而 ``Σ Δᵢ`` 对两个成员可交换，
        对三个成员则有 ~1e-16 的累加差。**差得极小，所以没人会看出来。**
        """
        out: list[N.Adapter] = []
        for lab in labels:
            if lab not in self.adapters:
                raise KeyError(f"zoo 里没有 Adapter {lab!r}")
            out.append(self.adapters[lab])
        return out

    def summary_lines(self) -> list[str]:
        lines = []
        for r in self.records:
            lines.append(
                f"  {r.label:<14} {r.capability:<8} {r.variant:<4} "
                f"r{r.rank:<3} 挂点 {len(r.targets)} 个  {r.n_params:>5} 参   "
                f"{r.n_bytes:>6} B  改动基座 {r.n_base_params_touched:>5}   {r.digest[:12]}"
            )
        return lines


def registry_digest(records: Iterable[Any]) -> str:
    """zoo 的**内容**摘要。

    只吃 ``label`` 与 ``digest``，不吃 ``train`` 那一块 —— 训练曲线是**过程**
    的记录，不是「zoo 是什么」的一部分。把它算进去的后果是：换一次
    ``steps`` 但产出的权重逐位相同，版本摘要却变了，于是整批产物被判为过期。
    """
    parts: list[str] = []
    for r in records:
        parts.append(f"{r.label}|{r.digest}|{r.rank}|{r.capability}|{r.variant}|{','.join(r.targets)}")
    return join_digest(parts)


# ----------------------------------------------------------------------
# 载入
# ----------------------------------------------------------------------
def load_registry(
    root: str | Path,
    cfg: Mapping[str, Any],
    shape: N.ModelShape,
    *,
    verify: bool = True,
) -> Registry:
    """从 ``data/adapters/`` 载入 zoo。

    **缺一个就报错，绝不静默少一个。** 少一个 Adapter 的后果不是「少一格」：
    组合网格是从 zoo 的大小推出来的，8 个变 7 个会让网格从 276 变成
    ``C(7,1)+C(7,2)+C(7,3)=63`` ×3 = 189 —— 于是所有跨运行的可比性一起消失，
    而报告里只会显示一张「稍微小一点」的表。
    """
    from .schema import AdapterRecord

    out_dir = Path(root) / "data" / "adapters"
    caps = [str(c) for c in cfg["data"]["capabilities"]]
    labels = expected_labels(caps)
    records: list[AdapterRecord] = []
    adapters: dict[str, N.Adapter] = {}
    missing: list[str] = []
    # 「这次的配方应该长什么样」由**配置**推出来，不读磁盘。
    want_specs = {s.label: s.to_dict() for s in C.zoo_specs_of(cfg)}
    want_train = C.zoo_recipe_of(cfg)
    for lab in labels:
        stem = out_dir / lab
        if not stem.with_suffix(".json").is_file():
            missing.append(lab)
            continue
        tensors, rec = Z.read_artifact(stem, verify=verify)
        if rec.base_key and rec.base_key != shape.key:
            raise ValueError(
                f"{lab}: 这份 Adapter 架在基座 {rec.base_key!r} 上，"
                f"而当前基座是 {shape.key!r}。跨基座的适配器不能组合 —— "
                f"这不是形状问题，是语义问题（见模块 docstring）。"
            )
        # **产物必须就是配置描述的那一份。** 放在这里，而不是只放在
        # ``produce`` 里，是因为「写的时候挡住」挡不住**已经写坏了的磁盘**：
        # ``zoo.rank`` 从 4 改成 8 后沿用旧产物，配置、报告、``registry``
        # 三处都写着 8，而权重是 4 —— 参数量、``MACs``、``rank_total`` 全跟着错，
        # 且没有任何一处会自相矛盾。只读路径是唯一每次都经过的地方，于是它是
        # 唯一能挡住这件事的地方。
        #
        # 报错而不是「登记下来继续跑」：一整轮 276 个组合的测量结果，如果
        # 描述的不是配置里那个系统，那它一个数都不该被写进 ``out/``。
        drift = Z.recipe_mismatch(
            rec, spec=want_specs.get(lab, {}), train=want_train, base_seed=None
        )
        if drift:
            raise ValueError(
                f"{lab}: 磁盘上的产物与当前配置不是同一份配方 —— {drift}\n"
                f"  这一份是配置推出来的 zoo 目录里的一员，但磁盘上的权重"
                f"是按另一套配方训出来的。\n"
                f"  修法：`python main.py produce`（配方变了会自动重训，并打印"
                f"每个被重训的标签与原因）。"
            )
        ad = Z.adapter_from_record(tensors, rec, label=lab)
        # 挂点以**产物**为准，不以配置为准：配置只规定变体的名字，
        # 实际挂了哪些模块是训练时的事实。配置说 `all` 而产物只挂了 qv，
        # 那是产物对；反过来才是错的。所以这里不比对，只登记。
        adapters[lab] = ad
        records.append(
            AdapterRecord(
                label=lab,
                capability=rec.capability,
                variant=rec.variant,
                targets=list(rec.targets),
                rank=rec.rank,
                alpha=rec.alpha,
                scale=rec.scale,
                digest=rec.digest,
                n_bytes=rec.n_bytes,
                n_params=rec.n_params,
                base_key=rec.base_key,
                source=rec.source,
                n_base_params_touched=n_base_params_touched(shape, rec.targets),
                train=dict(rec.train if hasattr(rec, "train") else {}),
            )
        )
    if missing:
        raise FileNotFoundError(
            f"zoo 缺 {len(missing)} 个 Adapter：{missing}\n"
            f"  期望目录：{out_dir}\n"
            f"  先跑 `python main.py produce`（一次性训练并冻结）。"
        )
    return Registry(records=records, adapters=adapters, base_key=shape.key, digest=registry_digest(records))


# ----------------------------------------------------------------------
# 边界：读真产物
# ----------------------------------------------------------------------
def read_foreign(stem: str | Path) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """读一份**外部格式**的产物（FineTuneFactory 的 ``artifacts/<spec>.bin|json``）。

    与 :func:`zoo.read_artifact` 的差别只在**元数据长在哪**：外部格式把
    ``rank`` / ``alpha`` / ``target_modules`` 放在 ``meta`` 下面，而本项目
    把它们平铺在顶层。**裸字节的解释方式完全相同**（这一点是被验证的，
    不是被假设的）—— ``meta.payload`` 里的 ``names/shapes/offsets/dtype/endian``
    与 :func:`zoo.pack` 写出来的逐字同构。

    刻意不复用 :class:`~adapterfactory.zoo.ZooRecord.from_dict`：那个函数
    要求 ``targets`` / ``file`` / ``bytes`` 这些**本项目才有**的键，
    复用它会让「读不进来」的原因指向缺字段，而真正的问题可能是格式漂移。
    两条读取路径分开，任一条坏掉都能被另一条照出来。
    """
    stem = Path(stem)
    p_bin, p_json = Z.artifact_paths(stem)
    if not p_json.is_file():
        raise FileNotFoundError(f"外部产物的元数据不存在：{p_json}")
    doc = json.loads(p_json.read_text(encoding="utf-8"))
    meta = doc.get("meta") if isinstance(doc.get("meta"), Mapping) else {}
    payload = (meta or {}).get("payload") or doc.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{p_json.name}: 找不到 payload 元数据（既不在 meta.payload 也不在顶层）")
    payload_bytes = p_bin.read_bytes()
    tensors = Z.unpack(payload_bytes, payload)
    info = {
        "file": p_bin.name,
        "spec_id": str(doc.get("spec_id", stem.stem)),
        "kind": str(doc.get("kind", "")),
        "label": str((meta or {}).get("label") or doc.get("spec_id", stem.stem)),
        "capability": str((meta or {}).get("capability", "")),
        "rank": (meta or {}).get("rank"),
        "alpha": (meta or {}).get("alpha"),
        "target_modules": list((meta or {}).get("target_modules") or []),
        "digest": str(doc.get("digest", "")),
        "n_bytes": int(doc.get("n_bytes", len(payload_bytes))),
        "n_tensors": len(tensors),
    }
    return tensors, info


#: 只有这两种 ``kind`` 是**适配器**（低秩增量）。``full`` / ``freeze`` 是整模型
#: 检查点 —— 它们压根不是适配器，把两类混在一起计数会得出一个假结论。
_ADAPTER_KINDS = ("lora", "qlora")


def base_identity_of(doc: Mapping[str, Any]) -> str:
    """从外部元数据里尽力找出「这份产物架在哪根基座上」。

    **结论是：找不到。** FineTuneFactory 的边车 JSON 里没有基座标识 ——
    既没有顶层键，也没有 ``meta`` 里的键。它记的是**张量形状**
    （``meta.payload.shapes``），而形状不足以确定基座：一个 ``d_model=16``、
    ``vocab=28`` 的基座与一个 ``d_model=16``、``vocab=26`` 的基座，
    所有线性层形状逐字相同，词表不同。见 :func:`import_report` 的
    ``base_identity`` 字段 —— 这条不是猜测，是扫遍 64 份产物得到的。
    """
    meta = doc.get("meta") if isinstance(doc.get("meta"), Mapping) else {}
    for key in ("base_key", "base", "model_key", "model_config"):
        for src in (doc, meta or {}):
            if isinstance(src, Mapping) and src.get(key):
                return str(src[key])
    return ""


def _shape_compat(tensors: Mapping[str, Any], shape: N.ModelShape) -> tuple[list[str], int]:
    """核对一份**适配器**产物与本项目基座是否形状相容。

    返回 ``(问题清单, 落在已知挂点上的键数)``。空清单 = 相容。

    第二个返回值存在的理由是给调用方一个**分母**：只说「11 个问题」看不出
    是「8 个张量全错」还是「40 个张量里错 11 个」。这里的 ``len(tensors)``
    在整模型检查点上没有意义（它们有 10 个非 LoRA 张量），所以只在
    :data:`_ADAPTER_KINDS` 上调用本函数。

    ⚠️ 那道「张量成对」的检查曾经写成 ``seen_hooks * 2 != len(tensors)``，
    而 ``seen_hooks`` 数的**就是键**、不是挂点对数 —— 于是它对每一份
    形状完好的适配器都成立，把**全部**适配器判成不相容。
    症状极具迷惑性：跨基座的产物本来也确实不相容，所以这个 bug 藏在一堆
    真实的 ``False`` 后面，直到有人拿一份同基座的产物去读才发现。
    现在按**挂点分组**核对两个因子都在，不再拿一个计数乘 2。
    """
    base = N.param_shapes(shape)
    problems: list[str] = []
    seen_keys = 0
    factors: dict[str, set[str]] = {}
    for key, value in tensors.items():
        head, sep, tail = key.rpartition(".")
        if not sep or tail not in ("lora_A", "lora_B"):
            problems.append(f"{key}: 不是 <挂点>.lora_A/.lora_B 形式的键")
            continue
        if head not in base:
            problems.append(f"{key}: 挂点 {head!r} 不在本项目基座里")
            continue
        seen_keys += 1
        factors.setdefault(head, set()).add(tail)
        want_in, want_out = base[head]
        got = Z.shape_of(value)
        if tail == "lora_A":
            # A 的形状是 [rank, out]。**out 是那个跟着基座 d_model 走的维度**，
            # 所以跨基座时它首先炸；rank 是适配器自己的，跨基座不变。
            if len(got) != 2 or got[1] != want_out:
                problems.append(f"{key}: 形状 {got} 与基座 [rank, {want_out}] 不符")
        else:
            if len(got) != 2 or got[0] != want_in:
                problems.append(f"{key}: 形状 {got} 与基座 [{want_in}, rank] 不符")
    if not tensors:
        problems.append("产物里一个张量都没有")
    else:
        if seen_keys != len(tensors):
            problems.append(f"有 {len(tensors) - seen_keys} 个键没有落在本项目的挂点上")
        half = sorted(h for h, s in factors.items() if s != {"lora_A", "lora_B"})
        if half:
            problems.append(f"挂点缺因子（低秩分解必须两个因子都在）：{half[:4]}")
    return problems, seen_keys


#: :func:`import_report` 每一行**保证**有的键。
#:
#: 三个分支（读不通 / 适配器 / 整模型检查点）各自只写自己那几个字段。把那些
#: 参差不齐的字典直接交给调用方，等于在赌调用方**记得住**哪一支有哪几个键 ——
#: 而它记不住。``main.py import`` 曾经打印 ``row['name']``，``name`` 却从来没有
#: 任何分支写过；同一行里的 ``row['kind']`` 在「读不通」那一支也同样不存在。
#: 于是那条命令**在它的打印循环上**炸掉（``错误：'name'``，exit 2），而前面几行
#: 统计早已打完 —— 看上去像是报告写到一半坏了，其实是渲染层读了一个幻字段。
#:
#: 契约在这里补齐，不在调用方补：生产者把键集说死，消费者索引写错一个键就会当场
#: ``KeyError``。反过来用 ``.get(k) or {}`` 兜住，会把「读错了地方」伪装成
#: 「本来就没有」—— 那正是坑 104。
_ROW_KEYS = (
    "file", "read", "error",
    "spec_id", "kind", "is_adapter", "label", "capability", "rank",
    "targets", "n_tensors", "digest", "base_identity",
    "compatible", "n_hooks", "n_problems", "problems", "note",
)


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """把一行的键集补齐到 :data:`_ROW_KEYS`，该行没有的填 ``None``。

    **``None`` 与 ``False`` 在这份报告里不是一回事。** ``compatible=None`` 说的是
    「这一项不适用」（整模型检查点根本不参与相容性判定），``compatible=False``
    才是「判过了，不相容」。这两者已经区分过一次（``n_adapters`` 与
    ``n_adapter_compatible`` 分开报），这里沿用同一条规矩 —— 补 ``None`` 而不是
    补 ``0``/``[]`` 正是为了不让那一处分明的区别在序列化时又被抹平。

    多出来的键**直接报错，不静默丢弃**：丢掉的话，以后有人在某个分支新加一个
    字段，它会无声消失在这份唯一的事实来源里。
    """
    extra = sorted(set(row) - set(_ROW_KEYS))
    if extra:
        raise AssertionError(f"import_report 的行里出现了未登记在 _ROW_KEYS 的键：{extra}")
    return {k: row.get(k) for k in _ROW_KEYS}


def import_report(src: str | Path, shape: N.ModelShape, *, limit: int = 40) -> dict[str, Any]:
    """扫一个外部产物目录，逐份尝试读入并核对形状相容性。

    **返回一份报告而不是一个布尔。** 「读进来了」与「能用」是两件事，
    压成一个 ``bool`` 会把后者吞掉 —— 而后者（形状不相容 ⇒ 跨基座的适配器
    不可搬运）正是本项目里更有意思的那条结论。

    ``--from`` 指向的目录既可以是产物所在目录本身，也可以是上一层
    （此时会去找 ``artifacts/`` 子目录、再找当前层的 ``*.bin``）。

    ### 分类是必须的，不是修饰

    一个目录里同时躺着**适配器**（``kind=lora``/``qlora``）与**整模型检查点**
    （``kind=full``/``freeze``）。把两者一起计入分母，得到的是
    「``n_compatible=0/64``」这种数字 —— 它看起来像「格式完全对不上」，
    实际上是「16 份根本不是适配器的东西当然不相容」+「48 份是适配器，
    格式逐字同构，只是架在别的基座上」。**同一个 0 背后是两个完全不同的
    结论**，所以 ``n_adapters`` 与 ``n_adapter_compatible`` 分开报。

    分类依据是产物**自报**的 ``kind``，不是文件名的前缀猜测：
    文件名是人在写脚本时定的，``kind`` 是导出时写进元数据的。

    ### ``rows`` 的键集是承诺

    ``rows`` 里每一行都恰好有 :data:`_ROW_KEYS` 那些键，不适用的填 ``None``
    （由 :func:`_normalize_row` 兑现）。调用方**按名索引就好，不要 ``.get`` 兜底**：
    写错键名应当当场炸，而不是安静地少一列 —— 这条命令的渲染层就是踩了
    「以为有个 ``name`` 字段」才坏的。
    """
    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError(f"目录不存在：{src}")
    root = src / "artifacts" if (src / "artifacts").is_dir() else src
    bins = sorted(root.glob("*.bin"))
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for b in bins:
        entry: dict[str, Any] = {"file": b.name}
        try:
            tensors, info = read_foreign(b)
        except Exception as exc:  # noqa: BLE001 - 报告要把失败原因带出来，不是吞掉
            entry.update({"read": False, "is_adapter": None, "error": f"{type(exc).__name__}: {exc}"})
            rows.append(entry)
            continue
        doc = json.loads(Z.artifact_paths(b)[1].read_text(encoding="utf-8"))
        identity = base_identity_of(doc)
        if identity:
            identities.add(identity)
        kind = info["kind"]
        is_adapter = kind in _ADAPTER_KINDS
        entry.update(
            {
                "read": True,
                "spec_id": info["spec_id"],
                "kind": kind,
                "is_adapter": is_adapter,
                "label": info["label"],
                "capability": info["capability"],
                "rank": info["rank"],
                "targets": sorted(
                    {k.rpartition(".")[0].rpartition(".")[2] for k in tensors if k.endswith(".lora_A")}
                ),
                "n_tensors": info["n_tensors"],
                "digest": info["digest"],
                "base_identity": identity,
            }
        )
        if is_adapter:
            probs, n_hooks = _shape_compat(tensors, shape)
            entry.update(
                {
                    "compatible": not probs,
                    "n_hooks": n_hooks,
                    "problems": probs[:6],
                    "n_problems": len(probs),
                }
            )
        else:
            # 整模型检查点：**不参与相容性判定**，但也不静默丢弃 —— 报告里
            # 要看得见「这些文件被读进来了、被认出来了、被排除在分母之外」。
            entry.update(
                {
                    "compatible": None,
                    "note": f"kind={kind}：整模型检查点，不是适配器，不参与相容性判定",
                }
            )
        rows.append(entry)
    # 三支写完再统一补齐键集：上面的分支代码因此不必为了「让调用方好读」而到处
    # 填 None，契约只在这一处兑现（见 :data:`_ROW_KEYS`）。
    rows = [_normalize_row(r) for r in rows]
    adapters = [r for r in rows if r.get("is_adapter")]
    n_read = sum(1 for r in rows if r.get("read"))
    n_ok = sum(1 for r in adapters if r.get("compatible"))
    return {
        "source": str(root),
        "n_files": len(bins),
        "n_read": n_read,
        "n_adapters": len(adapters),
        "n_adapter_compatible": n_ok,
        "n_checkpoints": sum(1 for r in rows if r.get("is_adapter") is False),
        "n_unreadable": sum(1 for r in rows if not r.get("read")),
        "base_key": shape.key,
        # 外部产物自报的基座标识。**FineTuneFactory 的产物这里是空的**，
        # 而空集与「有标识但不匹配」的处置完全不同：前者导致下面的
        # ``base_identity`` 判词，后者应当直接拒绝搬运。
        "foreign_base_identities": sorted(identities),
        "artifact_digest": join_digest([r["digest"] for r in rows if r.get("read")]) if n_read else "",
        "rows": rows[:limit],
        "n_rows_hidden": max(0, len(rows) - limit),
    }
