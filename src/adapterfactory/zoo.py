"""Adapter 产物的磁盘格式：**与 FineTuneFactory 逐字节兼容**。

### 格式（照抄 FineTuneFactory，不是「类似」）

``<name>.bin`` = 裸 ``float32`` 小端，**无魔数、无长度前缀、无对齐填充**，
各张量按顺序行主序拼接。``<name>.json`` = 旁证元数据，张量的名字 / 形状 /
**字节偏移** 全在里面。

为什么一个有自尊的格式不写魔数：产物的字节数在这里是一个**算术量**
（``Σ 参数量 × 4``，由参数表推出），而这条等式只有在文件里**只**放浮点数时才成立。
加一个魔数，等式就差出固定的几字节 —— 而那个差看起来完全像「正常的序列化开销」，
于是没人会去查它。

``digest = sha256(裸字节)``。**不是** sha256(整个文件)，也不是 sha256(JSON) ——
摘要必须锚在算术值上，否则「同一份权重」会因为 JSON 里多了一个展示字段而换一个身份。

### 组合后的适配器：仍然是合法的 LoRA

一个 ``stack`` 出来的组合是 ``Σ sᵢ·BᵢAᵢ``。这个和**一般不是低秩的**，
但它总能写成低秩形式：把各成员沿秩的方向拼起来

```
B_c = [ s₁B₁ | s₂B₂ | … | s_kB_k ]        # [in, k·rank]
A_c = [ A₁ ; A₂ ; … ; A_k ]               # [k·rank, out]
B_c @ A_c = Σ sᵢ BᵢAᵢ                     # 分块矩阵乘法，恒等
```

于是 ``compositions/<cid>.bin`` **不是一张快照，是一份可以再被组合、再被
FineTuneFactory 读进去的产物**。这条性质有个可执行的后果：
:mod:`audit` 的 ``disk_roundtrip`` 可以核对「从磁盘读回来的组合」与
「内存里算出来的组合」给出同一个前向 —— 若导出写成「把合并后的权重整个存下来」，
那条核对就退化成了自证。

**缩放被折进 B**，导出的 ``alpha`` 取 ``rank_total``（即 ``scale = 1.0``）。
折进去而不是留在 alpha 里，是因为 ``stack`` 与 ``merge`` 的差别恰恰在系数上：
留成一个统一的 scale，两者的产物就会长得一模一样。

### 一次往返是有损的，两次不是

``float64 → float32`` 第一次会丢精度（前向相差 ~1e-7）。所以「磁盘往返」
**不能**断言「读回来的浮点值与写下去的相等」—— 那条断言必然失败。
能断言的、也确实该断言的是：

1. 字节逐位相同（``sha256(读回的字节) == digest``）；
2. **一次往返是幂等的** —— ``f32(f32(x)) == f32(x)``，所以第二次写盘与第一次
   逐字节相同。:func:`to_f32` 就是这条性质的具名形式。
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .net import LORA_ABLE_MODULES, Adapter, ModelShape
from .rng import canonical_json, sha256_bytes

__all__ = [
    "ZooRecord",
    "pack",
    "unpack",
    "to_f32",
    "flatten",
    "shape_of",
    "write_artifact",
    "read_artifact",
    "artifact_paths",
    "compose_factors",
    "adapter_from_record",
    "load_adapter",
    "canonical_record",
    "recipe_mismatch",
    "VARIANT_QV",
    "VARIANT_ALL",
    "VARIANT_TARGETS",
    "AdapterSpec",
    "default_zoo",
]

DTYPE = "float32"
ENDIAN = "little"
LAYOUT = "concatenated row-major, no header"


def shape_of(value: Any) -> list[int]:
    """张量形状。**只认到两层** —— 参数一定是 1D 或 2D。"""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"张量必须是 list/tuple，得到 {type(value).__name__}")
    if value and isinstance(value[0], (list, tuple)):
        inner = len(value[0])
        for row in value:
            if len(row) != inner:
                raise ValueError("张量是锯齿状的：各行列数不同，无法按行主序拼接")
        return [len(value), inner]
    return [len(value)]


def flatten(value: Any, out: list[float]) -> None:
    """展平成 float 序列（行主序）。"""
    if value and isinstance(value[0], (list, tuple)):
        for row in value:
            for v in row:
                out.append(float(v))
    else:
        for v in value:
            out.append(float(v))


def to_f32(x: float) -> float:
    """把一个 float64 量化到 float32 再回到 Python float。

    **这是「一次往返是幂等的」这条性质的具名形式。** 没有它，测试只能写成
    「读回来与原来差不多」，那是一条既可能永远通过、也可能随机失败的断言。
    """
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def pack(tensors: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    """``{名字: 张量}`` → ``(裸字节, 旁证元数据)``。**顺序即拼接顺序**（dict 保序）。"""
    if not tensors:
        raise ValueError("pack 收到空的张量表")
    names: list[str] = []
    shapes: list[list[int]] = []
    offsets: list[int] = []
    chunks: list[bytes] = []
    off = 0
    for name, value in tensors.items():
        flat: list[float] = []
        flatten(value, flat)
        blob = struct.pack(f"<{len(flat)}f", *flat)
        names.append(name)
        shapes.append(shape_of(value))
        offsets.append(off)
        chunks.append(blob)
        off += len(blob)
    payload = b"".join(chunks)
    return payload, {
        "names": names,
        "shapes": shapes,
        "offsets": offsets,
        "dtype": DTYPE,
        "endian": ENDIAN,
        "layout": LAYOUT,
        "n_floats": sum(len(c) // 4 for c in chunks),
    }


def unpack(payload: bytes, meta: Mapping[str, Any]) -> dict[str, list[Any]]:
    """裸字节 + 元数据 → ``{名字: 张量}``。

    **每一个字段都要核对，绝不「读不到就当没有」。** 偏移写错了不会抛异常，
    它只会把张量切到错的地方 —— 而切出来的还是一堆合法的浮点数，
    于是「同一个 Adapter」静默地变成了另一个。这里的每一道检查都对应一种
    曾经真实发生过的错位：偏移越界、偏移不是升序、形状与字节数对不上。

    ⚠️ ``offsets`` 是**字节**偏移，不是元素偏移。写成元素偏移在 rank=4、
    形状都很小的本项目里只会错位一点点，直到某个张量的长度恰好让它越界。
    """
    req = ("names", "shapes", "offsets", "dtype", "endian", "n_floats")
    missing = [k for k in req if k not in meta]
    if missing:
        raise ValueError(f"旁证元数据缺字段 {missing}")
    if meta["dtype"] != DTYPE:
        raise ValueError(f"只认识 {DTYPE}，得到 {meta['dtype']!r}")
    if meta["endian"] != ENDIAN:
        raise ValueError(f"只认识 {ENDIAN} endian，得到 {meta['endian']!r}")
    names: list[str] = list(meta["names"])
    shapes: list[list[int]] = [list(s) for s in meta["shapes"]]
    offsets: list[int] = [int(o) for o in meta["offsets"]]
    if not (len(names) == len(shapes) == len(offsets)):
        raise ValueError(f"names/shapes/offsets 长度不一致：{len(names)}/{len(shapes)}/{len(offsets)}")
    if len(set(names)) != len(names):
        raise ValueError("names 里有重复：同一个张量名会出现两次，读的时候会静默丢掉一个")
    if len(payload) % 4 != 0:
        raise ValueError(f"裸字节长度 {len(payload)} 不是 4 的倍数，不可能是 float32 序列")
    if len(payload) // 4 != int(meta["n_floats"]):
        raise ValueError(f"n_floats={meta['n_floats']} 与实际浮点数 {len(payload) // 4} 不符")
    for i, off in enumerate(offsets):
        if off % 4 != 0:
            raise ValueError(f"第 {i} 个偏移 {off} 不是 4 的倍数（字节偏移一定是 4 的倍数）")
        if i and off <= offsets[i - 1]:
            raise ValueError(f"偏移不是严格升序：{offsets[i - 1]} → {off}")

    out: dict[str, list[Any]] = {}
    for i, name in enumerate(names):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(payload)
        n = end - start
        if n % 4 != 0:
            raise ValueError(f"{name}: 切片长度 {n} 不是 4 的倍数")
        vals = list(struct.unpack(f"<{n // 4}f", payload[start:end]))
        want = math.prod(shapes[i])
        if want != len(vals):
            raise ValueError(f"{name}: 形状 {shapes[i]} 需要 {want} 个数，切片给出 {len(vals)} 个")
        out[name] = _reshape(vals, shapes[i])
    return out


def _reshape(vals: Sequence[float], shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return list(vals)
    rows, cols = int(shape[0]), int(shape[1])
    return [list(vals[r * cols : (r + 1) * cols]) for r in range(rows)]


# ----------------------------------------------------------------------
# 磁盘
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ZooRecord:
    """一份 Adapter 产物的旁证元数据（``.json`` 的内容）。"""

    name: str
    path_bin: str
    digest: str
    n_bytes: int
    n_params: int
    targets: list[str]
    rank: int
    alpha: float
    capability: str
    variant: str
    base_key: str
    base_seed: int
    source: str
    payload: Mapping[str, Any]
    #: 这个标签是不是「组合出来的」而非「训练出来的」。它决定报告里的分组。
    composed_of: list[str]
    #: 训练设置与训练报告。**它们不参与任何身份比较**（见 :func:`canonical_record`），
    #: 但必须落盘 —— 「这个 Adapter 是哪来的」是一个必须永远可回答的问题，
    #: 而答案不是它的名字。
    train: Mapping[str, Any] = field(default_factory=dict)
    train_report: Mapping[str, Any] = field(default_factory=dict)
    #: **被请求的那份配方**（:meth:`AdapterSpec.to_dict` 的内容），原样落盘。
    #:
    #: 它与上面两个字段的分工很具体：``train`` 记「怎么训的」，``spec`` 记
    #: 「要的是哪一份」。两者都**不进身份比较**（同一份权重可以是不同配方
    #: 的输出），但**缺了后者，``produce`` 就没法回答「磁盘上这份产物是不是
    #: 我这次要的」** —— 而那个问题以前是靠「文件在不在」回答的。
    #:
    #: 后果不是「多训一次」那么轻：``rank`` 从 4 改成 8 之后，8 个产物被
    #: 原样沿用，配置、报告、registry 三处都写着 rank=8，而权重全是 rank=4 训出来的。
    #: 参数量、MACs、rank_total 全跟着错，且**没有任何一处会不一致**。
    spec: Mapping[str, Any] = field(default_factory=dict)

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.path_bin,
            "digest": self.digest,
            "bytes": self.n_bytes,
            "n_params": self.n_params,
            "targets": list(self.targets),
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "capability": self.capability,
            "variant": self.variant,
            "base": {"key": self.base_key, "seed": self.base_seed},
            "source": self.source,
            "composed_of": list(self.composed_of),
            "meta": {"payload": dict(self.payload)},
            "spec": dict(self.spec),
            "train": dict(self.train),
            "train_report": dict(self.train_report),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ZooRecord":
        """**显式取字段，绝不 ``(**d)``**。``to_dict`` 里有 ``scale`` 这种派生字段，
        ``(**d)`` 会直接 ``TypeError`` —— 而这条路径只有读盘时才走。"""
        payload = d["meta"]["payload"] if "meta" in d else d["payload"]
        base = d.get("base") or {}
        return cls(
            name=str(d["name"]),
            path_bin=str(d["file"]),
            digest=str(d["digest"]),
            n_bytes=int(d["bytes"]),
            n_params=int(d["n_params"]),
            targets=[str(t) for t in d["targets"]],
            rank=int(d["rank"]),
            alpha=float(d["alpha"]),
            capability=str(d.get("capability", "")),
            variant=str(d.get("variant", "")),
            base_key=str(base.get("key", "")),
            base_seed=int(base.get("seed", 0)),
            source=str(d.get("source", "")),
            payload=payload,
            composed_of=[str(c) for c in (d.get("composed_of") or [])],
            train=dict(d.get("train") or {}),
            train_report=dict(d.get("train_report") or {}),
            # 老产物没有这个键 ⇒ 空字典 ⇒ 与任何配方都不相等 ⇒ 会被判成
            # 「必须重训」。这是**故意**的：一份没有记下配方来源的产物，
            # 没资格被当成「就是你要的那一份」。
            spec=dict(d.get("spec") or {}),
        )


def artifact_paths(dir_or_stem: str | Path) -> tuple[Path, Path]:
    """``foo`` → ``(foo.bin, foo.json)``；``foo.bin`` / ``foo.json`` 也接受。"""
    p = Path(dir_or_stem)
    suf = p.suffix.lower()
    if suf == ".bin":
        return p, p.with_suffix(".json")
    if suf == ".json":
        return p.with_suffix(".bin"), p
    return p.with_suffix(".bin"), p.with_suffix(".json")


def write_artifact(
    stem: str | Path,
    tensors: Mapping[str, Any],
    *,
    name: str,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    capability: str = "",
    variant: str = "",
    base_key: str = "",
    base_seed: int = 0,
    source: str = "produce",
    composed_of: Sequence[str] = (),
    train: Mapping[str, Any] | None = None,
    train_report: Mapping[str, Any] | None = None,
    spec: Mapping[str, Any] | None = None,
) -> ZooRecord:
    """写 ``<stem>.bin`` + ``<stem>.json``，返回旁证元数据（含 digest）。"""
    path_bin, path_json = artifact_paths(stem)
    payload, meta = pack(tensors)
    digest = sha256_bytes(payload)
    rec = ZooRecord(
        name=name,
        path_bin=path_bin.name,
        digest=digest,
        n_bytes=len(payload),
        n_params=sum(math.prod(s) for s in meta["shapes"]),
        targets=[str(t) for t in targets],
        rank=int(rank),
        alpha=float(alpha),
        capability=capability,
        variant=variant,
        base_key=base_key,
        base_seed=int(base_seed),
        source=source,
        payload=meta,
        composed_of=[str(c) for c in composed_of],
        train=dict(train or {}),
        train_report=dict(train_report or {}),
        spec=dict(spec or {}),
    )
    path_bin.parent.mkdir(parents=True, exist_ok=True)
    path_bin.write_bytes(payload)
    path_json.write_text(
        json.dumps(rec.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return rec


def read_artifact(stem: str | Path, *, verify: bool = True) -> tuple[dict[str, list[Any]], ZooRecord]:
    """读回 ``(张量表, 旁证元数据)``。``verify=True`` 时核对字节与 digest。"""
    path_bin, path_json = artifact_paths(stem)
    if not path_bin.is_file():
        raise FileNotFoundError(f"产物不存在：{path_bin}")
    if not path_json.is_file():
        raise FileNotFoundError(f"旁证元数据不存在：{path_json}（没有它，裸字节无法解释）")
    rec = ZooRecord.from_dict(json.loads(path_json.read_text(encoding="utf-8")))
    payload = path_bin.read_bytes()
    if verify:
        got = sha256_bytes(payload)
        if got != rec.digest:
            raise ValueError(f"{path_bin.name} 的 sha256 {got[:12]}… 与旁证里的 digest {rec.digest[:12]}… 不符")
        if len(payload) != rec.n_bytes:
            raise ValueError(f"{path_bin.name} 字节数 {len(payload)} 与旁证 {rec.n_bytes} 不符")
    return unpack(payload, rec.payload), rec


def adapter_from_record(tensors: Mapping[str, Any], rec: ZooRecord, *, label: str = "") -> Adapter:
    """字节 → :class:`~adapterfactory.net.Adapter`。**键名要去掉张量后缀**。

    磁盘上是 ``layers.0.q_proj.lora_A`` 这样的全名，内存里 :class:`Adapter` 用的是
    ``{全名: [A, B]}`` 两层结构 —— 转换发生在这里，只发生一次。
    """
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        head, sep, tail = key.rpartition(".")
        if not sep or tail not in ("lora_A", "lora_B"):
            raise ValueError(f"产物里有不认识的张量名 {key!r}（应当是 <挂点>.lora_A / .lora_B）")
        grouped.setdefault(head, {})[tail] = value
    out: dict[str, Any] = {}
    for head, pair in grouped.items():
        if set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(f"{head} 缺少 {'lora_B' if 'lora_A' in pair else 'lora_A'}：低秩分解必须两个因子都在")
        # 内存里的顺序是 [A, B]；与 net.Adapter.delta 的读取顺序一致。
        out[head + ".lora_A"] = pair["lora_A"]
        out[head + ".lora_B"] = pair["lora_B"]
    return Adapter(
        tensors=out,
        targets=tuple(rec.targets),
        rank=int(rec.rank),
        alpha=float(rec.alpha),
        label=label or rec.name,
        capability=rec.capability,
    )


def compose_factors(
    adapters: Sequence[Adapter],
    coefs: Sequence[float],
    base: ModelShape,
) -> tuple[dict[str, list[Any]], int]:
    """把若干适配器折成**一份**合法的低秩适配器。返回 ``(张量表, 总秩)``。

    这是「组合后的产物仍然可读」这条设计的落点，见模块 docstring。

    系数折进 ``B``：``B_c[:, r] = cᵢ · Bᵢ[:, r]``。折进 ``B`` 而不是 ``A``
    是任意的，但**必须只折一边** —— 折进两边会变成 ``cᵢ²``，
    而那个错误在 ``cᵢ = 1`` 时（也就是 ``stack``）完全看不出来。
    """
    if len(adapters) != len(coefs):
        raise ValueError(f"系数个数 {len(coefs)} != 适配器个数 {len(adapters)}")
    if not adapters:
        raise ValueError("compose_factors 收到空的适配器表")
    rank_total = sum(int(a.rank) for a in adapters)
    orders = {int(a.rank) for a in adapters}
    if len(orders) != 1:
        # 混秩其实可以支持（只是拼的时候秩不同），但本项目的设计空间里不会出现，
        # 而**静默地当成同秩处理**会把不同秩的成员拼错位置。
        raise ValueError(f"组合的成员必须同秩，得到 {sorted(orders)}")
    if not base.tie_embeddings:
        raise ValueError("只有权重共享的基座才能导出组合产物：输出投影也要拼")

    # 挂点取并集，**按基座参数的落盘顺序**排 —— 顺序即产物字节顺序，
    # 用 set 迭代序会让同一份组合在不同进程里长出不同的字节。
    all_targets = sorted({t for a in adapters for t in a.targets}, key=_target_order)
    out: dict[str, list[Any]] = {}
    for t in all_targets:
        for layer in range(base.n_layers):
            full = f"layers.{layer}.{t}"
            As: list[list[float]] = []
            Bs_cols: list[list[float]] = []
            for a, c in zip(adapters, coefs):
                if t not in a.targets:
                    continue
                A = a.tensors[full + ".lora_A"]
                B = a.tensors[full + ".lora_B"]
                sc = a.scale * float(c)
                As.extend([list(row) for row in A])
                # B 是 [in, rank]；按列拼 → 转成 [rank, in] 再拼
                cols = [[B[i][r] * sc for i in range(len(B))] for r in range(a.rank)]
                Bs_cols.extend(cols)
            if not As:
                continue
            width = len(As[0])
            for row in As:
                if len(row) != width:
                    raise ValueError(f"{full}: 成员的 A 宽度不一致")
            out[full + ".lora_A"] = As
            for col in Bs_cols:
                if len(col) != len(Bs_cols[0]):
                    raise ValueError(f"{full}: 成员的 B 输入维不一致")
            # 转回 [in, rank_total]
            in_dim = len(Bs_cols[0])
            out[full + ".lora_B"] = [[Bs_cols[r][i] for r in range(len(Bs_cols))] for i in range(in_dim)]
    if not out:
        raise ValueError("组合的一份产物是空的：所有成员都没有落在共同的挂点上")
    return out, rank_total


def _target_order(t: str) -> int:
    """挂点在基座里的规范顺序。**用它而不是字母序** —— 产物字节顺序要跟
    ``net.PARAM_SUFFIXES`` 对齐，否则同一份组合在两次序列化之间会换一个 digest。"""
    from .net import PARAM_SUFFIXES

    try:
        return PARAM_SUFFIXES.index(t)
    except ValueError:
        return len(PARAM_SUFFIXES)


#: 进身份比较时要剥掉的字段。两类：
#:
#: - ``file`` / ``bytes`` —— 描述「文件放哪、多大」，不是「这份权重是什么」。
#:   留着 ``bytes`` 的后果很具体：同一份权重写进 ``data/adapters/`` 与
#:   ``out/compositions/``，两处的旁证会被判成两个不同的东西。
#: - ``train`` / ``train_report`` / ``spec`` —— **过程的记录**。改一次 ``steps``
#:   而权重逐位相同时，它们会变；把它算进身份会让「同一份 Adapter」换一个名字。
#:   ``spec`` 同理：rank=4 与 rank=8 是两份配方，但身份的依据始终是**权重本身**
#:   （digest 锚在裸字节上）—— 配方只是「这份权重是怎么来的」的档案。
_VOLATILE_RECORD_KEYS: tuple[str, ...] = ("file", "bytes", "train", "train_report", "spec")


def canonical_record(rec: ZooRecord) -> str:
    """旁证元数据的规范序列化，剥掉 :data:`_VOLATILE_RECORD_KEYS`。

    **它计算的是「这份权重是什么」，不是「它是怎么来的」。**
    两者都需要被记录（:meth:`ZooRecord.to_dict` 全都写了），
    但只有前者有资格参与身份比较。
    """
    d = rec.to_dict()
    for k in _VOLATILE_RECORD_KEYS:
        d.pop(k, None)
    return canonical_json(d)


def load_adapter(stem: str | Path) -> Adapter:
    """``.bin/.json`` → :class:`~adapterfactory.net.Adapter`（一步到位，供载入路径使用）。"""
    tensors, rec = read_artifact(stem)
    return adapter_from_record(tensors, rec)


def recipe_mismatch(
    rec: ZooRecord,
    *,
    spec: Mapping[str, Any],
    train: Mapping[str, Any] | None = None,
    base_key: str = "",
    base_seed: int | None = None,
) -> str:
    """磁盘上这份产物**是不是**这份配方的输出。返回「不符在哪」，``""`` 表示相符。

    **为什么比对配方，而不是比对「文件在不在」** —— 见 :attr:`ZooRecord.spec`
    上那段。一句话复述：``zoo.rank`` 从 4 改成 8 之后，沿用旧产物会让配置、
    报告、``registry`` 三处都写着 8，而权重是 4 训出来的。参数量、``MACs``、
    ``rank_total`` 全跟着错，**而没有任何一处会自相矛盾** ——
    每个数字单看都合理，只有拿来跟另一处比才不对。而这里比对的就是那件事。

    ``train`` 只核对**传进来的那些键**：调用方给的是配置派生出来的配方
    （:func:`adapterfactory.config.zoo_recipe_of`），``betas`` / ``eps`` 那些
    代码常量不在里面，于是也不会被误判。漏比一个已知键会放过真实的漂移，
    多比一个未知键会把「配方没提」判成「不符」—— 两侧都得防。

    ``base_key=""`` / ``base_seed=None`` 表示**这一项这次不查**。默认不查，
    是因为「在同一根基座上」这件事 :func:`~adapterfactory.registry.load_registry`
    已经单独报错了（它是语义错误，不是配方漂移），两处都报只会让报错信息重复。
    """
    if not rec.spec:
        return (
            "磁盘上这份产物没有配方索引（sidecar 的 spec 为空）—— 它来自"
            "没有记下配方的旧版本，而「没记下」不等于「就是你要的那份」，"
            "必须重训（`python main.py produce` 会自动这么做）"
        )
    got, want = dict(rec.spec), dict(spec)
    for k in sorted(set(got) | set(want)):
        if got.get(k) != want.get(k):
            return f"spec.{k}：磁盘 {got.get(k)!r} ≠ 本次 {want.get(k)!r}"
    if base_key and rec.base_key != base_key:
        return f"base.key：磁盘 {rec.base_key!r} ≠ 本次 {base_key!r}"
    if base_seed is not None and rec.base_seed != base_seed:
        return f"base.seed：磁盘 {rec.base_seed!r} ≠ 本次 {base_seed!r}"
    for k, v in sorted((train or {}).items()):
        if k not in rec.train:
            return f"train.{k}：磁盘上没记 ≠ 本次 {v!r}"
        if rec.train[k] != v:
            return f"train.{k}：磁盘 {rec.train[k]!r} ≠ 本次 {v!r}"
    return ""


# ----------------------------------------------------------------------
# zoo 的**目录定义**：8 个 Adapter 叫什么、挂哪里
# ----------------------------------------------------------------------
#
# 这一段曾经住在 ``produce.py`` 里，那是个错误的位置。``AdapterSpec`` 与
# ``default_zoo`` 是**纯描述**（能力 + 变体 + 秩 + alpha → 一份配方），
# 它们一次训练都不做。放在 producer 里的后果是：只读命令 ``main.py task``
# 为了打印「zoo 规格」这张表，必须 import 整个训练模块。
#
# 于是「只读路径不加载训练代码」这句话就只是文档里的一句话，而不是
# 一件能被检查的事 —— 而 audit 的 ``producer_contained`` 检查的正是后者。
# 搬到 ``zoo.py`` 之后，producer 只被 ``cli._cmd_produce`` 一个函数够得着，
# ``main.py task`` 走的路径上一个训练符号都没有。

#: 全模块覆盖的变体名。两个变体的差别就是**覆盖了多少挂点** —— 这是
#: 「同能力、不同覆盖」那组对照的全部内容。
VARIANT_QV = "qv"
VARIANT_ALL = "all"

#: 变体 → 挂点。``qv`` 只动注意力里的 value/query 投影，``all`` 动全部 7 个模块。
VARIANT_TARGETS: dict[str, tuple[str, ...]] = {
    VARIANT_QV: ("q_proj", "v_proj"),
    VARIANT_ALL: tuple(LORA_ABLE_MODULES),
}


@dataclass(frozen=True)
class AdapterSpec:
    """一个 Adapter 的**配方**。它是设计空间的原子，也是 zoo 的目录项。"""

    capability: str
    variant: str
    rank: int
    alpha: float

    @property
    def label(self) -> str:
        return f"{self.capability}-{self.variant}"

    @property
    def targets(self) -> tuple[str, ...]:
        return VARIANT_TARGETS[self.variant]

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "capability": self.capability,
            "variant": self.variant,
            "targets": list(self.targets),
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
        }


def default_zoo(capabilities: Sequence[str], *, rank: int, alpha: float) -> list[AdapterSpec]:
    """8 个 Adapter = 4 个能力 × 2 个覆盖变体。**顺序即报告里的顺序。**

    顺序按能力在索引里的顺序、再按变体，不用 ``sorted`` —— 字母序会让
    ``upper-all`` 排在 ``upper-qv`` 前面，于是 zoo 的目录序与设计的意图序不一致。
    """
    out: list[AdapterSpec] = []
    for cap in capabilities:
        for variant in (VARIANT_QV, VARIANT_ALL):
            out.append(AdapterSpec(capability=cap, variant=variant, rank=int(rank), alpha=float(alpha)))
    return out
