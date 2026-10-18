"""确定性随机与内容寻址（本项目可复现性的唯一底座）。

**禁止内置 ``hash()``。** ``PYTHONHASHSEED`` 会让它跨进程不稳定 —— 同一个配置
在两次运行里得到不同的种子，于是「同一份配置必得同一个组合矩阵」这句话会
**静默地**变成假的（它只在重启解释器之后才现形，而那时你不会怀疑种子）。

这里的每一样东西都只依赖 ``sha256``，所以它是**跨进程、跨机器、跨平台**稳定的。

**为什么需要 :func:`derive_seed` 而不是直接用主种子。** 组合空间里每个
Adapter、每个能力、每个探针集都各自需要一个种子。若它们共用主种子，
「换一个 Adapter 的种子」会同时改掉能力集与探针集 —— 那就不是一个受控比较了
（skill 坑 71：比较实验里唯一变量必须是你想比的那个变量）。

分隔符 ``\\x1f``（单元分隔符）不是装饰：``derive_seed("a:b", "c")`` 与
``derive_seed("a", "b:c")`` 必须给出不同的种子，否则带分隔符的字段名会在
某个巧合下撞车。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

__all__ = [
    "SEP",
    "canonical_json",
    "sha256_text",
    "sha256_bytes",
    "stable_hash64",
    "short_digest",
    "derive_seed",
    "digest_of",
    "join_digest",
    "quantize_for_hash",
]

#: 种子派生与摘要拼接的分隔符。见模块 docstring。
SEP = "\x1f"


def _fallback(obj: Any) -> Any:
    """``canonical_json`` 的兜底序列化。

    ``tuple`` / ``set`` 要降成 ``list``：``(1, 2)`` 与 ``[1, 2]`` 在 JSON 里
    必须是同一个东西，否则「同一份 IR 的不同表示给出不同指纹」——
    而这两种写法在本项目里都会出现（配置里写 list，dataclass 里存 tuple）。
    """
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"canonical_json 不知道怎么序列化 {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """稳定序列化：键排序、紧凑分隔符、拒绝 NaN。

    ``allow_nan=False`` 是刻意的：NaN 一旦进了指纹，两件不同的事会得到同一个
    摘要（``nan != nan`` 让比较永远失败），而报告里那些 NaN 列本来就是
    「算不出来」，它**不该**参与身份计算。
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=_fallback,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_hash64(*parts: Any) -> int:
    """把任意多个部件折成一个 64 位整数（稳定、跨进程）。"""
    return int(sha256_text(SEP.join(str(p) for p in parts))[:16], 16)


def short_digest(full: str, length: int = 12) -> str:
    """只用于**展示**的短摘要。身份比较一律用全长。"""
    return full[:length]


def derive_seed(*parts: Any) -> int:
    """从任意多个部件派生一个种子。**只依赖部件本身，不依赖遍历顺序。**

    调用点的口径（全项目统一）：
    ``derive_seed("init", 参数名, 模型名, seed)`` / ``derive_seed("lora", 名字, seed)`` /
    ``derive_seed("eval", 能力, seed)`` / ``derive_seed("probe", 能力, seed)``。
    """
    return stable_hash64(*parts)


def digest_of(obj: Any) -> str:
    """对象 → 内容寻址摘要（全长 hex）。"""
    return sha256_text(canonical_json(obj))


def join_digest(parts: Iterable[str]) -> str:
    """把若干摘要合成一个。**先排序** —— 否则拼接顺序会变成隐藏输入。"""
    return sha256_text(SEP.join(sorted(parts)))


def quantize_for_hash(value: float, digits: int = 6) -> int:
    """把浮点量化成整数再进指纹。

    浮点的最后几位会随平台/编译选项漂移，而本项目里有些量是**推导出来的
    配置**（比如 ``scale = alpha / rank``）。直接拿它进指纹会让「同一份配置」
    在不同机器上得到不同身份 —— 这正是「内容寻址」最不该出的错。
    """
    return int(round(float(value) * 10**digits))
