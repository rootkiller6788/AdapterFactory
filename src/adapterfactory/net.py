"""微型张量引擎 + 基座模型：**本项目里唯一会碰数值的东西**。

### 这个文件在边界上的位置

AdapterFactory 的核心命题是「Adapter 产出**之后**的生命周期」—— 登记、组合、
Merge、冲突、部署。**本项目不训练。** 这条边界由三件事共同守住：

1. :mod:`audit` 用 ``tokenize`` 扫 token，核对流水线模块里没有 ``train`` / ``backward``；
2. 本模块的反向传播**只被 :mod:`produce` 使用**（``produce`` 是 FineTuneFactory 的替身，
   在架构墙的另一侧）；
3. 组合与评测走的全是 :func:`forward` / :func:`forward_with_adapters` —— **纯前向**。

于是「本项目不训练」不是一句文档承诺，而是一张可执行的表。

### 张量表示（与同工作区 L2/L3 各项目一致）

- 张量 = 嵌套 ``list``（1D: ``list[float]``；2D: ``list[list[float]]``）。
- 反向 = 每个算子自带 backward 闭包 + 反向拓扑遍历。
- **内积一律走 :func:`math.sumprod`**（标准库，C 实现）。组合空间有几百个配置，
  每个配置要过 4 个能力 × 若干探针 —— 手写三重循环会让流水线从一分钟变成几十分钟。

### LoRA 的形状约定（**与 PyTorch 相反，别按记忆写**）

本模型所有线性权重是 ``[in, out]``（前向是 ``x @ W``）。于是适配器的两个因子是：

```
A: [rank, out_dim]        B: [in_dim, rank]        delta = scale · (B @ A)   # [in, out]
```

:func:`merge_into` 里落的就是这个式子。写反了**不会报错** —— 形状检查会放过
``[out,rank] @ [rank,in]``，只是结果是一个完全不同的矩阵，而「组合后能力变了」
在本项目里是一个**合法结论**（见 README 的干扰假设），两者在输出上一模一样。

### 浮点口径

``merge_error`` 比的是**「合并进权重再前向」**与**「不合并、逐层把 delta 加在
激活上」**两条路径。两者在实数上恒等（``x@W + Σ x@Δᵢ == x@(W + ΣΔᵢ)``），
差别只有累加顺序，实测在 1e-15 量级。**它证明的是「合并没有引入实现错误」，
不是「合并没有损失能力」** —— 后者是 :mod:`conflict` 里的 ``merge_loss``，
两者是两件事，报告里必须分开写。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Mapping, Sequence

from .rng import derive_seed

#: C 速度的内积。``math.sumprod`` 自 CPython 3.12 起提供。
_dot = math.sumprod

__all__ = [
    # 引擎
    "Tensor",
    "constant",
    "mac_counter",
    "MacCounter",
    # 模型结构
    "ModelShape",
    "EMB",
    "NORM_F",
    "PARAM_SUFFIXES",
    "LORA_ABLE_MODULES",
    "MASK_NEG",
    "ROPE_BASE",
    "layer_param_names",
    "lora_param_names",
    "param_shapes",
    "count_params",
    "init_base_weights",
    "causal_mask",
    "forward",
    "forward_with_adapters",
    "forward_routed",
    "forward_train",
    "routed_mixer",
    "gates_by_module",
    "gate_coeffs",
    "derive_gates",
    "mix_rows",
    "lora_delta_map",
    "logits_of",
    "merge_into",
    "expand_targets",
    "param_shapes_of",
    "max_abs_diff",
    "adapter_shapes",
    "init_adapter",
    "Adapters",
    "Adapter",
]

#: 损失序列的掩码填充值。**用 -1e9 不用 -inf**：``-inf`` 进 softmax 会让
#: ``exp(-inf - max)`` 变成 ``0 * inf``，一旦某行全是掩码就出 NaN。
MASK_NEG = -1e9

#: RoPE 的底。它必须是一个**具名常量**，不能是散落在算式里的 10000.0。
ROPE_BASE = 10000.0

EMB = "emb"
NORM_F = "norm_f"

#: 每个 Transformer 块里的非适配器参数，**顺序即落盘顺序**（量化分组与产物字节序都看它）。
PARAM_SUFFIXES: tuple[str, ...] = (
    "norm1",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "norm2",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: 可以被适配器挂载的模块。**注意这里没有 ``norm1``/``norm2``/``norm_f``/``emb``** ——
#: 归一化的缩放向量不是矩阵，挂不了 ``B@A`` 这种低秩分解。
LORA_ABLE_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: 全模块覆盖的简称，出现在配置文件里。
ALL_MODULES = LORA_ABLE_MODULES


# ======================================================================
# 一、引擎
# ======================================================================
def _is2d(x: Sequence[Any]) -> bool:
    return isinstance(x, list) and bool(x) and isinstance(x[0], list)


def _is1d(x: Any) -> bool:
    return isinstance(x, list)


def _zeros_like(x: Any) -> Any:
    if not _is1d(x):
        return 0.0
    if _is2d(x):
        return [[0.0] * len(row) for row in x]
    return [0.0] * len(x)


def _ones_like(x: Any) -> Any:
    """标量也要能走这里 —— 损失是一个 float，而「直接对损失求导」是梯度校验的第一步。"""
    if not _is1d(x):
        return 1.0
    if _is2d(x):
        return [[1.0] * len(row) for row in x]
    return [1.0] * len(x)


def _cols(x: Sequence[Any]) -> int:
    if not x:
        return 0
    return len(x[0]) if _is2d(x) else 1  # type: ignore[arg-type]


class MacCounter:
    """累计引擎**实际执行**的乘加次数。

    它数的是「真的跑了多少」，不是「按公式应该跑多少」。组合后的 ``parameter_cost``
    与 ``latency`` 两条轴都从这里来 —— 于是「组合的代价」是**量出来的**，
    不是按参数量乘一个系数**估出来的**。
    """

    def __init__(self) -> None:
        self.by_op: dict[str, int] = {}
        self.total: int = 0

    def add(self, op: str, macs: int) -> None:
        self.by_op[op] = self.by_op.get(op, 0) + int(macs)
        self.total += int(macs)

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "by_op": dict(sorted(self.by_op.items()))}


_ACTIVE_COUNTER: MacCounter | None = None


class _MacCounterContext:
    def __enter__(self) -> MacCounter:
        global _ACTIVE_COUNTER
        self._saved = _ACTIVE_COUNTER
        _ACTIVE_COUNTER = MacCounter()
        return _ACTIVE_COUNTER

    def __exit__(self, *exc: object) -> bool:
        global _ACTIVE_COUNTER
        _ACTIVE_COUNTER = self._saved
        return False


def mac_counter() -> _MacCounterContext:
    """``with mac_counter() as c: ...`` —— 只统计这个块里跑的算子。"""
    return _MacCounterContext()


def _count(op: str, macs: int) -> None:
    if _ACTIVE_COUNTER is not None:
        _ACTIVE_COUNTER.add(op, macs)


class Tensor:
    """一个带梯度的值。``data`` 与 ``grad`` 同形。叶子用 :func:`constant` 造。"""

    __slots__ = ("data", "grad", "name", "_backward", "_prev", "_frozen")

    def __init__(self, data: Any, *, name: str | None = None, prev: Sequence["Tensor"] = ()) -> None:
        self.data = data
        self.grad: Any = None
        self.name = name
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(prev)
        self._frozen = False

    @property
    def shape(self) -> tuple[int, ...]:
        if _is2d(self.data):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    @property
    def numel(self) -> int:
        if _is2d(self.data):
            return len(self.data) * (len(self.data[0]) if self.data else 0)
        return len(self.data)

    def zero_grad(self) -> None:
        self.grad = None

    def freeze(self) -> "Tensor":
        """标记为「不接收梯度」。用于 :mod:`produce` 里被冻结的基座叶子。"""
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def backward(self, seed: Any | None = None) -> None:
        """反向传播。**只有 :mod:`produce` 会调它**（见模块 docstring 的边界说明）。"""
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(t: Tensor) -> None:
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                build(p)
            topo.append(t)

        build(self)
        self.grad = _ones_like(self.data) if seed is None else seed
        for t in reversed(topo):
            t._backward()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        tag = f" name={self.name!r}" if self.name else ""
        return f"Tensor(shape={self.shape}{tag})"


def _copy_grad(g: Any) -> Any:
    """按梯度形状拷一层。**这个拷贝不是性能优化，是正确性。**

    梯度是嵌套 list，而 list 可变。``t.grad = g`` 挂上去的是**消费者的那个对象**，
    而同一个 ``_bw`` 常常把自己收到的那一份分给多个输入（``add`` 最典型），
    于是两个不同的节点从此共享同一个 list —— 之后任何一次原地累加都会同时改掉另一个。

    这个错误不报错、不产生 NaN、形状也一直对。它只在「一个节点有多个消费者、
    且它的兄弟与它共享了同一份梯度对象」时出现，模型越大越容易撞上，
    而单看某条曲线又完全正常。
    """
    if _is2d(g):
        return [list(row) for row in g]
    return list(g)


def _acc(t: Tensor, g: Any) -> None:
    """梯度累加。**必须累加不能覆盖** —— 一个张量被用两次时覆盖会静默少一半梯度。

    ⚠️ 首次赋值也必须**拷贝**（见 :func:`_copy_grad`）。
    """
    if t._frozen:
        return
    if t.grad is None:
        t.grad = _copy_grad(g)
    elif _is2d(g):
        for i, row in enumerate(g):
            for j, v in enumerate(row):
                t.grad[i][j] += v
    elif isinstance(g, list):
        for i, v in enumerate(g):
            t.grad[i] += v
    else:
        t.grad += g


def _as_tensor(x: Any, *, name: str | None = None) -> Tensor:
    """把裸 list 包成常量。**所有算子都从这里进来。**

    这是「冻结基座 = 图里的常量」这条设计的落点：调用点不需要知道
    「这个权重是可训练的还是冻结的」，它只需要知道「这是个值」。
    """
    if isinstance(x, Tensor):
        return x
    return constant(x, name=name)


def constant(data: Any, *, name: str | None = None) -> Tensor:
    return Tensor(data, name=name)


# ---- 线性代数 ----------------------------------------------------------
def matmul(a: Any, b: Any) -> Tensor:
    """2D 矩阵乘。MACs = ``rows(a) × cols(a) × cols(b)``。"""
    ta, tb = _as_tensor(a), _as_tensor(b)
    A, B = ta.data, tb.data
    if not _is2d(A) or not _is2d(B):
        raise ValueError(f"matmul 只支持 2D，得到 {ta.shape} × {tb.shape}")
    m, k, n = len(A), len(B), len(B[0]) if B else 0
    if len(A[0]) != k:
        raise ValueError(f"matmul 形状不匹配：{ta.shape} × {tb.shape}")
    Bt = list(zip(*B))
    out = [[_dot(row, col) for col in Bt] for row in A]
    _count("matmul", m * k * n)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        if not ta._frozen:
            _acc(ta, [[_dot(row, col) for col in B] for row in G])
        if not tb._frozen:
            At, Gt = list(zip(*A)), list(zip(*G))
            _acc(tb, [[_dot(row, col) for col in Gt] for row in At])

    t._backward = _bw
    return t


def transpose(a: Any) -> Tensor:
    ta = _as_tensor(a)
    out = [list(col) for col in zip(*ta.data)]
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, [list(col) for col in zip(*t.grad)])

    t._backward = _bw
    return t


def split_cols(x: Any, sizes: Sequence[int]) -> tuple[Tensor, ...]:
    """沿最后一维切块。**注意力按头切分靠它。**

    ⚠️ **每个输出各自负责自己那一段。** 引擎对**每一个节点**各调一次 ``_backward``
    （N 个输出就是 N 个节点），若让所有输出共用一个「把整条梯度拼回来」的闭包，
    同一条合并梯度会被累加 N 次 —— ``tx.grad`` 直接变成 **N 倍**。
    按头切分时 ``nh=4``，于是 ``q`` 的梯度是 4 倍：它让 ``norm1`` 的梯度在每一层
    都不对，而 ``q_proj``/``k_proj``/``v_proj`` 看起来是对的；损失曲线、准确率
    全都正常 —— 只有梯度数值差分能照出它。
    """
    tx = _as_tensor(x)
    width = _cols(tx.data)
    if sum(sizes) != width:
        raise ValueError(f"split_cols 宽度对不上：{width} vs sum({list(sizes)})")
    parts: list[list[Any]] = []
    offsets: list[int] = []
    off = 0
    for s in sizes:
        offsets.append(off)
        parts.append([row[off : off + s] for row in tx.data])
        off += s
    outs = tuple(Tensor(p, prev=(tx,)) for p in parts)

    def _make_bw(o: Tensor, start: int, s: int) -> Any:
        def _bw() -> None:
            if o.grad is None:
                return
            # 只填自己那一段，其余补 0：N 个输出各累加一次，合起来正好是一次拼接。
            rows = len(tx.data)
            merged = [[0.0] * width for _ in range(rows)]
            for i in range(rows):
                for j in range(s):
                    merged[i][start + j] = o.grad[i][j]
            _acc(tx, merged)

        return _bw

    for o, start, s in zip(outs, offsets, sizes):
        o._backward = _make_bw(o, start, s)
    return outs


def concat_cols(parts: Sequence[Any]) -> Tensor:
    ts = tuple(_as_tensor(p) for p in parts)
    rows = len(ts[0].data)
    sizes = [len(t.data[0]) for t in ts]
    out = [sum((t.data[i] for t in ts), []) for i in range(rows)]
    t = Tensor(out, prev=ts)

    def _bw() -> None:
        if t.grad is None:
            return
        off = 0
        for tt, s in zip(ts, sizes):
            _acc(tt, [row[off : off + s] for row in t.grad])
            off += s

    t._backward = _bw
    return t


# ---- 逐元素 ------------------------------------------------------------
def add(a: Any, b: Any) -> Tensor:
    ta, tb = _as_tensor(a), _as_tensor(b)
    if ta.shape != tb.shape:
        raise ValueError(f"add 形状不匹配：{ta.shape} vs {tb.shape}")
    if _is2d(ta.data):
        out = [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(ta.data, tb.data)]
    else:
        out = [x + y for x, y in zip(ta.data, tb.data)]
    _count("add", ta.numel)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, t.grad)
        _acc(tb, t.grad)

    t._backward = _bw
    return t


def mul(a: Any, b: Any) -> Tensor:
    """逐元素相乘（同形，或 1D 沿最后一维广播）。"""
    ta, tb = _as_tensor(a), _as_tensor(b)
    if ta.shape == tb.shape:
        if _is2d(ta.data):
            out = [[x * y for x, y in zip(r1, r2)] for r1, r2 in zip(ta.data, tb.data)]
        else:
            out = [x * y for x, y in zip(ta.data, tb.data)]
        bc = False
    elif _is2d(ta.data) and len(tb.data) == _cols(ta.data):
        out = [[x * y for x, y in zip(row, tb.data)] for row in ta.data]
        bc = True
    else:
        raise ValueError(f"mul 形状不匹配：{ta.shape} vs {tb.shape}")
    _count("mul", ta.numel)
    t = Tensor(out, prev=(ta, tb))

    def _bw() -> None:
        if t.grad is None:
            return
        if bc:
            gb = [0.0] * len(tb.data)
            for i, row in enumerate(t.grad):
                for j, g in enumerate(row):
                    gb[j] += g * ta.data[i][j]
            _acc(tb, gb)
            _acc(ta, [[g * y for g, y in zip(row, tb.data)] for row in t.grad])
        else:
            _acc(ta, _ewise(t.grad, tb.data, "*"))
            _acc(tb, _ewise(t.grad, ta.data, "*"))

    t._backward = _bw
    return t


def _ewise(g: Any, other: Any, op: str) -> Any:
    if _is2d(g):
        if _is2d(other):
            return [[(x * y if op == "*" else x / y) for x, y in zip(r1, r2)] for r1, r2 in zip(g, other)]
        return [[(x * y if op == "*" else x / y) for x, y in zip(row, other)] for row in g]
    return [(x * y if op == "*" else x / y) for x, y in zip(g, other)]


def scale(a: Any, s: float) -> Tensor:
    ta = _as_tensor(a)
    out = [[v * s for v in row] for row in ta.data] if _is2d(ta.data) else [v * s for v in ta.data]
    _count("scale", ta.numel)
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        _acc(ta, [[v * s for v in row] for row in t.grad] if _is2d(t.grad) else [v * s for v in t.grad])

    t._backward = _bw
    return t


def mix_rows(parts: Sequence[Any], coeffs: Sequence[Sequence[float]]) -> Tensor:
    """按 token 加权混合若干同形的 ``[T, out]``：``out[t][j] = Σᵢ c[t][i]·parts[i][t][j]``。

    **这是 ``route`` 独占的算子，而且它不能由现成的算子拼出来。**
    :func:`mul` 的广播只沿**最后一维**（一个 ``[out]`` 向量乘整张矩阵），
    而这里要的是**每一行一个不同的系数向量** —— 形状 ``[T, k]`` 对 ``[k, T, out]``。
    用 ``scale`` 循环 k 次再相加，得到的是「全 token 同一个系数」，
    那正是 ``stack`` 的特例而**不是**路由：症状是 route 的数值会与 stack 只差
    一个常数缩放，两者在报告里都「跑通了」。

    它带完整的反向（虽然当前只有推理路径用它）：手写一半的反向会让「借用这个
    算子的下一个人」得到一份静默错误的梯度，而不是一条报错。
    """
    ts = tuple(_as_tensor(p) for p in parts)
    if not ts:
        raise ValueError("mix_rows 至少要一个分量")
    k = len(ts)
    T = len(ts[0].data)
    out_dim = len(ts[0].data[0])
    for i, t in enumerate(ts):
        if len(t.data) != T or (T and len(t.data[0]) != out_dim):
            raise ValueError(f"mix_rows 第 {i} 个分量的形状不是 [{T},{out_dim}]")
    if len(coeffs) != T:
        raise ValueError(f"mix_rows 系数行数 {len(coeffs)} != token 数 {T}")
    for row in coeffs:
        if len(row) != k:
            raise ValueError(f"mix_rows 每行系数个数必须等于分量数 {k}，得到 {len(row)}")

    out = [[_dot([coeffs[t][i] for i in range(k)], [ts[i].data[t][j] for i in range(k)]) for j in range(out_dim)] for t in range(T)]
    _count("mix_rows", T * k * out_dim)
    res = Tensor(out, prev=ts)

    def _bw() -> None:
        if res.grad is None:
            return
        for i in range(k):
            ti = ts[i]
            # d parts[i][t][j] = c[t][i] · grad[t][j]
            _acc(ti, [[coeffs[t][i] * g for g in res.grad[t]] for t in range(T)])
        # 系数的梯度不进图：coeffs 是 softmax 出来的**数据**，不是叶子。
        # 它依赖 gate 向量 uᵢ，而 uᵢ 按设计是种子派生的常量、不训练 ——
        # 若哪天要让 uᵢ 可学，这里必须补一条往 softmax 里走的反向，
        # 而缺了它的症状是「uᵢ 有梯度但恒为 0」，训练曲线看起来只是学得慢。

    res._backward = _bw
    return res


def silu(a: Any) -> Tensor:
    ta = _as_tensor(a)

    def f(v: float) -> float:
        return v / (1.0 + math.exp(-v)) if v > -30 else 0.0

    def df(v: float) -> float:
        if v < -30:
            return 0.0
        s = 1.0 / (1.0 + math.exp(-v))
        return s * (1.0 + v * (1.0 - s))

    out = [[f(v) for v in row] for row in ta.data] if _is2d(ta.data) else [f(v) for v in ta.data]
    _count("silu", ta.numel)
    t = Tensor(out, prev=(ta,))

    def _bw() -> None:
        if t.grad is None:
            return
        if _is2d(ta.data):
            _acc(ta, [[g * df(v) for g, v in zip(gr, vr)] for gr, vr in zip(t.grad, ta.data)])
        else:
            _acc(ta, [g * df(v) for g, v in zip(t.grad, ta.data)])

    t._backward = _bw
    return t


# ---- 归一化与 softmax ---------------------------------------------------
def rmsnorm(x: Any, w: Any, *, eps: float = 1e-6) -> Tensor:
    """``w * x / sqrt(mean(x²) + eps)``。

    分母是 **mean** 不是 sum：用 sum 会让输出尺度随 ``d_model`` 变化，
    于是「同一个组合在不同宽度下表现不同」—— 而本项目要横向比不同配置，
    那种差异会被误读成组合方式的差异。
    """
    tx, tw = _as_tensor(x), _as_tensor(w)
    if not _is2d(tx.data):
        raise ValueError("rmsnorm 只支持 2D 输入")
    d = _cols(tx.data)
    if len(tw.data) != d:
        raise ValueError(f"rmsnorm 宽度不匹配：{tw.shape} vs {tx.shape}")
    out: list[list[float]] = []
    inv: list[float] = []
    for row in tx.data:
        ms = _dot(row, row) / d
        r = 1.0 / math.sqrt(ms + eps)
        inv.append(r)
        out.append([v * r * wi for v, wi in zip(row, tw.data)])
    _count("rmsnorm", tx.numel)
    t = Tensor(out, prev=(tx, tw))

    def _bw() -> None:
        if t.grad is None:
            return
        G = t.grad
        gw = [0.0] * d
        for i, row in enumerate(tx.data):
            for j, v in enumerate(row):
                gw[j] += G[i][j] * v * inv[i]
        _acc(tw, gw)
        gx: list[list[float]] = []
        for i, row in enumerate(tx.data):
            r = inv[i]
            s = 0.0
            for j in range(d):
                s += G[i][j] * tw.data[j] * row[j]
            s = s * (r**3) / d
            gx.append([G[i][j] * tw.data[j] * r - row[j] * s for j in range(d)])
        _acc(tx, gx)

    t._backward = _bw
    return t


def softmax_rows(x: Any) -> Tensor:
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("softmax_rows 只支持 2D 输入")
    out: list[list[float]] = []
    for row in tx.data:
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        s = sum(ex)
        out.append([v / s for v in ex])
    _count("softmax", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        gx: list[list[float]] = []
        for p, g in zip(out, t.grad):
            dot = _dot(p, g)
            gx.append([pi * (gi - dot) for pi, gi in zip(p, g)])
        _acc(tx, gx)

    t._backward = _bw
    return t


def softmax_rows_data(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """只对裸数据做行 softmax（**不进图**）。供路由门使用。"""
    out: list[list[float]] = []
    for row in rows:
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        s = sum(ex)
        out.append([v / s for v in ex])
    return out


def _ce_parts(
    logits_data: Sequence[Sequence[float]],
    targets: Sequence[int],
    mask: Sequence[float] | None,
) -> tuple[list[list[float]], float, float]:
    """交叉熵的共享核心：返回（逐行 softmax 概率、掩码平均损失、分母）。

    抽出来是为了让标量版与批量版**共用同一份数学**。两份实现漂移的后果是
    「单条样本的损失」与「一个 batch 的损失」口径不一致 —— 而两者都不报错。
    """
    T = len(logits_data)
    if len(targets) != T:
        raise ValueError(f"targets 长度 {len(targets)} != logits 行数 {T}")
    if mask is None:
        mask = [1.0 if t >= 0 else 0.0 for t in targets]
    if len(mask) != T:
        raise ValueError(f"mask 长度 {len(mask)} != logits 行数 {T}")
    probs: list[list[float]] = []
    acc = 0.0
    for row, tgt, mi in zip(logits_data, targets, mask):
        if tgt < 0:
            probs.append([0.0] * len(row))
            continue
        m = max(row)
        ex = [math.exp(v - m) for v in row]
        s = sum(ex)
        p = [v / s for v in ex]
        probs.append(p)
        if mi != 0.0:
            acc += mi * -math.log(max(p[tgt], 1e-30))
    denom = sum(mask)
    if denom <= 0:
        raise ValueError("mask 全为 0：这条样本没有任何监督信号，不该进 batch")
    return probs, acc / denom, denom


def batch_cross_entropy(
    pairs: Sequence[tuple[Any, Sequence[int], Sequence[float] | None]],
    *,
    weight: float = 1.0,
) -> Tensor:
    """一个 batch 的平均损失，返回 ``[loss]``（只有一个元素的 list）。

    **batch 损失一律取平均，绝不取和**：取和会让梯度范数正比于 batch 大小，
    于是 ``grad_clip`` 几乎总是把梯度整体裁掉。症状极隐蔽 —— loss 曲线在降，
    数值「合理」，但参数几乎没更新，而且「改 batch 大小」会等效于「改学习率」。

    每条样本**先在自己的掩码分母上归一化，再对样本取平均**；顺序反了会让回答
    短的那条样本被系统性低估。
    """
    if not pairs:
        raise ValueError("batch_cross_entropy 收到空的 batch")
    tensors = [_as_tensor(p[0]) for p in pairs]
    parts: list[tuple[list[list[float]], float, float]] = []
    total_rows = 0
    for tl, (_, targets, mask) in zip(tensors, pairs):
        if not _is2d(tl.data):
            raise ValueError("batch_cross_entropy 的 logits 必须是 2D")
        parts.append(_ce_parts(tl.data, targets, mask))
        total_rows += len(tl.data)
    n = len(parts)
    loss = weight * sum(p[1] for p in parts) / n
    _count("ce", total_rows * (len(tensors[0].data[0]) if tensors[0].data else 0))
    t = Tensor([loss], prev=tuple(tensors))

    def _bw() -> None:
        if t.grad is None:
            return
        g = weight * t.grad[0] / n
        for tl, (_, targets, mask), (probs, _loss, denom) in zip(tensors, pairs, parts):
            if mask is None:
                mask = [1.0 if tg >= 0 else 0.0 for tg in targets]
            scale_i = g / denom
            gx: list[list[float]] = []
            for i, (p, tgt) in enumerate(zip(probs, targets)):
                mi = mask[i]
                if mi == 0 or tgt < 0:
                    gx.append([0.0] * len(p))
                    continue
                row_g = [mi * scale_i * pi for pi in p]
                row_g[tgt] -= mi * scale_i
                gx.append(row_g)
            _acc(tl, gx)

    t._backward = _bw
    return t


def embed(table: Any, ids: Sequence[int]) -> Tensor:
    """按行取嵌入。反向是「把梯度散加回去」—— 同一个 id 出现两次必须加两次。"""
    tt = _as_tensor(table)
    if not _is2d(tt.data):
        raise ValueError("embed 只支持 2D 表")
    n, d = len(tt.data), len(tt.data[0])
    for i in ids:
        if not (0 <= i < n):
            raise IndexError(f"embed 下标越界：{i} 不在 [0, {n})")
    out = [list(tt.data[i]) for i in ids]
    _count("embed", len(ids) * d)
    t = Tensor(out, prev=(tt,))

    def _bw() -> None:
        if t.grad is None:
            return
        g = [[0.0] * d for _ in range(n)]
        for row_g, i in zip(t.grad, ids):
            for j, v in enumerate(row_g):
                g[i][j] += v
        _acc(tt, g)

    t._backward = _bw
    return t


def rope(x: Any, positions: Sequence[int], *, base: float = ROPE_BASE) -> Tensor:
    """把每个头内的通道两两配对做旋转。

    **它存在的原因是参数量公式里没有位置嵌入。** :func:`param_shapes` 只数了
    ``emb + 各层 + 最终 norm``，所以任何「可学习的位置表」都会让
    「参数量公式 == 实测参数量」这条互校当场破掉。RoPE 不带参数，于是它既给了
    模型位置信息，又不破坏那条互校。

    配对方式是相邻两个通道一组（``2j`` 与 ``2j+1``），所以 ``head_dim`` 必须是偶数。

    反向是**同一个旋转的逆**（旋转矩阵正交，转置即逆）：``gx = R(-θ)·g``。
    写出这一步最容易犯的错是只对 ``g`` 用 ``cos`` 而忘了 ``sin`` 项要变号 ——
    它**不报错**，只让梯度比真值小一点，表现为「训练慢」，而「训练慢」在本项目里
    是一个合法结论（Adapter 学不动），于是这种 bug 会被当成实验现象。
    """
    tx = _as_tensor(x)
    if not _is2d(tx.data):
        raise ValueError("rope 只支持 2D 输入（单头：[T, head_dim]）")
    T = len(tx.data)
    hd = _cols(tx.data)
    if hd % 2 != 0:
        raise ValueError(f"rope 要求 head_dim 为偶数，得到 {hd}")
    if len(positions) != T:
        raise ValueError(f"rope 的 positions 长度 {len(positions)} != 行数 {T}")

    half = hd // 2
    theta = [base ** (-2.0 * j / hd) for j in range(half)]
    table: list[list[tuple[float, float]]] = []
    for p in positions:
        table.append([(math.cos(float(p) * theta[j]), math.sin(float(p) * theta[j])) for j in range(half)])

    out: list[list[float]] = []
    for i, row in enumerate(tx.data):
        new_row = [0.0] * hd
        for j in range(half):
            c, s = table[i][j]
            a, b = row[2 * j], row[2 * j + 1]
            new_row[2 * j] = a * c - b * s
            new_row[2 * j + 1] = a * s + b * c
        out.append(new_row)
    _count("rope", tx.numel)
    t = Tensor(out, prev=(tx,))

    def _bw() -> None:
        if t.grad is None:
            return
        gx: list[list[float]] = []
        for i, grow in enumerate(t.grad):
            new_row = [0.0] * hd
            for j in range(half):
                c, s = table[i][j]
                ga, gb = grow[2 * j], grow[2 * j + 1]
                new_row[2 * j] = ga * c + gb * s
                new_row[2 * j + 1] = -ga * s + gb * c
            gx.append(new_row)
        _acc(tx, gx)

    t._backward = _bw
    return t


# ======================================================================
# 二、模型结构
# ======================================================================
@dataclass(frozen=True)
class ModelShape:
    """基座的**结构**。这里没有权重 —— 权重由 :func:`init_base_weights` 从 (结构, 种子) 确定性地生成。

    落盘一份几万浮点只会让「同一份基座」多出一个可以漂移的自由度，而它换不来任何东西。

    形状（每一项都与 :func:`param_shapes` 逐位对齐）::

        emb [vocab, d]
        每层: norm1 [d]
              q_proj/k_proj/v_proj/o_proj            [d, d]      ×4
              norm2 [d]
              gate_proj/up_proj [d, d_ff], down_proj [d_ff, d]   ×3   (SwiGLU)
        norm_f [d]
    """

    name: str = "tiny"
    vocab: int = 32
    d_model: int = 24
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 48
    max_len: int = 64
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} 必须能被 n_heads={self.n_heads} 整除")
        if self.head_dim % 2 != 0:
            # RoPE 按相邻两通道配对（见 net.rope）。奇数会让位置编码无法表达。
            raise ValueError(f"head_dim 必须为偶数（RoPE），得到 {self.head_dim}")
        for f in ("vocab", "d_model", "n_layers", "n_heads", "d_ff", "max_len"):
            if getattr(self, f) < 1:
                raise ValueError(f"{f} 必须为正，得到 {getattr(self, f)}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def key(self) -> str:
        """基座的结构指纹（**不含种子**）：同一个 key 必然给出同一张参数表。"""
        return (
            f"{self.name}:{self.n_layers}l-d{self.d_model}-h{self.n_heads}"
            f"-f{self.d_ff}-v{self.vocab}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocab": self.vocab,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_len": self.max_len,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ModelShape":
        """**显式按字段取值，绝不 `(**d)`。**

        ``to_dict`` 里多写一个展示用的字段，``(**d)`` 就会 ``TypeError``；而这条路径
        平时走不到（载入路径才用），只有 round-trip 测试抓得到。
        """
        if isinstance(d, str):
            raise TypeError(f"ModelShape.from_dict 收到字符串 {d!r}，应当是一个映射")
        return cls(
            name=str(d["name"]),
            vocab=int(d["vocab"]),
            d_model=int(d["d_model"]),
            n_layers=int(d["n_layers"]),
            n_heads=int(d["n_heads"]),
            d_ff=int(d["d_ff"]),
            max_len=int(d["max_len"]),
            tie_embeddings=bool(d.get("tie_embeddings", True)),
        )


def layer_param_names(shape: ModelShape) -> list[str]:
    """基座的全部参数名，**顺序即落盘顺序**。"""
    names = [EMB]
    for i in range(shape.n_layers):
        names.extend(f"layers.{i}.{s}" for s in PARAM_SUFFIXES)
    names.append(NORM_F)
    return names


def lora_param_names(shape: ModelShape, targets: Sequence[str]) -> list[str]:
    """LoRA 挂点的全名。**短名 → 全名的跨越只发生在这里**（模板与实例的边界只跨一次）。

    循环是**层在外、目标在内**，于是落盘顺序是
    ``layers.0.q_proj, layers.0.v_proj, layers.1.q_proj, …``。
    """
    known = set(LORA_ABLE_MODULES)
    for t in targets:
        if t not in known:
            raise ValueError(f"未知的适配器挂点 {t!r}（可选：{sorted(known)}）")
    out: list[str] = []
    for i in range(shape.n_layers):
        for t in targets:
            out.append(f"layers.{i}.{t}")
    return out


def param_shapes(shape: ModelShape) -> dict[str, tuple[int, ...]]:
    """基座参数的形状表。线性权重一律 ``[in, out]``（前向是 ``x @ W``）。"""
    d, ff = shape.d_model, shape.d_ff
    out: dict[str, tuple[int, ...]] = {EMB: (shape.vocab, d)}
    for i in range(shape.n_layers):
        p = f"layers.{i}."
        out[p + "norm1"] = (d,)
        out[p + "q_proj"] = (d, d)
        out[p + "k_proj"] = (d, d)
        out[p + "v_proj"] = (d, d)
        out[p + "o_proj"] = (d, d)
        out[p + "norm2"] = (d,)
        out[p + "gate_proj"] = (d, ff)
        out[p + "up_proj"] = (d, ff)
        out[p + "down_proj"] = (ff, d)
    out[NORM_F] = (d,)
    return out


def adapter_shapes(shape: ModelShape, targets: Sequence[str], rank: int) -> dict[str, tuple[int, ...]]:
    """适配器张量的形状表：``A: [rank, out]``、``B: [in, rank]``。

    ``delta = scale · (B @ A)``，形状 ``[in, out]``，与基座权重同形。
    """
    if rank < 1:
        raise ValueError(f"rank 必须 >= 1，得到 {rank}")
    base = param_shapes(shape)
    out: dict[str, tuple[int, ...]] = {}
    for name in lora_param_names(shape, targets):
        in_dim, out_dim = base[name]
        out[name + ".lora_A"] = (rank, out_dim)
        out[name + ".lora_B"] = (in_dim, rank)
    return out


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def count_params(weights: Any) -> int:
    """实测参数量。**与 :func:`param_shapes` 的公式互校**（公式与实例化漂移是静默的）。

    ``weights`` 可以是 ``{名字: 张量}`` 的映射，也可以是一个形状表（值当成形状读）。
    """
    if isinstance(weights, dict):
        total = 0
        for v in weights.values():
            if isinstance(v, (tuple, list)) and v and isinstance(v[0], int):
                total += _numel(v)  # 形状表
            else:
                total += _numel(_shape_of(v))
        return total
    raise TypeError(f"count_params 不认识 {type(weights).__name__}")


def _shape_of(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return (len(value), len(value[0]))
        return (len(value),)
    raise TypeError(f"不认识这个张量表示：{type(value).__name__}")


def init_base_weights(shape: ModelShape, seed: int) -> dict[str, list[Any]]:
    """从 (结构, 种子) 确定性地生成基座权重。

    每个参数的种子来自 ``derive_seed("init", 参数名, 结构 key, seed)`` ——
    **只依赖参数名与结构，不依赖遍历顺序**。否则「同一个 key 必得同一张权重表」
    会多出一个可以漂移的自由度。
    """
    shapes = param_shapes(shape)
    out: dict[str, list[Any]] = {}
    for name, shp in shapes.items():
        rng = random.Random(derive_seed("init", name, shape.key, int(seed)))
        if name.endswith("norm1") or name.endswith("norm2") or name == NORM_F:
            out[name] = [1.0] * shp[0]
            continue
        fan_in = shp[0]
        if name == EMB:
            fan_in = shp[1]
        std = 0.02 / math.sqrt(max(1.0, fan_in / 16.0))
        rows = shp[0]
        cols = shp[1] if len(shp) > 1 else 1
        out[name] = [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]
    return out


def init_adapter(shape: ModelShape, targets: Sequence[str], rank: int, seed: int) -> "Adapter":
    """新建一个适配器：``A ~ N(0, 1/sqrt(out))``、``B = 0``。

    **B 初始化为 0** ⇒ ``scale·B@A = 0`` ⇒ 适配器在第 0 步与基座**逐位相同**。
    这不是为了好看：它让「组合的起点就是基座」成为一条可断言的性质，
    否则「组合涨了多少」这个数字里会混进初始化的运气。
    """
    shp = adapter_shapes(shape, targets, rank)
    tensors: dict[str, list[Any]] = {}
    for name, s in shp.items():
        rng = random.Random(derive_seed("lora", name, shape.key, int(seed)))
        if name.endswith(".lora_B"):
            tensors[name] = [[0.0] * s[1] for _ in range(s[0])]
        else:
            std = 1.0 / math.sqrt(max(1.0, float(s[1])))
            tensors[name] = [[rng.gauss(0.0, std) for _ in range(s[1])] for _ in range(s[0])]
    return Adapter(tensors=tensors, targets=tuple(targets), rank=int(rank))


# ---- 前向 --------------------------------------------------------------
_MASKS: dict[int, Tensor] = {}


def causal_mask(T: int) -> Tensor:
    """因果掩码（上三角 ``MASK_NEG``）。缓存起来 —— 它是 ``T`` 的纯函数。"""
    hit = _MASKS.get(T)
    if hit is not None:
        return hit
    m = [[0.0] * T for _ in range(T)]
    for i in range(T):
        for j in range(i + 1, T):
            m[i][j] = MASK_NEG
    t = constant(m).freeze()
    _MASKS[T] = t
    return t


def _split_heads(x: Tensor, shape: ModelShape) -> list[Tensor]:
    return list(split_cols(x, [shape.head_dim] * shape.n_heads))


def _merge_heads(heads: Sequence[Tensor]) -> Tensor:
    return concat_cols(list(heads))


@dataclass
class Adapter:
    """一份适配器：张量表 + 挂点 + 秩 + **缩放系数**。

    ``scale = alpha / rank``。**它和 rank 必须分开存**：同一个 rank 配不同 alpha
    是完全不同的适配强度，而产物里只记 rank 会让「为什么要这个强度」
    变成不可追溯的（见 :mod:`zoo` 的边车 JSON）。
    """

    tensors: dict[str, list[Any]]
    targets: tuple[str, ...]
    rank: int
    alpha: float = 0.0
    #: 这份适配器的来源标签（哪个能力、哪个变体）。**它不参与数值，只参与身份**。
    label: str = ""
    capability: str = ""

    def __post_init__(self) -> None:
        if self.alpha == 0.0:
            # 默认 alpha = 2·rank，与同工作区 FineTuneFactory 的口径一致。
            self.alpha = 2.0 * float(self.rank)

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)

    @property
    def n_params(self) -> int:
        return sum(_numel(_shape_of(v)) for v in self.tensors.values())

    def delta(self, name: str, base_shape: Sequence[int]) -> list[list[float]]:
        """权重空间的增量 ``scale · (B @ A)``，形状 ``[in, out]``。

        **这是「组合」真正在操作的东西**，也是 :mod:`conflict` 里参数空间那一路
        预测冲突的输入。写成独立函数（而不是内联在两个调用点）是为了让
        「合并进权重」与「算 delta 用于冲突预测」共用同一份数学。
        """
        A = self.tensors[name + ".lora_A"]
        B = self.tensors[name + ".lora_B"]
        rank = len(A)
        in_dim, out_dim = base_shape[0], base_shape[1]
        if len(B) != in_dim or (B and len(B[0]) != rank):
            raise ValueError(f"{name}: B 的形状与基座对不上（期望 [{in_dim}, {rank}]）")
        if len(A) != rank or (A and len(A[0]) != out_dim):
            raise ValueError(f"{name}: A 的形状与基座对不上（期望 [{rank}, {out_dim}]）")
        s = self.scale
        # delta[i][j] = s · Σ_r B[i][r] · A[r][j]
        out: list[list[float]] = []
        for i in range(in_dim):
            bi = B[i]
            row = [_dot(bi, [A[r][j] for r in range(rank)]) * s for j in range(out_dim)]
            out.append(row)
        return out


#: 一组适配器 = 名字 → 适配器。
Adapters = dict[str, Adapter]


def _linear(x: Tensor, w: Any, *, name: str | None = None) -> Tensor:
    return matmul(x, _as_tensor(w, name=name))


def forward(
    weights: dict[str, Any],
    ids: Sequence[int],
    shape: ModelShape,
    *,
    extra: dict[str, Any] | None = None,
    mixer: Any = None,
) -> Tensor:
    """跑一次前向，返回整条序列的 logits ``[T, vocab]``。

    ``extra`` 是「激活空间」的附加项：``{权重名: [delta 矩阵, ...]}``。
    每个 delta 会在**那个线性层**的输出上再加一次 —— 这就是
    :func:`forward_with_adapters` 的实现路径，也是「不合并」的那条参照。

    ``mixer`` 见 :func:`_linear_add`：**``stack`` 与 ``route`` 走的是这一段
    前向**，只在每个线性层的混合方式上分岔。这是「路由 vs 合并」这个比较
    能成立的前提。

    ⚠️ **``extra`` 里的项数直接决定 MACs。** 组合的 ``latency`` 轴就该把
    「不合并时要多跑 k 次小矩阵乘」算进去 —— 若把合并前后的 MACs 算成一样，
    「Route 比 Merge 贵」这件事在报告里就看不见了。
    """
    pos = list(range(len(ids)))
    x = embed(_as_tensor(weights[EMB]), ids)
    mask = causal_mask(len(ids))
    extra = extra or {}

    for i in range(shape.n_layers):
        p = f"layers.{i}."
        h = rmsnorm(x, _as_tensor(weights[p + "norm1"]))
        q = _linear_add(h, weights[p + "q_proj"], extra.get(p + "q_proj"), shape, mixer, p + "q_proj")
        k = _linear_add(h, weights[p + "k_proj"], extra.get(p + "k_proj"), shape, mixer, p + "k_proj")
        v = _linear_add(h, weights[p + "v_proj"], extra.get(p + "v_proj"), shape, mixer, p + "v_proj")
        qh, kh, vh = _split_heads(q, shape), _split_heads(k, shape), _split_heads(v, shape)
        outs = []
        for qi, ki, vi in zip(qh, kh, vh):
            qr = rope(qi, pos)
            kr = rope(ki, pos)
            scores = matmul(qr, transpose(kr))
            _count("attn_scale", len(scores.data) * len(scores.data[0]))
            scaled = scale(scores, 1.0 / math.sqrt(float(shape.head_dim)))
            probs = softmax_rows(add(scaled, mask))
            outs.append(matmul(probs, vi))
        attn = _linear_add(_merge_heads(outs), weights[p + "o_proj"], extra.get(p + "o_proj"), shape, mixer, p + "o_proj")
        x = add(x, attn)

        h2 = rmsnorm(x, _as_tensor(weights[p + "norm2"]))
        g = _linear_add(h2, weights[p + "gate_proj"], extra.get(p + "gate_proj"), shape, mixer, p + "gate_proj")
        u = _linear_add(h2, weights[p + "up_proj"], extra.get(p + "up_proj"), shape, mixer, p + "up_proj")
        mlp = _linear_add(mul(silu(g), u), weights[p + "down_proj"], extra.get(p + "down_proj"), shape, mixer, p + "down_proj")
        x = add(x, mlp)

    xf = rmsnorm(x, _as_tensor(weights[NORM_F]))
    emb = _as_tensor(weights[EMB])
    # 权重共享：输出投影用 emb 的转置。非共享时形状检查会拦住（本项目的 IR 里没有这一支）。
    return matmul(xf, transpose(emb)) if shape.tie_embeddings else matmul(xf, _as_tensor(weights["lm_head"]))


def gate_coeffs(x: Tensor, gates: Sequence[Sequence[float]], temperature: float) -> list[list[float]]:
    """``softmax_i(⟨h_t, uᵢ⟩ / T)`` —— 逐 token 的挂载权重。

    门向量 ``uᵢ`` 是**种子派生的固定随机向量，不训练**。这条必须在代码里
    和报告里都说清楚：它是**结构对照**，用来回答「把适配器保持在激活空间、
    不做任何合并，光靠一个固定随机门分派，能不能白拿收益」。
    把它读成「学出来的路由」会让 ``route`` 那一列变成对某个真实路由方案的
    成绩单 —— 而这里从来没有训练过任何路由。

    返回的是**数据**（嵌套 list），不是 :class:`Tensor`：系数由 x 决定，
    但这条路径里 x 不进反向（route 只用于推理）。见 :func:`mix_rows` 末尾
    关于「若要让 uᵢ 可学得补什么」的说明。
    """
    if not gates:
        raise ValueError("gate_coeffs 至少需要一个门向量")
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正，得到 {temperature}")
    k = len(gates)
    dim = len(gates[0])
    for i, u in enumerate(gates):
        if len(u) != dim:
            raise ValueError(f"第 {i} 个门向量长度 {len(u)} != {dim}")
    rows: list[list[float]] = []
    for t, h in enumerate(x.data):
        if len(h) != dim:
            raise ValueError(f"第 {t} 个 token 的激活维度 {len(h)} != 门向量维度 {dim}")
        rows.append([_dot(h, u) / temperature for u in gates])
    return softmax_rows_data(rows) if k > 1 else [[1.0] for _ in rows]


def _linear_add(
    x: Tensor,
    w: Any,
    deltas: Sequence[Any] | None,
    shape: ModelShape,
    mixer: Any = None,
    name: str = "",
) -> Tensor:
    """``x @ W + Σ x @ Δᵢ``。无 delta 时退化成一次普通 ``matmul``。

    ``mixer`` 是**可选的混合器**：给了它就由它决定「这些 Δ 怎么进这个线性层」，
    否则走默认的**逐项相加**。这是 ``route`` 与 ``stack`` 唯一的差别所在 ——
    刻意做成同一个函数上的一个参数，而不是两条前向：

    若 ``route`` 另写一份 ``forward``，那么「路由比合并好／差」这个结论就同时
    混进了「两份前向实现之间的差别」—— 而两份实现是必然要漂移的
    （一层漏掉、一处 mask 顺序不同），漂移的症状是数值合理、结论错误。
    **同一条前向里换一个混合器，比较的才是混合器本身。**

    ``name`` 是当前线性层的全名（``layers.0.k_proj``），**只传给 mixer**。
    混合器需要它，因为「挂在这一层上的 delta 有几个」是逐层不同的：
    一个只挂 ``q,v`` 的适配器在 ``k_proj`` 上根本不出现。不知道名字的混合器
    只能按「适配器总数」去算系数，而系数个数与 delta 个数不等时
    :func:`mix_rows` 会抛错（这条路径实测抛过，见 README 附录 C）。

    默认路径的操作顺序（``((out + p₁) + p₂)``）被刻意保留：它与浮点加法
    不可结合性有关，改成「先求和再加一次」会让既有产物的每一位数字都变，
    而看起来只是等价重写。
    """
    out = _linear(x, w)
    if not deltas:
        return out
    if mixer is not None:
        return add(out, mixer(x, deltas, name))
    for d in deltas:
        out = add(out, matmul(x, _as_tensor(d)))
    return out


def routed_mixer(gate_of: Any, temperature: float) -> Any:
    """造一个 ``route`` 用的混合器：``Σᵢ softmax(⟨h,uᵢ⟩/T)ᵢ · (x @ Δᵢ)``。

    ``gate_of(name)`` 给出**该模块上**那 k 个 delta 各自的门向量，顺序与
    delta 一致。这样定是因为「软最大在谁之间做」这个问题必须逐层回答：
    在 ``k_proj`` 上只有一个适配器挂着时，答案只能是 1.0 —— 那里没有可做的
    选择，route 退化成 stack。一个全局 softmax 会让没挂的适配器也占一份权重，
    于是 route 在每一层都在稀释，而报告把它读成「路由的代价」。

    每个 Δ 的矩阵乘**照样要做**（MACs 全额计入），所以 route 的 ``latency``
    必然高于同规模的 stack —— 这正是「不合并要付什么代价」这个问题的答案。
    若把它优化成「先按系数合并 Δ 再乘一次」，latency 会掉到与 stack 相同，
    报告里 route 的高成本就凭空消失了。
    """

    def mixer(x: Tensor, deltas: Sequence[Any], name: str) -> Tensor:
        us = gate_of(name)
        if len(us) != len(deltas):
            raise ValueError(
                f"{name}: 门向量 {len(us)} 个与 delta {len(deltas)} 个对不上 —— "
                f"混合器必须按模块给出该模块上的门向量"
            )
        parts = [matmul(x, _as_tensor(d)) for d in deltas]
        if not parts:
            raise ValueError("mixer 收到了空的 delta 列表")
        return mix_rows(parts, gate_coeffs(x, us, temperature))

    return mixer


def gates_by_module(
    adapters: Sequence[Adapter],
    gates: Mapping[str, Mapping[str, Sequence[float]]],
    shape: ModelShape,
    *,
    temperature: float = 1.0,
) -> Any:
    """把 ``{标签: {模块: 门向量}}`` 摊成 ``{模块全名: [该模块上的门向量, ...]}``。

    展开用的是与 :func:`deltas_for` **同一个** :func:`expand_targets`，
    顺序也是同一套（适配器序、层号升序）。两处若各写一份展开逻辑，
    「门向量与 delta 一一对应」这句话就会在某个变体上悄悄失效，
    而症状是 route 的系数被配给了错误的适配器 —— 数值照常，结论错位。

    挂点一个都没有的模块**不进这张表**，于是 :func:`routed_mixer` 的
    ``gate_of`` 返回空、:func:`_linear_add` 因 ``deltas`` 为空而根本不调
    混合器。两条路都通向「那里没有路由可做」。
    """
    base = param_shapes(shape)
    per_module: dict[str, list[Sequence[float]]] = {}
    for ad in adapters:
        per = gates.get(ad.label)
        if per is None:
            raise KeyError(f"缺少适配器 {ad.label!r} 的门向量（有：{sorted(gates)}）")
        for full in expand_targets(base, ad.targets):
            u = per.get(full)
            if u is None:
                raise KeyError(f"适配器 {ad.label!r} 缺少模块 {full!r} 的门向量")
            per_module.setdefault(full, []).append(u)
    return routed_mixer(per_module.get, temperature)


def forward_routed(
    weights: dict[str, Any],
    adapters: Sequence[Adapter],
    ids: Sequence[int],
    shape: ModelShape,
    *,
    gates: Mapping[str, Mapping[str, Sequence[float]]],
    temperature: float = 1.0,
) -> Tensor:
    """``route``：**不合并**，逐层按 ``softmax(⟨h,uᵢ⟩/T)`` 在激活空间混合。

    ``gates`` 是 ``{标签: {模块: 门向量}}``（见 :func:`derive_gates`）。
    按标签取，不按下标取 —— 下标会在 zoo 变化时静默错配。
    """
    return forward(
        weights,
        ids,
        shape,
        extra=deltas_for(weights, adapters),
        mixer=gates_by_module(adapters, gates, shape, temperature=temperature),
    )


def module_indims(shape: ModelShape) -> dict[str, int]:
    """每个可挂载模块的**输入维度**。``down_proj`` 是 48，其余是 24。

    这个函数存在的唯一理由是：**门向量必须活在这层的输入空间里**。
    第一版把门向量一律定成 ``d_model=24`` 维，于是在 ``down_proj`` 上
    拿一个 24 维的门去点一个 48 维的激活 —— :func:`gate_coeffs` 抛错拦住了它
    （``第 0 个 token 的激活维度 48 != 门向量维度 24``）。
    若当初写的是 ``zip`` 截断，它会静默变成「只看前 24 维的点积」，
    ``down_proj`` 上的路由照样出数，只是那半层路由在拿前一半通道做决定。
    """
    return {full: s[0] for full, s in param_shapes(shape).items() if full.rpartition(".")[2] in LORA_ABLE_MODULES}


def derive_gates(
    shape: ModelShape,
    labels: Sequence[str],
    *,
    seed: int,
) -> dict[str, dict[str, list[float]]]:
    """门向量：**由 (seed, 标签, 模块) 派生，不训练、不落盘、不进指纹文件。**

    返回 ``{标签: {模块全名: u}}``，``u`` 的维度等于那个模块的输入维度。

    用**标签**而不是下标派生，所以「zoo 里多了/少了一个 Adapter」不会让别的
    Adapter 的门向量悄悄换一个 —— 用下标的话，删掉第 2 个会让第 3 个顶上来，
    而 route 的全部结果都会变，且没有任何东西会提示这件事。

    ⚠️ 每次调用都重新派生，**不缓存、不落盘**。把它写进产物会让「门向量」
    看起来是一个可以调参的自由度，而它是种子派生的常量：报告里 ``route``
    那一列的含义正建立在这上面（见 :func:`gate_coeffs`）。
    """
    dims = module_indims(shape)
    gates: dict[str, dict[str, list[float]]] = {}
    for lab in labels:
        per_module: dict[str, list[float]] = {}
        for full, dim in dims.items():
            rng = random.Random(derive_seed("gate", shape.key, lab, full, seed))
            per_module[full] = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        gates[lab] = per_module
    return gates


def expand_targets(base: Mapping[str, Any], targets: Sequence[str]) -> list[str]:
    """把短挂点名（``"q_proj"``）展开成实际存在的全名（``"layers.0.q_proj"`` …）。

    **层号从 0 开始数，而且上界是「探测出来的」不是一个常量。**

    这两点各自都是一条防线。曾经这里写的是 ``for i in range(1, 64)``：起点写成 1，
    于是**第 0 层被静默跳过**。它不报错、不抛异常，只是组合的效果少算了一层 ——
    而 :func:`merge_into` 与 :func:`deltas_for` 用同一段展开逻辑，
    于是「合并」与「逐个施加」两条路径**一起**少算第 0 层，``merge_error`` 依然
    是 1e-17，恒等式照样成立。**共享 bug 的两条路径不是互相独立的证据。**

    用一个「探测到哪一层没有了就停」的循环取代常量上界：给定 ``n_layers=2`` 时
    写 64 只是运气好，写 1 就是灾难，而两种写法在代码里长得一样。
    """
    out: list[str] = []
    i = 0
    while True:
        prefix = f"layers.{i}."
        if not any(k.startswith(prefix) for k in base):
            break
        for t in targets:
            if prefix + t in base:
                out.append(prefix + t)
        i += 1
    return out


def deltas_for(weights: dict[str, Any], adapters: Sequence[Adapter]) -> dict[str, list[list[list[float]]]]:
    """把若干适配器折成 ``{权重名: [delta 矩阵, ...]}``（**激活空间**用的形式）。"""
    base = param_shapes_of(weights)
    out: dict[str, list[list[list[float]]]] = {}
    for ad in adapters:
        for full in expand_targets(base, ad.targets):
            out.setdefault(full, []).append(ad.delta(full, base[full]))
    return out


def param_shapes_of(weights: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """从一张权重表反推形状。**只用于载入路径** —— 结构的事实来源是 :class:`ModelShape`。"""
    return {k: _shape_of(v) for k, v in weights.items()}


def forward_with_adapters(
    weights: dict[str, Any],
    adapters: Sequence[Adapter],
    ids: Sequence[int],
    shape: ModelShape,
) -> Tensor:
    """**不合并**：把每个适配器的 delta 直接加在激活上（「逐个施加」那条参照路径）。"""
    return forward(weights, ids, shape, extra=deltas_for(weights, adapters))


def lora_delta_map(
    tensors: dict[str, Any],
    targets: Sequence[str],
    shape: ModelShape,
    *,
    scale_value: float,
) -> dict[str, list[Tensor]]:
    """把 LoRA 张量**在图里**折成 ``delta = scale_value · (B @ A)``。

    它与 :meth:`Adapter.delta` 算的是同一个东西，但走的是两条路：

    - ``Adapter.delta`` 用**手写三重循环**，产出裸 list，供**推理与冲突预测**使用；
    - 这个函数用引擎的 :func:`matmul`，产出 :class:`Tensor`，**梯度能流回 B 与 A**。

    两条路都留着不是为了冗余，而是为了**互相校对**。只留图里那一条的话，
    :mod:`produce` 训出来的适配器没有任何独立实现能验证「它等于我刚训的那个」；
    只留裸 list 那条的话，训练根本跑不起来（没有梯度）。

    ⚠️ **它每次调用都新建张量。** 复用同一批 delta 张量跨多个样本前向，会让
    同一个节点的 ``_backward`` 被多调几次 —— 梯度是对的（累加），但
    ``_prev`` 里挂着上一轮图里的节点，于是这次反向会顺着一根**已经失效的边**
    继续往回走。症状是参数更新量偏大而损失曲线看起来正常。
    """
    base = param_shapes(shape)
    out: dict[str, list[Tensor]] = {}
    for full in expand_targets(base, targets):
        A = _as_tensor(tensors[full + ".lora_A"])
        B = _as_tensor(tensors[full + ".lora_B"])
        out[full] = [scale(matmul(B, A), scale_value)]
    return out


def forward_train(
    weights: dict[str, Any],
    tensors: dict[str, Any],
    targets: Sequence[str],
    ids: Sequence[int],
    shape: ModelShape,
    *,
    scale_value: float,
) -> Tensor:
    """可求导的前向：基座是常量，LoRA 张量是叶子。**只被 :mod:`produce` 使用。**"""
    return forward(weights, ids, shape, extra=lora_delta_map(tensors, targets, shape, scale_value=scale_value))


def logits_of(
    weights: dict[str, Any],
    ids: Sequence[int],
    shape: ModelShape,
    *,
    adapters: Sequence[Adapter] = (),
) -> list[list[float]]:
    """前向的裸数据出口。有适配器时走「逐个施加」那条路径。

    **它和训练走的是同一段前向代码** —— 若评测另写一份，两份会漂移，
    而漂移的症状是「训练时的 loss 与评测出的 acc 对不上」，两者都不报错。
    """
    t = forward_with_adapters(weights, adapters, ids, shape) if adapters else forward(weights, ids, shape)
    return t.data


def merge_into(
    weights: dict[str, Any],
    adapters: Sequence[Adapter],
    *,
    weights_by_adapter: Sequence[float] | None = None,
) -> dict[str, list[Any]]:
    """**合并**：把 ``Σ wᵢ · Δᵢ`` 加进基座权重，产出一张新的权重表。

    ``weights_by_adapter`` 是每个适配器的**合成权重**（默认全 1）：

    - 全 1 ⇒ ``stack``（求和）—— 「加性假设」；
    - 全 ``1/k`` ⇒ ``merge``（平均）—— 「稀释共识」。

    两个方法共用这一个函数、只在系数上不同，是刻意的：**「求平均」不是另一套
    算法，它只是求和的一个特例**。拆成两份实现的话，「merge 的能力损失到底来自
    平均还是来自别的地方」就说不清了。

    手写三重循环（不走 :func:`matmul`）是为了让「合并结果」与引擎的算子路径
    **互相独立** —— 两者共用一个实现的话，``merge_error`` 那条互校会退化成自证。
    """
    base = param_shapes_of(weights)
    coefs = list(weights_by_adapter) if weights_by_adapter is not None else [1.0] * len(adapters)
    if len(coefs) != len(adapters):
        raise ValueError(f"系数个数 {len(coefs)} != 适配器个数 {len(adapters)}")

    acc: dict[str, list[list[float]]] = {}
    for ad, c in zip(adapters, coefs):
        for full in expand_targets(base, ad.targets):
            d = ad.delta(full, base[full])
            cur = acc.get(full)
            if cur is None:
                acc[full] = [[v * c for v in row] for row in d]
            else:
                for r, row in enumerate(d):
                    cr = cur[r]
                    for j, v in enumerate(row):
                        cr[j] += v * c

    out: dict[str, list[Any]] = {}
    for name, w in weights.items():
        d = acc.get(name)
        if d is None:
            out[name] = [list(r) for r in w] if _is2d(w) else list(w)
            continue
        if not _is2d(w):
            raise ValueError(f"{name} 是 1D 参数，适配器不该挂在它上面")
        out[name] = [[w[i][j] + d[i][j] for j in range(len(w[0]))] for i in range(len(w))]
    return out


def max_abs_diff(a: Any, b: Any) -> float:
    """两个同形嵌套 list 的最大绝对差。**形状不同直接抛错**，不返回 inf。"""
    if _is2d(a) != _is2d(b):
        raise ValueError("max_abs_diff 的形状阶数不同")
    if _is2d(a):
        if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
            raise ValueError(f"max_abs_diff 形状不同：{len(a)}×{len(a[0]) if a else 0} vs {len(b)}×{len(b[0]) if b else 0}")
        return max((abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb)), default=0.0)
    if len(a) != len(b):
        raise ValueError(f"max_abs_diff 长度不同：{len(a)} vs {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)
