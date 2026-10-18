"""能力族、词表、样本与**两条基线**。

### 与 FineTuneFactory 的关系：能力语义必须逐字相同

本项目的四个能力（``upper`` / ``shift`` / ``reverse`` / ``sort``）与 FineTuneFactory
是**同一套函数**（逐字符映射 + 置换），词表与 payload 口径也相同。这不是偷懒：
README 允许用 ``main.py import --from ../L3-finetunefactory/out`` 把真产物换进来，
若两个项目对「upper」的定义差分毫，那份对照就变成了两件不同的事在比大小。

### 为什么这里比 FineTuneFactory 少两个维度

FF 的样本有 ``template``（chat / plain）与 ``objective``（all_tokens / response_only）
两条轴。**本项目两条都砍掉了**，各留一个固定取值：

- 模板不是本项目的自变量。设计空间是 Adapter 子集，模板挂了两个值只会让
  「276 个组合」变成「552 个」，而多出来的那一半不回答任何一个新问题。
- 目标口径由 producer 侧固定为 response-only。组合实验比较的是**权重**，
  不是损失口径。

砍掉的是**没有被任何实验使用的自由度**。留下它们不会报错，只会让配置文件里
多出两个没人拧的旋钮，而「配置里有的东西一定是被试过的」是本工作区的一条约定。

### 基线为什么必须是两条

``chance`` = ``1/|payload_chars|``（**不是** ``1/vocab``，理由见 :func:`chance_level`），
``heuristic`` = ``max(majority, copy)``。验收盯的是**更强**的那条（``heuristic``）——
一条 Adapter 的增益如果只是超过了盲猜，那说明不了任何事。只报 ``chance`` 是本
工作区明确记过的一个坑。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .rng import canonical_json, derive_seed, sha256_text

__all__ = [
    "DEFAULT_SPECIALS",
    "TASK_FNS",
    "TASK_TOKENS",
    "Vocab",
    "Example",
    "DataIndex",
    "load_index",
    "build_vocab",
    "build_examples",
    "build_eval_sets",
    "excluded_payloads",
    "build_train_set",
    "batch_of",
    "chance_level",
    "baselines",
    "fit_majority",
    "content_positions",
    "response_positions",
    "token_accuracy",
    "mean_accuracy",
    "describe",
    "iter_capabilities",
    "seq_len_of",
]

DEFAULT_SPECIALS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", ":", "=")

PAD_TOK = "<pad>"
BOS_TOK = "<bos>"
EOS_TOK = "<eos>"

#: 任务 token。**它必须进词表**（:func:`_encode_sequence` 会取它的 id），
#: 所以它是 :data:`DEFAULT_INDEX` 的 ``chars`` 的一部分。放在这里的唯一理由是
#: 「能力 → token」的映射与「能力 → 函数」的映射必须写在同一个屏幕内，
#: 否则加一个能力要改两个文件、漏一个就在运行时变成 KeyError。
TASK_TOKENS: dict[str, str] = {
    "upper": "1",
    "shift": "2",
    "reverse": "3",
    "sort": "4",
}


def _upper(p: str) -> str:
    return p.upper()


def _shift(p: str) -> str:
    return "".join(chr(ord(c) + 1) for c in p)


def _reverse(p: str) -> str:
    return p[::-1]


def _sort(p: str) -> str:
    return "".join(sorted(p))


#: 内置能力。**名字 → 一条纯函数**，没有例外、没有参数。
#:
#: 四个能力是刻意选的两组：``upper`` / ``shift`` 是**逐位置**映射（不需要跨位置
#: 搬运信息），``reverse`` / ``sort`` 是**置换**（必须靠 attention 把别的位置的
#: 信息搬过来）。
#:
#: 这个分组在 AdapterFactory 里有一个具体用途：它让「干扰」这件事有了**两种可
#: 区分的机制假设** —— 逐位置能力共享一套「读出输入字符」的底层特征，于是应该
#: 互相干扰；置换能力不共享，于是应该互不干扰。:mod:`conflict` 的参数空间那一路
#: （``cos(Δᵢ, Δⱼ)``）正是对这条假设的**独立测量**。**如果实测出来的图景不是
#: 这样，那才是有意思的结果** —— 报告里必须照写。
TASK_FNS: dict[str, Callable[[str], str]] = {
    "upper": _upper,
    "shift": _shift,
    "reverse": _reverse,
    "sort": _sort,
}


# ----------------------------------------------------------------------
# 词表
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Vocab:
    """定长词表。**顺序即 id**，所以它必须来自文件而不是代码里的字面量。

    改一个字面量就等于改掉全部 token id，于是所有已落盘的产物静默失效。
    走文件至少留下了一次 diff。
    """

    specials: tuple[str, ...]
    chars: str

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.specials + tuple(self.chars)

    @property
    def size(self) -> int:
        return len(self.tokens)

    def id(self, tok: str) -> int:
        try:
            return self._stoi[tok]
        except KeyError as exc:
            raise KeyError(f"词表里没有 token {tok!r}") from exc

    def has(self, tok: str) -> bool:
        return tok in self._stoi

    def __post_init__(self) -> None:
        toks = self.tokens
        dupes = sorted({t for t in toks if toks.count(t) > 1})
        if dupes:
            raise ValueError(f"词表里有重复 token（会静默改变 id 分配）：{dupes}")
        object.__setattr__(self, "_stoi", {t: i for i, t in enumerate(toks)})

    # frozen dataclass 上的缓存字段：在 __post_init__ 里用 object.__setattr__ 放。
    _stoi: Mapping[str, int] = field(default_factory=dict, repr=False, compare=False)

    @property
    def pad(self) -> int:
        return self.id(PAD_TOK)

    @property
    def bos(self) -> int:
        return self.id(BOS_TOK)

    @property
    def eos(self) -> int:
        return self.id(EOS_TOK)

    def to_dict(self) -> dict[str, Any]:
        return {"specials": list(self.specials), "chars": self.chars}


@dataclass(frozen=True)
class DataIndex:
    """``data/datasets/index.json`` 的内容。

    它声明「有哪些能力、任务 token 是哪个、字符表多大、payload 多长」。
    **能力名必须在 :data:`TASK_FNS` 里有实现** —— 索引只能选，不能发明。
    """

    specials: tuple[str, ...]
    chars: str
    payload_chars: str
    payload_len: int
    tasks: Mapping[str, str]  # task_id token -> capability name
    source: str = ""

    #: **不参与身份、也不参与往返**的字段。声明它不是仪式：
    #: 「``from_dict∘to_dict`` 严格互逆」是本工作区的一条硬性质（坑 41/42/43），
    #: 而 ``source`` 是唯一一个会破坏它的字段。若不把它显式列出来，
    #: 那条性质就变成了「除了 source 以外互逆」—— 一句没人能执行的话。
    #: 列出来之后，测试可以精确地断言「除去已声明的易变字段，往返相等」。
    VOLATILE: tuple[str, ...] = ("source",)

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self.tasks.values())

    def task_id_of(self, capability: str) -> str:
        for tok, name in self.tasks.items():
            if name == capability:
                return tok
        raise KeyError(f"数据索引里没有能力 {capability!r}（可选：{sorted(self.tasks.values())}）")

    def to_dict(self) -> dict[str, Any]:
        """**不含 ``source``**（见 :attr:`VOLATILE`）—— 路径是「从哪读的」，
        不是「读到了什么」。

        让它进指纹会让同一份索引在两个不同的挂载点下算出两个版本摘要，
        而那正是「内容寻址」这个机制要避免的事情。
        """
        return {
            "specials": list(self.specials),
            "chars": self.chars,
            "payload_chars": self.payload_chars,
            "payload_len": self.payload_len,
            "tasks": dict(sorted(self.tasks.items())),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, source: str = "") -> "DataIndex":
        """**显式取字段，绝不 ``(**d)``。**

        索引文件里带 ``_comment`` / ``version`` 这类展示字段，``(**d)`` 会直接
        ``TypeError``；而这条路径只有载入时才走，平时测不到。
        """
        tasks = d["tasks"]
        if not isinstance(tasks, Mapping):
            raise TypeError(f"index.tasks 必须是映射，得到 {type(tasks).__name__}")
        unknown = sorted(set(tasks.values()) - set(TASK_FNS))
        if unknown:
            raise ValueError(
                f"数据索引声明的能力 {unknown} 在 TASK_FNS 里没有实现（可选：{sorted(TASK_FNS)}）"
            )
        idx = cls(
            specials=tuple(str(s) for s in d["specials"]),
            chars=str(d["chars"]),
            payload_chars=str(d["payload_chars"]),
            payload_len=int(d["payload_len"]),
            tasks={str(k): str(v) for k, v in tasks.items()},
            source=source,
        )
        # 词表必须覆盖任务 token 与全部 payload 字符，否则错误会推迟到编码时才炸。
        v = Vocab(specials=idx.specials, chars=idx.chars)
        missing = [t for t in idx.tasks if not v.has(t)]
        if missing:
            raise ValueError(f"词表里没有任务 token {missing}（chars 不足以承载 tasks）")
        for c in idx.payload_chars:
            if not v.has(c):
                raise ValueError(f"词表里没有 payload 字符 {c!r}")
        return idx


def load_index(path: str | Path | Mapping[str, Any], *, source: str = "") -> DataIndex:
    """读数据索引。路径或已解析的映射都接受（后者供测试直接喂结构）。"""
    if isinstance(path, Mapping):
        return DataIndex.from_dict(path, source=source)
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"数据索引不存在：{p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return DataIndex.from_dict(raw, source=str(p))


def build_vocab(idx: DataIndex) -> Vocab:
    return Vocab(specials=idx.specials, chars=idx.chars)


# ----------------------------------------------------------------------
# 样本
# ----------------------------------------------------------------------
@dataclass
class Example:
    """一条定长样本。

    ``ids`` 与 ``targets`` 长度都是 ``seq_len``；``targets[t]`` 是「在位置 t+1
    要预测的 token」，PAD 处为 -1。

    ### ``eos_t``：为什么它要单独拿出来

    ``targets`` 里有一个位置的目标**永远是 EOS**，于是它 100% 可学，一条样本
    白送一次命中。把它算进准确率的后果是：所有能力的准确率被一起抬高，
    量程被压扁，而「模型学会了回答内容」与「模型学会了数到第 4 个就停」
    这两件难度天差地别的事在指标上**无法区分** —— 而本项目要比的正是
    「多少能力被组合保留下来」，量程被压扁会直接压掉组合之间的差异。

    所以：**损失里保留 EOS**（不会停的模型没用），**准确率里剔除 EOS**，
    而 EOS 的命中率单独汇报。

    ### ``payload``：为什么存字符串而不是回头去切 ``ids``

    启发式基线里的 ``copy`` 需要知道「回答段第 i 个 token 对应输入的哪个字符」。
    从 ``ids`` 里按偏移切片也能拿到，但那个偏移依赖 :func:`_encode_sequence` 的
    布局 —— 布局一改，切片**不会报错**，只会静默地复制错的东西，于是 ``copy``
    基线悄悄变了而没人知道。直接把 payload 存下来，这个耦合就消失了。
    """

    ids: list[int]
    targets: list[int]
    mask: list[float]
    capability: str
    payload: str
    response_start: int
    response_len: int
    #: 目标为 EOS 的那个位置下标；理论上有且仅有一个，找不到时为 -1。
    eos_t: int = -1

    def __len__(self) -> int:
        return len(self.ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ids": list(self.ids),
            "targets": list(self.targets),
            "mask": list(self.mask),
            "capability": self.capability,
            "payload": self.payload,
            "response_start": self.response_start,
            "response_len": self.response_len,
            "eos_t": self.eos_t,
        }


def _encode_sequence(vocab: Vocab, task_id: str, payload: str, response: str) -> tuple[list[int], int]:
    """``<bos> 任务 : payload = 回答 <eos>``。返回（ids, 回答起点）。"""
    s = [vocab.bos, vocab.id(task_id), vocab.id(":")]
    s.extend(vocab.id(c) for c in payload)
    s.append(vocab.id("="))
    start = len(s)
    s.extend(vocab.id(c) for c in response)
    s.append(vocab.eos)
    return s, start


def _make_example(vocab: Vocab, idx: DataIndex, capability: str, payload: str, seq_len: int) -> Example:
    response = TASK_FNS[capability](payload)
    ids, start = _encode_sequence(vocab, idx.task_id_of(capability), payload, response)
    if len(ids) > seq_len:
        raise ValueError(
            f"序列长度 {len(ids)} 超过 seq_len={seq_len}（能力 {capability}、payload 长 {idx.payload_len}）"
        )
    pad = vocab.pad
    n_real = len(ids)
    ids = ids + [pad] * (seq_len - len(ids))
    # targets[t] = 位置 t+1 的 token。最后一条真实 token 是 EOS，它后面是 PAD，
    # 于是「预测 EOS」这条监督被保留，而「在 PAD 后面预测 PAD」被丢弃。
    targets = [ids[t + 1] if t + 1 < n_real else -1 for t in range(seq_len)]
    # 回答段 = 回答 token 本身 + 结尾的 EOS。EOS 必须算进损失：不含它的训练会让
    # 模型永远学不会「该停了」，而推理时的贪心解码正是靠它停下来。
    mask = [1.0 if (t + 1 >= start and t + 1 < n_real) else 0.0 for t in range(seq_len)]
    eos_t = n_real - 2  # 目标为「最后一个真实 token（EOS）」的那个位置
    if eos_t < 0 or targets[eos_t] != vocab.eos:
        raise ValueError("样本里找不到 EOS 的监督位置：编码逻辑与 seq_len 对不上了")
    return Example(
        ids=ids,
        targets=targets,
        mask=mask,
        capability=capability,
        payload=payload,
        response_start=start,
        response_len=len(response) + 1,
        eos_t=eos_t,
    )


def _payloads(n: int, *, seed: int, alphabet: str, length: int, exclude: frozenset[str] = frozenset()) -> list[str]:
    """确定性抽取 n 个 payload。**用拒绝采样保证与 ``exclude`` 不相交。**

    ``exclude`` 是「训练集与评测集不重叠」的实现方式。没有它，一条训练样本会
    同时出现在评测集里，于是「涨了多少」被记忆污染 —— 这个错误不报任何错，
    只会让所有组合的 gain 一起偏高、把组合之间的差异压扁，
    而那正是本项目唯一要测的东西。
    """
    rng = random.Random(int(seed))
    out: list[str] = []
    seen: set[str] = set()
    space = len(alphabet) ** length
    if n + len(exclude) > space:
        raise ValueError(f"要求 {n} 条且排除 {len(exclude)} 条，但样本空间只有 {space}")
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 100 * (n + 1):
            raise ValueError("拒绝采样没有收敛：样本空间几乎被 exclude 占满")
        p = "".join(rng.choice(alphabet) for _ in range(length))
        if p in seen or p in exclude:
            continue
        seen.add(p)
        out.append(p)
    return out


def build_examples(
    vocab: Vocab,
    idx: DataIndex,
    capability: str,
    *,
    seq_len: int,
    n: int,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> list[Example]:
    """按 ``seed`` 抽 payload 并编码。**同一个 seed 必然给出同一批样本，顺序也一样。**"""
    pays = _payloads(n, seed=seed, alphabet=idx.payload_chars, length=idx.payload_len, exclude=exclude)
    return [_make_example(vocab, idx, capability, p, seq_len) for p in pays]


def seq_len_of(idx: DataIndex) -> int:
    """一份样本的**精确**长度：``<bos> 任务 : L×payload = L×回答 <eos>`` = ``5 + 2L``。

    回答长度恒等于 payload 长度（四个能力都是逐字符映射），所以不需要为
    ``shift`` 留余量 —— ``shift`` 只改字符值不改个数。

    **它返回的是下界而不是「随便给个够大的数」**：这个函数的全部意义就是让
    ``seq_len`` 有一个可推导的来源。写成常量会把「编码布局」与「配置值」之间的
    耦合藏起来，而那个耦合一旦断了，症状只是「多了一截 PAD」——不影响任何数值。
    """
    return 5 + 2 * idx.payload_len


# ----------------------------------------------------------------------
# 数据集构建（按内容摘要缓存）
# ----------------------------------------------------------------------
_CACHE: dict[tuple[Any, ...], list[Example]] = {}


def _cache_key(
    idx: DataIndex, capability: str, seq_len: int, n: int, seed: int, exclude: frozenset[str] = frozenset()
) -> tuple[Any, ...]:
    """缓存键。**用索引内容的摘要，不用 ``id(idx)``。**

    ``id()`` 会在对象被回收后复用，于是「换了索引但缓存里还留着上一个索引的数据」
    会在某个完全无关的时刻冒出来 —— 而它表现成「数据和索引对不上」，
    离真正的原因隔着十万八千里。

    ⚠️ ``exclude`` **必须进键**。它曾经不在键里，于是
    ``build_train_set(..., exclude=评测 payload)`` 与「不带 exclude 的同一次调用」
    撞了同一个键 —— 后者先跑（``pipeline.prepare`` 为了算基线必须拿到训练集），
    前者就静默地拿回了**含评测样本**的那一份。整个过程一句警告都没有，
    直到 ``produce_zoo`` 里那句显式的重叠检查把它抓住。

    这正是「缓存键不是输出的完整函数」这类 bug 的标准长相：它不报错，
    它只是让一次调用的**输入**看起来没起作用。修法只有一条 ——
    凡是能改变输出的参数，全都进键。
    """
    return (
        sha256_text(canonical_json(idx.to_dict())),
        capability,
        seq_len,
        n,
        seed,
        tuple(sorted(exclude)),
    )


def build_eval_sets(
    idx: DataIndex,
    vocab: Vocab,
    *,
    seq_len: int,
    n_eval: int,
    seed: int,
    capabilities: Sequence[str] | None = None,
) -> dict[str, list[Example]]:
    """**每一个**能力都要有评测集，因为「干扰」就定义在别的能力上。

    评测 payload 与训练 payload 用**不同的种子**并互相排除，于是它们是真正的不相交集。
    """
    caps = list(capabilities) if capabilities is not None else list(idx.capabilities)
    out: dict[str, list[Example]] = {}
    for cap in caps:
        s = derive_seed("eval", cap, seed)
        key = _cache_key(idx, cap, seq_len, n_eval, s)
        if key not in _CACHE:
            _CACHE[key] = build_examples(vocab, idx, cap, seq_len=seq_len, n=n_eval, seed=s)
        out[cap] = _CACHE[key]
    return out


def excluded_payloads(eval_sets: Mapping[str, Sequence[Example]]) -> dict[str, frozenset[str]]:
    """从评测集里抽出「训练集必须排除」的 payload 集合。

    **这是「训练集与评测集不相交」这条性质的唯一实现处。** 曾经它写在
    :func:`~adapterfactory.produce.produce_zoo` 里，而 :func:`~adapterfactory.pipeline.prepare`
    另建了一份不排除的训练集 —— 两份都「对」，但只有一份是干净的，
    于是「不相交」从一个不变量退化成了「取决于你问的是哪一份」。
    抽到这里之后，两个调用方拿的是同一个映射、进的是同一个缓存键。
    """
    return {cap: frozenset(ex.payload for ex in exs) for cap, exs in eval_sets.items()}


def build_train_set(
    idx: DataIndex,
    vocab: Vocab,
    *,
    seq_len: int,
    n_train: int,
    seed: int,
    capabilities: Sequence[str] | None = None,
    exclude: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, list[Example]]:
    """训练集。**只被 :mod:`produce` 使用**（流水线不训练，也就不需要它）。

    ``exclude`` 让 producer 能显式地把评测 payload 排除掉 —— 而不是靠「种子不同
    所以大概不会撞」这种概率论证。8 选 4 的空间只有 4096，撞上是会发生的。
    """
    caps = list(capabilities) if capabilities is not None else list(idx.capabilities)
    ex = exclude or {}
    out: dict[str, list[Example]] = {}
    for cap in caps:
        s = derive_seed("train", cap, seed)
        keep_out = ex.get(cap, frozenset())
        key = _cache_key(idx, cap, seq_len, n_train, s, keep_out)
        if key not in _CACHE:
            _CACHE[key] = build_examples(
                vocab, idx, cap, seq_len=seq_len, n=n_train, seed=s, exclude=keep_out
            )
        out[cap] = _CACHE[key]
    return out


def batch_of(examples: Sequence[Example]) -> tuple[list[list[int]], list[list[int]], list[list[float]]]:
    """把一个 batch 摊平成三个平行表。**空 batch 直接抛错，不返回空表。**

    空 batch 流进 :func:`~adapterfactory.net.batch_cross_entropy` 会得到
    ``0/0``；那里也会抛错，但错误信息会指向损失函数，而真正的原因在这里。
    """
    if not examples:
        raise ValueError("batch_of 收到空的 batch")
    ids = [list(e.ids) for e in examples]
    tgts = [list(e.targets) for e in examples]
    masks = [list(e.mask) for e in examples]
    for m in masks:
        if not any(v > 0 for v in m):
            raise ValueError("batch 里有一条样本的掩码全 0：它没有任何监督信号")
    return ids, tgts, masks


# ----------------------------------------------------------------------
# 位置口径
# ----------------------------------------------------------------------
def response_positions(ex: Example) -> list[int]:
    """这条样本里**被回答掩码监督**的那些位置（回答内容 + EOS）。用于损失口径的核对。"""
    return [t for t in range(len(ex.targets)) if ex.mask[t] > 0 and ex.targets[t] >= 0]


def content_positions(ex: Example) -> list[int]:
    """这条样本里**算准确率**的那些位置 —— 只算回答内容，**剔除 EOS**。

    理由见 :class:`Example` 的类文档。剔除之后，准确率才真的在问
    「回答的内容对不对」。EOS 的命中率没有被丢掉，它进了
    :func:`token_accuracy` 的第二个返回值。
    """
    return [t for t in response_positions(ex) if t != ex.eos_t]


def token_accuracy(logits: Sequence[Sequence[float]], ex: Example) -> tuple[float, float, int]:
    """逐 token 贪婪准确率。返回（内容准确率, EOS 准确率, 内容位置数）。

    **内容与 EOS 分开算**，见 :class:`Example`。EOS 那条不是装饰：它是
    「模型还会不会停」的唯一信号，而一条不会停的 Adapter 在部署时是废的 ——
    组合让它从 1.0 掉到 0.0 是必须看得见的信息。
    """
    pos = content_positions(ex)
    hit = 0
    for t in pos:
        row = logits[t]
        best = max(range(len(row)), key=row.__getitem__)
        if best == ex.targets[t]:
            hit += 1
    eos_hit = 0.0
    if ex.eos_t >= 0 and ex.eos_t < len(logits):
        row = logits[ex.eos_t]
        best = max(range(len(row)), key=row.__getitem__)
        eos_hit = 1.0 if best == ex.targets[ex.eos_t] else 0.0
    n = len(pos)
    return (hit / n if n else 0.0), eos_hit, n


def mean_accuracy(model_logits: Sequence[Sequence[Sequence[float]]], examples: Sequence[Example]) -> dict[str, float]:
    """一组样本上的平均准确率 + EOS 命中率。**两者一起返回，绝不合成一个数。**"""
    if len(model_logits) != len(examples):
        raise ValueError(f"logits 条数 {len(model_logits)} != 样本数 {len(examples)}")
    if not examples:
        raise ValueError("mean_accuracy 收到空的样本集")
    tot = 0.0
    eos = 0.0
    n = 0
    for lg, ex in zip(model_logits, examples):
        acc, e, k = token_accuracy(lg, ex)
        tot += acc * k
        eos += e
        n += k
    if n == 0:
        raise ValueError("这组样本没有任何内容位置：准确率无定义，不该静默返回 0.0")
    return {"acc": tot / n, "eos_acc": eos / len(examples), "n_positions": float(n)}


# ----------------------------------------------------------------------
# 基线
# ----------------------------------------------------------------------
def chance_level(idx: DataIndex) -> float:
    """机会水平：逐位置均匀猜。

    **不是 ``1/词表大小``。** 回答位置上可能的 token 只有「payload 字母 + 它的
    变换」那一小撮，用整个词表算会把机会水平压得毫无意义地低，于是「已经学会」
    看起来像「还没学会」。这里按 ``1/|payload_chars|`` 计，于是它是一个
    **保守偏高**（更难被超越）的机会下界 —— 偏高是安全的，偏低不是。
    """
    return 1.0 / max(1, len(idx.payload_chars))


def _argmax(counts: Mapping[int, int]) -> int:
    """众数。**平局时取小的 token id** —— 平局判定必须是确定的，否则
    「同一份数据两次运行得到同一个基线」这句话就不成立，而基线会进报告。"""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def fit_majority(idx: DataIndex, train_sets: Mapping[str, Sequence[Example]]) -> dict[str, list[int]]:
    """在**训练集**上拟合「每个回答位置出现最多的那一个 token」。

    **必须在训练集上拟合。** 在评测集上取每位置的众数是一个泄漏的基线：
    众数会显著高于真实的盲猜水平，于是启发式基线看起来比它实际能做到的高得多，
    而 gain 被系统性地低估。
    """
    out: dict[str, list[int]] = {}
    for cap, examples in train_sets.items():
        examples = list(examples)
        if not examples:
            continue
        counts: list[dict[int, int]] = [{} for _ in range(len(examples[0].targets))]
        for ex in examples:
            for t in content_positions(ex):
                d = counts[t]
                d[ex.targets[t]] = d.get(ex.targets[t], 0) + 1
        out[cap] = [(_argmax(c) if c else -1) for c in counts]
    return out


def baselines(
    idx: DataIndex,
    train_sets: Mapping[str, Sequence[Example]],
    eval_sets: Mapping[str, Sequence[Example]],
) -> dict[str, dict[str, float]]:
    """每个能力的两个基线（外加两个分量）。

    - ``chance``：逐位置均匀猜（解析值）。
    - ``majority``：**最强的不看输入的常数回答** —— 训练集上拟合每位置众数，
      在评测集上打分。它把「这任务是不是靠背一个常数就能拿分」变成可比较的数字。
    - ``copy``：**照抄输入的启发式** —— 回答第 i 个 token 直接用输入第 i 个
      payload 字符。在 ``sort`` 这类「排序后常常有些位置跟原串一样」的任务上它
      不是 0，而它到底是几分正是这一项要量出来的东西。
    - ``heuristic``：``max(majority, copy)``。**验收盯的就是它**（更强的那条）。

    四个都按逐 token 准确率算，且都**剔除 EOS**，与 :func:`mean_accuracy` 的主指标
    同口径。口径不同的基线与实测值不是同一个量，放在一起比较毫无意义。
    """
    fitted = fit_majority(idx, train_sets)
    out: dict[str, dict[str, float]] = {}
    for cap, examples in eval_sets.items():
        examples = list(examples)
        if not examples:
            continue
        runs: list[tuple[Example, list[int]]] = []
        n_pos = 0
        for ex in examples:
            pos = content_positions(ex)
            if not pos:
                continue
            runs.append((ex, pos))
            n_pos += len(pos)
        mode = fitted.get(cap) or []
        n_major = 0
        n_copy = 0
        for ex, pos in runs:
            payload_ids = [ex.ids[ex.response_start - 1 - len(ex.payload) + i] for i in range(len(ex.payload))]
            for t in pos:
                if t < len(mode) and mode[t] >= 0 and ex.targets[t] == mode[t]:
                    n_major += 1
                i = t + 1 - ex.response_start  # 这是回答段的第几个 token
                if 0 <= i < len(payload_ids) and ex.targets[t] == payload_ids[i]:
                    n_copy += 1
        denom = float(n_pos or 1)
        out[cap] = {
            "chance": chance_level(idx),
            "majority": n_major / denom,
            "copy": n_copy / denom,
            "heuristic": max(n_major, n_copy) / denom,
            "n_positions": float(n_pos),
        }
    return out


def describe(idx: DataIndex) -> str:
    caps = ", ".join(f"{tok}={name}" for tok, name in sorted(idx.tasks.items()))
    return (
        f"数据索引 {idx.source or '<内存>'}：词表 {len(idx.specials)} 特殊 + {len(idx.chars)} 字符，"
        f"payload {idx.payload_len} 位 / 字母表 {len(idx.payload_chars)}，能力 {caps}"
    )


def iter_capabilities(idx: DataIndex, wanted: Iterable[str]) -> list[str]:
    """把配置里的能力名对齐到索引，**顺序照配置给**（顺序即报告里的顺序）。"""
    caps = list(wanted)
    known = set(idx.capabilities)
    unknown = [c for c in caps if c not in known]
    if unknown:
        raise ValueError(f"配置里有数据索引不认识的能力 {unknown}（索引：{sorted(known)}）")
    seen: set[str] = set()
    out: list[str] = []
    for c in caps:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
