"""**架构墙的另一侧。** 这是全项目唯一会训练的地方，也是唯一 import 反向传播的地方。

### 它为什么存在，以及它为什么必须被隔离

AdapterFactory 的核心命题是「Adapter **产出之后**的生命周期」。要让这句话可执行，
就必须有一个「产出」的终点和一个「生命周期」的起点，且两者之间有一道**可以
执行的**边界，而不是文档里的一句承诺。

于是：

- ``produce.py`` 是 **FineTuneFactory 的替身**。它一次性真实训练出 8 个 LoRA，
  按 FineTuneFactory 的产物格式冻结进 ``data/adapters/``，之后**再也不运行**
  （除非显式改 `--force`）。
- :mod:`audit` 用 ``tokenize`` 扫 token，核对 ``registry / compose / evaluate /
  conflict / graph / pipeline`` 里**没有** ``backward`` / ``train`` / ``produce``。
- ``main.py import --from ../L3-finetunefactory/out`` 走的是另一条路：
  把**真的** FineTuneFactory 产物换进 zoo。边界因此既能从结构上守住，
  也能从功能上被验证 —— 否则「不重复 FineTuneFactory」只是一句自我声明。

### 哪些东西是刻意写在这里而不是放进引擎的

优化器、梯度裁剪、批顺序。它们是**训练**的概念，不是**张量**的概念。
放进 :mod:`net` 会让「引擎有多大」这件事失去意义，而引擎的大小决定了
「一次前向要花多少心思」—— 组合空间有 276 个配置，每个都要跑前向。

### 标定：这个文件最重要的不只是「能跑」

FineTuneFactory 的实测记录里有一条刺眼的数字：它的 LoRA 在 ``lr=0.03`` 时
**恰好等于乱猜**。也就是说，一个「跑通了、损失在降、不报错」的训练过程，
完全可能什么都没学到。

所以本模块的验收不是「跑完不报错」，而是 :func:`calibrate` 里的三条硬条件：
基座贴 chance、单 Adapter 显著超过 **heuristic** 基线、同能力两个 Adapter 次可加。
**不过关就不往下写。**
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import data as D
from . import net as N
from . import zoo as Z
from .rng import derive_seed

#: zoo 的**目录定义**（``AdapterSpec`` / ``default_zoo`` / 变体表）住在 ``zoo.py``：
#: 它们是纯描述、一次训练都不做，而放在本模块里会让只读命令 ``main.py task``
#: 为了打印「zoo 规格」那张表而必须导入整个训练模块。这里转出它们，
#: 好让既有调用方（``cli`` 的写路径、``tests/conftest``）不必改口。
from .zoo import VARIANT_ALL, VARIANT_QV, VARIANT_TARGETS, AdapterSpec, default_zoo

__all__ = [
    "TrainSettings",
    "AdapterSpec",
    "ZOO",
    "TrainReport",
    "default_zoo",
    "train_adapter",
    "build_base_weights",
    "produce_zoo",
    "zoo_dir",
]



#: 配置文件里 ``zoo.capabilities`` 为空时用的默认序。
ZOO_CAPABILITIES: tuple[str, ...] = ("upper", "shift", "reverse", "sort")


# ======================================================================
# 训练设置
# ======================================================================
@dataclass(frozen=True)
class TrainSettings:
    """训练超参。**每一项都进产物的旁证元数据** —— 不然「这个 Adapter 是哪来的」
    就只剩一个名字，而名字不解释任何数字。"""

    lr: float = 0.05
    steps: int = 240
    batch: int = 8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    n_train: int = 256

    def __post_init__(self) -> None:
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(f"未知优化器 {self.optimizer!r}（可选：adamw / sgd）")
        if self.lr <= 0:
            raise ValueError(f"lr 必须为正，得到 {self.lr}")
        if self.steps < 1 or self.batch < 1 or self.n_train < 1:
            raise ValueError("steps / batch / n_train 必须为正")

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "steps": self.steps,
            "batch": self.batch,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "optimizer": self.optimizer,
            "betas": list(self.betas),
            "n_train": self.n_train,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainSettings":
        """显式取字段 —— ``n_train`` 与 ``betas`` 都有默认值，但 ``lr`` 与 ``steps``
        没有。用 ``(**d)`` 的话，配置里多一个展示字段就会 ``TypeError``。"""
        return cls(
            lr=float(d["lr"]),
            steps=int(d["steps"]),
            batch=int(d["batch"]),
            weight_decay=float(d.get("weight_decay", 0.0)),
            grad_clip=float(d.get("grad_clip", 1.0)),
            optimizer=str(d.get("optimizer", "adamw")),
            betas=(float(d.get("betas", (0.9, 0.95))[0]), float(d.get("betas", (0.9, 0.95))[1])),
            n_train=int(d.get("n_train", 256)),
        )


# ======================================================================
# 优化器（**只在训练这一侧存在**）
# ======================================================================
def _apply(leaf: N.Tensor, fn: Any) -> None:
    """把 ``fn(值, 梯度, i, j)`` 逐元素作用到叶子上。**1D 用 j 作下标、i 恒为 0。**"""
    g = leaf.grad
    d = leaf.data
    if d and isinstance(d[0], list):
        if g is None:
            return
        for i, row in enumerate(d):
            gi = g[i]
            for j in range(len(row)):
                row[j] = fn(row[j], gi[j], i, j)
    else:
        for j in range(len(d)):
            d[j] = fn(d[j], 0.0 if g is None else g[j], 0, j)


def _numel(data: Any) -> int:
    if data and isinstance(data[0], list):
        return len(data) * len(data[0])
    return len(data)


def _zeros_like(data: Any) -> dict[tuple[int, int], float]:
    """按 ``(i, j)`` 建零状态。**1D 也用 ``(0, j)`` 而不是 ``j``。**

    这个细节看着无关紧要，但两处只要有一处不一致，优化器的状态就会在 1D
    参数上读到 ``KeyError``（好），或者 —— 如果那里恰好用了 ``defaultdict`` ——
    每次都从 0 开始，于是 ``norm_f`` 的参数永远走的是「第一步」的更新量。
    后者不报错，症状是「归一化的缩放学不动」，离原因隔了十万八千里。
    所以 :func:`_apply` 与 :func:`_zeros_like` 必须共用同一个键格式。
    """
    if data and isinstance(data[0], list):
        return {(i, j): 0.0 for i in range(len(data)) for j in range(len(data[0]))}
    return {(0, j): 0.0 for j in range(len(data))}


class _Optimizer:
    """**它只认识「叶子列表」，不认识配方。**"""

    name = "base"

    def __init__(self, leaves: Sequence[tuple[str, N.Tensor]], *, lr: float, weight_decay: float) -> None:
        self.leaves = list(leaves)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_steps = 0

    def zero_grad(self) -> None:
        """**每一步都必须调。** 引擎是累加进 ``.grad`` 的（梯度累积就靠这一点），
        漏掉清零不会报错，只会让梯度越滚越大，表现为「跑几十步之后突然发散」。"""
        for _n, t in self.leaves:
            t.grad = None

    def step(self, lr: float) -> None:
        raise NotImplementedError


class AdamW(_Optimizer):
    """Adam + **解耦**权重衰减。

    权重衰减是解耦的（``p ← p − lr·wd·p``），不是加进梯度里。两者在有自适应缩放
    时是实质性的区别：加进梯度会被 ``1/sqrt(v)`` 缩放，于是「同一份 wd」
    对不同量级的参数效果不同。
    """

    name = "adamw"

    def __init__(self, leaves: Any, *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
        super().__init__(leaves, lr=lr, weight_decay=weight_decay)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.m = [_zeros_like(t.data) for _n, t in self.leaves]
        self.v = [_zeros_like(t.data) for _n, t in self.leaves]

    def step(self, lr: float) -> None:
        self.n_steps += 1
        bc1 = 1.0 - self.b1**self.n_steps
        bc2 = 1.0 - self.b2**self.n_steps
        for idx, (_name, leaf) in enumerate(self.leaves):
            if leaf.grad is None:
                continue
            m, v = self.m[idx], self.v[idx]
            b1, b2, eps, wd = self.b1, self.b2, self.eps, self.weight_decay

            def upd(p: float, gi: float, i: int, j: int) -> float:
                key = (i, j)
                mi = b1 * m[key] + (1.0 - b1) * gi
                vi = b2 * v[key] + (1.0 - b2) * gi * gi
                m[key] = mi
                v[key] = vi
                p2 = p - lr * wd * p if wd else p
                return p2 - lr * (mi / bc1) / (math.sqrt(vi / bc2) + eps)

            _apply(leaf, upd)


class SGD(_Optimizer):
    """带动量的 SGD。存在的唯一理由是让「优化器」这条轴不至于是硬编码的。"""

    name = "sgd"

    def __init__(self, leaves: Any, *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> None:
        super().__init__(leaves, lr=lr, weight_decay=weight_decay)
        self.momentum = float(betas[0])
        self.buf = [_zeros_like(t.data) for _n, t in self.leaves]

    def step(self, lr: float) -> None:
        self.n_steps += 1
        for idx, (_name, leaf) in enumerate(self.leaves):
            if leaf.grad is None:
                continue
            buf = self.buf[idx]
            mu, wd = self.momentum, self.weight_decay

            def upd(p: float, gi: float, i: int, j: int) -> float:
                key = (i, j)
                bi = mu * buf[key] + gi
                buf[key] = bi
                p2 = p - lr * wd * p if wd else p
                return p2 - lr * bi

            _apply(leaf, upd)


def make_optimizer(leaves: Any, s: TrainSettings) -> _Optimizer:
    """按设置造优化器。**名字不认识就报错，绝不静默退回默认值** ——
    静默退回的后果是「配置里写的是 sgd，实际跑的是 adamw」，而报告里那一行看着很正常。"""
    cls = {"adamw": AdamW, "sgd": SGD}.get(s.optimizer)
    if cls is None:
        raise ValueError(f"未知优化器 {s.optimizer!r}（可选：adamw / sgd）")
    return cls(leaves, lr=s.lr, betas=s.betas, eps=s.eps, weight_decay=s.weight_decay)


def clip_grad_norm_(leaves: Sequence[tuple[str, N.Tensor]], max_norm: float) -> float:
    """全局范数裁剪，**返回裁剪前的范数**。

    返回裁剪前的范数不是为了好看：它是「训练是否在正常范围」的第一手信号，
    它必须进训练报告。只裁剪不记录的话，一个**每次都被裁满**的训练过程与一个
    健康的训练过程在报告里长得一模一样 —— 前者的损失也会降，只是降得比它能
    达到的慢。``max_norm <= 0`` 表示关闭，此时返回 0 并不做任何事。
    """
    if max_norm <= 0:
        return 0.0
    total = 0.0
    for _n, leaf in leaves:
        g = leaf.grad
        if g is None:
            continue
        if g and isinstance(g[0], list):
            for row in g:
                for v in row:
                    total += v * v
        else:
            for v in g:
                total += v * v
    norm = math.sqrt(total)
    if norm <= max_norm or norm == 0.0:
        return norm
    factor = max_norm / norm
    for _n, leaf in leaves:
        g = leaf.grad
        if g is None:
            continue
        if g and isinstance(g[0], list):
            for row in g:
                for j in range(len(row)):
                    row[j] *= factor
        else:
            for j in range(len(g)):
                g[j] *= factor
    return norm


def batch_order(n: int, batch: int, *, seed: int, steps: int) -> list[list[int]]:
    """确定性的批顺序，**每个 epoch 洗一次牌**。

    用 ``derive_seed("batch", seed, epoch)`` 而不是一个跨 epoch 连续的随机数流：
    后者在「两次运行步数不同」时会让前面的批完全一样、只在结尾分叉，
    于是「改了 steps 之后前面的数字应该逐位不变」这条直觉成立 ——
    而它其实反过来说明洗牌没有真的覆盖到全部样本。
    """
    out: list[list[int]] = []
    epoch = 0
    while len(out) < steps:
        rng = random.Random(derive_seed("batch", seed, epoch))
        order = list(range(n))
        rng.shuffle(order)
        for start in range(0, n - batch + 1, batch):
            if len(out) >= steps:
                break
            out.append(order[start : start + batch])
        epoch += 1
    return out


# ======================================================================
# 基座
# ======================================================================
def build_base_weights(shape: N.ModelShape, seed: int) -> dict[str, Any]:
    """基座权重。**它是 (结构, 种子) 的纯函数**，所以不落盘 —— 落盘只会多出一个
    可以漂移的自由度，而它换不来任何东西（本项目里没有「同一个结构的两份基座」）。"""
    return N.init_base_weights(shape, seed)


# ======================================================================
# 训练
# ======================================================================
@dataclass
class TrainReport:
    """一次训练的可核对记录。**它进产物的旁证元数据**，于是「这个 Adapter 是
    怎么来的」永远可回答。"""

    label: str
    steps: int
    first_loss: float
    last_loss: float
    grad_norm_max: float
    grad_norm_first: float
    n_params: int
    n_train: int
    #: 训练结束时在**评测集**上的逐能力准确率 —— 这是标定表的原料。
    eval_acc: dict[str, float] = field(default_factory=dict)
    #: EOS 命中率。它单独列，理由是「一条不会停的 Adapter 在部署时是废的」。
    eval_eos: dict[str, float] = field(default_factory=dict)

    @property
    def loss_drop(self) -> float:
        return self.first_loss - self.last_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "steps": self.steps,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "loss_drop": self.loss_drop,
            "grad_norm_first": self.grad_norm_first,
            "grad_norm_max": self.grad_norm_max,
            "n_params": self.n_params,
            "n_train": self.n_train,
            "eval_acc": dict(sorted(self.eval_acc.items())),
            "eval_eos": dict(sorted(self.eval_eos.items())),
        }


def train_adapter(
    shape: N.ModelShape,
    base_weights: dict[str, Any],
    spec: AdapterSpec,
    train_sets: Mapping[str, Sequence[D.Example]],
    eval_sets: Mapping[str, Sequence[D.Example]],
    settings: TrainSettings,
    *,
    seed: int,
) -> tuple[N.Adapter, TrainReport]:
    """训一个 LoRA。

    **每一步都从 ``base_weights`` 重新前向**（它是常量），只更新 LoRA 叶子。
    换句话说：这里没有「基座被悄悄改动」这种可能 —— 但也没有任何机制阻止你
    把 ``base_weights`` 换成别的，所以 :func:`produce_zoo` 对 8 个 Adapter
    用的是**同一个** ``base_weights`` 对象。它们是同一根基座上的八个插件，
    这是「组合」这件事有意义的前提。
    """
    if spec.capability not in train_sets:
        raise KeyError(f"训练集里没有能力 {spec.capability!r}（有：{sorted(train_sets)}）")
    examples = list(train_sets[spec.capability])
    if settings.n_train and len(examples) > settings.n_train:
        examples = examples[: settings.n_train]

    ad = N.init_adapter(shape, spec.targets, spec.rank, derive_seed("lora-init", spec.label, seed))
    ad.alpha = float(spec.alpha)
    ad.label = spec.label
    ad.capability = spec.capability
    leaves: list[tuple[str, N.Tensor]] = [(k, N.constant(v, name=k)) for k, v in ad.tensors.items()]

    opt = make_optimizer(leaves, settings)
    order = batch_order(len(examples), settings.batch, seed=derive_seed("order", spec.label, seed), steps=settings.steps)

    first_loss = float("nan")
    last_loss = float("nan")
    gnorm_first = 0.0
    gnorm_max = 0.0
    for step in range(settings.steps):
        idxs = order[step]
        pairs: list[tuple[Any, Sequence[int], Sequence[float]]] = []
        for i in idxs:
            ex = examples[i]
            logits = N.forward_train(
                base_weights, dict(leaves), spec.targets, ex.ids, shape, scale_value=ad.scale
            )
            pairs.append((logits, ex.targets, ex.mask))
        loss = N.batch_cross_entropy(pairs)
        opt.zero_grad()
        loss.backward()
        gn = clip_grad_norm_(leaves, settings.grad_clip)
        if step == 0:
            gnorm_first = gn
        gnorm_max = max(gnorm_max, gn)
        opt.step(settings.lr)
        v = float(loss.data[0])
        if step == 0:
            first_loss = v
        last_loss = v

    # 训练结束：把叶子里的值写回一个 Adapter（**用同一份数据，不是重新初始化**）
    trained = N.Adapter(
        tensors={k: t.data for k, t in leaves},
        targets=spec.targets,
        rank=spec.rank,
        alpha=float(spec.alpha),
        label=spec.label,
        capability=spec.capability,
    )
    rep = TrainReport(
        label=spec.label,
        steps=settings.steps,
        first_loss=first_loss,
        last_loss=last_loss,
        grad_norm_first=gnorm_first,
        grad_norm_max=gnorm_max,
        n_params=trained.n_params,
        n_train=len(examples),
    )
    for cap, exs in eval_sets.items():
        lgs = [N.logits_of(base_weights, ex.ids, shape, adapters=[trained]) for ex in exs]
        m = D.mean_accuracy(lgs, exs)
        rep.eval_acc[cap] = m["acc"]
        rep.eval_eos[cap] = m["eos_acc"]
    return trained, rep


# ======================================================================
# zoo 落盘
# ======================================================================
def zoo_dir(root: str | Path = ".") -> Path:
    return Path(root) / "data" / "adapters"


def produce_zoo(
    *,
    root: str | Path = ".",
    shape: N.ModelShape,
    base_seed: int,
    specs: Sequence[AdapterSpec],
    settings: TrainSettings,
    vocab: D.Vocab,
    idx: D.DataIndex,
    seq_len: int,
    eval_sets: Mapping[str, Sequence[D.Example]],
    train_seed: int,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """训出整个 zoo 并冻结进 ``data/adapters/``。

    **训练集显式排除评测 payload。** 8 选 4 的空间只有 4096，靠「种子不同所以
    大概不会撞」是不够的 —— 一旦撞上，那条 Adapter 的 gain 里就混进了记忆，
    而所有组合的 gain 会一起偏高把差异压扁。

    ``force=False`` 时，已有的产物**不重训**（沿用磁盘上的）。这让 ``produce``
    成为一次性动作：AdapterFactory 的日常运行不该重新训练任何东西。
    """
    out_dir = zoo_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_weights = build_base_weights(shape, base_seed)

    exclude = D.excluded_payloads(eval_sets)
    train_sets = D.build_train_set(
        idx,
        vocab,
        seq_len=seq_len,
        n_train=settings.n_train,
        seed=train_seed,
        capabilities=sorted({s.capability for s in specs}),
        exclude=exclude,
    )
    # 拒绝采样写错了不会报错，只会让一批训练样本同时出现在评测集里，
    # 于是所有 gain 一起偏高、差异被压扁 —— 而差异正是本项目唯一要测的东西。
    # 按 **payload** 比对（不是 ids：ids 含任务 token 与 PAD，两条不同 payload
    # 的 ids 前缀总是相同的，按 ids 比会得到一堆假阳性）。
    for cap, exs in eval_sets.items():
        if not exs:
            continue
        overlap = {e.payload for e in exs} & {e.payload for e in train_sets.get(cap, [])}
        if overlap:
            raise ValueError(f"能力 {cap} 的评测集与训练集重叠了 {len(overlap)} 条：{sorted(overlap)[:3]}…")

    reports: dict[str, Any] = {}
    for spec in specs:
        stem = out_dir / spec.label
        if stem.with_suffix(".json").is_file() and not force:
            _tensors, rec = Z.read_artifact(stem)
            # 比的是**配方**：这份产物要的是哪一份（``spec``）、怎么训的
            # （``settings``）、架在哪根基座上。三项全同才叫「已经有货了」。
            stale = Z.recipe_mismatch(
                rec,
                spec=spec.to_dict(),
                train=settings.to_dict(),
                base_key=shape.key,
                base_seed=base_seed,
            )
            if not stale:
                if verbose:
                    print(f"  [skip] {spec.label}（配方逐字段相同，force=False）")
                reports[spec.label] = {"reused": True, "digest": rec.digest, "n_params": rec.n_params}
                continue
            # **沿用是靠比对配方，不是靠文件在不在。** 这一行以前写的是
            # ``if stem.with_suffix(".json").is_file()``，于是把 ``zoo.rank``
            # 从 4 改成 8 之后，8 个产物被原样沿用：配置、报告、registry 三处
            # 都写着 rank=8，权重却全是 rank=4 训出来的。参数量、MACs、
            # rank_total 全跟着错，而**没有任何一处会不一致** ——
            # 每个数字单看都合理，只有拿来比才不对。
            print(f"  [重训] {spec.label}：磁盘上的产物与本次配方不符 —— {stale}")
        ad, rep = train_adapter(shape, base_weights, spec, train_sets, eval_sets, settings, seed=base_seed)
        tensors, rank_total = Z.compose_factors([ad], [1.0], shape)
        rec = Z.write_artifact(
            stem,
            tensors,
            name=spec.label,
            targets=list(spec.targets),
            rank=rank_total,
            # 训练出来的产物 scale 已经是 1.0（系数折进了 B），见 zoo 的模块 docstring。
            alpha=float(rank_total),
            capability=spec.capability,
            variant=spec.variant,
            base_key=shape.key,
            base_seed=base_seed,
            source="produce",
            # 训练设置与报告一起进旁证 —— 「这个 Adapter 是哪来的」必须永远可回答。
            # 两者**分开两个字段**，不合并：它们都有 ``steps`` / ``n_train``，
            # 用 ``dict(a, **b)`` 合会在两边静默覆盖掉一个，而覆盖掉的是哪一边
            # 取决于写的顺序 —— 那种 bug 看不出来。
            train=settings.to_dict(),
            train_report=rep.to_dict(),
            spec=spec.to_dict(),
        )
        reports[spec.label] = {"reused": False, **rep.to_dict(), "digest": rec.digest}
        if verbose:
            accs = " ".join(f"{c}={a:.3f}" for c, a in sorted(rep.eval_acc.items()))
            print(
                f"  [train] {spec.label:14s} loss {rep.first_loss:.4f}→{rep.last_loss:.4f} "
                f"|grad|max {rep.grad_norm_max:.2f} n={rep.n_params}  {accs}"
            )
    return {"reports": reports, "dir": str(out_dir), "base_key": shape.key}


def _json_dump(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
