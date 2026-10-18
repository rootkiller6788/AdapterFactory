"""一次性脚本，跑完即弃：核对引擎的形状、梯度与 MACs。"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adapterfactory import data as D
from adapterfactory import net as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAIL: list[str] = []


def chk(name: str, cond: bool, extra: str = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + name + (("  " + extra) if extra else ""))
    if not cond:
        FAIL.append(name)


print("=== 1. 数据索引 ===")
idx = D.load_index(os.path.join(ROOT, "data", "datasets", "index.json"))
vocab = D.build_vocab(idx)
print(" ", D.describe(idx))
chk("词表大小 == 26", vocab.size == 26, f"实际 {vocab.size}")
chk("seq_len == 13", D.seq_len_of(idx) == 13, f"实际 {D.seq_len_of(idx)}")

SL = 16
ev = D.build_eval_sets(idx, vocab, seq_len=SL, n_eval=8, seed=7)
tr = D.build_train_set(idx, vocab, seq_len=SL, n_train=8, seed=7)
chk("四个能力都有评测集", sorted(ev) == ["reverse", "shift", "sort", "upper"], str(sorted(ev)))
ex0 = ev["shift"][0]
print("  示例 shift:", ex0.payload, "->", D.TASK_FNS["shift"](ex0.payload))
print("  ids:", ex0.ids)
print("  positions:", D.content_positions(ex0), "eos_t:", ex0.eos_t)
chk("内容位置数 == payload 长", len(D.content_positions(ex0)) == idx.payload_len)
chk("mask 位置数 == payload+1", len(D.response_positions(ex0)) == idx.payload_len + 1)

print("\n=== 2. 基座前向 ===")
shape = N.ModelShape(vocab=vocab.size, d_model=24, n_layers=2, n_heads=4, d_ff=48, max_len=SL)
print("  shape:", shape.key, " head_dim:", shape.head_dim)
shapes = N.param_shapes(shape)
w = N.init_base_weights(shape, seed=1234)
n_formula = sum(math.prod(s) for s in shapes.values())
n_actual = sum(math.prod(N._shape_of(v)) for v in w.values())
chk("参数量公式 == 实测", n_formula == n_actual, f"{n_formula} vs {n_actual}")
print("  参数量:", n_formula)
chk("权重名 == 参数名表", list(w.keys()) == N.layer_param_names(shape))
chk("基座键确定性", N.init_base_weights(shape, seed=1234)[N.EMB][0][0] == w[N.EMB][0][0])

with N.mac_counter() as mc:
    lg = N.logits_of(w, ex0.ids, shape)
chk("logits 形状 [T, vocab]", len(lg) == SL and len(lg[0]) == vocab.size, f"{len(lg)}x{len(lg[0])}")
print("  MACs/前向:", mc.total)
acc, eos, k = D.token_accuracy(lg, ex0)
print(f"  未训练基座: acc={acc:.4f} eos={eos:.0f} n={k}")
print("  baselines:", {c: round(b["heuristic"], 3) for c, b in D.baselines(idx, tr, ev).items()})

print("\n=== 3. 梯度数值差分（最关键的一步）===")


def loss_at(weights: dict, example) -> float:
    logits = N.forward(weights, example.ids, shape)
    t = N.batch_cross_entropy([(logits, example.targets, example.mask)])
    return t.data[0]


# 引擎是「闭包建图 + 拓扑反向」。要在同一张图上做数值差分，必须用同一个参数对象。
def make_param(weights: dict, name: str) -> N.Tensor:
    t = N.constant(weights[name], name=name)
    t.freeze()
    return t


ex = ev["upper"][0]
worst = 0.0
for name in ["layers.0.q_proj", "layers.0.down_proj", N.EMB, "layers.1.o_proj", "layers.1.norm1"]:
    wt = {k: v for k, v in w.items()}
    leaf = N.constant(wt[name], name=name)
    wt[name] = leaf
    logits = N.forward(wt, ex.ids, shape)
    t = N.batch_cross_entropy([(logits, ex.targets, ex.mask)])
    t.backward()
    g = leaf.grad
    eps = 1e-5
    maxrel = 0.0
    probes = [(0, 0)] if not N._is2d(w[name]) else [(0, 0), (1, 5), (len(w[name]) - 1, len(w[name][0]) - 1)]
    for (i, j) in probes:
        base = leaf.data  # 直接改叶子张量的数据；改的是同一个对象，所以下次前向会看到
        if N._is2d(base):
            base[i][j] += eps
            lp = loss_at(wt, ex)
            base[i][j] -= 2 * eps
            lm = loss_at(wt, ex)
            base[i][j] += eps
            ana = g[i][j]
        else:
            base[i] += eps
            lp = loss_at(wt, ex)
            base[i] -= 2 * eps
            lm = loss_at(wt, ex)
            base[i] += eps
            ana = g[i]
        num = (lp - lm) / (2 * eps)
        rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
        maxrel = max(maxrel, rel)
    print(f"  {name:24s} 最大相对误差 {maxrel:.3e}")
    worst = max(worst, maxrel)
chk("梯度与数值差分一致（<1e-5）", worst < 1e-5, f"worst {worst:.3e}")

print("\n=== 4. 适配器：B=0 时与基座逐位相同 ===")
ad = N.init_adapter(shape, ["q_proj", "v_proj"], rank=4, seed=99)
ad.label = "upper-qv"
print("  适配器参数:", ad.n_params, " scale:", ad.scale)
lg_a = N.logits_of(w, ex0.ids, shape, adapters=[ad])
d0 = N.max_abs_diff(lg, lg_a)
chk("B=0 ⇒ 适配器初始与基座逐位相同", d0 == 0.0, f"maxdiff {d0}")
merged0 = N.merge_into(w, [ad])
d1 = max(N.max_abs_diff(w[k], merged0[k]) for k in w)
chk("B=0 ⇒ 合并不改权重", d1 == 0.0, f"maxdiff {d1}")

print("\n=== 5. 合并 vs 逐个施加 ===")
import random as _r

for name in ad.tensors:
    if name.endswith(".lora_B"):
        rng = _r.Random(N.derive_seed("test", name))
        ad.tensors[name] = [[rng.gauss(0, 0.05) for _ in range(len(ad.tensors[name][0]))] for _ in range(len(ad.tensors[name]))]
merged = N.merge_into(w, [ad])
lg_seq = N.logits_of(w, ex0.ids, shape, adapters=[ad])
lg_mrg = N.logits_of(merged, ex0.ids, shape)
err = N.max_abs_diff(lg_seq, lg_mrg)
print(f"  merge_error = {err:.3e}")
chk("合并 == 逐个施加（<1e-12）", err < 1e-12)
chk("合并不是空操作", err > 0.0 or N.max_abs_diff(w["layers.0.q_proj"], merged["layers.0.q_proj"]) > 0)

print("\n=== 6. 两个适配器的可加性恒等 ===")
ad2 = N.init_adapter(shape, ["q_proj", "v_proj"], rank=4, seed=100)
for name in ad2.tensors:
    if name.endswith(".lora_B"):
        rng = _r.Random(N.derive_seed("test2", name))
        ad2.tensors[name] = [[rng.gauss(0, 0.05) for _ in range(len(ad2.tensors[name][0]))] for _ in range(len(ad2.tensors[name]))]
m2 = N.merge_into(w, [ad, ad2])
seq2 = N.logits_of(w, ex0.ids, shape, adapters=[ad, ad2])
err2 = N.max_abs_diff(seq2, N.logits_of(m2, ex0.ids, shape))
print(f"  merge_error(2 adapters) = {err2:.3e}")
chk("两适配器合并 == 逐个施加（<1e-12）", err2 < 1e-12)
chk("与单适配器结果不同", N.max_abs_diff(lg_seq, seq2) > 0)

print("\n=== 7. merge 系数与层覆盖 ===")
m_sum = N.merge_into(w, [ad, ad2])
m_avg = N.merge_into(w, [ad, ad2], weights_by_adapter=[0.5, 0.5])
chk("平均 != 求和", N.max_abs_diff(m_sum["layers.0.q_proj"], m_avg["layers.0.q_proj"]) > 0)
# 这一条专门钉住「第 0 层被跳过」那个 bug：它不报错，只能靠「每一层都真的变了」抓。
base_keys = N.param_shapes_of(w)
touched = N.expand_targets(base_keys, ["q_proj", "v_proj"])
print("  展开的挂点名:", touched)
chk("第 0 层与最后一层都在展开结果里", touched[:2] == ["layers.0.q_proj", "layers.0.v_proj"] and "layers.1.v_proj" in touched)
for k in touched:
    if N.max_abs_diff(w[k], m_sum[k]) == 0.0:
        chk(f"合并在 {k} 上真的改了权重", False)
        break
else:
    chk("合并在每个挂点上都真的改了权重", True)
ad_all = N.init_adapter(shape, list(N.LORA_ABLE_MODULES), rank=4, seed=101)
for name in ad_all.tensors:
    if name.endswith(".lora_B"):
        rng = _r.Random(N.derive_seed("test3", name))
        ad_all.tensors[name] = [[rng.gauss(0, 0.05) for _ in range(len(ad_all.tensors[name][0]))] for _ in range(len(ad_all.tensors[name]))]
t_all = N.expand_targets(base_keys, ad_all.targets)
chk("全模块覆盖展开出 2 层 × 7 模块 = 14 个挂点", len(t_all) == 2 * len(N.LORA_ABLE_MODULES), f"实际 {len(t_all)}")

print("\n=== 8. 缓存不串味 ===")
chk("derive_seed 稳定", N.derive_seed("eval", "upper", 7) == N.derive_seed("eval", "upper", 7))
chk("不同部件不同种子", N.derive_seed("eval", "upper", 7) != N.derive_seed("eval", "upper", 8))

print()
if FAIL:
    print(f"!!! {len(FAIL)} 项失败: {FAIL}")
    sys.exit(1)
print("全部通过。")
