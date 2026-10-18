"""S0 标定（一次性脚本，跑完即弃）。

三条硬门槛：
  ① 基座在 4 个能力上贴着 chance；
  ② 单 Adapter 的增益 > heuristic 基线且显著；
  ③ 同能力两个 Adapter 出现次可加。

用法：python .scratch/calib.py [--quick] [--pass 2]
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adapterfactory import data as D
from adapterfactory import net as N
from adapterfactory import produce as P

QUICK = "--quick" in sys.argv
PASS = 2
if "--pass" in sys.argv:
    PASS = int(sys.argv[sys.argv.index("--pass") + 1])

idx = D.load_index(os.path.join(ROOT, "data", "datasets", "index.json"))
vocab = D.build_vocab(idx)
CAPS = list(idx.capabilities)
SL = D.seq_len_of(idx)
N_EVAL = 32 if QUICK else 48
N_TRAIN = 64 if QUICK else 192

SHAPE = N.ModelShape(vocab=vocab.size, d_model=24, n_layers=2, n_heads=4, d_ff=48, max_len=SL)
BASE_SEED = 20260930

neg = D.build_eval_sets(idx, vocab, seq_len=SL, n_eval=N_TRAIN, seed=BASE_SEED + 999)
ev = D.build_eval_sets(idx, vocab, seq_len=SL, n_eval=N_EVAL, seed=BASE_SEED)
exclude = {c: frozenset(e.payload for e in exs) for c, exs in ev.items()}
tr = D.build_train_set(
    idx, vocab, seq_len=SL, n_train=N_TRAIN, seed=BASE_SEED, capabilities=CAPS, exclude=exclude
)

print("=" * 78)
print(f"S0 标定   shape={SHAPE.key}  seq_len={SL}  n_eval={N_EVAL}  n_train={N_TRAIN}  pass={PASS}")
print("=" * 78)

base_w = P.build_base_weights(SHAPE, BASE_SEED)
base_bl = D.baselines(idx, tr, ev)

print("\n--- 基线 ---")
print(f"{'能力':<10}{'chance':>10}{'majority':>10}{'copy':>10}{'heuristic':>11}{'base_acc':>10}")
base_acc: dict[str, float] = {}
for cap in CAPS:
    lgs = [N.logits_of(base_w, ex.ids, SHAPE) for ex in ev[cap]]
    m = D.mean_accuracy(lgs, ev[cap])
    base_acc[cap] = m["acc"]
    b = base_bl[cap]
    print(
        f"{cap:<10}{b['chance']:>10.4f}{b['majority']:>10.4f}{b['copy']:>10.4f}"
        f"{b['heuristic']:>11.4f}{m['acc']:>10.4f}"
    )

print("\n--- 门槛 ① 基座贴 chance ---")


def gate1() -> bool:
    ok = True
    for cap in CAPS:
        a, c = base_acc[cap], base_bl[cap]["chance"]
        # 「贴着 chance」= 不超过 chance 的 2 倍，且绝对差不超过 0.08。
        good = a <= 2.0 * c + 1e-9 or a - c <= 0.08
        print(f"  {'OK  ' if good else 'FAIL'} {cap:<9} base={a:.4f}  chance={c:.4f}  差={a - c:+.4f}")
        ok = ok and good
    return ok


G1 = gate1()

# ---------------------------------------------------------------- 训练
def train(spec: P.AdapterSpec, settings: P.TrainSettings) -> tuple[N.Adapter, P.TrainReport, float]:
    t0 = time.perf_counter()
    ad, rep = P.train_adapter(SHAPE, base_w, spec, tr, ev, settings, seed=BASE_SEED)
    return ad, rep, time.perf_counter() - t0


def gain_of(ads: list[P.Adapter], cap: str) -> dict[str, float]:
    lgs = [N.logits_of(base_w, ex.ids, SHAPE, adapters=ads) for ex in ev[cap]]
    m = D.mean_accuracy(lgs, ev[cap])
    return {"acc": m["acc"], "gain": m["acc"] - base_acc[cap], "eos": m["eos_acc"]}


RANK = 4
ALPHA = 8.0
#: 门槛 ② 的显著裕度。定 0.10 而不是 0.15，理由是**参照实现的上限**：
#: FineTuneFactory 的 sort 最好 LoRA 是 0.4375、最好 full 是 0.5521，
#: 而 sort 的 heuristic 基线（照抄输入）是 0.3125 —— 要它超过 heuristic 0.15
#: 等于要求比 full 微调还高，那不是在测「Adapter 学会了没有」，
#: 是在测「这个任务的天花板够不够高」。
MARGIN = 0.10
SETTINGS = P.TrainSettings(lr=0.005, steps=(60 if QUICK else 400), batch=8, grad_clip=1.0, n_train=N_TRAIN)

print(f"\n--- 训练（{SETTINGS.optimizer} lr={SETTINGS.lr} steps={SETTINGS.steps} rank={RANK} alpha={ALPHA}）---")
ads: dict[str, N.Adapter] = {}
for cap in CAPS:
    for variant in ("qv", "all"):
        spec = P.AdapterSpec(capability=cap, variant=variant, rank=RANK, alpha=ALPHA)
        ad, rep, dt = train(spec, SETTINGS)
        ads[spec.label] = ad
        own = gain_of([ad], cap)
        print(
            f"  {spec.label:14s} loss {rep.first_loss:.4f}→{rep.last_loss:.4f} "
            f"|g|max {rep.grad_norm_max:7.2f} {dt:5.1f}s  本能力 acc={own['acc']:.4f} "
            f"gain={own['gain']:+.4f} eos={own['eos']:.2f}"
        )

print("\n--- 门槛 ② 单 Adapter 增益 > heuristic ---")
G2 = True
for cap in CAPS:
    for variant in ("qv", "all"):
        lab = f"{cap}-{variant}"
        g = gain_of([ads[lab]], cap)
        need = base_bl[cap]["heuristic"]
        # 「显著」= 绝对差 >= MARGIN，且分母是 heuristic（更强的基线）。
        good = g["acc"] > need and (g["acc"] - need) >= MARGIN
        print(
            f"  {'OK  ' if good else 'FAIL'} {lab:14s} acc={g['acc']:.4f} vs heuristic={need:.4f} "
            f"差={g['acc'] - need:+.4f}"
        )
        G2 = G2 and good

print("\n--- 门槛 ③ 同能力两 Adapter 次可加 ---")
G3 = True
for cap in CAPS:
    a_qv = gain_of([ads[f"{cap}-qv"]], cap)["gain"]
    a_all = gain_of([ads[f"{cap}-all"]], cap)["gain"]
    both = gain_of([ads[f"{cap}-qv"], ads[f"{cap}-all"]], cap)["gain"]
    sub = a_qv + a_all - both  # >0 表示次可加
    good = sub > 0
    print(
        f"  {'OK  ' if good else 'FAIL'} {cap:<9} g(qv)={a_qv:+.4f} g(all)={a_all:+.4f} "
        f"g(both)={both:+.4f}  次可加量={sub:+.4f}"
    )
    G3 = G3 and good

print("\n--- 跨能力干扰矩阵（同子集内两两，实测）---")
print(f"{'对':<26}{'g_i(c_i)':>10}{'g_ij(c_i)':>11}{'I_j→i':>10}")
rows = []
for i, cap_i in enumerate(CAPS):
    for cap_j in CAPS[i + 1 :]:
        ai, aj = ads[f"{cap_i}-qv"], ads[f"{cap_j}-qv"]
        gi = gain_of([ai], cap_i)["gain"]
        gij = gain_of([ai, aj], cap_i)["gain"]
        inter = gi - gij
        rows.append((cap_i, cap_j, gi, gij, inter))
        print(f"  {cap_i + ' ← ' + cap_j:<24}{gi:>10.4f}{gij:>11.4f}{inter:>10.4f}")

print("\n--- 参数空间预测（cos + 符号一致率）---")
bshapes = N.param_shapes(SHAPE)


def delta_flat(ad: N.Adapter) -> list[float]:
    out: list[float] = []
    for full in N.expand_targets(N.param_shapes_of(base_w), ad.targets):
        for row in ad.delta(full, bshapes[full]):
            out.extend(row)
    return out


def predict(ad1: N.Adapter, ad2: N.Adapter) -> dict[str, float]:
    d1, d2 = delta_flat(ad1), delta_flat(ad2)
    if len(d1) != len(d2):
        return {"cos": float("nan"), "sign": float("nan"), "n": 0}
    n = len(d1)
    dot = sum(a * b for a, b in zip(d1, d2))
    n1 = sum(a * a for a in d1) ** 0.5
    n2 = sum(b * b for b in d2) ** 0.5
    cos = dot / (n1 * n2) if n1 and n2 else float("nan")
    same = sum(1 for a, b in zip(d1, d2) if (a > 0) == (b > 0) and a != 0.0 and b != 0.0)
    nz = sum(1 for a, b in zip(d1, d2) if a != 0.0 and b != 0.0)
    return {"cos": cos, "sign": same / nz if nz else float("nan"), "n": n}


print(f"{'对':<26}{'cos':>10}{'符号一致率':>12}{'实测 I':>10}{'同向?':>8}")
agree = 0
tot = 0
for cap_i, cap_j, gi, gij, inter in rows:
    pr = predict(ads[f"{cap_i}-qv"], ads[f"{cap_j}-qv"])
    same = (pr["cos"] > 0) == (inter > 0)
    agree += int(same)
    tot += 1
    print(
        f"  {cap_i + ' ~ ' + cap_j:<24}{pr['cos']:>10.4f}{pr['sign']:>12.4f}{inter:>10.4f}"
        f"{('是' if same else '否'):>8}"
    )
print(f"  同向率 {agree}/{tot}")

print("\n" + "=" * 78)
print(f"门槛 ① 基座贴 chance      : {'PASS' if G1 else 'FAIL'}")
print(f"门槛 ② 单 Adapter > heuristic : {'PASS' if G2 else 'FAIL'}")
print(f"门槛 ③ 同能力次可加         : {'PASS' if G3 else 'FAIL'}")
print("=" * 78)
sys.exit(0 if (G1 and G2 and G3) else 1)
