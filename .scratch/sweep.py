"""S0 超参扫描（一次性脚本，跑完即弃）：为什么 -all 学不动、sort 要多少步。

用法：python .scratch/sweep.py <组名>     组名 ∈ {all-lr, sort-steps}
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from adapterfactory import data as D
from adapterfactory import net as N
from adapterfactory import produce as P

GROUP = sys.argv[1] if len(sys.argv) > 1 else "all-lr"

idx = D.load_index(os.path.join(ROOT, "data", "datasets", "index.json"))
vocab = D.build_vocab(idx)
SL = D.seq_len_of(idx)
SEED = 20260930
CAPS = list(idx.capabilities)

SHAPE = N.ModelShape(vocab=vocab.size, d_model=24, n_layers=2, n_heads=4, d_ff=48, max_len=SL)
base_w = P.build_base_weights(SHAPE, SEED)

N_TRAIN, N_EVAL = 96, 32
ev = D.build_eval_sets(idx, vocab, seq_len=SL, n_eval=N_EVAL, seed=SEED)
exclude = {c: frozenset(e.payload for e in exs) for c, exs in ev.items()}
tr = D.build_train_set(idx, vocab, seq_len=SL, n_train=N_TRAIN, seed=SEED, capabilities=CAPS, exclude=exclude)
bl = D.baselines(idx, tr, ev)
base_acc = {c: D.mean_accuracy([N.logits_of(base_w, e.ids, SHAPE) for e in ev[c]], ev[c])["acc"] for c in CAPS}
print(f"chance={D.chance_level(idx):.4f}")
print(f"heuristic={{ {', '.join(f'{c}: {bl[c][chr(104)+chr(101)+chr(117)+chr(114)+chr(105)+chr(115)+chr(116)+chr(105)+chr(99)]:.3f}' for c in CAPS)} }}")
print(f"base={{ {', '.join(f'{c}: {base_acc[c]:.3f}' for c in CAPS)} }}\n")

RUNS: list[tuple[str, str, int, float, float, int]] = []
if GROUP == "all-lr":
    for cap in ("shift", "reverse"):
        for lr in (0.005, 0.02, 0.06):
            RUNS.append((cap, "all", 4, 8.0, lr, 200))
elif GROUP == "sort-steps":
    for cap in ("sort", "upper"):
        for variant in ("qv", "all"):
            RUNS.append((cap, variant, 4, 8.0, 0.005, 400))
elif GROUP == "lr2":
    for cap, variant in (("sort", "all"), ("upper", "all"), ("sort", "qv"), ("reverse", "qv")):
        for lr in (0.002, 0.005, 0.012):
            RUNS.append((cap, variant, 4, 8.0, lr, 400))
elif GROUP == "rank":
    for r, a in ((2, 4.0), (4, 8.0), (8, 16.0)):
        RUNS.append(("sort", "qv", r, a, 0.005, 400))
        RUNS.append(("reverse", "all", r, a, 0.005, 400))

print(f"{'配置':<34}{'loss':>16}{'|g|max':>9}{'本能力acc':>11}{'gain':>9}{'eos':>7}{'heuristic':>11}{'秒':>8}")
for cap, variant, rank, alpha, lr, steps in RUNS:
    spec = P.AdapterSpec(capability=cap, variant=variant, rank=rank, alpha=alpha)
    s = P.TrainSettings(lr=lr, steps=steps, batch=8, grad_clip=1.0, n_train=N_TRAIN)
    t0 = time.perf_counter()
    ad, rep = P.train_adapter(SHAPE, base_w, spec, tr, ev, s, seed=SEED)
    dt = time.perf_counter() - t0
    lgs = [N.logits_of(base_w, e.ids, SHAPE, adapters=[ad]) for e in ev[cap]]
    m = D.mean_accuracy(lgs, ev[cap])
    gain = m["acc"] - base_acc[cap]
    tag = f"{cap}-{variant} r{rank} a{alpha:g} lr{lr:g} s{steps}"
    print(
        f"{tag:<34}{rep.first_loss:>7.3f}→{rep.last_loss:<7.3f}{rep.grad_norm_max:>9.2f}"
        f"{m['acc']:>11.4f}{gain:>+9.4f}{m['eos_acc']:>7.2f}{bl[cap]['heuristic']:>11.4f}{dt:>8.1f}"
    )
