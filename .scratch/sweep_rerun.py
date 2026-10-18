"""S0 标定扫描：让 reverse-qv / sort-qv 真的学会那两件事。

只训练**两个** Adapter，训到临时目录，读回 ``eval_acc``，算
``gap = acc - base_acc - heuristic``（也就是 ``min_single_gain_margin`` 那一栏）。

不改仓里的任何东西：临时根目录只用来接住产物。
"""
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(r"D:\Opencode\my-project\factory-everything\L3-adapterfactory")
sys.path.insert(0, str(ROOT / "src"))

from adapterfactory import config as C, data as D, pipeline as PL, produce as PR, zoo as Z

cfg = json.loads(C.canonical_json(C.DEFAULTS))
ctx = PL.prepare(root=ROOT, cfg=cfg)
shape, vocab, idx = ctx["shape"], ctx["vocab"], ctx["idx"]
ev, tr = ctx["eval_sets"], ctx["train_sets"]

# 基座在四个能力上的实测（由 size-1 stack 的 acc - gain 反推，见 af_calib.py）
BASE = {"upper": 0.1198, "shift": 0.0938, "reverse": 0.1042, "sort": 0.1823}
HEUR = {c: float(b["heuristic"]) for c, b in ctx["baselines"].items()}
print("  基线：", "  ".join(f"{c}=chance {ctx['baselines'][c]['chance']:.4f}/"
                            f"majority {ctx['baselines'][c]['majority']:.4f}/"
                            f"copy {ctx['baselines'][c]['copy']:.4f}/"
                            f"heur {HEUR[c]:.4f}" for c in cfg["data"]["capabilities"]))
print()
print(f"  {'trial':<28}{'cap':<9}{'own acc':>9}{'gap':>9}   loss")
print("  " + "-" * 62)

TRIALS = [
    ("rank4 lr5e-3 steps400", 4, 5e-3, 400),
    ("rank4 lr5e-3 steps1200", 4, 5e-3, 1200),
    ("rank4 lr1e-2 steps800", 4, 1e-2, 800),
    ("rank4 lr5e-3 steps400 wd.01", 4, 5e-3, 400),
    ("rank8 lr5e-3 steps800", 8, 5e-3, 800),
]

for name, rank, lr, steps in TRIALS:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "proj"
    tmp.mkdir(parents=True)
    for cap in ("reverse", "sort"):
        spec = Z.AdapterSpec(capability=cap, variant=Z.VARIANT_QV, rank=rank, alpha=2.0 * rank)
        rep = PR.produce_zoo(
            root=tmp, shape=shape, base_seed=ctx["base_seed"], specs=[spec],
            settings=PR.TrainSettings(lr=lr, steps=steps, batch=8, weight_decay=0.01 if "wd" in name else 0.0,
                                      grad_clip=1.0, optimizer="adamw", n_train=int(cfg["data"]["n_train"])),
            vocab=vocab, idx=idx, seq_len=ctx["seq_len"], eval_sets=ev,
            train_seed=int(cfg["data"]["seed"]), force=True, verbose=False,
        )["reports"][spec.label]
        acc = float(rep["eval_acc"][cap])
        gap = acc - BASE[cap] - HEUR[cap]
        flag = "  OK" if gap >= 0.10 else "  <== 未过"
        print(f"  {name:<28}{cap:<9}{acc:>9.4f}{gap:>+9.4f}   "
              f"{rep['first_loss']:.3f}→{rep['last_loss']:.3f}{flag}")
        shutil.rmtree(tmp, ignore_errors=True)
