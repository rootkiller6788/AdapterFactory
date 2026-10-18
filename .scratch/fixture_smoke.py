import json, shutil, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
import conftest as CT
from adapterfactory import config as C, pipeline as PL, produce as PR

tmp = Path(tempfile.mkdtemp(prefix="tiny-"))
cfg = CT.C.deep_merge(json.loads(C.canonical_json(C.DEFAULTS)), CT.TINY_OVERRIDE)
(tmp/"data"/"datasets").mkdir(parents=True)
shutil.copyfile("data/datasets/index.json", tmp/"data"/"datasets"/"index.json")
ctx = PL.prepare(root=tmp, cfg=cfg)
rank = int(cfg["zoo"]["rank"]); alpha = float(cfg["zoo"]["alpha_mult"])*rank
specs = PR.default_zoo(C.capabilities_of(cfg), rank=rank, alpha=alpha)
st = PR.TrainSettings(lr=cfg["train"]["lr"], steps=cfg["train"]["steps"], batch=cfg["train"]["batch"],
    weight_decay=cfg["train"]["weight_decay"], grad_clip=cfg["train"]["grad_clip"],
    optimizer=cfg["train"]["optimizer"], n_train=cfg["data"]["n_train"])
t0=time.time()
rep = PR.produce_zoo(root=tmp, shape=ctx["shape"], base_seed=ctx["base_seed"], specs=specs,
    settings=st, vocab=ctx["vocab"], idx=ctx["idx"], seq_len=ctx["seq_len"],
    eval_sets=ctx["eval_sets"], train_seed=cfg["data"]["seed"], force=False, verbose=False)
print(f"produce {time.time()-t0:.1f}s")
for lab,r in sorted(rep["reports"].items()):
    print("  ", lab, {k: round(v,3) for k,v in sorted(r["eval_acc"].items())}, f"loss {r['first_loss']:.3f}->{r['last_loss']:.3f}")
print("baselines:", {k:{kk:round(vv,4) for kk,vv in v.items() if kk!='n_positions'} for k,v in ctx["baselines"].items()})

t0=time.time()
res = PL.run_pipeline(root=tmp, cfg=cfg, export=True)
print(f"\nrun {time.time()-t0:.1f}s")
print("runs:", len(res.runs), "conflicts:", len(res.conflicts), "frontier:", len(res.frontier))
print("counters:", {k:v for k,v in res.counters.items() if k not in ("checks","acceptance","graph","predictor_stats","predictor_rows")})
for ch in res.counters["checks"]:
    print(f"  {'OK ' if ch['ok'] else 'BAD'} req={ch['required']} {ch['name']}: {ch['detail'][:120]}")
for a in res.counters["acceptance"]:
    print(f"  {'OK ' if a['ok'] else 'BAD'} {a['name']}: {a['detail'][:95]}")
print("\nfindings:", [(f.key, f.verdict) for f in res.findings])
out = tmp/"out"
print("\nfiles:", sorted(p.name for p in out.iterdir()))
print("compositions:", len(list((out/"compositions").glob("*.bin"))))
