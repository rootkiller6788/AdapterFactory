"""S0.5：**三个方法在同一个成员集上到底给出什么。**

S0 标定跑出两条互相印证、但都很意外的结果：

1. 同能力的一对（``upper-qv`` + ``upper-all``）stack 之后，增益是**负的**
   （shift −0.0260、sort −0.1458）—— 不是「次可加」，是掉到基座以下。
2. 参数空间的预测（cos ≈ 0、符号一致率 ≈ 0.50）与实测干扰**完全对不上**（1/6）。

两条合起来指向一个与「能力冲突」不同的机制：**stack 把每个独立训练出来的
delta 原样相加，于是在共用挂点上做了两倍于「训练时见过的」更新**。
若这个解释成立，那么：

- ``merge``（系数 1/k，更新量减半）应当把同能力的一对**救回来** ——
  两个 delta 都在解同一个任务，取平均仍在解的邻域里；
- 跨能力的一对，平均之后应当**仍然受损**（两个解方向无关，各摊一半
  等于都没做到）；
- ``route`` 不合并权重、只在激活空间按门混合，应当落在两者之间。

这是一个**可证伪的预测**。本脚本就是去证伪它 —— 若 merge 也救不回来，
那么上面那个机制解释就是错的，报告里必须换成别的说法。

用法：``python .scratch/method_probe.py [steps]``（默认 800）。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapterfactory import compose as P  # noqa: E402
from adapterfactory import config as C  # noqa: E402
from adapterfactory import data as D  # noqa: E402
from adapterfactory import evaluate as E  # noqa: E402
from adapterfactory import net as N  # noqa: E402
from adapterfactory import produce as PR  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800

cfg = C.load_config(root=".")
idx = D.load_index("data/datasets/index.json")
vocab = D.build_vocab(idx)
caps = C.capabilities_of(cfg)
SL = cfg["base"]["seq_len"]
seed = cfg["base"]["seed"]
shape = N.ModelShape(
    vocab=cfg["base"]["vocab"],
    d_model=cfg["base"]["d_model"],
    n_layers=cfg["base"]["n_layers"],
    n_heads=cfg["base"]["n_heads"],
    d_ff=cfg["base"]["d_ff"],
    max_len=SL,
)
base_weights = N.init_base_weights(shape, seed)
ev = D.build_eval_sets(idx, vocab, seq_len=SL, n_eval=cfg["data"]["n_eval"], seed=cfg["data"]["seed"])
tr = D.build_train_set(idx, vocab, seq_len=SL, n_train=cfg["data"]["n_train"], seed=cfg["data"]["seed"])
bl = D.baselines(idx, tr, ev)

rank = cfg["zoo"]["rank"]
alpha = cfg["zoo"]["alpha_mult"] * rank
settings = PR.TrainSettings(
    lr=cfg["train"]["lr"],
    steps=STEPS,
    batch=cfg["train"]["batch"],
    grad_clip=cfg["train"]["grad_clip"],
    n_train=cfg["data"]["n_train"],
    optimizer=cfg["train"]["optimizer"],
    weight_decay=cfg["train"]["weight_decay"],
)

#: 只要四个：upper 的两种覆盖（同能力对）+ sort 的两种覆盖（跨能力对用上 sort-qv）。
WANT = ["upper-qv", "upper-all", "sort-qv", "sort-all"]
specs = [s for s in PR.default_zoo(caps, rank=rank, alpha=alpha) if s.label in WANT]
assert len(specs) == len(WANT), [s.label for s in specs]

print("=" * 78)
print(f"S0.5 方法探针   steps={STEPS} lr={settings.lr} rank={rank} alpha={alpha}")
print("=" * 78)

zoo: dict[str, N.Adapter] = {}
gates_all = N.derive_gates(shape, [s.label for s in specs], seed=seed)


def evaluate_members(labels: list[str], method: str) -> E.Adjudication:
    """构造一个组合并评测。**这一步就是 compose.realize 的正规路径。**"""
    members = tuple(sorted(labels))
    plan = P.Plan(cid=P.cid_of(method, members), method=method, members=members)
    ads = [zoo[m] for m in members]
    gates = {m: gates_all[m] for m in members}
    r = P.realize(
        base_weights, ads, plan, shape, gates=gates, temperature=float(cfg["route"]["temperature"])
    )
    return E.measure(r, ev, shape, caps)


def macs_for(labels: list[str], method: str) -> int:
    """一次前向的 MACs。**另建一个 Realized 是有意的** —— 评测那个已经把
    探针样本灌进缓存了，共用它的话计数会命中缓存而报 0。"""
    members = tuple(sorted(labels))
    plan = P.Plan(cid=P.cid_of(method, members), method=method, members=members)
    r = P.realize(
        base_weights, [zoo[m] for m in members], plan, shape,
        gates={m: gates_all[m] for m in members},
        temperature=float(cfg["route"]["temperature"]),
    )
    return E.macs_of(r, list(ev[caps[0]][0].ids), shape)


t0 = time.time()
base_adj = E.measure(P.realize_base(base_weights, shape), ev, shape, caps)
print("\n基座 tok_acc：" + "  ".join(f"{c}={base_adj.acc(c):.4f}" for c in caps))
print("基线 heuristic：" + "  ".join(f"{c}={bl[c]['heuristic']:.4f}" for c in caps))
print(f"\n--- 训练（adamw lr={settings.lr} steps={STEPS} rank={rank} alpha={alpha}）---")

for spec in specs:
    t = time.time()
    ad, rep = PR.train_adapter(shape, base_weights, spec, tr, ev, settings, seed=seed)
    zoo[spec.label] = ad
    solo = evaluate_members([spec.label], "stack")
    m = solo.measures[spec.capability]
    marg = m.acc - bl[spec.capability]["heuristic"]
    print(
        f"  {spec.label:<11} loss {rep.first_loss:.4f}→{rep.last_loss:.4f} "
        f"|g|max {rep.grad_norm_max:6.2f} {time.time()-t:6.1f}s  "
        f"{spec.capability} tok_acc={m.acc:.4f} exact={m.exact:.4f} 差={marg:+.4f}"
    )
print(f"\n训练总耗时 {time.time()-t0:.0f}s")

solo_gain = {(lab, c): None for lab in zoo for c in caps}
for lab in zoo:
    a = evaluate_members([lab], "stack")
    for c in caps:
        solo_gain[(lab, c)] = a.acc(c) - base_adj.acc(c)

print("\n单干增益表：")
for lab in sorted(zoo):
    print(f"  {lab:<11} " + "  ".join(f"{c}={solo_gain[(lab,c)]:+.4f}" for c in caps))


def report_sets(title: str, members: list[str], focus: list[str]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for method in ("stack", "merge", "route"):
        a = evaluate_members(members, method)
        parts = "  ".join(f"{c}={a.acc(c)-base_adj.acc(c):+.4f}" for c in focus)
        ex = "  ".join(f"{c}.exact={a.exact(c):.4f}" for c in focus)
        print(f"  {method:<6} {parts}   {ex}   macs={macs_for(members, method)}")
    for c in focus:
        singles = "  ".join(f"{lab}={solo_gain[(lab,c)]:+.4f}" for lab in members)
        print(f"  （单干 {c}： {singles}）")


report_sets(
    "同能力对：upper-qv + upper-all（两者都在解 upper）",
    ["upper-qv", "upper-all"],
    ["upper"],
)
report_sets(
    "跨能力对：upper-qv + sort-qv（共挂 q,v，但目标不同）",
    ["upper-qv", "sort-qv"],
    ["upper", "sort"],
)
report_sets(
    "三元：upper-qv + sort-qv + sort-all",
    ["upper-qv", "sort-qv", "sort-all"],
    ["upper", "sort"],
)

print("\n" + "=" * 78)
print("delta 幅度（解释「为什么相加会过冲」）")
print("=" * 78)
bs = N.param_shapes(shape)
for lab in sorted(zoo):
    ad = zoo[lab]
    tot = 0.0
    base_tot = 0.0
    for full in N.expand_targets(bs, ad.targets):
        d = ad.delta(full, bs[full])
        tot += sum(x * x for row in d for x in row)
        w = base_weights[full]
        base_tot += sum(x * x for row in w for x in row)
    print(
        f"  {lab:<11} ‖Δ‖={tot**0.5:9.4f}  ‖W_挂点‖={base_tot**0.5:9.4f}  "
        f"比值={tot**0.5/base_tot**0.5:.4f}  挂 {len(ad.targets)} 个模块"
    )

print("\n--- 参数空间：cos 与符号一致率（成对，共同挂点上）---")
for i, a_lab in enumerate(sorted(zoo)):
    for b_lab in sorted(zoo)[i + 1 :]:
        da = P.delta_of_pair(zoo[a_lab], base_weights, shape)
        db = P.delta_of_pair(zoo[b_lab], base_weights, shape)
        cos, rate, n = P.cosine_and_sign(da, db)
        print(
            f"  {a_lab} ~ {b_lab:<11} cos={cos:+.4f}  符号一致率={rate:.4f}  "
            f"共同挂点={n}"
        )
