"""附录 C 坑号 109–142 → 126–159（+17）。

为什么：`L4-preferencefactory` 今天先交付，它占了 109–125（它的 README 写着
「编号接着 L3-finetunefactory 的 108」）——两个项目同一天从同一个前提出发，
编号撞了。工作区约定的意义是「坑号全局唯一」，所以本项目（还没交付的那个）
整体后移。

**为什么用「区域 + 3 位数字」这一条规则就够、而不必逐条列。**
附录 C 是本项目自己的坑号命名空间；在那一区域里，孤立的 3 位数**要么是坑号
要么是它的一部分**。防误伤靠两侧的断言：
- 前面不是数字或小数点（`0.1198` 里的 `1198`、`1.2552` 都不会被匹到）；
- 后面不是数字、小数点或百分号（`1198` 这种四位数不会被截断成 `119`）。
这份脚本**先写候选文件**，跑完自检再替换 —— 工作区不是 git 仓库（坑 89），
没有版本控制兜底。
"""
from __future__ import annotations

import re
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "README.md"
DST = ROOT / ".scratch" / "README.renumbered.md"
BACKUP = ROOT / ".scratch" / "README.before_renumber.md"

LO, HI, SHIFT = 109, 142, 17
HEADER = "## 附录 C｜坑"

text = SRC.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(HEADER))
head, tail = "".join(lines[:start]), "".join(lines[start:])

# 区域内的孤立 3 位数：前后都不能是数字/小数点
pat = re.compile(r"(?<![\d.])1[0-4][0-9](?![\d.%])")


def bump(m: re.Match[str]) -> str:
    n = int(m.group(0))
    return str(n + SHIFT) if LO <= n <= HI else m.group(0)


new_tail = pat.sub(bump, tail)
DST.write_text(head + new_tail, encoding="utf-8")

# ---------------------------------------------------------------- 自检
checks: list[str] = []
before, after = tail, new_tail

# 1) 34 条条目头，且 126..159 连续
before_heads = [int(x) for x in re.findall(r"^\*\*(\d{3})｜", before, re.M)]
after_heads = [int(x) for x in re.findall(r"^\*\*(\d{3})｜", after, re.M)]
checks.append(f"条目头 {len(before_heads)} 条：{before_heads[0]}–{before_heads[-1]} → "
              f"{len(after_heads)} 条：{after_heads[0]}–{after_heads[-1]}")
assert len(before_heads) == len(after_heads) == 34, len(after_heads)
assert after_heads == list(range(126, 160)), after_heads

# 2) 区域里不再有 109–125 的孤立 3 位数
leftover = sorted({int(x) for x in pat.findall(after) if 109 <= int(x) <= 125})
checks.append(f"区域内残留的 109–125：{leftover or '无'}")
assert not leftover, leftover

# 3) 区域里的 3 位数只允许落在两段：≤108（复用的工作区坑号，不动）或 ≥126（本项目的）
alln = sorted({int(x) for x in pat.findall(after)})
checks.append(f"区域内 3 位数取值域：{alln[0]}–{alln[-1]}；非法段（109–125）："
              f"{[n for n in alln if 109 <= n <= 125] or '无'}")
assert not [n for n in alln if 109 <= n <= 125], alln

# 4) 复用工作区的坑号（≤108）在新区域里一个都没动
reused_before = sorted({int(x) for x in re.findall(r"坑 (\d{3})", before)})
reused_after = sorted({int(x) for x in re.findall(r"坑 (\d{3})", after)})
checks.append(f"区域内「坑 NNN」引用：{reused_before} → {reused_after}")
assert all(n <= 108 for n in reused_after), reused_after

# 5) 区域以外的部分逐字节未动，且那些 ≤108 的引用还在
assert head == "".join(lines[:start])
outside = re.findall(r"坑 (\d{3})", head)
checks.append(f"区域外「坑 NNN」引用（不动）：{sorted(set(outside))}")
assert all(int(n) <= 108 for n in outside), outside

# 6) 总行数不变
checks.append(f"行数 {len(before.splitlines())} → {len(new_tail.splitlines())}")
assert len(before.splitlines()) == len(new_tail.splitlines())

print("\n".join("  ✓ " + c for c in checks))
print(f"\n候选写在 {DST.name}；确认后覆盖 {SRC.name}（备份 {BACKUP.name}）")
if "--apply" in sys.argv:
    BACKUP.write_text(text, encoding="utf-8")
    SRC.write_text(head + new_tail, encoding="utf-8")
    print("已覆盖。")
