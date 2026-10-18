"""AdapterFactory —— **Adapter 产出之后**发生了什么。

FineTuneFactory 是 ``Dataset → Train → Adapter``；
AdapterFactory 是 ``Adapter → Registry → Compose → Merge → Evaluate → Deploy``。

它研究的是 **Adapter 生命周期**：登记、版本、能力标注、组合、冲突检测与部署，
**而不是再训练一遍**。这条边界不是文档里的一句话，是被代码守着的：

- 流水线模块（``registry / compose / evaluate / conflict / graph / pipeline``）
  一律不导入 :mod:`adapterfactory.produce`，也不含 ``train`` / ``backward``；
- :mod:`adapterfactory.audit` 用 ``tokenize`` 扫 token（**不是扫文本** —— 那样会
  扫到自己的文档字符串）在每次运行上核对这件事，结果进 ``result.counters``
  并且是 ``acceptance`` 的门槛之一。

形状是**扇出-收拢型**的第三形态：设计空间不是「算法 × 规模」也不是「某个 IR
字段的取值」，而是 **Adapter 集合上的子集格** —— 8 个 Adapter 的全部 size 1..3
子集（92 个）× 3 种组合方法 = 276 个组合。

这个形状独有的两件事：

1. **子集格就是血缘图。** ``combo:<cid>`` 的父节点恰好是它的成员 Adapter，于是
   README 要的那个「Adapter Graph」不是另画的独立产物，它就是 ``lineage`` DAG
   去掉 baseline 那一支的子图。
2. **两条独立路径必须对质。** 路径 A 在**参数空间**由 ``cos(Δᵢ,Δⱼ)`` 与符号一致率
   *预测*冲突；路径 B 在**能力空间**实测 ``Interference(i,j)``。两者必须同向 ——
   对得上，「冲突」这个说法才有来源；对不上，那本身就是最有信息量的输出。

导出核心对象，方便 ``from adapterfactory import ...``。
"""

from .compose import Plan, cid_of, grid_size, plan_grid
from .config import (
    METHODS,
    VARIANTS,
    capabilities_of,
    load_config,
    methods_of,
    protocol_digest,
    short_version,
    validate_config,
    version_digest,
    weight_scheme,
)
from .graph import pareto_frontier
from .lineage import Lineage, build_lineage
from .net import Adapter, ModelShape
from .pipeline import load_full, load_result, run_pipeline
from .registry import Registry, load_registry
from .report import render_markdown
from .rng import canonical_json, digest_of, join_digest, sha256_text, short_digest
from .schema import (
    AdapterRecord,
    CapabilityMetric,
    CompositionRun,
    ConflictEdge,
    Finding,
    PipelineResult,
    Version,
)
from .visualize import render_html

__all__ = [
    # 版本与摘要
    "METHODS",
    "VARIANTS",
    "canonical_json",
    "digest_of",
    "join_digest",
    "protocol_digest",
    "short_digest",
    "short_version",
    "sha256_text",
    "version_digest",
    # 配置
    "capabilities_of",
    "load_config",
    "methods_of",
    "validate_config",
    "weight_scheme",
    # 基座与适配器
    "Adapter",
    "ModelShape",
    "Registry",
    "load_registry",
    # 组合空间
    "Plan",
    "cid_of",
    "grid_size",
    "plan_grid",
    # 图与结果
    "Lineage",
    "build_lineage",
    "pareto_frontier",
    "AdapterRecord",
    "CapabilityMetric",
    "CompositionRun",
    "ConflictEdge",
    "Finding",
    "PipelineResult",
    "Version",
    # 端到端与报告
    "run_pipeline",
    "load_result",
    "load_full",
    "render_markdown",
    "render_html",
]

__version__ = "0.1.0"
