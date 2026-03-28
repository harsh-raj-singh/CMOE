"""CMoE From Scratch — Conditional Mixture of Experts implemented from first principles."""

from .experts import Expert, ExpertPool, SharedExpert
from .router import TopKRouter, ExpertChoiceRouter, compute_routing_stats
from .layers import CMoELayer, CMoETransformerBlock, replace_ffn_with_cmoe
from .config import CMoEConfig, TrainConfig, ExperimentConfig
from .trainer import train, evaluate
from .utils import count_parameters, analyze_routing, get_expert_specialization_score

__version__ = "1.0.0"
__all__ = [
    "Expert",
    "ExpertPool",
    "SharedExpert",
    "TopKRouter",
    "ExpertChoiceRouter",
    "compute_routing_stats",
    "CMoELayer",
    "CMoETransformerBlock",
    "replace_ffn_with_cmoe",
    "CMoEConfig",
    "TrainConfig",
    "ExperimentConfig",
    "train",
    "evaluate",
    "count_parameters",
    "analyze_routing",
    "get_expert_specialization_score",
]
