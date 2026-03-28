"""Utility functions for CMoE analysis and metrics."""

import torch
import torch.nn as nn
from .layers import CMoELayer
from .router import compute_routing_stats


def count_parameters(model: nn.Module) -> dict:
    """Count total, trainable, active-per-forward-pass parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate active params (shared expert + top_k sparse experts per CMoE layer)
    active = 0
    for module in model.modules():
        if isinstance(module, CMoELayer):
            d_model = module.experts.experts[0].w1.in_features
            d_ff = module.experts.experts[0].w1.out_features
            expert_params = d_model * d_ff + d_ff * d_model  # single expert
            expert_params *= module.top_k  # top-k active experts
            if module.use_shared_expert:
                expert_params += d_model * d_ff + d_ff * d_model  # shared expert
            active += expert_params
        elif isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            active += sum(p.numel() for p in module.parameters())

    return {
        "total": total,
        "trainable": trainable,
        "active_per_forward": active,
        "total_gb": total * 4 / (1024 ** 3),
        "active_gb": active * 4 / (1024 ** 3),
        "sparsity": 1.0 - (active / total) if total > 0 else 0.0,
    }


def analyze_routing(model: nn.Module, dataloader, device: torch.device, num_batches: int = 10) -> dict:
    """Run inference and collect routing statistics across CMoE layers."""
    model.eval()
    layer_stats = {}
    layer_idx = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            _ = model(input_ids=input_ids)

            for module in model.modules():
                if isinstance(module, CMoELayer):
                    key = f"layer_{layer_idx}"
                    if key not in layer_stats:
                        layer_stats[key] = {"num_experts": module.num_experts, "top_k": module.top_k}

                    # Get routing decisions
                    expert_indices, expert_weights, _ = module.router(model.ln2(input_ids) if hasattr(model, 'ln2') else input_ids)
                    stats = compute_routing_stats(expert_indices, module.num_experts)

                    if "utilization_history" not in layer_stats[key]:
                        layer_stats[key]["utilization_history"] = []
                    layer_stats[key]["utilization_history"].append(stats["utilization"])

                    layer_idx += 1

            layer_idx = 0  # reset for next batch

    # Aggregate
    for key in layer_stats:
        hist = layer_stats[key]["utilization_history"]
        avg_util = [sum(x[i] for x in hist) / len(hist) for i in range(layer_stats[key]["num_experts"])]
        layer_stats[key]["avg_utilization"] = avg_util
        layer_stats[key]["max_skew"] = max(avg_util) / (min(avg_util) + 1e-8)
        del layer_stats[key]["utilization_history"]

    return layer_stats


def get_expert_specialization_score(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 5,
) -> float:
    """Measure how much experts specialize (0=all same, 1=perfect specialization).

    Computes correlation between token positions and expert assignments.
    Higher correlation = more specialization.
    """
    model.eval()
    specializations = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            _ = model(input_ids=input_ids)

            for module in model.modules():
                if isinstance(module, CMoELayer):
                    expert_indices, _, _ = module.router(input_ids)
                    # Measure entropy of routing decisions
                    stats = compute_routing_stats(expert_indices, module.num_experts)
                    # Lower normalized entropy = more specialization
                    specialization = 1.0 - stats["normalized_entropy"]
                    specializations.append(specialization)

    return sum(specializations) / len(specializations) if specializations else 0.0
