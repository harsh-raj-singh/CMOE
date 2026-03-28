"""
Expert ablation: sweep number of experts and top-k, measure parameter count & throughput.

Usage:
    python -m experiments.expert_ablation
"""

import json
import logging
import time
from pathlib import Path

import torch
from transformers import GPT2Model

from cmoe_from_scratch import replace_ffn_with_cmoe, count_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_ablation(output_path: str = "results/expert_ablation.json"):
    expert_configs = [
        {"num_experts": 4, "top_k": 1},
        {"num_experts": 4, "top_k": 2},
        {"num_experts": 8, "top_k": 1},
        {"num_experts": 8, "top_k": 2},
        {"num_experts": 8, "top_k": 4},
        {"num_experts": 16, "top_k": 2},
        {"num_experts": 16, "top_k": 4},
        {"num_experts": 32, "top_k": 2},
        {"num_experts": 32, "top_k": 4},
        {"num_experts": 64, "top_k": 4},
    ]

    results = []

    for cfg in expert_configs:
        ne, tk = cfg["num_experts"], cfg["top_k"]
        logger.info(f"--- Experts={ne}, Top-{tk} ---")

        model = GPT2Model.from_pretrained("gpt2")
        model = replace_ffn_with_cmoe(model, num_experts=ne, top_k=tk, verbose=False)

        info = count_parameters(model)

        dummy = torch.randn(1, 8, 768)
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy)
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(20):
                start = time.perf_counter()
                _ = model(dummy)
                times.append(time.perf_counter() - start)
        avg_ms = sum(times) / len(times) * 1000

        result = {
            "num_experts": ne,
            "top_k": tk,
            "total_params": info["total"],
            "active_params": info["active_per_forward"],
            "sparsity": info["sparsity"],
            "active_pct": 100.0 * info["active_per_forward"] / info["total"],
            "forward_ms": avg_ms,
            "param_efficiency": info["active_per_forward"] / ne,  # params per expert
        }
        results.append(result)

        logger.info(
            f"  Total: {info['total']:,} | Active: {info['active_per_forward']:,} "
            f"({100*info['active_per_forward']/info['total']:.1f}%) | "
            f"Sparsity: {info['sparsity']:.1%} | Fwd: {avg_ms:.2f}ms"
        )

        del model

    out = {"model": "gpt2", "base_ffn_params": 768 * 3072 * 2, "ablation": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print table
    print("\n" + "=" * 110)
    print(f"{'Experts':>8} | {'Top-K':>6} | {'Total Params':>14} | {'Active Params':>14} | {'Active %':>10} | {'Sparsity':>10} | {'Fwd (ms)':>10}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['num_experts']:>8} | {r['top_k']:>6} | {r['total_params']:>14,} | "
            f"{r['active_params']:>14,} | {r['active_pct']:>9.1f}% | "
            f"{r['sparsity']:>9.1%} | {r['forward_ms']:>9.2f}"
        )
    print("=" * 110)


if __name__ == "__main__":
    run_ablation()
