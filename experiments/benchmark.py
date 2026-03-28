"""
Benchmark: dense GPT-2 vs CMoE variants on parameter count, memory, and throughput.

Usage:
    python -m experiments.benchmark
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


def measure_forward(model, batch_size=4, seq_len=128, n_iters=50):
    dummy = torch.randn(batch_size, seq_len, 768)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = model(dummy)
            times.append(time.perf_counter() - start)
    return sum(times) / len(times) * 1000


def measure_memory(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)


def run_benchmark(output_path: str = "results/benchmark.json"):
    results = {}
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    logger.info(f"Device: {device_name}")

    # Dense baseline
    logger.info("Benchmarking dense GPT-2...")
    model_dense = GPT2Model.from_pretrained("gpt2")
    fwd_dense = measure_forward(model_dense)
    mem_dense = measure_memory(model_dense)
    total_params_dense = sum(p.numel() for p in model_dense.parameters())
    results["dense_gpt2"] = {
        "total_params": total_params_dense,
        "active_params": total_params_dense,
        "sparsity": 0.0,
        "forward_ms": fwd_dense,
        "model_size_mb": mem_dense,
    }
    del model_dense

    # CMoE variants
    configs = [
        {"num_experts": 4, "top_k": 2},
        {"num_experts": 8, "top_k": 2},
        {"num_experts": 16, "top_k": 2},
        {"num_experts": 16, "top_k": 4},
        {"num_experts": 32, "top_k": 2},
        {"num_experts": 32, "top_k": 4},
    ]

    for cfg in configs:
        ne, tk = cfg["num_experts"], cfg["top_k"]
        name = f"cmoe_{ne}e_top{tk}"
        logger.info(f"Benchmarking {name}...")

        model = GPT2Model.from_pretrained("gpt2")
        model = replace_ffn_with_cmoe(model, num_experts=ne, top_k=tk, verbose=False)
        info = count_parameters(model)
        fwd = measure_forward(model)
        mem = measure_memory(model)

        results[name] = {
            "num_experts": ne,
            "top_k": tk,
            "total_params": info["total"],
            "active_params": info["active_per_forward"],
            "sparsity": info["sparsity"],
            "active_pct": 100.0 * info["active_per_forward"] / info["total"],
            "forward_ms": fwd,
            "model_size_mb": mem,
            "fwd_overhead_pct": 100.0 * (fwd - fwd_dense) / fwd_dense if fwd_dense > 0 else 0,
        }
        logger.info(
            f"  Total: {info['total']:,} | Active: {info['active_per_forward']:,} "
            f"({100*info['active_per_forward']/info['total']:.1f}%) | "
            f"Fwd: {fwd:.2f}ms ({100*(fwd-fwd_dense)/fwd_dense:+.1f}% vs dense)"
        )
        del model

    results["device"] = device_name

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print table
    print("\n" + "=" * 120)
    print(f"{'Model':>25} | {'Total':>12} | {'Active':>12} | {'Active %':>10} | {'Sparsity':>10} | {'Fwd (ms)':>10} | {'vs Dense':>10}")
    print("-" * 120)
    d = results["dense_gpt2"]
    print(f"{'Dense GPT-2':>25} | {d['total_params']:>12,} | {d['active_params']:>12,} | {'100.0':>9}% | {'0.0':>9}% | {d['forward_ms']:>9.2f} | {'baseline':>10}")
    for cfg in configs:
        ne, tk = cfg["num_experts"], cfg["top_k"]
        name = f"cmoe_{ne}e_top{tk}"
        r = results[name]
        print(
            f"{'CMoE '+str(ne)+'E top-'+str(tk):>25} | {r['total_params']:>12,} | {r['active_params']:>12,} | "
            f"{r['active_pct']:>9.1f}% | {r['sparsity']*100:>9.1f}% | {r['forward_ms']:>9.2f} | "
            f"{r['fwd_overhead_pct']:>+9.1f}%"
        )
    print("=" * 120)


if __name__ == "__main__":
    run_benchmark()
