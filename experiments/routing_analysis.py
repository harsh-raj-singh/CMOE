"""
Routing analysis: measure expert utilization, load balance, and specialization.

Usage:
    python -m experiments.routing_analysis
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Model

from cmoe_from_scratch import replace_ffn_with_cmoe, count_parameters
from cmoe_from_scratch.router import compute_routing_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def analyze_routing(
    num_experts: int = 8,
    top_k: int = 2,
    output_path: str = "results/routing_analysis.json",
):
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    model = replace_ffn_with_cmoe(model, num_experts=num_experts, top_k=top_k, verbose=False)
    model.eval()

    logger.info("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length", return_tensors="pt")

    test_ds = dataset["test"].map(tokenize, batched=True, remove_columns=["text"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # Collect routing decisions
    all_expert_counts = torch.zeros(num_experts)
    layer_routing = {}
    total_tokens = 0
    num_batches = 20

    logger.info("Analyzing routing patterns...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            # Gather routing from CMoE layers
            for name, module in model.named_modules():
                if hasattr(module, 'router'):
                    indices, weights, _ = module.router(input_ids.float())
                    stats = compute_routing_stats(indices, num_experts)

                    if name not in layer_routing:
                        layer_routing[name] = {
                            "utilizations": [],
                            "entropy": [],
                            "load_balance": [],
                        }

                    layer_routing[name]["utilizations"].append(stats["utilization"])
                    layer_routing[name]["entropy"].append(stats["normalized_entropy"])
                    layer_routing[name]["load_balance"].append(stats["load_balance_ratio"])

                    total_counts = sum(stats["expert_counts"])
                    all_expert_counts += torch.tensor(stats["expert_counts"])
                    total_tokens += total_counts

    # Aggregate results
    results = {
        "num_experts": num_experts,
        "top_k": top_k,
        "total_tokens_routed": total_tokens,
        "global_expert_utilization": (all_expert_counts / all_expert_counts.sum()).tolist(),
        "global_entropy": -(all_expert_counts / all_expert_counts.sum()).log().sum().item() / num_experts,
        "per_layer": {},
    }

    for name, data in layer_routing.items():
        avg_util = [sum(u[i] for u in data["utilizations"]) / len(data["utilizations"]) for i in range(num_experts)]
        results["per_layer"][name] = {
            "avg_utilization": avg_util,
            "avg_entropy": sum(data["entropy"]) / len(data["entropy"]),
            "avg_load_balance": sum(data["load_balance"]) / len(data["load_balance"]),
            "max_skew": max(avg_util) / (min(avg_util) + 1e-8),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Routing analysis saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Routing Analysis: {num_experts} experts, top-{top_k}")
    print("=" * 60)
    print(f"\nGlobal Expert Utilization:")
    for i, u in enumerate(results["global_expert_utilization"]):
        bar = "#" * int(u * 200)
        print(f"  Expert {i:2d}: {u:.3f} |{bar}")
    print(f"\nNormalized Entropy: {results['global_entropy']:.4f} (1.0 = perfect balance)")
    print(f"Max Skew Ratio:     {max(results['global_expert_utilization']) / (min(results['global_expert_utilization']) + 1e-8):.2f}")


if __name__ == "__main__":
    analyze_routing()
