#!/usr/bin/env bash
set -euo pipefail

# CMoE From Scratch — Experiment Runner
# Usage: bash scripts/run_experiment.sh

echo "=== CMoE From Scratch: Conditional Mixture of Experts on GPT-2 ==="
echo ""

NUM_EXPERTS=${1:-8}
TOP_K=${2:-2}

echo "Config: num_experts=$NUM_EXPERTS, top_k=$TOP_K"
echo ""

# Expert ablation
echo "--- Expert Ablation Study ---"
python -m experiments.expert_ablation
echo ""

# Routing analysis
echo "--- Routing Analysis ---"
python -m experiments.routing_analysis
echo ""

# Full training
echo "--- Training CMoE (${NUM_EXPERTS} experts, top-${TOP_K}) ---"
python -m experiments.train_gpt2_cmoe --num_experts $NUM_EXPERTS --top_k $TOP_K --epochs 3
echo ""

# Benchmark
echo "--- Benchmarking ---"
python -m experiments.benchmark
echo ""

echo "=== All experiments complete. Check results/ directory. ==="
