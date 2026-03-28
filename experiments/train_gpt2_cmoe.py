"""
Train GPT-2 with CMoE layers on WikiText-2.

Usage:
    python -m experiments.train_gpt2_cmoe --num_experts 8 --top_k 2 --epochs 3
"""

import argparse
import logging
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel

from cmoe_from_scratch import (
    replace_ffn_with_cmoe, train, count_parameters,
    ExperimentConfig, CMoEConfig, TrainConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize_dataset(tokenizer, dataset, max_length):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 with CMoE")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k routing")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    config = ExperimentConfig(
        cmoe=CMoEConfig(num_experts=args.num_experts, top_k=args.top_k),
        train=TrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            seed=args.seed,
            output_dir=args.output_dir,
        ),
    )

    logger.info(f"Loading {config.train.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.train.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(config.train.model_name)

    logger.info("Replacing FFN layers with CMoE...")
    model = replace_ffn_with_cmoe(
        model,
        num_experts=config.cmoe.num_experts,
        top_k=config.cmoe.top_k,
        use_shared_expert=config.cmoe.use_shared_expert,
    )

    info = count_parameters(model)
    logger.info(f"Total params:     {info['total']:,}")
    logger.info(f"Active per fwd:   {info['active_per_forward']:,}")
    logger.info(f"Sparsity:         {info['sparsity']:.1%}")

    logger.info(f"Loading {config.train.dataset}:{config.train.dataset_config}...")
    dataset = load_dataset(config.train.dataset, config.train.dataset_config)
    train_ds = tokenize_dataset(tokenizer, dataset["train"], config.train.max_seq_length)
    val_ds = tokenize_dataset(tokenizer, dataset["validation"], config.train.max_seq_length)

    from torch.utils.data import DataLoader
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, drop_last=True)

    logger.info("Starting training...")
    metrics = train(model, train_loader, val_loader, config)
    logger.info(f"Final eval perplexity: {metrics['eval_perplexity'][-1]:.2f}")


if __name__ == "__main__":
    main()
