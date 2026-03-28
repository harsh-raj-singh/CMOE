"""Training loop for CMoE models with auxiliary loss tracking."""

import math
import time
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .layers import CMoELayer

logger = logging.getLogger(__name__)


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: ExperimentConfig,
) -> dict:
    """Train CMoE model with combined CE loss + load-balancing auxiliary loss."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.train.learning_rate, weight_decay=config.train.weight_decay)

    total_steps = len(train_loader) * config.train.num_epochs // config.train.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.train.warmup_steps, num_training_steps=total_steps
    )

    metrics = {"train_loss": [], "aux_loss": [], "eval_loss": [], "eval_perplexity": []}
    global_step = 0
    start_time = time.time()

    for epoch in range(config.train.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_aux = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids)
            logits = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

            # Causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # Collect auxiliary losses from CMoE layers
            aux_loss = torch.tensor(0.0, device=device)
            for module in model.modules():
                if isinstance(module, CMoELayer):
                    aux_loss = aux_loss + module.aux_loss

            loss = ce_loss + aux_loss
            loss = loss / config.train.gradient_accumulation_steps
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, config.train.max_grad_norm)

            if (step + 1) % config.train.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                metrics["train_loss"].append(ce_loss.item())
                metrics["aux_loss"].append(aux_loss.item())

                if global_step % config.train.log_every == 0:
                    elapsed = time.time() - start_time
                    avg = sum(metrics["train_loss"][-config.train.log_every:]) / len(metrics["train_loss"][-config.train.log_every:])
                    logger.info(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"CE {avg:.4f} | Aux {aux_loss.item():.5f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
                    )

            epoch_loss += ce_loss.item()
            epoch_aux += aux_loss.item()

        avg_ce = epoch_loss / len(train_loader)
        avg_aux = epoch_aux / len(train_loader)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(min(val_loss, 20))
            metrics["eval_loss"].append(val_loss)
            metrics["eval_perplexity"].append(ppl)
            logger.info(f"Epoch {epoch+1} | CE {avg_ce:.4f} | Aux {avg_aux:.5f} | Val PPL {ppl:.2f}")

    total_time = time.time() - start_time
    metrics["total_time_seconds"] = total_time

    out_dir = Path(config.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids)
        logits = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        total_loss += loss.item()
    return total_loss / len(dataloader)
