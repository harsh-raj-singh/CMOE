"""Configuration dataclasses for CMoE experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CMoEConfig:
    """CMoE architecture configuration."""

    num_experts: int = 8
    top_k: int = 2
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.0
    noise_std: float = 0.1
    capacity_factor: float = 1.0
    use_shared_expert: bool = True
    aux_loss_weight: float = 0.01


@dataclass
class TrainConfig:
    """Training configuration."""

    model_name: str = "gpt2"
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 200
    max_seq_length: int = 512
    seed: int = 42
    output_dir: str = "results/checkpoints"
    log_every: int = 50
    eval_every: int = 500


@dataclass
class ExperimentConfig:
    """Combined experiment configuration."""

    cmoe: CMoEConfig = field(default_factory=CMoEConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: Optional[str] = None
