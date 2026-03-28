"""
Expert network implementations for Conditional Mixture of Experts.

Each expert is a feed-forward network (FFN) that can be sparsely activated
by the conditional router based on input token representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Expert(nn.Module):
    """Single expert: a 2-layer FFN with GELU activation (GPT-2 style)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


class ExpertPool(nn.Module):
    """Pool of N identical-capacity experts with independent parameters."""

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Route tokens to their assigned experts.

        Args:
            x: (batch, seq_len, d_model)
            expert_indices: (batch, seq_len, top_k) — expert IDs per token

        Returns:
            (batch, seq_len, d_model) — expert outputs
        """
        batch, seq_len, d_model = x.shape
        top_k = expert_indices.shape[-1]

        # Flatten for parallel expert computation
        x_flat = x.view(-1, d_model)  # (B*S, d_model)
        indices_flat = expert_indices.view(-1, top_k)  # (B*S, top_k)

        # Process each expert position (top_k loop — simple, not batched)
        outputs = torch.zeros_like(x_flat)
        for k_idx in range(top_k):
            for expert_id in range(self.num_experts):
                mask = (indices_flat[:, k_idx] == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_out = self.experts[expert_id](expert_input)
                    outputs[mask] += expert_out

        return outputs.view(batch, seq_len, d_model)

    def forward_batched(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted combination of top-k expert outputs.

        Args:
            x: (batch, seq_len, d_model)
            expert_indices: (batch, seq_len, top_k)
            expert_weights: (batch, seq_len, top_k) — routing probabilities

        Returns:
            (batch, seq_len, d_model) — weighted expert outputs
        """
        batch, seq_len, d_model = x.shape
        top_k = expert_indices.shape[-1]

        output = torch.zeros(batch, seq_len, d_model, device=x.device, dtype=x.dtype)

        for k_idx in range(top_k):
            for expert_id in range(self.num_experts):
                mask = (expert_indices[:, :, k_idx] == expert_id)  # (B, S)
                if mask.any():
                    expert_input = x[mask]  # (N, d_model)
                    expert_out = self.experts[expert_id](expert_input)  # (N, d_model)
                    weights = expert_weights[:, :, k_idx][mask].unsqueeze(-1)  # (N, 1)
                    output[mask] += expert_out * weights

        return output


class SharedExpert(nn.Module):
    """A shared expert that always activates (dense), combined with sparse experts.

    From DeepSeek-MoE: shared + routed experts prevents information loss
    from sparse routing on generic token patterns.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.expert = Expert(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)
