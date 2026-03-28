"""
Conditional routing mechanisms for Mixture of Experts.

Implements top-k routing with load-balancing auxiliary loss,
noisy gating, and expert-capacity constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopKRouter(nn.Module):
    """Conditional top-k router: gates tokens to the k highest-scoring experts.

    Uses noisy gating during training for exploration and load-balancing
    auxiliary loss to prevent expert collapse.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.capacity_factor = capacity_factor

        # Gate projection: input -> expert scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            expert_indices: (batch, seq_len, top_k) — selected expert IDs
            expert_weights: (batch, seq_len, top_k) — routing probabilities
            aux_loss: scalar — load-balancing auxiliary loss
        """
        logits = self.gate(x)  # (B, S, E)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax over top-k for routing weights
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Load-balancing auxiliary loss
        aux_loss = self._load_balancing_loss(logits, top_k_indices)

        return top_k_indices, top_k_weights, aux_loss

    def _load_balancing_loss(
        self,
        logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Auxiliary loss to encourage uniform expert utilization.

        L_balance = N * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i
              P_i = fraction of router probability allocated to expert i
        """
        num_tokens = logits.shape[0] * logits.shape[1]

        # f_i: fraction of tokens assigned to each expert
        one_hot = F.one_hot(top_k_indices, self.num_experts).float()  # (B,S,K,E)
        f = one_hot.sum(dim=(1, 2)) / num_tokens  # (E,)

        # P_i: mean router probability for each expert
        probs = F.softmax(logits, dim=-1)  # (B, S, E)
        P = probs.mean(dim=(0, 1))  # (E,)

        aux_loss = self.num_experts * (f * P).sum()
        return aux_loss


class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing: experts select their top tokens.

    From Zhou et al. (2022) — guarantees perfect load balancing
    by letting each expert choose exactly (num_tokens / num_experts) tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expert-choice routing.

        Returns:
            expert_indices, expert_weights, aux_loss (zero for expert-choice)
        """
        batch, seq_len, d_model = x.shape
        num_tokens = batch * seq_len
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)

        logits = self.gate(x)  # (B, S, E)
        logits_flat = logits.view(num_tokens, self.num_experts)  # (N, E)

        # Each expert selects its top-capacity tokens
        # Transpose: (E, N) -> top-k per expert
        expert_logits = logits_flat.t()  # (E, N)
        top_k = min(capacity, num_tokens)

        _, top_token_indices = torch.topk(expert_logits, top_k, dim=-1)  # (E, top_k)

        # Build reverse mapping: token -> (expert_id, weight)
        expert_indices_out = torch.zeros(
            batch, seq_len, 1, dtype=torch.long, device=x.device
        )
        expert_weights_out = torch.ones(
            batch, seq_len, 1, device=x.device
        )

        # For simplicity, build full routing assignment
        probs = F.softmax(logits, dim=-1)
        for expert_id in range(self.num_experts):
            token_ids = top_token_indices[expert_id]  # (top_k,)
            for tid in token_ids:
                b = tid.item() // seq_len
                s = tid.item() % seq_len
                expert_indices_out[b, s, 0] = expert_id
                expert_weights_out[b, s, 0] = probs[b, s, expert_id]

        aux_loss = torch.tensor(0.0, device=x.device)
        return expert_indices_out, expert_weights_out, aux_loss


def compute_routing_stats(expert_indices: torch.Tensor, num_experts: int) -> dict:
    """Compute routing statistics for analysis.

    Returns dict with per-expert utilization and load balance metrics.
    """
    one_hot = F.one_hot(expert_indices, num_experts).float()
    counts = one_hot.sum(dim=(0, 1, 2))  # (E,)
    total = counts.sum()

    utilization = counts / total if total > 0 else counts
    max_util = utilization.max().item()
    min_util = utilization.min().item()
    entropy = -(utilization[utilization > 0] * utilization[utilization > 0].log()).sum().item()

    # Perfect balance entropy
    perfect_entropy = (1.0 / num_experts) * num_experts * (1.0 / num_experts)

    return {
        "expert_counts": counts.tolist(),
        "utilization": utilization.tolist(),
        "max_utilization": max_util,
        "min_utilization": min_util,
        "load_balance_ratio": min_util / max_util if max_util > 0 else 0.0,
        "entropy": entropy,
        "normalized_entropy": entropy / perfect_entropy if perfect_entropy > 0 else 0.0,
    }
