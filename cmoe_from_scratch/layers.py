"""
CMoE Layer: integrates conditional router with expert pool.

Drop-in replacement for transformer FFN layers that adds
sparse conditional computation with load balancing.
"""

import torch
import torch.nn as nn
from typing import Optional

from .experts import ExpertPool, SharedExpert
from .router import TopKRouter


class CMoELayer(nn.Module):
    """Conditional Mixture of Experts layer.

    Replaces a standard FFN with:
        output = x + sparse_experts(x) + shared_expert(x)

    The sparse experts are conditionally routed via top-k gating.
    The shared expert always fires (captures generic patterns).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        noise_std: float = 0.1,
        capacity_factor: float = 1.0,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = use_shared_expert

        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=noise_std,
            capacity_factor=capacity_factor,
        )
        self.experts = ExpertPool(
            num_experts=num_experts,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        if use_shared_expert:
            self.shared_expert = SharedExpert(d_model, d_ff, dropout)

        self.aux_loss_weight = 0.01
        self._aux_loss = torch.tensor(0.0)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_indices, expert_weights, aux_loss = self.router(x)
        self._aux_loss = aux_loss * self.aux_loss_weight

        sparse_out = self.experts.forward_batched(x, expert_indices, expert_weights)
        output = sparse_out

        if self.use_shared_expert:
            output = output + self.shared_expert(x)

        return output


class CMoETransformerBlock(nn.Module):
    """Single transformer block with CMoE replacing the FFN.

    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> CMoE -> Residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        use_shared_expert: bool = True,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cmoe = CMoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
            use_shared_expert=use_shared_expert,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + attn_out

        # CMoE FFN with pre-norm
        residual = x
        x_norm = self.ln2(x)
        cmoe_out = self.cmoe(x_norm)
        x = residual + cmoe_out

        return x


def replace_ffn_with_cmoe(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    target_modules: Optional[list] = None,
    use_shared_expert: bool = True,
    verbose: bool = True,
) -> nn.Module:
    """Replace FFN layers in a transformer model with CMoE layers.

    Targets the intermediate MLP layers by name.
    """
    if target_modules is None:
        target_modules = ["mlp", "ffn", "feed_forward"]

    count = 0

    def _replace(module: nn.Module, prefix: str = ""):
        nonlocal count
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this looks like an FFN/MLP
            if any(t in name.lower() for t in target_modules):
                # Detect dimensions from the child's layers
                if hasattr(child, '__getitem__'):
                    continue
                if hasattr(child, 'c_fc') and hasattr(child, 'c_proj'):
                    # GPT-2 style MLP: c_fc (768->3072), c_proj (3072->768)
                    d_model = child.c_fc.out_features if hasattr(child.c_fc, 'out_features') else 768
                    d_ff = d_model * 4

                    cmoe = CMoELayer(
                        d_model=768,
                        d_ff=d_ff,
                        num_experts=num_experts,
                        top_k=top_k,
                        use_shared_expert=use_shared_expert,
                    )
                    setattr(module, name, cmoe)
                    count += 1
                    if verbose:
                        print(f"  [CMoE] {full_name}: FFN -> {num_experts} experts (top-{top_k})")
                continue

            _replace(child, full_name)

    _replace(model)
    if verbose:
        print(f"\nReplaced {count} FFN layer(s) with CMoE ({num_experts} experts, top-{top_k})")
    return model
