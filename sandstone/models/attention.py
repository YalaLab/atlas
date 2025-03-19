"""
Reference: https://github.com/huggingface/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import List, Optional, Union, Tuple
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, LayerNorm2d


def init_attn_impl(ATTN_IMPL, verbose=True):
    if verbose:
        print(f"Attention implementation: {ATTN_IMPL}")
    torch_scaled_dot_product_attention = F.scaled_dot_product_attention
    if ATTN_IMPL == "flash_attention3":
        from flash_attn_interface import flash_attn_func
        def scaled_dot_product_attention(q, k, v, dropout_p=0.0, causal=False):
            out, _ = flash_attn_func(torch.permute(q, [0, 2, 1, 3]), torch.permute(k, [0, 2, 1, 3]), torch.permute(v, [0, 2, 1, 3]), causal=causal)
            return torch.permute(out, [0, 2, 1, 3])

        F.scaled_dot_product_attention = scaled_dot_product_attention
    elif ATTN_IMPL == "flash_attention2":
        # Need to install flash attention2: https://github.com/Dao-AILab/flash-attention/tree/v2.2.2
        from flash_attn import flash_attn_func

        torch_scaled_dot_product_attention = F.scaled_dot_product_attention

        def scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        ):
            # torch convention: B, num heads, seq len, C
            # print(f"Using flash attention, query: {query.shape}, key: {key.shape}, value: {value.shape}")
            assert attn_mask is None, attn_mask
            # On A100/H100, FlashAttn2 supports up to 256 head dim. Otherwise up to 192.
            if query.shape[-1] > 256:
                return torch_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
            return torch.permute(
                flash_attn_func(
                    torch.permute(query, [0, 2, 1, 3]),
                    torch.permute(key, [0, 2, 1, 3]),
                    torch.permute(value, [0, 2, 1, 3]),
                    dropout_p=dropout_p,
                    causal=is_causal,
                ),
                [0, 2, 1, 3],
            )

        F.scaled_dot_product_attention = scaled_dot_product_attention

        # Use memory efficient attention as a fallback
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif ATTN_IMPL == "cudnn_sdpa":
        torch.backends.cuda.enable_cudnn_sdp(True)
        # print(torch.backends.cuda.cudnn_sdp_enabled())
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    elif ATTN_IMPL == "flash_attention":
        # Note: please use precision 16-mixed or bf16-mixed
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    elif ATTN_IMPL == "memory_efficient":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif ATTN_IMPL == "math":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif ATTN_IMPL == "memory_efficient_math":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    elif ATTN_IMPL == "torch":
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        raise ValueError(
            f"Unknown attention implementation (ATTN_IMPL environment variable): {ATTN_IMPL}"
        )




class CrossWindowAttention(nn.Module):
    """Cross-window attention where queries come from a separate input."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        # Separate Q projection for query input
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # KV projection for context input
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Output projection
        self.proj = nn.Linear(dim, dim)

        # Dropouts
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_q: Query input tensor (B, Nq, C)
            x_kv: Key-value input tensor (B, Nkv, C)
            mask: Optional attention mask
        """
        B, Nq, C = x_q.shape
        _, Nkv, _ = x_kv.shape

        # Generate Q from x_q
        q = (
            self.q(x_q)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # Generate K,V from x_kv
        kv = (
            self.kv(x_kv)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)  # Each shape: (B, num_heads, Nkv, head_dim)

        if (
            torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.cudnn_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
            or torch.backends.cuda.math_sdp_enabled()
        ) and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        return x

    def run_attn(self, q, k, v, mask=None):
        B, H, Nq, D = q.shape
        C = H * D
        if (
            torch.backends.cuda.flash_sdp_enabled()
            or torch.backends.cuda.cudnn_sdp_enabled()
            or torch.backends.cuda.mem_efficient_sdp_enabled()
            or torch.backends.cuda.math_sdp_enabled()
        ) and mask is None:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.attn_drop)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        return x

    def get_qkv(self, x_q, x_kv):
        B, Nq, C = x_q.shape
        _, Nkv, _ = x_kv.shape
        q = (
            self.q(x_q)
            .reshape(B, Nq, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv(x_kv)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return q, k, v

    def get_q(self, x):
        B, Nq, C = x.shape
        q = self.q(x).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return q

    def get_kv(self, x):
        B, Nkv, C = x.shape
        kv = (
            self.kv(x)
            .reshape(B, Nkv, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        return [k, v]


class CrossWindowBlock(nn.Module):
    """Transformer block with cross-window attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        # Cross window attention
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x_q: torch.Tensor, x_kv: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x_q: Query input tensor
            x_kv: Key-value input tensor
            mask: Optional attention mask
        """
        # Cross window attention with residual
        x = x_q + self.drop_path(
            self.attn(self.norm1_q(x_q), self.norm1_kv(x_kv), mask)
        )

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def get_qkv(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
        x_q = self.norm1_q(x_q)
        x_kv = self.norm1_kv(x_kv)
        q, k, v = self.attn.get_qkv(x_q, x_kv)
        return q, k, v

    def get_qkv_tokens(self, x, key="q"):
        if key == "q":
            return self.attn.get_q(self.norm1_q(x))
        if key == "kv":
            return self.attn.get_kv(self.norm1_kv(x))

    def xattn_qkv(self, q, k, v, mask=None):
        x = self.attn.run_attn(q, k, v, mask)
        return x

    def mlp_residual(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def skip_with_drop(self, x, skip):
        x = x + self.drop_path(skip)
        return x
