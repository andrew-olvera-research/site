"""One scaled-dot-product-attention backend contract for all Starscream models."""

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Literal

import torch
import torch.nn.functional as F


AttentionBackend = Literal["auto", "flash", "mem_efficient", "math"]
_BACKEND: AttentionBackend = os.environ.get("STARSCREAM_ATTENTION", "auto")  # type: ignore[assignment]


def set_attention_backend(backend: AttentionBackend) -> None:
    if backend not in {"auto", "flash", "mem_efficient", "math"}:
        raise ValueError(f"unknown attention backend: {backend}")
    global _BACKEND
    _BACKEND = backend


def get_attention_backend() -> AttentionBackend:
    return _BACKEND


def _kernel_context(backend: AttentionBackend, device: torch.device):
    if backend == "auto":
        return nullcontext()
    if backend in {"flash", "mem_efficient"} and device.type != "cuda":
        raise RuntimeError(f"attention backend {backend!r} requires CUDA")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as error:
        if backend != "math":
            raise RuntimeError("explicit fused SDPA backends require a newer PyTorch") from error
        return nullcontext()
    selected = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }[backend]
    return sdpa_kernel(selected)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend: AttentionBackend | None = None,
) -> torch.Tensor:
    """SDPA with PyTorch bool-mask semantics (``True`` means allowed)."""

    selected = backend or _BACKEND
    if attn_mask is not None and is_causal:
        raise ValueError("pass either an explicit attention mask or is_causal=True, not both")
    with _kernel_context(selected, query.device):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=float(dropout_p),
            is_causal=is_causal,
        )
