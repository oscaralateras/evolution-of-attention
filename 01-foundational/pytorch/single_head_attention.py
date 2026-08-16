"""Single-head scaled dot-product attention, in PyTorch.

The maths is documented in numpy/single_head_attention.py. This file is the
idiomatic version of the same computation and the baseline the CUDA kernels
are checked against.

Three things differ from the NumPy reference:

  * torch.softmax is used rather than rewritten. The NumPy file exists to show
    the stable-softmax construction; here the point is to use the primitive
    everyone else uses, so any disagreement between the two is a real bug in
    one of them rather than two versions of the same handwritten code.
  * It is differentiable, so correctness can be checked on the backward pass
    too, not just the forward.
  * Shapes are written for arbitrary leading dimensions from the start, so
    batch and head axes need no rewrite in multi_head_attention.py.

torch.nn.functional.scaled_dot_product_attention already does all of this,
fused. Writing it out is the point: the fused version is a stage 3 benchmark
competitor, not a baseline.
"""

import math

import torch
from torch import Tensor, nn


def causal_mask(
    L_q: int,
    L_k: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Additive causal mask, 0.0 where allowed and -inf where not.

    Rectangular: queries are the last L_q positions of the sequence, so the
    diagonal starts at column L_k - L_q. Square is the prefill special case.
    """
    offset = L_k - L_q
    i = torch.arange(L_q, device=device).unsqueeze(1)
    j = torch.arange(L_k, device=device).unsqueeze(0)

    mask = torch.zeros(L_q, L_k, device=device, dtype=dtype)
    return mask.masked_fill_(j > i + offset, float("-inf"))


class SingleHeadAttention(nn.Module):
    """Scaled dot-product attention over a single head.

    No parameters: Q, K and V arrive projected. Subclasses nn.Module anyway so
    it composes with the rest of the model — .to(device), .eval(), and being
    holdable as a child module.
    """

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        Q : (..., L_q, d_k)
        K : (..., L_k, d_k)
        V : (..., L_k, d_v)
        mask : (..., L_q, L_k) additive, or None. Broadcast against the
            scores, so a (L_q, L_k) mask applies to every batch and head.

        Returns
        -------
        out : (..., L_q, d_v)
        weights : (..., L_q, L_k)
        """
        d_k = Q.shape[-1]

        # transpose(-2, -1) rather than .T, which is 2-D only and deprecated
        # for higher rank. This form is already correct for (B, H, L, d).
        scores = Q @ K.transpose(-2, -1)

        # math.sqrt keeps this a Python float, so the division is a fused
        # scalar op. torch.sqrt would allocate a tensor for a value already
        # known from the shape, and a CPU scalar tensor dividing a CUDA tensor
        # costs a host-to-device transfer.
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores + mask

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        return out, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # float64 so the invariants are tested without float32 noise as a
    # confound. Precision behaviour is a profiling question, not a
    # correctness one.
    dt = torch.float64

    attn = SingleHeadAttention()

    # Every dimension distinct, so a transposed matmul cannot produce a valid
    # shape and slip through.
    L_q, L_k, d_k, d_v = 4, 6, 8, 5
    Q = torch.randn(L_q, d_k, dtype=dt)
    K = torch.randn(L_k, d_k, dtype=dt)
    V = torch.randn(L_k, d_v, dtype=dt)

    out, weights = attn(Q, K, V)
    assert out.shape == (L_q, d_v), out.shape
    assert weights.shape == (L_q, L_k), weights.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(L_q, dtype=dt))
    assert (weights >= 0).all()

    # Batched, to confirm the leading dimensions really are free.
    B, H = 2, 3
    bq = torch.randn(B, H, L_q, d_k, dtype=dt)
    bk = torch.randn(B, H, L_k, d_k, dtype=dt)
    bv = torch.randn(B, H, L_k, d_v, dtype=dt)
    b_out, b_w = attn(bq, bk, bv)
    assert b_out.shape == (B, H, L_q, d_v), b_out.shape
    assert torch.allclose(b_w.sum(dim=-1), torch.ones(B, H, L_q, dtype=dt))

    # Overflow: without the max subtraction inside softmax this is nan.
    _, huge_w = attn(Q * 1e3, K, V)
    assert torch.isfinite(huge_w).all()

    # Causal self-attention leaks nothing above the diagonal.
    L = 6
    Qs = torch.randn(L, d_k, dtype=dt)
    Ks = torch.randn(L, d_k, dtype=dt)
    Vs = torch.randn(L, d_v, dtype=dt)
    cm = causal_mask(L, L, dtype=dt)
    full_out, full_w = attn(Qs, Ks, Vs, cm)
    assert torch.allclose(torch.triu(full_w, diagonal=1), torch.zeros(L, L, dtype=dt))

    # Prefill/decode equivalence: recomputing only the last position against
    # the full history reproduces the last row of the full pass. This is the
    # assumption the KV cache is built on.
    step_out, _ = attn(Qs[-1:], Ks, Vs, causal_mask(1, L, dtype=dt))
    assert torch.allclose(step_out[0], full_out[-1])

    # Gradients flow and are finite. The NumPy reference cannot check this,
    # and a kernel that is right forward and wrong backward is a real failure
    # mode once these are used in training.
    Qg = Qs.clone().requires_grad_(True)
    o, _ = attn(Qg, Ks, Vs, cm)
    o.sum().backward()
    assert Qg.grad is not None and torch.isfinite(Qg.grad).all()

    print("all checks passed")
