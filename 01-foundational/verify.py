"""Cross-verify the NumPy and PyTorch single-head references.

Step 3 of the process: the two references must agree before either is used to
judge a CUDA kernel. If they disagree, one of them is wrong and there is no
baseline at all.

Both modules are loaded by file path rather than imported normally. The
reference lives in a directory called `numpy/`, which would shadow the real
NumPy package the moment `01-foundational/` landed on sys.path. Loading by
path never touches sys.path, so the collision cannot happen.

Run: uv run python 01-foundational/verify.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch

STAGE = Path(__file__).parent

# Tight, because both sides run float64 over identical operations. Anything
# larger than rounding noise means a genuine difference in the maths.
TOL = 1e-12


def load(name: str, path: Path):
    """Load a module from an explicit file path, bypassing sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


np_sha = load("np_sha", STAGE / "numpy" / "single_head_attention.py")
pt_sha = load("pt_sha", STAGE / "pytorch" / "single_head_attention.py")


def compare(label: str, a: np.ndarray, b: torch.Tensor) -> float:
    """Report and assert the largest elementwise gap."""
    b = b.detach().numpy()
    assert a.shape == b.shape, f"{label}: shape {a.shape} vs {b.shape}"

    # Masks carry -inf, and -inf minus -inf is nan, so subtraction cannot be
    # applied blindly. Check that the non-finite entries sit in the same
    # places and hold the same values, then measure the finite ones.
    finite = np.isfinite(a)
    assert np.array_equal(finite, np.isfinite(b)), f"{label}: -inf in different places"
    assert np.array_equal(a[~finite], b[~finite]), f"{label}: -inf values differ"

    diff = np.abs(a[finite] - b[finite]).max() if finite.any() else 0.0
    print(f"  {label:<24} max|Δ| = {diff:.3e}")
    assert diff < TOL, f"{label} exceeded {TOL}"
    return diff


def case(name: str, L_q: int, L_k: int, d_k: int, d_v: int, causal: bool) -> None:
    print(f"\n{name}  (L_q={L_q}, L_k={L_k}, d_k={d_k}, d_v={d_v}, causal={causal})")

    rng = np.random.default_rng(0)
    Q = rng.standard_normal((L_q, d_k))
    K = rng.standard_normal((L_k, d_k))
    V = rng.standard_normal((L_k, d_v))

    # Same numbers on both sides. torch.from_numpy shares memory rather than
    # copying, so there is no chance of the two seeing different inputs.
    tQ, tK, tV = (torch.from_numpy(x) for x in (Q, K, V))

    np_mask = np_sha.causal_mask(L_q, L_k) if causal else None
    pt_mask = pt_sha.causal_mask(L_q, L_k, dtype=torch.float64) if causal else None

    if causal:
        compare("causal_mask", np_mask, pt_mask)

    np_out, np_w = np_sha.SingleHeadAttention()(Q, K, V, np_mask)
    pt_out, pt_w = pt_sha.SingleHeadAttention()(tQ, tK, tV, pt_mask)

    compare("weights", np_w, pt_w)
    compare("output", np_out, pt_out)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Square self-attention, both masked and not.
    case("prefill, unmasked", 8, 8, 16, 16, causal=False)
    case("prefill, causal", 8, 8, 16, 16, causal=True)

    # Distinct dimensions throughout, so a transpose cannot survive by
    # accidentally producing a valid shape.
    case("rectangular", 4, 6, 8, 5, causal=False)

    # Decode: one query against the whole cache. The shape that matters from
    # stage 2 onward.
    case("decode step", 1, 12, 16, 16, causal=True)

    # Chunked prefill: several new queries against a longer history.
    case("chunked prefill", 3, 12, 16, 16, causal=True)

    # A long row, where an unstable softmax would show up as drift rather
    # than as an obvious nan.
    case("long context", 2, 512, 64, 64, causal=True)

    print(f"\nNumPy and PyTorch agree to within {TOL:g}")
