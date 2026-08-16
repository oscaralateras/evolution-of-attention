# Stage 01 — Foundational Attention

**Status: In progress**

## Paper

**Attention Is All You Need** — Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez,
Kaiser, Polosukhin (2017).
arXiv: https://arxiv.org/abs/1706.03762

## Scope

This single paper covers the whole foundational vocabulary of the series:

- **Single-head scaled dot-product attention** — `softmax(QKᵀ / √d_k) V`
- **Multi-head attention (MHA)** — parallel heads, projections, concatenation
- **Self-attention** — Q, K, V from the same sequence
- **Cross-attention** — Q from one sequence, K/V from another
- **Causal / masked attention** — enforcing autoregressive structure

For each of these the stage follows the standard six-step process: derive the maths,
write NumPy then PyTorch references, verify those references agree numerically,
implement progressive CUDA and Triton kernels checked against them, profile on real
NVIDIA hardware, and analyse KV-cache and memory-traffic costs.

Out of scope for Stage 1: any KV-cache compression scheme (MQA/GQA/MLA), IO-aware
exact attention (FlashAttention), and anything sub-quadratic. Those are Stages 02–07.

## Layout

| Folder | Contents |
|--------|----------|
| `math/` | Polished derivations and teaching notes — scaled dot-product, softmax stability, multi-head decomposition, shape diagrams. |
| `numpy/` | Pure NumPy reference implementations. Executable mathematics; follows the paper equations directly, no performance tricks. |
| `pytorch/` | Clean, idiomatic PyTorch reference. The correctness baseline every kernel is measured against, and the layer the shared decoder wrapper is built in. |
| `kernels/cuda/` | CUDA kernels written from first principles, in progressive versions (`v1`, `v2`, …), each commented back to the maths and the hardware decision it encodes. |
| `kernels/triton/` | The same kernels in Triton. CUDA is thread-level — shared memory, warps, synchronisation placed by hand. Triton is block-level: you write tiles and the compiler handles coalescing, register allocation and intra-block scheduling. Written after the CUDA so it is clear what Triton is doing for you rather than what it is hiding. |
| `profiling/` | Benchmark scripts and measured results: achieved bandwidth, occupancy, arithmetic intensity, cross-implementation comparisons, hardware implications. Every result records the exact GPU, driver, CUDA toolkit, and PyTorch version it was measured on — numbers are not comparable without it. |
| `notes/` | Working notes — failed attempts, open questions, sketches, paper quotes. Deliberately unpolished. |

`verify.py` sits at the top of the stage and cross-checks every backend against
the same fp64 reference. It runs in float64 so that rounding noise stays far
below any real disagreement: a genuine bug is then unmistakable rather than lost
in the last two digits. Benchmarks use bf16, which is what production inference
runs and what the tensor cores want — verification and measurement want opposite
dtypes for opposite reasons.

The stage summary lives in the top-level `docs/` folder. Long technical write-ups are
published in the accompanying blog series rather than in the repository.

## Checkpoint

`checkpoint.md` is the stage's definition of done, and predictions are committed to
`profiling/predictions.md` *before* anything is measured — the git history is what makes
them predictions rather than hindsight.

Mechanism: predict what changes versus the previous stage before implementing it;
validate every derived formula against a real published model rather than only against
your own numbers; and treat the `math/` write-up as the test, since a section that keeps
coming out vague is pointing at a gap.

Hardware: achieved bandwidth as a percentage of theoretical peak, measured occupancy
against what register and shared-memory usage predicted, the top stall reason from
Nsight Compute, and an account of every remaining gap between the best hand-written
kernel and `F.sdpa`.
