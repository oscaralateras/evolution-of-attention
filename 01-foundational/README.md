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
write NumPy then PyTorch references, implement progressive CUDA kernels, profile on
real NVIDIA hardware, analyse KV-cache and memory-traffic costs, and verify correctness
across all three implementations.

Out of scope for Stage 1: any KV-cache compression scheme (MQA/GQA/MLA), IO-aware
exact attention (FlashAttention), and anything sub-quadratic. Those are Stages 02–07.

## Layout

| Folder | Contents |
|--------|----------|
| `math/` | Polished derivations and teaching notes — scaled dot-product, softmax stability, multi-head decomposition, shape diagrams. |
| `numpy/` | Pure NumPy reference implementations. Executable mathematics; follows the paper equations directly, no performance tricks. |
| `pytorch/` | Clean, idiomatic PyTorch reference. The main correctness baseline before CUDA. |
| `cuda/` | CUDA kernels written from first principles, in progressive versions (`v1`, `v2`, …), each commented back to the maths and the hardware decision it encodes. |
| `profiling/` | Benchmark scripts and measured results: achieved bandwidth, occupancy, arithmetic intensity, cross-implementation comparisons, hardware implications. |
| `notes/` | Working notes — failed attempts, open questions, sketches, paper quotes. Deliberately unpolished. |

Polished long-form write-ups for this stage live in the top-level `docs/` folder, not here.
