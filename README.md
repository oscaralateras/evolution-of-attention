# Evolution of Attention Mechanisms

**Status: In Progress — Stage 1 (Foundational)**

A public, systematic walk through the major practical lineages of attention used in
modern LLMs and inference systems — from the 2017 scaled dot-product formulation
through to today's linear and hybrid designs.

The goal is **deep ownership, not surface coverage**.

---

## Why this exists

I want a durable mental model. Specifically: the ability to look at high-level Python
code and immediately see the corresponding **memory traffic**, **tiling opportunities**,
**bandwidth vs compute behaviour**, and **systems trade-offs** underneath it.

That kind of intuition doesn't come from reading papers. It comes from re-deriving the
maths, writing the kernels, and measuring what the hardware actually does.

## The process (applied to every stage)

For each major stage, the process is always the same:

1. **Understand the math** and explain it simply
2. **Write clean NumPy + PyTorch reference implementations**
3. **Verify the references agree** numerically — NumPy ↔ PyTorch
4. **Implement GPU kernels** from first principles — CUDA for thread-level control,
   Triton for the block-level view — each verified against those references
5. **Profile on real NVIDIA hardware**
6. **Analyse KV-cache costs**, memory traffic, arithmetic intensity, and hardware implications

Correctness is established before anything is measured — a fast kernel that computes the
wrong thing is not a result.

---

## Series spine

| Stage | Focus | Status |
|-------|-------|--------|
| 01 – Foundational | Single-head scaled dot-product → Multi-Head Attention (MHA) | In progress |
| 02 – KV-cache compact | MQA → GQA → MLA | Planned |
| 03 – FlashAttention family | IO-aware exact attention (tiling, online softmax, progressive kernels) | Planned |
| 04 – Paged / Block Attention | Serving-oriented KV management | Planned |
| 05 – Sparse attention | Practical sparse patterns | Planned |
| 06 – Linear & sub-quadratic | Linear attention, DeltaNet-style, gated variants | Planned |
| 07 – Hybrids | Interleaved full / linear / sparse designs | Planned |

Each chosen topic is covered in depth rather than every paper being covered equally.
Papers are cited in each stage's own README as that stage is worked.

---

## Repository structure

```
evolution-of-attention/
├── README.md
├── LICENSE
├── .gitignore
├── docs/                     # per-stage summaries and the index
├── model/                    # shared decoder wrapper every variant plugs into
├── scripts/                  # shared utilities, helpers, profiling scripts
└── 01-foundational/
    ├── README.md
    ├── verify.py             # cross-checks every backend against the reference
    ├── math/                 # written explanations & derivations
    ├── numpy/                # pure NumPy reference (executable maths)
    ├── pytorch/              # clean PyTorch reference
    ├── kernels/
    │   ├── cuda/             # CUDA kernels (progressive versions)
    │   └── triton/           # Triton kernels (block-level counterpart)
    ├── profiling/            # measurements, benchmark results, analysis
    └── notes/                # working / process notes
```

---

## Stage 1 — Foundational

One paper — *Attention Is All You Need* (Vaswani et al., 2017) — already contains most
of the vocabulary the rest of the series depends on:

- single-head scaled dot-product attention
- multi-head attention (MHA)
- self-attention
- cross-attention
- causal / masked attention

Getting all five genuinely solid, at the maths, code, kernel, and hardware level, is
the whole of Stage 1.

### What Stage 1 will produce

- Teaching-quality derivation of scaled dot-product attention and its extension to multi-head
- A pure NumPy reference implementation covering single-head, multi-head, self-, cross-,
  and causal attention
- A clean PyTorch reference implementation used as the correctness baseline
- Numerical verification across every backend against a single fp64 reference
- Progressive CUDA kernels (naïve → tiled → optimised), each with its hardware rationale
- Triton kernels for the same computation, to contrast thread-level control with
  block-level abstraction — including a comparison against the Triton that
  `torch.compile` generates for the eager reference
- Profiling results on real NVIDIA hardware: achieved bandwidth, arithmetic intensity, occupancy
- A KV-cache and memory-traffic analysis for the baseline MHA formulation
- A stage summary in `docs/`, with the long technical write-up published as part of the
  accompanying blog series

---

## Feedback

Feedback is very welcome — corrections, better framings, sharper profiling methodology,
"you've misunderstood X". Open an issue or reach out. Progress is shared publicly on
Twitter alongside longer technical write-ups.

## License

MIT — see [LICENSE](LICENSE).

---

**Work in progress.** Structure, contents, and conclusions will change as the series
develops. Anything here may be revised or rewritten as understanding improves.
