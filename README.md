# Evolution of Attention Mechanisms

**Status: In Progress — Day 0 / Stage 1 (Foundational)**

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
3. **Implement CUDA kernels** from first principles
4. **Profile on real NVIDIA hardware**
5. **Analyse KV-cache costs**, memory traffic, arithmetic intensity, and hardware implications
6. **Verify correctness** across NumPy ↔ PyTorch ↔ CUDA

## What this is not

This is a **learning and portfolio project**. It is **not** a production inference
library. Nothing here is tuned for deployment, API stability, or breadth of feature
coverage. Where a choice exists between "fast" and "obvious", the code picks obvious —
except in `cuda/`, where the whole point is to earn the speed and explain where it came from.

---

## Series spine

| Stage | Focus | Key Papers | Status |
|-------|-------|------------|--------|
| 01 – Foundational | Single-head scaled dot-product → Multi-Head Attention (MHA) | Attention Is All You Need (Vaswani et al., 2017) | In progress |
| 02 – KV-cache compact | MQA → GQA → MLA | Shazeer 2019, Ainslie et al. 2023, DeepSeek-V2 | Planned |
| 03 – FlashAttention family | IO-aware exact attention (tiling, online softmax, progressive kernels) | Dao et al. | Planned |
| 04 – Paged / Block Attention | Serving-oriented KV management | vLLM PagedAttention | Planned |
| 05 – Sparse attention | Practical sparse patterns | Selected modern designs | Planned |
| 06 – Linear & sub-quadratic | Linear attention, DeltaNet-style, gated variants, Kimi Delta Attention etc. | Katharopoulos, DeltaNet papers, related work | Planned |
| 07 – Hybrids | Interleaved full / linear / sparse designs | Current hybrid architectures | Planned |

Each chosen topic is covered in depth rather than every paper being covered equally.

---

## Repository structure

```
evolution-of-attention/
├── README.md
├── LICENSE
├── .gitignore
├── docs/                     # polished long-form technical write-ups
├── scripts/                  # shared utilities, helpers, profiling scripts
└── 01-foundational/
    ├── README.md
    ├── math/                 # written explanations & derivations
    ├── numpy/                # pure NumPy reference (executable maths)
    ├── pytorch/              # clean PyTorch reference
    ├── cuda/                 # CUDA kernels (progressive versions)
    ├── profiling/            # measurements, benchmark results, analysis
    └── notes/                # working / process notes
```

### What goes where

- **`math/`** — polished teaching notes. Step-by-step derivations, shape diagrams,
  intuition, "why this formula". Should read like good technical teaching. No rough
  thoughts or debugging notes here.
- **`numpy/`** — executable mathematics. Follows the paper equations as closely as
  possible. Clean, readable, no performance tricks. The code should make the maths obvious.
- **`pytorch/`** — clean, idiomatic PyTorch reference. Clarity over speed. This is the
  main correctness baseline before CUDA.
- **`cuda/`** — kernels from first principles, in progressive versions (`v1`, `v2`, …).
  Comments link each decision back to the maths and to the hardware.
- **`profiling/`** — benchmark scripts and measured results. Achieved bandwidth,
  occupancy, arithmetic intensity, NumPy vs PyTorch vs CUDA comparisons, hardware implications.
- **`notes/`** — working notes only. "I tried X and it failed because…", sketches,
  open questions, paper quotes and links. Allowed to be messy; it records the thinking process.
- **`docs/`** — the long, polished write-ups intended for publication. These weave the
  maths, key code, diagrams, profiling insight, and hardware analysis into a single narrative.

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
- Progressive CUDA kernels (naïve → tiled → optimised), each with its hardware rationale
- Profiling results on real NVIDIA hardware: achieved bandwidth, arithmetic intensity, occupancy
- A KV-cache and memory-traffic analysis for the baseline MHA formulation
- Numerical verification across NumPy ↔ PyTorch ↔ CUDA
- A published long-form write-up in `docs/`

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
