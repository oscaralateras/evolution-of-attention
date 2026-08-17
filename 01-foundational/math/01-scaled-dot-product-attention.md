# Scaled dot-product attention

Assumes linear algebra and no prior exposure to attention. Every design choice in

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

is justified here, including the ones that look arbitrary.

---

## 1. The problem

Before attention, sequence-to-sequence models were recurrent. An encoder RNN read the
input one token at a time, updating a hidden state; the final hidden state was handed to
a decoder RNN, which generated output from it.

Three things were wrong with that.

**The bottleneck.** The decoder saw one fixed-size vector. A five-word sentence and a
five-hundred-word document were both compressed into the same number of floats.
Everything the decoder could ever know about the input had to survive that compression.

**Path length.** For information in token 1 to influence the output at token 500, it
passes through 499 sequential transformations. Gradients flowing back along that path
shrink multiplicatively. LSTMs and GRUs made the decay slower, not absent — long-range
dependencies stayed hard to learn.

**Sequential execution.** Hidden state $h_t$ depends on $h_{t-1}$, so the tokens must be
processed in order. No parallelism across the sequence, on hardware whose entire
advantage is parallelism.

Attention addresses all three. There is no summary vector, so nothing is compressed.
Any token reaches any other token in **one step**, so gradients do not decay with
distance. And every position is computed simultaneously, as one matrix multiply.

The price is quadratic cost, which is what the rest of this series is about.

---

## 2. The shape of a solution

Each token needs to build a representation of itself that incorporates the other tokens
it cares about. Two things are needed:

1. A way to decide **how relevant** every other token is — a number per pair.
2. A way to **combine** the other tokens according to those numbers.

Step 2 is a weighted average. Step 1 is the interesting part.

---

## 3. Queries, keys and values

Each token's hidden state $x_i$ is projected three ways:

$$q_i = x_i W_q \qquad k_i = x_i W_k \qquad v_i = x_i W_v$$

The intuition is a lookup. The **query** is what a token is looking for. The **key** is
what a token advertises about itself. The **value** is what it contributes if selected.

That intuition is worth holding onto — it makes the mechanism memorable and it is how
most people first understand attention. But it explains what the three vectors *are*
rather than why three are needed at all, so the rest of this section supplies the
argument the intuition skips.

### Why not use $x$ directly?

Suppose we skip $W_q$ and $W_k$ and score with $XX^\top$. Entry $[i,j]$ is
$x_i \cdot x_j$, and entry $[j,i]$ is $x_j \cdot x_i$. **The matrix is symmetric.**

Token $i$'s relevance to token $j$ would be forced equal to $j$'s relevance to $i$.
Language is not like that. In

> the trophy didn't fit in the suitcase because **it** was too big

"it" must attend strongly to "trophy" to resolve at all. "Trophy" has little reason to
attend back to "it". A symmetric score matrix cannot represent a one-directional
dependency, and most linguistic dependencies are one-directional.

There is a second problem. The diagonal entry $x_i \cdot x_i = |x_i|^2$ is the
**largest entry in its row** — a vector is more aligned with itself than with anything
else, which is the Cauchy–Schwarz inequality. Before learning anything, every token
would attend mostly to itself.

Separate projections fix both. $(x_i W_q) \cdot (x_j W_k)$ has no reason to equal
$(x_j W_q) \cdot (x_i W_k)$, so asymmetry is available, and nothing forces the diagonal
to dominate.

### Why is $V$ separate from $K$?

Because what makes a token **findable** is not what it **contributes**.

A library catalogue entry is not the book. The key is the index — the properties by
which a token should be retrieved. The value is the payload — what gets passed on once
it is. Tying them would force a token to be findable only by the content it delivers.

### What the projections also buy

They are learned, so the model chooses *which* subspace similarity is measured in
rather than being stuck with whatever the residual stream happens to encode. And they
reduce dimension from $d_{model}$ to $d_k$, which is what makes multiple heads
affordable.

---

## 4. Scoring with the dot product

Relevance of key $j$ to query $i$ is their dot product:

$$s_{ij} = q_i \cdot k_j = |q_i|\,|k_j|\cos\theta$$

Magnitude times alignment:

| $\theta$ | $\cos\theta$ | score | meaning |
|---|---|---|---|
| 0° | +1 | large positive | aligned — very relevant |
| 90° | 0 | zero | orthogonal — unrelated |
| 180° | −1 | large negative | opposed — actively irrelevant |

A negative score means the vectors point in opposite directions. It is the strongest
possible statement of *irrelevance*, and it must end up with near-zero weight.

**A known weakness:** the dot product conflates direction with magnitude. A key with
large norm scores highly against every query regardless of angle. Nothing in the
mechanism prevents this; normalisation elsewhere in the network is what keeps
magnitudes in a sane range.

All pairs at once:

$$S = QK^\top$$

$$(L_q, d_k) \times (d_k, L_k) \rightarrow (L_q, L_k)$$

The $d_k$ axis is contracted — summed away inside each dot product. What remains is one
scalar per (query, key) pair.

---

## 5. From scores to weights

Scores are arbitrary reals. To average with them they must be non-negative and sum to 1.

### Why not just divide by the sum?

Because scores can be negative. Take $s = [3, -1, -2]$:

$$\sum s = 0$$

Division by zero. Try $s = [3, -1, -1]$, which sums to 1:

$$w = [3, -1, -1]$$

A weight of 3 and two weights of $-1$. Both meaningless — you cannot take $-1$ of a
value vector, and a weight above 1 amplifies rather than averages.

**Linear normalisation cannot survive negative inputs.** That is the whole reason for
the exponential.

### Why exponentiate

$\exp$ maps $\mathbb{R} \rightarrow (0, \infty)$. Every score, however negative, becomes
positive. Once positive, dividing by the sum is safe:

$$w_{ij} = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}$$

Note $\exp$ alone does **not** bound anything to $[0,1]$ — $e^5 = 148$. The division
does that. Any positive numbers divided by their own total sum to 1, because you are
dividing the total by itself:

$$\frac{a}{a+b+c} + \frac{b}{a+b+c} + \frac{c}{a+b+c} = \frac{a+b+c}{a+b+c} = 1$$

A second property comes free. Since $e^a / e^b = e^{a-b}$, the **ratio** between two
weights depends only on the **difference** between their scores. Absolute values are
irrelevant; only gaps matter. This lets the distribution sharpen toward hard selection
as gaps grow, which linear normalisation cannot do.

### Why sum-to-1 is the point

The output is $\text{weights} \times V$. Because the weights are non-negative and total
1, each output row is a **convex combination** of the value vectors — a weighted average
whose weights are a probability distribution. The consequence is that the output lands
*inside* the region the value vectors span, on the same scale as $V$, regardless of how
large the raw scores were.

If weights summed to 5, outputs would be roughly 5× the values. A transformer adds each
layer's attention output back into a running representation that passes through the
whole network, so a layer that multiplies magnitude by 5 compounds: sixty such layers
would produce numbers no float can hold. Normalisation is what makes attention a
*mixing* operation rather than a *scaling* one.

---

## 6. Numerical stability

Written directly, the softmax overflows. In float64, $\exp$ overflows above about 709;
$e^{1000}$ is `inf`, and `inf/inf` is `nan`. One large score destroys its entire row.

The fix is to subtract the row maximum first:

$$w_i = \frac{e^{s_{ij} - m_i}}{\sum_{j'} e^{s_{ij'} - m_i}}, \qquad m_i = \max_j s_{ij}$$

**This is exact, not an approximation:**

$$\frac{e^{s-m}}{\sum e^{s'-m}} = \frac{e^{s}e^{-m}}{e^{-m}\sum e^{s'}} = \frac{e^{s}}{\sum e^{s'}}$$

The $e^{-m}$ factors out of numerator and denominator and cancels. Every gap is
preserved, so every weight is identical — but the largest exponent is now 0, so nothing
exceeds $e^0 = 1$ and overflow is impossible.

Very negative entries underflow to exactly 0. Harmless: they were getting negligible
weight anyway.

> This cancellation is the seed of the **online softmax** in FlashAttention (stage 3),
> where the row is never fully materialised so $m$ is not known in advance. Because
> correcting by $e^{m_{old} - m_{new}}$ is exact rather than lossy, the maximum can be
> revised as it is discovered.

---

## 7. The scale factor $\sqrt{d_k}$

Take $q$ and $k$ with independent components of mean 0 and variance 1. Then

$$q \cdot k = \sum_{c=1}^{d_k} q_c k_c$$

Each term has mean $E[q_c]E[k_c] = 0$ and variance
$E[q_c^2]E[k_c^2] = 1$. Independent variances add, so

$$\text{Var}(q \cdot k) = d_k, \qquad \text{SD}(q \cdot k) = \sqrt{d_k}$$

**Scores grow with head width.** At $d_k = 64$ they typically span ±8; at $d_k = 512$,
±23.

Now recall that only *gaps* between scores matter. Wider heads mean larger gaps:

| $d_k$ | gap of one SD | weight ratio $e^{\text{gap}}$ |
|---|---|---|
| 64 | 8 | ~3,000 |
| 512 | 23 | ~10,000,000,000 |

Taking the gap as one standard deviation understates it — two independent draws differ
by about $1.4\,\sigma$ on average, and the largest gap in a row is larger still. The
table is therefore a floor, and the real ratios are worse.

At $d_k = 512$ the softmax is effectively one-hot. A saturated softmax has near-zero
gradient — the model stops learning.

Dividing by $\sqrt{d_k}$ restores unit variance for any $d_k$, keeping the distribution
in a usable range as the architecture scales.

**Why $\sqrt{d_k}$ and not $d_k$:** you are correcting a *standard deviation*, and the
standard deviation is $\sqrt{d_k}$. Dividing by $d_k$ over-corrects, crushing scores
toward zero and producing a near-uniform distribution that cannot discriminate.

---

## 8. Which axis the softmax normalises

Over **keys** — the last axis of the $(L_q, L_k)$ score matrix.

Each query independently distributes a fixed budget of attention across the available
keys. Row $i$ of the weights is query $i$'s probability distribution, so row $i$ sums
to 1.

Normalising the other way would make each *key* distribute itself across queries. The
shapes are identical, the outputs look plausible, and the operation is meaningless. This
is the single easiest error to make here and the hardest to detect by inspection — which
is why the reference implementation asserts that rows sum to 1.

---

## 9. Masking

Some positions must be excluded — a token must not see its own future, and padding must
be ignored. This is done by adding a mask to the scores **before** the softmax:

$$S' = \frac{QK^\top}{\sqrt{d_k}} + M$$

where $M_{ij} = 0$ if allowed and $-\infty$ if not.

It works because $e^{-\infty} = 0$ exactly. Masked positions contribute nothing to the
weights and nothing to the denominator, so the surviving weights renormalise themselves
with no extra step.

**Why additive rather than assignment.** Masks compose — causal plus padding is a sum.
Finite values also work, which is how ALiBi and relative-position biases attach. And it
is the interface GPU kernels expect: a fused add is cheap, scattered conditional writes
are not.

So causality is a *choice of argument*, not a separate code path.

### The causal mask, and why it is rectangular

Query $i$ may attend to key $j$ only if $j$ is at or before it. When all tokens are
processed at once this is lower-triangular. But $Q$ and $K$ need not hold the same
tokens.

$K$ holds every token so far. $Q$ holds only the ones being computed now. Every token
produces a query, a key and a value, but they have different lifetimes: once a token's
output exists it can never change, because nothing later may influence it, so its query
is finished with. Its key and value must persist — every future token still needs them.

**That asymmetry is why inference caches K and V and not Q.**

So $Q$'s rows are the *last* $L_q$ positions, and its local index $i$ must be converted
to an absolute position:

$$\text{offset} = L_k - L_q, \qquad \text{allowed} \iff j \le i + \text{offset}$$

| | $L_q$ | $L_k$ | offset | mask |
|---|---|---|---|---|
| Prefill | 6 | 6 | 0 | lower triangular |
| Chunked | 2 | 6 | 4 | triangle starting at column 4 |
| Decode | 1 | 6 | 5 | all zeros — nothing to hide |

A hardcoded square mask silently breaks decode, blocking most of the cache.

---

## 10. The whole thing

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

```
Q (L_q, d_k) ─┐
              ├─ QKᵀ ─> scores (L_q, L_k) ─ /√d_k ─ +mask ─ softmax ─> weights (L_q, L_k)
K (L_k, d_k) ─┘                                                              │
                                                                             │
V (L_k, d_v) ────────────────────────────────────────────────────────────────┴─> out (L_q, d_v)
```

Two contractions, one in each matmul:

| step | shape | contracted |
|---|---|---|
| $QK^\top$ | $(L_q, L_k)$ | $d_k$ |
| softmax | $(L_q, L_k)$ | — |
| $WV$ | $(L_q, d_v)$ | $L_k$ |

$Q$ and $K$ must share $d_k$, since a dot product needs equal-length vectors. $K$ and
$V$ must share $L_k$, since each key indexes one value. $d_v$ is otherwise free — $V$ is
never dotted with $Q$, only averaged.

---

## 11. What it costs

**Time.** The score matrix has $L_q L_k$ entries, each costing $d_k$ multiply-adds, and
the second matmul costs the same again:

$$O(L^2 d)$$

**Memory.** The intermediate is $O(L^2)$, independent of $d$ — and this is the harder
constraint. Time you can wait for; memory you cannot exceed. At $L = 8192$ in fp32 one
head's score matrix is 268 MB, and with 32 heads that is **8.6 GB of transient
allocation for a single layer**.

**Compared to everything else.** The projections cost $O(L\,d_{model}^2)$ — *linear* in
sequence length. Their ratio to attention is roughly $L / 2d_{model}$, so in **FLOP
terms** attention only overtakes them once sequence length is around twice the model
width: about 8k tokens for a 4096-wide model.

That is a statement about arithmetic, not about what limits you in practice, and the two
come apart in both directions. The $O(L^2)$ allocation can exhaust memory long before
attention dominates the FLOP count, so attention can be the binding constraint while
still being the smaller share of the work. And during decode, where a single query
attends over the whole cache, the cost is dominated by *reading* the cache rather than
by arithmetic at all — a regime this counting says nothing about.

The useful conclusion is narrower than "optimise attention": know which resource you are
short of before deciding what to optimise.

---

## 12. What follows

Everything after this stage attacks one of two costs.

The $O(L^2)$ **memory** term is attacked by never materialising the score matrix —
FlashAttention (stage 3), which is where the online softmax from §6 is needed.

The **KV cache** implied by §9 is attacked by storing less per token — MQA, GQA and MLA
(stage 2) — or by managing it better across sequences — PagedAttention (stage 4).

The $O(L^2)$ **time** term is attacked by not computing every pair: sparse patterns
(stage 5), or a fixed-size recurrent state instead of a growing cache (stage 6).

---

## References

Vaswani et al., *Attention Is All You Need*, 2017. https://arxiv.org/abs/1706.03762

The executable form of everything here is
[`numpy/single_head_attention.py`](../numpy/single_head_attention.py), cross-checked
against the PyTorch reference by [`verify.py`](../verify.py).
