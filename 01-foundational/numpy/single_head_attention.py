"""Single-head scaled dot-product attention, in NumPy.

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) V

Eq. 1 of Vaswani et al. (2017), https://arxiv.org/abs/1706.03762.

This is a reference implementation, which means it does two jobs. It explains
the equation, and it is the oracle every later implementation gets checked
against. Both jobs argue for the same style: one operation per line, nothing
fused, no vectorisation tricks. A shortcut here is a place a bug can hide in
the thing certifying everything else.

Naming: L is a length (how many vectors), d is a dimension (how wide each one
is). L comes from the input and changes every call; d is fixed when the model
is designed.

    L_q   query positions
    L_k   key/value positions
    d_k   width of each query and key vector, e.g. 64
    d_v   width of each value vector, usually == d_k

    Q     (L_q, d_k)
    K     (L_k, d_k)
    V     (L_k, d_v)
    out   (L_q, d_v)

Q and K must share d_k, since q·k needs equal-length vectors. K and V must
share L_k, since each key indexes one value. d_v is free — V is never dotted
with Q, only weighted-summed.

L_q and L_k are equal only when every token is being processed at once. Every
token produces a query, a key and a value, but the three have different
lifetimes. Once a token's output is computed it never changes, because causal
masking means nothing later can affect it, so its query is finished with and
thrown away. Its key and value stay: every future token still needs to look at
them. That asymmetry is why inference caches K and V and not Q, and why during
decode Q holds one row while K holds the whole history. The shapes here are
rectangular so that case needs no second code path.
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    Turns arbitrary real scores into a probability distribution: exp makes
    every entry positive, dividing by the row sum makes each row total 1.
    Both halves earn their place. Positive weights that sum to 1 make
    `weights @ V` a weighted average of the value vectors, so the output stays
    on the same scale as V no matter how large the raw scores were. Without
    the normalisation, stacked layers would rescale the residual stream on
    every pass.

    exp(x) / sum(exp(x)) written directly overflows: exp(1000) is inf, and
    inf/inf is nan, so one large score destroys its whole row. Subtracting the
    row max first is exact rather than an approximation, since the resulting
    exp(-max) factor cancels between numerator and denominator. The largest
    exponent becomes 0, so nothing exceeds exp(0) = 1.

    The same cancellation is what makes FlashAttention's online softmax
    possible in stage 3, where the row is never fully materialised so the max
    is not known up front and has to be corrected as it grows.

    Parameters
    ----------
    x : (..., n) scores
    axis : axis to normalise over, collapsed by the max and the sum

    Returns
    -------
    (..., n) non-negative, summing to 1 along `axis`
    """
    # keepdims on both reductions. Without it the (L_q, 1) intermediates
    # collapse to (L_q,) and broadcast against the wrong axis — which raises
    # when L_q != L_k, but silently computes nonsense when they are equal.
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def causal_mask(L_q: int, L_k: int) -> np.ndarray:
    """Additive mask letting query i attend only to keys at or before it.

    Rectangular on purpose. K holds the full history while Q holds only the
    positions currently being computed, so the queries are always the *last*
    L_q positions of the sequence — offset by however many keys precede them.

    A square lower-triangular mask is the L_q == L_k special case. Hardcoding
    that shape breaks decode, where one query attends over a long cache and
    should mask nothing at all.

    Returns
    -------
    (L_q, L_k) of 0.0 where attention is allowed, -inf where it is not
    """
    # j indexes K, which spans the whole sequence, so j is an absolute
    # position. i indexes Q, which holds only the new rows, so i is local and
    # restarts at 0. offset converts one to the other.
    offset = L_k - L_q

    i = np.arange(L_q)[:, None]  # (L_q, 1)
    j = np.arange(L_k)[None, :]  # (1, L_k)

    # <= rather than < so a token can attend to itself.
    return np.where(j <= i + offset, 0.0, -np.inf)


class SingleHeadAttention:
    """Scaled dot-product attention over a single head.

    Holds no parameters. Q, K and V arrive already projected, because the
    W_q/W_k/W_v projections belong to multi-head attention.

    It is a class for interface consistency: everything in this series exposes
    the same `forward`, so the transformer wrapper can hold any of them
    without knowing which. Two independent things get swapped through that
    slot, and they are worth keeping apart.

    Architecture — MHA, MQA, GQA, MLA, linear. Mutually exclusive; a model
    picks one. These compute different functions and differ in what state
    they carry, which from stage 2 is threaded through a `decode_step`
    alongside `forward`: a KV cache here, a compressed latent for MLA, a
    fixed-size recurrent matrix for DeltaNet.

    Implementation — naive, tiled, FlashAttention. Same architecture, same
    outputs, different memory schedule. So FlashAttention does not compete
    with GQA; production systems run both at once. Comparisons only mean
    something along one axis at a time.
    """

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Attend queries over keys and mix the corresponding values.

        Parameters
        ----------
        Q : (L_q, d_k)
        K : (L_k, d_k)
        V : (L_k, d_v)
        mask : (L_q, L_k) additive, or None. 0.0 where allowed, -inf where
            not, added to the scaled scores before the softmax so masked
            positions come out at exactly zero weight.

            Additive rather than assignment for three reasons. Masks compose,
            so causal plus padding is a sum. Finite values work too, which is
            how ALiBi and relative position biases plug in. And it is the
            interface every CUDA kernel expects, since a fused add is cheap
            while scattered conditional writes are not.

            Causality is therefore a choice of argument, not a code path.
            Cross-attention, padding and the stage 5 sparse patterns all reuse
            this one slot.

        Returns
        -------
        out : (L_q, d_v)
        weights : (L_q, L_k), rows summing to 1

        Returning the weights is free and makes the invariants testable and
        the behaviour plottable.
        """
        d_k = Q.shape[-1]

        # Every query against every key in one matmul. d_k is contracted: the
        # dot product sums it away, leaving one scalar score per (query, key).
        # This (L_q, L_k) matrix is the quadratic term — at 8k context in fp32
        # it is 268 MB, allocated again for `weights`. Removing it is the
        # whole point of stage 3. Batched shapes need swapaxes(-1, -2)
        # instead of .T, which reverses every axis.
        scores = Q @ K.T

        # Scores grow with head width: with unit-variance inputs the dot
        # product sums d_k independent terms, so its standard deviation is
        # √d_k. Dividing restores unit variance whatever d_k is. Softmax
        # depends only on the gaps between scores, so unscaled scores at large
        # d_k saturate it onto one key and the gradient vanishes. √d_k rather
        # than d_k because the correction is to a standard deviation.
        scores = scores / np.sqrt(d_k)

        # `is not None`, never `if mask:` — truth-testing an array of more
        # than one element raises. Masking after the scale, so -inf is not
        # rescaled and finite biases are applied on the same footing as the
        # scores they adjust.
        if mask is not None:
            scores = scores + mask

        # axis=-1 is the key axis: each query independently distributes its
        # attention across the keys. Normalising the other way would spread
        # each key across queries, which is a different and meaningless
        # operation with identical shapes. -1 rather than 1 so the code
        # survives the batch and head axes that MHA adds in front.
        weights = softmax(scores, axis=-1)

        # L_k contracted. Row i is the convex combination of value vectors
        # picked out by query i's distribution.
        out = weights @ V

        return out, weights

    def __call__(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """`attn(Q, K, V)` for `attn.forward(Q, K, V)`, as nn.Module does."""
        return self.forward(Q, K, V, mask)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    attn = SingleHeadAttention()

    # Every dimension distinct. Square inputs let a transposed matmul produce
    # a valid shape and pass unnoticed. L_q != L_k is not an exotic case: it
    # is what decode looks like, one query row against the whole cache.
    L_q, L_k, d_k, d_v = 4, 6, 8, 5
    Q = rng.standard_normal((L_q, d_k))
    K = rng.standard_normal((L_k, d_k))
    V = rng.standard_normal((L_k, d_v))

    out, weights = attn(Q, K, V)
    assert out.shape == (L_q, d_v), out.shape
    assert weights.shape == (L_q, L_k), weights.shape
    assert np.allclose(weights.sum(axis=-1), 1.0), "rows must sum to 1"
    assert (weights >= 0).all(), "weights must be non-negative"

    # Overflow: scores this large give nan without the max subtraction.
    huge = rng.standard_normal((L_q, d_k)) * 1e3
    _, huge_w = attn(huge, K, V)
    assert np.isfinite(huge_w).all(), "softmax overflowed"

    # Causal self-attention: nothing above the diagonal survives.
    L = 6
    Qs = rng.standard_normal((L, d_k))
    Ks = rng.standard_normal((L, d_k))
    Vs = rng.standard_normal((L, d_v))
    full_out, full_w = attn(Qs, Ks, Vs, causal_mask(L, L))
    assert np.allclose(np.triu(full_w, k=1), 0.0), "leaked into the future"

    # Prefill/decode equivalence. Recomputing only the last position, with one
    # query row against the full history, must reproduce the last row of the
    # full pass. This is the property the KV cache relies on, and the reason
    # causal_mask is rectangular: causal_mask(1, L) is all zeros, since the
    # newest token has no future to hide from.
    step_out, _ = attn(Qs[-1:], Ks, Vs, causal_mask(1, L))
    assert np.allclose(step_out[0], full_out[-1]), "decode diverged from prefill"

    print("all checks passed")
