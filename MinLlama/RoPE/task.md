### Learning objectives

By completing this task, you will be able to:
- **Understand rotary positional embeddings** - Learn how RoPE encodes positional information in transformers more effectively than traditional approaches.
- **Implement complex tensor operations** - Apply rotation matrices and frequency-based transformations to high-dimensional tensor data.
- **Work with positional encoding mathematics** - Understand the mathematical foundation of how position information is encoded in modern transformers.
- **Debug transformer components** - Use provided unit tests to verify correctness of mathematical implementations in deep learning contexts.

### Problem context

Traditional positional embeddings in transformers use either learned or sinusoidal position encodings that are added to token embeddings. However, these approaches have limitations when dealing with longer sequences or when trying to maintain relative positional relationships between tokens.

**Why rotary positional embeddings matter:**
- **Relative position encoding** - RoPE encodes relative distances between tokens more naturally than absolute position methods.
- **Length extrapolation** - Models trained with RoPE can better handle sequences longer than those seen during training.
- **Computational efficiency** - RoPE integrates position information directly into attention computations without requiring additional parameters.
- **State-of-the-art performance** - Used in leading language models like LLaMA, PaLM, and other modern architectures for improved performance.

**What makes this challenging:**
- **Complex mathematics** - Involves rotation matrices, frequency transformations, and multi-dimensional tensor operations.
- **Tensor manipulation precision** - Small errors in tensor reshaping or rotation can lead to dramatically different model behavior.
- **Frequency computation** - Requires careful implementation of frequency scaling and rotation angle calculations.
- **Integration complexity** - Must work seamlessly with existing attention mechanisms and maintain compatibility with pretrained weights.

## Implementation requirements

Implement the rotary positional embedding mechanism that allows the LLaMA model to encode positional information more effectively than traditional approaches. This is a critical component that significantly impacts model performance on long sequences.

### Specific requirements:

1. **Implement `apply_rotary_emb` function** - Apply rotation matrices to query and key vectors based on their positions.
2. **Handle tensor reshaping** - Properly manage tensor dimensions for rotation operations.
3. **Compute rotation frequencies** - Calculate appropriate rotation frequencies for different embedding dimensions.
4. **Maintain mathematical precision** - Ensure rotation operations preserve the mathematical properties required for effective positional encoding.

### Expected deliverables:

- Completed `apply_rotary_emb` function that correctly applies rotary embeddings.
- Implementation that passes all provided unit tests.
- Code that maintains compatibility with LLaMA pretrained weights (no parameter renaming).
- Proper handling of batch dimensions and sequence lengths.

## Mathematical foundation

Rotary Positional Embedding (RoPE) represents position information by rotating token embeddings in a rotation matrix defined by their position and dimension.

**Core concept:**
For a token at position `m` and embedding dimension `d`, RoPE applies a rotation in 2D subspaces of the embedding vector. Each pair of consecutive dimensions is rotated by an angle that depends on both the position and the frequency associated with those dimensions.

**Key mathematical components:**

1. **Frequency calculation**: Each dimension pair has an associated frequency that determines rotation speed.
2. **Rotation matrix**: 2D rotation matrices are applied to pairs of embedding dimensions.
3. **Position-dependent rotation**: Rotation angles increase with token position, encoding relative distances.

**Implementation approach:**
- Reshape embeddings to separate even/odd dimensions for rotation pairs.
- Compute rotation angles based on position and per-dimension frequencies.
- Apply rotation transformations efficiently using sine/cosine operations.
- Reshape back to original embedding dimensions.

<div class="hint" title="Tensor Reshaping Strategy">

**Tip**: RoPE works by rotating pairs of dimensions. You'll need to reshape your embeddings from `[..., d]` to `[..., d//2, 2]` to separate dimension pairs, apply rotations, then reshape back. Pay careful attention to tensor dimension ordering.

</div>

<div class="hint" title="Frequency Computation">

**Tip**: Each dimension pair has a frequency `theta = 1.0 / (10000 ** (2 * i / d))` where `i` is the pair index and `d` is the embedding dimension. Use these frequencies to compute rotation angles: `angle = position * frequency`.

</div>

<div class="hint" title="Rotation Implementation">

**Tip**: For each dimension pair `[x, y]`, the rotation is `[x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ)]` where θ is the rotation angle. Consider using `torch.stack` and broadcasting for efficient computation.

</div>

## Implementation guidance

**Resources for understanding RoPE:**
- Section 3 of [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Slide 22 of the [Transformers lecture](https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf#page=22)

**Testing your implementation:**
Use the provided unit test in `RoPE_test.py` for modular verification. The test checks your implementation against reference outputs to ensure mathematical correctness.

**Critical implementation note:**
You may reorganize functions within the class, but **never rename variables corresponding to LLaMA-2 parameters**. Renaming these variables will prevent loading pretrained weights, breaking the entire model.

**Common implementation pitfalls:**
- Incorrect tensor reshaping that misaligns dimension pairs.
- Wrong frequency computation leading to incorrect rotation angles.
- Broadcasting errors when applying rotations across batch dimensions.
- Forgetting to handle both query and key tensors consistently.

<div class="hint" title="Complete RoPE Implementation Guide">

**Complete mathematical formulation and implementation steps:**

The RoPE algorithm applies 2D rotations to pairs of dimensions in the embedding space. For a position `m` and embedding dimension `d`, the rotation is defined by:

<math>
<mrow>
<msubsup>
<mi>θ</mi>
<mi>i</mi>
<mrow><mo>(</mo><mi>m</mi><mo>)</mo></mrow>
</msubsup>
<mo>=</mo>
<mi>m</mi>
<mo>·</mo>
<msup>
<mn>10000</mn>
<mrow>
<mo>-</mo>
<mfrac>
<mrow><mn>2</mn><mi>i</mi></mrow>
<mi>d</mi>
</mfrac>
</mrow>
</msup>
</mrow>
</math>

where `i` is the dimension pair index (0, 1, 2, ..., d/2-1).

**Step-by-step implementation approach:**

```python
def apply_rotary_emb(query, key, head_dim, max_seq_len, theta=10000.0):
    _, seqlen, _, _ = query.shape
    device = query.device
    
    # Step 1: Reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # Step 2: Compute rotation frequencies using the formula from slide 22
    freqs = theta ** (-2 * torch.arange(0, head_dim / 2, device=device).float() / head_dim)
    m = torch.arange(seqlen, device=device)
    angles = torch.outer(m, freqs).float()
    cos, sin = torch.cos(angles), torch.sin(angles)
    cos, sin = reshape_for_broadcast(cos, query_real), reshape_for_broadcast(sin, query_real)

    # Step 3: Apply the rotation using complex number multiplication
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    query_out_1_real, query_out_1_imag = query_real * cos, query_imag * cos
    query_out_2_real, query_out_2_imag = query_real * sin, -query_imag * sin
    query_out_1 = query_out_1_real + query_out_2_imag  # Real part of result
    query_out_2 = query_out_1_imag + query_out_2_real  # Imaginary part of result

    # Same for key tensors
    key_out_1_real, key_out_1_imag = key_real * cos, key_imag * cos
    key_out_2_real, key_out_2_imag = key_real * sin, -key_imag * sin
    key_out_1 = key_out_1_real + key_out_2_imag
    key_out_2 = key_out_1_imag + key_out_2_real

    # Step 4: Merge back into single tensors
    target = list(query_out_1.shape)
    target[-1] = -1

    query_stacked = torch.stack((query_out_1, query_out_2), dim=-1)
    query_out = query_stacked.reshape(target)

    key_stacked = torch.stack((key_out_1, key_out_2), dim=-1)
    key_out = key_stacked.reshape(target)

    return query_out, key_out
```

**Key insight:** The implementation treats embedding dimensions as complex numbers (pairs of real/imaginary parts) and applies rotation by multiplying with `e^(i*θ) = cos(θ) + i*sin(θ)`.

</div>
