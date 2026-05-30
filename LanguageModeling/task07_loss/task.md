
### Learning objectives

By completing this task, you will be able to:
- **Understand sequence loss computation** - Learn how to properly calculate loss for variable-length sequences while handling padding and special tokens.
- **Implement masked cross-entropy loss** - Apply masks to exclude irrelevant positions (padding, BOS tokens) from loss computation.
- **Handle tensor alignment** - Master the critical skill of aligning predictions with targets in sequence-to-sequence tasks.
- **Work with PyTorch loss functions** - Gain experience with `F.cross_entropy` and understand its input requirements and tensor shape expectations.

### Problem context

Training neural language models requires careful loss computation that goes far beyond simply applying cross-entropy to all positions. The fundamental challenge is that not all positions in your sequences should contribute equally (or at all) to the loss function.

**Why proper loss computation matters:**
- **Avoiding bias from padding** - Padding tokens should not influence model training, as they don't represent real language content.
- **Sequence alignment** - Predictions at position t should be compared against targets at position t+1 for language modeling.
- **Training stability** - Improper masking can lead to unstable training and poor model performance.
- **Computational efficiency** - Excluding irrelevant positions reduces computational overhead and improves training speed.

**What makes this challenging:**
- **Multi-dimensional tensor manipulation** - Need to properly reshape and align tensors of different dimensions for cross-entropy computation.
- **Mask application** - Must correctly apply boolean masks to filter out padding and special tokens.
- **Sequence boundaries** - Handle BOS (beginning-of-sequence) and EOS (end-of-sequence) tokens appropriately during loss calculation.
- **Shape compatibility** - Ensure logits and targets have compatible shapes for PyTorch's cross-entropy function.

## Implementation requirements

Build a custom cross-entropy loss function that properly handles masked sequences for language model training. This is a critical component that directly impacts model quality and training effectiveness.

### Specific requirements:

1. **Implement `forward` method** - Calculate masked cross-entropy loss for sequence data.
2. **Apply proper sequence masking** - Use masks from `compute_mask` to exclude padding and irrelevant tokens.
3. **Handle sequence alignment** - Correctly align predictions with targets for next-token prediction.
4. **Support batch processing** - Handle multiple sequences simultaneously with variable lengths.
5. **Integrate with PyTorch training** - Return scalar loss tensor compatible with PyTorch's automatic differentiation.

### Expected deliverables:

- Completed `CrossEntropyLoss.forward` method that correctly computes masked loss.
- Proper handling of BOS token exclusion and sequence alignment.
- Implementation that works with variable-length sequences and batch processing.
- Code that passes provided test cases and enables effective model training.
- Loss function that integrates seamlessly with PyTorch's optimization framework.

### Method specifications

#### **`forward`**

Calculate masked cross-entropy loss for sequence language modeling, ensuring that only valid positions contribute to the training signal.

**Critical Processing Steps:**

1. **Generate masks** - Use `compute_mask` to identify valid positions in sequences.
2. **Align sequences** - Exclude BOS from targets and last position from logits for proper prediction-target alignment.
3. **Apply masking** - Filter out padding tokens and irrelevant positions using boolean indexing.
4. **Compute loss** - Use `F.cross_entropy` on the filtered tensors to get the final loss value.

**Parameters:**
- `logits`: Tensor of shape `[batch_size, seq_len, n_tokens]` containing model predictions.
- `input_idx`: Tensor of shape `[batch_size, seq_len]` containing target token indices.

**Returns:**
- Scalar tensor representing the mean cross-entropy loss over all valid positions.

<div class="hint" title="Sequence Alignment Strategy">

**Tip**: For language modeling, you predict the next token at each position. This means `logits[:, :-1, :]` (all but last position) should predict `input_idx[:, 1:]` (all but first position). The BOS token at position 0 doesn't have a prediction, and the last logit position has no target to predict.

</div>

<div class="hint" title="Masking and Filtering">

**Tip**: After computing masks with `compute_mask`, remember to exclude the BOS token: `masks = masks[:, 1:]`. Then flatten both your filtered logits and targets before applying `F.cross_entropy`. Use boolean indexing like `logits_flat[mask_flat]` to select only valid positions.

</div>

<div class="hint" title="Tensor Shape Management">

**Tip**: `F.cross_entropy` expects logits of shape [N, C] and targets of shape [N] where N is the number of samples and C is the number of classes. You'll need to reshape: logits from [batch, seq-1, vocab] → [batch*(seq-1), vocab] and targets from [batch, seq-1] → [batch*(seq-1)].

</div>

**Usage Example:**
```python
# Model predictions and targets
logits = torch.randn(2, 5, 10)  # [batch_size=2, seq_len=5, n_tokens=10]
input_idx = torch.tensor([[1, 2, 3, 4, 0], [2, 3, 0, 0, 0]])  # Target sequences

# Initialize and compute loss
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits, input_idx)
print(f"Masked cross-entropy loss: {loss:.4f}")
```

**Expected Output:**
```python
Masked cross-entropy loss: 2.3054  # Scalar loss value
```

**Important Notes:**
- The loss should only include contributions from valid (non-padding) positions.
- BOS tokens at sequence start should be excluded from both predictions and targets.
- Use `F.cross_entropy` for the actual loss computation after proper tensor preparation.
- The returned loss should be a scalar tensor suitable for PyTorch's automatic differentiation.

After completing your implementation, run `run.py` to verify that your loss function enables effective training of the RNN language model.
