
### Learning objectives

By completing this task, you will be able to:
- **Understand recurrent neural networks** - Learn how RNNs process sequential data and maintain memory across time steps for language modeling.
- **Implement neural language models** - Build your first neural network-based language model using PyTorch's RNN components.
- **Work with embeddings** - Understand how token embeddings convert discrete symbols into continuous vector representations.
- **Apply sequence-to-sequence modeling** - Learn how RNNs predict the next token in a sequence by processing previous tokens sequentially.

### Problem context

Traditional n-gram models, while effective for basic language modeling, suffer from fundamental limitations that become apparent in real-world applications. They struggle with **long-range dependencies**, require extensive smoothing for unseen sequences, and cannot capture complex linguistic patterns effectively.

**Why recurrent neural networks matter:**
- **Long-range memory** - RNNs can theoretically remember information from much earlier in the sequence, enabling better context understanding.
- **Continuous representations** - Instead of discrete token counts, RNNs work with dense vector embeddings that capture semantic relationships.
- **Parameter sharing** - The same weights are used across all time steps, making the model more efficient and generalizable.
- **Scalability** - Neural models can handle much larger vocabularies and longer sequences than traditional approaches.

**What makes this challenging:**
- **Sequential processing** - Unlike n-grams which look at fixed windows, RNNs must process sequences step by step.
- **Gradient flow** - Training RNNs requires careful handling of gradients flowing through time to avoid vanishing or exploding gradients.
- **Architecture design** - Choosing appropriate embedding sizes, hidden dimensions, and RNN variants (vanilla RNN, LSTM, GRU) significantly impacts performance.
- **Memory vs. computation trade-offs** - Larger hidden states capture more information but require more computational resources.

## Implementation requirements

Build a recurrent neural network for character-level language modeling that can process variable-length sequences and predict probability distributions over the next character. This represents a significant step up from n-gram models, introducing neural network concepts and sequential processing.

### Specific requirements:

1. **Implement `__init__` method** - Initialize RNN architecture with embedding layer, recurrent layer, and output projection.
2. **Implement `__call__` method** - Process input sequences through the full model pipeline to produce token logits.
3. **Implement `get_possible_next_tokens` method** - Generate probability distributions for next token prediction.
4. **Use character-level tokenization** - Continue working with character-level processing from previous tasks.
5. **Support batch processing** - Handle multiple sequences simultaneously using batch-first tensor operations.

### Expected deliverables:

- Completed `RNNLanguageModel` class with all three required methods.
- Proper integration with existing `TextTools` utilities for preprocessing.
- Model that can generate coherent probability distributions for next character prediction.
- Code that handles variable-length sequences and batch processing correctly.
- Implementation that passes provided test cases and demonstrates improved performance over n-gram models.

### Method specifications

#### **`__init__`**

Initialize the RNN language model with the core neural network components needed for character-level sequence modeling.

**Required Architecture Components:**
- **Token Embeddings** - Converts discrete token indices into dense vectors of size `emb_size`.
- **RNN Layer** - Processes sequences with hidden layer size of `hid_size`, operating in batch-first mode (`[batch_size, sequence_length, features]`).
  - You can choose from: `nn.RNN`, `nn.LSTM`, or `nn.GRU`.
- **Output Linear Layer** - Maps RNN hidden states to token logits for vocabulary prediction.

**Parameters:**
- `tokens`: List of all unique tokens in the vocabulary.
- `emb_size`: Size of the token embedding vectors (default = 16).
- `hid_size`: Size of the RNN hidden layer (default = 256).

<div class="hint" title="Architecture Design Choice">

**Tip**: For character-level language modeling, GRU often provides a good balance between performance and computational efficiency. Start with `nn.GRU(input_size=emb_size, hidden_size=hid_size, batch_first=True)` and ensure the linear layer maps from `hid_size` to `len(tokens)`.

</div>

#### **`__call__`**

Standard PyTorch forward pass method that processes input sequences through the complete model pipeline.

**Processing Pipeline:**
1. **Embedding lookup** - Maps token indices in `input_ix` to dense embeddings using the embedding layer.
2. **Sequential processing** - Passes embeddings through the RNN layer to capture temporal dependencies.
3. **Output projection** - Transforms RNN outputs into token logits using the linear layer.

**Parameters:**
- `input_ix`: Tensor of shape `[batch_size, sequence_length]` containing token indices.

**Returns:**
- Tensor of shape `[batch_size, sequence_length, n_tokens]` containing logits for each position and token.

<div class="hint" title="Tensor Shapes Through Pipeline">

**Tip**: Track tensor shapes carefully: `input_ix` [B, T] → `embeddings` [B, T, E] → `rnn_out` [B, T, H] → `logits` [B, T, V] where B=batch, T=time, E=embedding, H=hidden, V=vocabulary size. The RNN returns both output and hidden states - you only need the output for the linear layer.

</div>

**Usage Example:**
```python
input_ix = torch.tensor([[0, 1, 2], [1, 2, 3]])  # Batch of token indices
logits = model(input_ix)
print(logits.shape)  # Expected: torch.Size([2, 3, vocab_size])
```

#### **`get_possible_next_tokens`**

Generate probability distribution over all tokens for next character prediction given a text prefix.

**Processing Steps:**
1. **Preprocessing** - Convert the `prefix` string into token indices using `TextTools.to_matrix`.
2. **Model inference** - Feed the indices through the model to compute logits.
3. **Probability conversion** - Apply softmax to the final position's logits to get probability distribution.

**Parameters:**
- `prefix`: String containing the input character sequence.

**Returns:**
- Dictionary mapping each token to its predicted probability.

<div class="hint" title="Handling the Last Position">

**Tip**: You only need the logits from the last time step for next token prediction. If your model outputs shape [1, seq_len, vocab_size], extract logits[-1, -1, :] or logits[0, -1, :] for the final position, then apply softmax to get probabilities.

</div>

<div class="hint" title="Converting Back to Tokens">

**Tip**: Remember that your model works with token indices, but the output should map actual characters to probabilities. Use the `TOKEN_TO_ID` and `ID_TO_TOKEN` mappings from TextTools to convert between indices and actual characters.

</div>

**Usage Example:**
```python
prefix = "hello"
probs = model.get_possible_next_tokens(prefix)
print(probs)  # Expected: {' ': 0.4, '!': 0.3, EOS: 0.2, ...}
```

After completing your implementation, test your RNN model's ability to generate coherent character-level predictions and compare its performance with the previous n-gram approaches.
