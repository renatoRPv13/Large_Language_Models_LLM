
### Learning objectives

By completing this task, you will be able to:
- **Understand character-level tokenization** - Learn why character-level processing is important for certain NLP tasks and how it differs from word-level tokenization.
- **Implement tensor preprocessing** - Build utilities to convert variable-length text sequences into fixed-size tensor matrices suitable for neural network processing.
- **Handle sequence padding and masking** - Apply proper padding strategies and create attention masks to handle variable-length sequences in batch processing.
- **Work with PyTorch tensors** - Gain experience manipulating tensor data structures and understanding their role in deep learning pipelines.

### Problem context

Text preprocessing is a critical but often overlooked component of NLP pipelines. Raw text comes in highly variable formats - different lengths, mixed cases, inconsistent spacing - but neural networks require structured, fixed-size numerical inputs. This task introduces you to **character-level tokenization**, a fundamental shift from word-based approaches.

**Why this problem matters:**
- **Handling out-of-vocabulary words** - Character-level models can process any text, including rare words, typos, and new vocabulary.
- **Cross-lingual applications** - Character-level processing works across languages without language-specific word segmentation.
- **Memory efficiency** - Smaller vocabulary size compared to word-level tokenization, especially for morphologically rich languages.

**What makes this challenging:**
- **Sequence length variation** - Text lines have different lengths but neural networks need fixed-size inputs.
- **Padding strategy** - Must pad sequences without interfering with model learning through proper masking.
- **Character encoding** - Need efficient mapping between characters and numerical token IDs.
- **Batch processing** - Handle multiple sequences simultaneously while preserving individual sequence boundaries.

**Important Transition:** Starting from this task we work with char-level tokenization:
- **Before**: "go back" → ["go", "back"] (word tokens)
- **After**: "go back" → ["g", "o", " ", "b", "a", "c", "k"] (character tokens)

## Implementation requirements

Build essential text preprocessing utilities that convert raw text into tensor format suitable for neural network training. You'll implement two core functions that handle the fundamental challenges of sequence processing: variable-length inputs and proper masking.

### Specific requirements:

1. **Implement `to_matrix` method** - Convert list of text strings to padded tensor matrix.
2. **Implement `compute_mask` method** - Generate boolean masks to identify valid vs. padded tokens.
3. **Handle character-level tokenization** - Process text at character granularity using provided TOKEN_TO_ID mapping.
4. **Apply proper padding strategy** - Use EOS tokens for padding and sequence termination.
5. **Support configurable parameters** - Allow customization of max length, padding tokens, and data types.

### Expected deliverables:

- Completed `to_matrix` method that correctly processes variable-length text inputs.
- Completed `compute_mask` method that generates accurate boolean masks.
- Proper handling of character-level tokenization and EOS token placement.
- Code that passes provided test cases and handles edge cases gracefully.
- Functions work correctly with batched input processing.

### Method specifications

#### **`to_matrix`**

This method converts a list of text lines into a tensor matrix suitable for model processing. The method works on the **character level** and pads sequences with the **EOS** token.

**Processing Steps:**
1. Converts all lines to lowercase for consistent processing
2. Sets a default maximum length (`max_len`) based on the longest line if not provided
3. Initializes a matrix filled with the padding value (`pad`), defaulting to the **EOS** token ID
4. Maps each character in the line to its corresponding ID using `TOKEN_TO_ID` and fills the matrix
5. Ensures each line ends with the **EOS** token if the maximum length is not reached

**Parameters:**
- `lines`: List of strings to be converted.
- `max_len`: Maximum length of each line in the matrix (optional, auto-computed if None).
- `pad`: Padding token ID (optional, defaults to EOS token ID).
- `dtype`: Data type of the output matrix (default: `np.int64`).

**Returns:**
- A PyTorch tensor matrix where each row corresponds to a tokenized and padded line

<div class="hint" title="Matrix Initialization Strategy">

**Tip**: Start by creating a matrix filled with padding values, then overwrite positions with actual token IDs. This ensures that any unfilled positions automatically contain the correct padding value. Remember to handle the case where sequences might be longer than max_len.

</div>

**Usage Example:**
```python
lines = ["hello", "go back"]
matrix = TextTools.to_matrix(lines, max_len=5)
print(matrix)
```

**Expected Output:**
```python
tensor([
 [12, 34, 56, 56, 78],  # Token IDs for "hell" with EOS (truncate "o")
 [90, 12, 45, 67, 78]   # Token IDs for "go b" with EOS (truncate "ack")
])
```

#### **`compute_mask`**

This method generates a boolean mask indicating valid tokens until the **EOS** token. This is crucial for attention mechanisms and loss computation, ensuring the model ignores padding tokens.

**Processing Steps:**
1. Identifies positions of the **EOS** token in the input tensor
2. Creates a cumulative sum across the sequence dimension, marking positions after the first **EOS** as invalid
3. Generates a boolean mask where valid tokens are marked as `True` and padding/post-EOS tokens as `False`

**Parameters:**
- `input_idx`: A tensor of token IDs (e.g., output from `to_matrix`).
- `eos_idx`: Token ID for **EOS** (optional, defaults to `TOKEN_TO_ID[EOS]`).

**Returns:**  
- A boolean PyTorch tensor where:
  - `True` represents valid tokens (content before and including EOS).
  - `False` represents padding or tokens after the first EOS.

<div class="hint" title="Cumulative Sum for Masking">

**Tip**: The key insight is using cumulative sum on EOS positions. When you find EOS tokens (`input_idx == eos_idx`), create a cumulative sum along the sequence dimension. Positions where cumsum <= 1 are valid (before or at first EOS), positions where cumsum > 1 are invalid (after first EOS).

</div>

**Usage Example:**
```python
input_idx = torch.tensor([
 [12, 34, 56, 78],  # Line with token IDs ending with EOS (78)
 [90, 12, 78, 78]   # Line with EOS in position 2, position 3 is padding
])
mask = TextTools.compute_mask(input_idx, eos_idx=78)
print(mask)
```

**Expected Output:**
```python
tensor([
 [ True,  True,  True,  True],   # All valid (EOS at end)
 [ True,  True,  True, False]    # Invalid after first EOS
])
```

<div class="hint" title="Edge Cases to Consider">

**Tip**: Consider edge cases like: sequences with no EOS token, sequences where EOS appears multiple times, and empty sequences. Your implementation should handle these gracefully without breaking the tensor operations.

</div>

After completing your implementation, run the provided tests to verify your text processing utilities work correctly with various input formats and edge cases.