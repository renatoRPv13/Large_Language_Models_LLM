### Learning Objectives

By completing this task, you will be able to:
- **Understand n-gram language models** - Learn how these foundational models predict the next word based on previous n-1 words
- **Implement n-gram counting** - Build efficient data structures to count sequence patterns in text corpora
- **Calculate conditional probabilities** - Transform raw counts into probability distributions for language modeling
- **Handle text preprocessing** - Work with special tokens (UNK, BOS, EOS) essential for proper language model boundaries

### Problem Context

N-gram language models are among the most fundamental techniques in natural language processing. They work on a simple but powerful principle: predict the next word by looking at patterns in the previous n-1 words. Despite their simplicity, n-gram models remain surprisingly effective and serve as the foundation for understanding more complex language models.

**Why this problem matters:**
- N-grams are the building blocks of statistical language modeling, used in everything from autocomplete to machine translation
- They provide an intuitive introduction to the core concepts of language modeling: context, prediction, and probability
- Understanding n-grams is essential before moving to neural language models like transformers

**What makes this challenging:**
- Efficiently handling sparse data structures with nested dictionaries and counters
- Properly managing special tokens and text boundaries
- Converting raw counts to normalized probability distributions
- Dealing with variable-length prefixes and padding

### Understanding Special Tokens

N-gram language models use special tokens to handle text boundaries and unknown words properly:

**UNK (Unknown Token)**:
- Represents words not seen during training or used for padding
- In this implementation, also used to pad short prefixes to the required n-1 length
- Example: For a 3-gram model, the prefix "hi" becomes `(UNK, "hi")` when looking up probabilities

**BOS (Beginning of Sentence)**:
- Marks the start of each text sequence
- Allows the model to learn what words commonly begin sentences
- Added at the beginning of each line during preprocessing

**EOS (End of Sentence)**:
- Marks the end of each text sequence  
- Allows the model to learn sentence termination patterns
- Added at the end of each line during preprocessing

**Token Processing Example**:
```python
# Original text: "hello world"
# After adding special tokens: [BOS, "hello", "world", EOS]
# For a 3-gram model, this creates:
# - (UNK, UNK, BOS) -> "hello"
# - (UNK, BOS, "hello") -> "world" 
# - (BOS, "hello", "world") -> EOS
```

## Implementation Requirements

Your task is to implement the core functionality of an N-gram language model that can learn patterns from text and predict probable next words.

**Setup**: Extract the data by running `download_data.py` before starting implementation.

1. **Implement `count_ngrams` static method** - Create efficient n-gram counting from text corpus
2. **Implement `__init__` constructor** - Initialize the model and convert counts to probabilities  
3. **Implement `get_possible_next_tokens` method** - Predict next tokens given a prefix
4. **Handle special tokens properly** - Manage UNK, BOS, EOS tokens for text boundaries
5. **Follow existing code patterns** - Pay attention to docstrings, type annotations, and comments in TODO blocks

### Expected Deliverables:
- Complete `NGramLanguageModel` class with all three methods implemented
- Properly functioning n-gram counting that handles text preprocessing 
- Accurate probability calculations from raw counts
- Working next-token prediction for variable-length prefixes

### Method Specifications:

#### **`count_ngrams`** - Static Method
Build n-gram frequency counts from a text corpus.

**Implementation Steps**:
1. **Initialize data structure**: Create a `defaultdict(Counter)` to store n-gram counts
2. **Process each line**: 
   - Split the line into tokens using `line.split()`
   - Add special tokens: prepend `(n-1)` UNK tokens and BOS, append EOS
   - Example: `"hi world"` → `[UNK, UNK, BOS, "hi", "world", EOS]` for n=3
3. **Create sliding windows**: 
   - Use consecutive slices of length n+1 (n-gram + next token)
   - The first n tokens form the n-gram key, the last token is the target
4. **Count occurrences**: For each window, increment the count in your data structure

**Detailed Algorithm**:
```python
# For line "hi world" with n=3:
tokens = [UNK, UNK, BOS, "hi", "world", EOS]
# Windows: [(UNK,UNK,BOS,"hi"), (UNK,BOS,"hi","world"), (BOS,"hi","world",EOS)]
# Results: (UNK,UNK,BOS)->"hi", (UNK,BOS,"hi")->"world", (BOS,"hi","world")->EOS
```

**Returns**: `defaultdict(Counter)` where:
- Keys are n-gram tuples (e.g., `(UNK, BOS, 'hi')`)
- Values are `Counter` objects mapping next tokens to their frequencies

#### **`__init__`** - Constructor  
Initialize the n-gram language model and compute probability distributions.

**Implementation Steps**:
1. **Store parameters**: Save the n-gram size (`self.n = n`)
2. **Get raw counts**: Call `count_ngrams(lines, n)` to get frequency data
3. **Convert to probabilities**: For each n-gram prefix in the counts:
   - Calculate the total count: `total = sum(counter.values())`
   - Normalize each next-token count: `probability = count / total`
   - Store results in `self.probs` as a `defaultdict(Counter)`

**Normalization Example**:
```python
# Raw counts: (UNK, 'hi'): Counter({'there': 1, 'world': 2})
# Total count: 1 + 2 = 3
# Probabilities: (UNK, 'hi'): Counter({'there': 0.33, 'world': 0.67})
```

**Key Implementation Detail**: Each Counter in `self.probs` should contain probabilities (floats between 0 and 1) that sum to 1.0 for each n-gram prefix.

#### **`get_possible_next_tokens`** - Prediction Method
Predict next tokens and their probabilities for a given text prefix.

**Implementation Steps**:
1. **Tokenize input**: Split the prefix string into tokens using `prefix.split()`
2. **Create lookup key**: 
   - Take the last `(n-1)` tokens from the prefix
   - If fewer than `(n-1)` tokens, pad with UNK tokens at the beginning
   - Convert to tuple for dictionary lookup
3. **Query probabilities**: Look up the key in `self.probs`
4. **Return results**: Convert Counter to regular dict and return

**Detailed Algorithm**:
```python
# For n=3 model, prefix="hello world":
tokens = ["hello", "world"]  # After split
key = ("hello", "world")     # Last n-1=2 tokens, no padding needed

# For n=3 model, prefix="hi":  
tokens = ["hi"]              # After split
key = (UNK, "hi")           # Pad with UNK to get n-1=2 tokens

# For n=3 model, prefix="":
tokens = []                  # After split
key = (UNK, UNK)            # Pad with UNK tokens to get n-1=2 tokens
```

**Edge Cases to Handle**:
- Empty prefix string → pad completely with UNK tokens
- Prefix not seen during training → return empty dict `{}`
- Single token prefix → pad with UNK as needed

### Complete Implementation Example

Here's a walkthrough showing how all the pieces fit together:

```python
# Example: Processing "hi world" with n=3

# Step 1: count_ngrams preprocessing
line = "hi world"
tokens = line.split()  # ["hi", "world"]
# Add special tokens: [UNK, UNK, BOS, "hi", "world", EOS]
full_tokens = [UNK] * (n-1) + [BOS] + tokens + [EOS]

# Step 2: Create sliding windows of size n+1
# Window 1: [UNK, UNK, BOS, "hi"] → (UNK, UNK, BOS) -> "hi"
# Window 2: [UNK, BOS, "hi", "world"] → (UNK, BOS, "hi") -> "world" 
# Window 3: [BOS, "hi", "world", EOS] → (BOS, "hi", "world") -> EOS

# Step 3: Count and normalize (after processing multiple lines)
# Raw counts: {(UNK, "hi"): Counter({"there": 1, "world": 2})}
# Probabilities: {(UNK, "hi"): Counter({"there": 0.33, "world": 0.67})}

# Step 4: Prediction
model.get_possible_next_tokens("hi")
# Process: "hi" → ["hi"] → (UNK, "hi") → lookup in probs
# Returns: {"there": 0.33, "world": 0.67}
```

<div class="hint" title="Sliding Window Implementation">

**Tip**: For creating sliding windows, you can use list slicing in a loop: `for i in range(len(tokens) - n): window = tokens[i:i+n+1]`. Alternatively, use `zip` with offset iterators: `zip(tokens, tokens[1:], tokens[2:], ...)` for cleaner code.

</div>

<div class="hint" title="Probability Normalization">

**Tip**: To convert counts to probabilities, iterate through each n-gram prefix and its counter. Calculate `total = sum(counter.values())`, then create a new counter with `{token: count/total for token, count in counter.items()}`. This ensures all probabilities for each prefix sum to 1.0.

</div>

<div class="hint" title="Prefix Handling and Padding">

**Tip**: For `get_possible_next_tokens`, use this pattern: `tokens = prefix.split(); key_tokens = tokens[-(self.n-1):]; key = ([UNK] * (self.n-1-len(key_tokens)) + key_tokens)`. This handles both padding and extracting the right number of tokens in one go.

</div>

<div class="hint" title="Data Structure Access">

**Tip**: When accessing your probability dictionary, remember that `defaultdict(Counter)` returns an empty Counter for missing keys. Convert the result to a regular dict before returning: `dict(self.probs.get(key, {}))` or use `dict(self.probs[key])` if you want the defaultdict behavior.

</div>

### Common Pitfalls and Edge Cases

**Token Processing Issues**:
- **Empty lines**: Handle empty strings in the input gracefully - they should still get BOS and EOS tokens
- **Single word lines**: A line with one word should produce valid n-grams with proper padding
- **Whitespace**: Use `line.split()` which handles multiple spaces automatically

**Sliding Window Errors**:
- **Off-by-one errors**: Remember that for n-grams, your sliding window should be size `n+1` (n tokens + next token)
- **Range bounds**: When iterating, ensure `range(len(tokens) - n)` to avoid index errors
- **Empty sequences**: Handle cases where preprocessed token list has fewer than n+1 elements

**Probability Calculation**:
- **Division by zero**: This shouldn't happen with proper counting, but be aware that empty counters could cause issues
- **Precision**: Use proper float division, not integer division (`/` not `//`)
- **Dictionary types**: Ensure your final probabilities are in regular dicts, not Counters, for the prediction method

**Prefix Handling**:
- **Case sensitivity**: The model is case-sensitive - "Hi" and "hi" are different tokens
- **Padding logic**: For prefix shorter than n-1, pad at the beginning with UNK, not the end
- **Tuple creation**: Ensure your lookup keys are tuples, not lists, for proper dictionary access

**Memory and Performance**:
- **Large vocabularies**: With real datasets, your n-gram dictionary can become very large
- **Sparse data**: Most n-gram combinations won't appear in training data - this is normal and expected

