
### Learning Objectives

By completing this task, you will be able to:
- **Understand smoothing techniques** - Learn why raw probability estimates from limited data can be unreliable and how Laplace smoothing addresses this fundamental problem
- **Implement probability smoothing** - Apply Laplace (add-one) smoothing to redistribute probability mass from observed to unobserved events
- **Handle unseen events** - Ensure your language model assigns non-zero probabilities to previously unseen token sequences, preventing division by zero and improving generalization

### Problem Context

In natural language processing, one of the biggest challenges with n-gram language models is the **sparse data problem**. Even with large training corpora, many valid word sequences will never appear in your training data. Without smoothing, your model would assign zero probability to these unseen sequences, leading to several critical issues:

**Why this problem matters:**
- **Real-world robustness** - Production language models must handle previously unseen text gracefully
- **Mathematical stability** - Zero probabilities cause numerical instability in downstream applications
- **Better generalization** - Smoothing helps models make reasonable predictions about rare or unseen events

**What makes this challenging:**
- **Probability mass redistribution** - You need to "steal" probability from observed events to give to unobserved ones
- **Vocabulary handling** - The entire vocabulary must be considered, not just observed tokens
- **Parameter tuning** - The smoothing parameter δ controls the trade-off between trusting observed data vs. assuming uniform distribution

## Implementation Requirements

Implement Laplace smoothing for n-gram language models. This technique adds a small constant (δ) to all token counts, ensuring that even unseen token sequences receive non-zero probability.

### Specific Requirements:

1. **Initialize the `__init__` method** in the `LaplaceLanguageModel` class
2. **Apply Laplace smoothing** to all n-gram probability calculations using the provided δ parameter
3. **Handle the complete vocabulary** - ensure all tokens in `self.vocab` are considered for each prefix
4. **Store smoothed probabilities** in `self.probs` dictionary structure

### Expected Deliverables:

- Completed `__init__` method that correctly implements Laplace smoothing
- Proper handling of the vocabulary size in denominator calculations
- All tokens in vocabulary receive non-zero probabilities for each prefix
- Code passes the provided test cases and achieves expected probability distributions

### Mathematical Foundation

**Before Laplace Smoothing:**
The probability of a token w given a prefix is calculated as:

<math>
  <mi>p</mi>
  <mo>(</mo>
  <mi>w</mi>
  <mo>|</mo>
  <mtext>prefix</mtext>
  <mo>)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mtext>count</mtext>
      <mo>[</mo>
      <mtext>prefix</mtext>
      <mo>]</mo>
      <mo>[</mo>
      <mi>w</mi>
      <mo>]</mo>
    </mrow>
    <mrow>
      <munder>
        <mo>∑</mo>
        <msup>
          <mi>w</mi>
          <mo>′</mo>
        </msup>
      </munder>
      <mtext>count</mtext>
      <mo>[</mo>
      <mtext>prefix</mtext>
      <mo>]</mo>
      <mo>[</mo>
      <msup>
        <mi>w</mi>
        <mo>′</mo>
      </msup>
      <mo>]</mo>
    </mrow>
  </mfrac>
</math>

Where:
- count[prefix][w] is the number of occurrences of w following the given prefix
- The denominator sums over all tokens that follow the prefix in training data

**After Laplace Smoothing:**
With Laplace smoothing, the formula becomes:

<math>
  <mi>p</mi>
  <mo>(</mo>
  <mi>w</mi>
  <mo>|</mo>
  <mtext>prefix</mtext>
  <mo>)</mo>
  <mo>=</mo>
  <mfrac>
    <mrow>
      <mtext>count</mtext>
      <mo>[</mo>
      <mtext>prefix</mtext>
      <mo>]</mo>
      <mo>[</mo>
      <mi>w</mi>
      <mo>]</mo>
      <mo>+</mo>
      <mi>δ</mi>
    </mrow>
    <mrow>
      <munder>
        <mo>∑</mo>
        <msup>
          <mi>w</mi>
          <mo>′</mo>
        </msup>
      </munder>
      <mo>(</mo>
      <mtext>count</mtext>
      <mo>[</mo>
      <mtext>prefix</mtext>
      <mo>]</mo>
      <mo>[</mo>
      <msup>
        <mi>w</mi>
        <mo>′</mo>
      </msup>
      <mo>]</mo>
      <mo>+</mo>
      <mi>δ</mi>
      <mo>)</mo>
    </mrow>
  </mfrac>
</math>

Where:
- δ is the smoothing parameter (typically δ = 1.0 for add-one smoothing)
- The denominator now includes δ for EVERY token in the vocabulary, not just observed ones
- This can be simplified to: (`count[prefix][w] + `δ`) / (total_count + `δ` `×` |vocabulary|`)

### Parameters:
- `lines`: List of text lines for training the model
- `n`: Size of n-grams (context length)
- `delta`: Smoothing parameter controlling probability redistribution

<div class="hint" title="Vocabulary Size Matters">

**Tip**: The key insight is that the denominator must account for the ENTIRE vocabulary, not just tokens seen after each prefix. When calculating `total_count + delta * len(self.vocab)`, remember that `len(self.vocab)` includes all possible tokens, ensuring proper probability distribution.

</div>

<div class="hint" title="Implementation Strategy">

**Tip**: Start by understanding what `counts` contains - it maps each prefix to a Counter of following tokens. Then, for each prefix, you need to create probabilities for ALL tokens in the vocabulary, even those with zero count for that specific prefix.

</div>

### Usage Example:

```python
lines = ['a b c', 'a b a']
model = LaplaceLanguageModel(lines=lines, n=1, delta=1.0)
print(model.probs)
```

Expected output structure:
```python
defaultdict(collections.Counter,
            {(): {'a': 0.3333333333333333,
              'b': 0.25,
              'c': 0.16666666666666666,
              EOS: 0.25}})
```

After completing your implementation, run `run.py` to verify your model's performance and compare against expected results.