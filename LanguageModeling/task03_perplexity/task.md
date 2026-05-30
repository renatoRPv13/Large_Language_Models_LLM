
### Learning Objectives

By completing this task, you will be able to:
- **Understand perplexity as a model evaluation metric** - Learn how perplexity measures a language model's uncertainty and predictive performance
- **Implement perplexity calculation** - Apply the mathematical formula to evaluate language model quality using log probabilities
- **Handle numerical stability** - Use minimum probability thresholds to prevent computational underflow in log calculations
- **Connect evaluation to model quality** - Interpret perplexity scores to assess and compare language model performance

### Problem Context

Perplexity is one of the most important metrics for evaluating language models. It measures how "surprised" or "perplexed" a model is when predicting the next word in a sequence. Lower perplexity indicates better predictive performance.

**Why perplexity matters:**
- Provides a standardized way to compare different language models
- Directly relates to the model's predictive accuracy and confidence
- Widely used in research and industry for model evaluation
- Foundation for understanding more advanced evaluation techniques

**What makes this challenging:**
- Requires careful handling of log probabilities to avoid numerical underflow
- Must properly account for sequence boundaries and special tokens
- Involves understanding the relationship between probability and information theory

## Implementation Requirements

Implement the `perplexity` method in the `Evaluator` class to calculate perplexity for language model evaluation.

### Specific Requirements:
1. Iterate through each line in the corpus and tokenize by splitting on spaces
2. Append an EOS (End of Sentence) token to each tokenized sequence
3. For each token, compute log probability using `model.get_next_token_prob`
4. Apply minimum log probability threshold to prevent numerical issues
5. Calculate final perplexity using the exponential of negative average log probability

### Expected Deliverables:
- Complete `perplexity` method in the `Evaluator` class
- Handle edge cases with appropriate minimum probability thresholds
- Return accurate perplexity values that can be used to compare model performance

The perplexity is calculated using this formula:

<math>
  <mtext>perplexity</mtext>
  <mo>=</mo>
  <mi>exp</mi>
  <mo>(</mo>
  <mo>-</mo>
  <mfrac>
    <mtext>total log probability</mtext>
    <mtext>total number of tokens</mtext>
  </mfrac>
  <mo>)</mo>
</math>

where:

<math>
  <mtext>total log probability</mtext>
  <mo>=</mo>
  <munderover>
    <mo>∑</mo>
    <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow>
    <mi>N</mi>
  </munderover>
  <mi>log</mi>
  <mi>p</mi>
  <mo>(</mo>
  <msub>
    <mi>w</mi>
    <mi>i</mi>
  </msub>
  <mo>∣</mo>
  <msub>
    <mi>w</mi>
    <mrow><mi>i</mi><mo>-</mo><mn>1</mn></mrow>
  </msub>
  <mo>,</mo>
  <mo>…</mo>
  <mo>,</mo>
  <msub>
    <mi>w</mi>
    <mrow><mi>i</mi><mo>-</mo><mi>n</mi><mo>+</mo><mn>1</mn></mrow>
  </msub>
  <mo>)</mo>
</math>

- **N** = total number of tokens
- **p(wᵢ | wᵢ₋₁, …, wᵢ₋ₙ₊₁)** = probability from `model.get_next_token_prob`

### Parameters:
- `model`: The language model instance (`BaseLanguageModel`) to evaluate
- `lines`: List of strings, each representing a line in the corpus with space-separated tokens  
- `min_logprob`: Minimum log probability threshold to avoid numerical underflow (default: log(10⁻⁵⁰))

<div class="hint" title="Numerical Stability">

**Tip**: When working with log probabilities, very small probabilities can cause numerical underflow. Use the `min_logprob` parameter to clamp extremely small values. Also remember that `math.log()` of very small numbers can be problematic, so apply the threshold before taking logarithms.

</div>

<div class="hint" title="Token Processing">

**Tip**: For each line, split into tokens and append the EOS token. Then iterate through each token in the sequence, using the preceding tokens as context to get the next token probability from the model.

</div>

*Usage example:*
```python
from LanguageModeling.task01_ngrams.task import NGramLanguageModel

lines = ['hi world', 'hi there']
model = NGramLanguageModel(lines, n=3)

evaluator = Evaluator()
perplexity_value = evaluator.perplexity(model, lines)
perplexity_value
```

*Output (example):*
```python
12.34
```

After completing your implementation, run `run.py` to test your perplexity calculation and verify the results.