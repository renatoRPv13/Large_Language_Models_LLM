### Learning Objectives

By completing this task, you will be able to:
- **Understand temperature sampling** - Control the randomness in text generation by adjusting probability distributions
- **Implement nucleus sampling** - Use cumulative probability thresholds to balance quality and diversity in generated text
- **Build a text generator** - Create a complete sequence generation system that can produce coherent text continuations
- **Apply sampling strategies** - Choose appropriate generation techniques based on desired output characteristics

### Problem Context

Text generation is a fundamental capability of language models, allowing them to produce human-like text continuations from a given prompt. However, simply selecting the most probable next token at each step (greedy decoding) often leads to repetitive, boring, or unrealistic text.

**Why this problem matters:**
- **Creative applications**: Enables chatbots, story generators, and creative writing assistants
- **Controllable generation**: Different sampling strategies produce different styles of output
- **Foundation for advanced NLP**: Understanding generation is crucial for working with modern language models

**What makes this challenging:**
- **Balancing quality and diversity**: Too deterministic = boring, too random = nonsensical
- **Probability manipulation**: Requires careful mathematical transformations of probability distributions
- **Sequence-level coherence**: Each token choice affects the entire generated sequence

### Implementation Requirements

Implement three interconnected methods that work together to generate text sequences using different sampling strategies.

### Specific Requirements:

1. **`get_next_token_sample`** - Implement temperature-based sampling
   - Handle temperature = 0 case (return most probable token)
   - Apply temperature scaling to probability distribution
   - Sample from the adjusted distribution

2. **`get_next_token_nucleus`** - Implement nucleus (top-p) sampling
   - Sort tokens by probability in descending order
   - Find nucleus threshold where cumulative probability ≤ p
   - Ensure at least one token is always included
   - Sample from the filtered and normalized distribution

3. **`generate_sequence`** - Build complete sequence generator
   - Support both 'sample' and 'nucleus' modes
   - Continue until EOS token or max_len reached
   - Return complete sequence including EOS token

### Expected Deliverables:
- All three methods fully implemented in the `Generator` class
- Proper handling of edge cases (temperature=0, nucleus=1.0, empty contexts)
- Generated sequences should be coherent and respect the specified constraints

### Mathematical Foundation

**Temperature Sampling Formula:**

<math>
  <msub>
    <mover>
      <mi>p</mi>
      <mo>^</mo>
    </mover>
    <mi>i</mi>
  </msub>
  <mo>=</mo>
  <mfrac>
    <msubsup>
      <mi>p</mi>
      <mi>i</mi>
      <mfrac>
        <mn>1</mn>
        <mi>T</mi>
      </mfrac>
    </msubsup>
    <mrow>
      <munder>
        <mo>∑</mo>
        <mi>j</mi>
      </munder>
      <msubsup>
        <mi>p</mi>
        <mi>j</mi>
        <mfrac>
          <mn>1</mn>
          <mi>T</mi>
        </mfrac>
      </msubsup>
    </mrow>
  </mfrac>
</math>
where:
- $p_i$ is the original probability of token $i$
- $\hat{p}_i$ is the temperature-adjusted probability
- $T$ is the temperature parameter

### Usage Examples

**Temperature Sampling:**
```python
model = NGramLanguageModel(lines=['hi world', 'hi there'], n=3)
generator = Generator(model)
token = generator.get_next_token_sample('hi', temperature=0.5)
# Output: 'world' (with some randomness)
```

**Nucleus Sampling:**
```python
token = generator.get_next_token_nucleus('hi', nucleus=0.9)
# Output: 'there' (from top 90% probability mass)
```

**Sequence Generation:**
```python
sequence = generator.generate_sequence('hi', mode='sample', temperature=0.5, max_len=10)
# Output: ['hi', 'world', EOS]
```

<div class="hint" title="Temperature Implementation">

**Tip**: For temperature sampling, remember that when temperature approaches 0, you should return the most probable token directly. For the general case, use `np.power()` to apply the temperature transformation and `np.random.choice()` with probability weights for sampling.

</div>

<div class="hint" title="Nucleus Sampling Logic">

**Tip**: Start by sorting tokens in descending probability order, then use `np.cumsum()` to compute cumulative probabilities. The nucleus includes all tokens up to where the cumulative sum first exceeds the threshold, but always include at least the most probable token.

</div>

<div class="hint" title="Sequence Generation Flow">

**Tip**: Use a while loop that continues until you hit EOS or max_len. For each iteration, get the next token using the specified mode, add it to your sequence, and update the context for the next prediction. Don't forget to include the EOS token in your final output.

</div>