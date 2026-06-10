### Learning objectives

By completing this task, you will be able to:
- **Understand transfer learning approaches** - Learn the difference between zero-shot classification, fine-tuning, and feature extraction with pre-trained models.
- **Implement classification heads** - Build neural network components that transform language model representations into classification decisions.
- **Work with pre-trained embeddings** - Extract and utilize rich contextual representations from large language models for downstream tasks.
- **Apply dropout for regularization** - Understand how dropout prevents overfitting when adapting pre-trained models to new tasks.

### Problem context

Large language models like LLaMA learn rich representations of language during pre-training on massive text corpora. These representations can be leveraged for downstream tasks like classification in several ways, each with different trade-offs.

**Why LLaMA-based classification matters:**
- **Rich contextual understanding** - Pre-trained models capture complex linguistic patterns that simple classifiers cannot learn from small datasets.
- **Transfer learning efficiency** - Leverages billions of parameters of knowledge without training from scratch.
- **Multiple adaptation strategies** - Supports both parameter-efficient (frozen) and full fine-tuning approaches.
- **State-of-the-art performance** - Modern NLP applications rely on transformer-based classification for high accuracy.

**What makes this challenging:**
- **Representation extraction** - Must correctly identify and extract the most informative hidden states from the sequence.
- **Dimensionality mismatch** - LLaMA's hidden dimensions don't match classification output dimensions.
- **Overfitting prevention** - Pre-trained features can lead to overfitting on small downstream datasets without proper regularization.
- **Training strategy selection** - Choosing between freezing parameters vs. fine-tuning affects both performance and computational cost.

## Implementation requirements

Implement two different approaches for using LLaMA as a text classifier: zero-shot classification and embedding-based classification with optional fine-tuning.

### Specific requirements:

1. **Complete the `LlamaEmbeddingClassifier.forward()` method** - Extract final token representations and transform them into classification predictions.
2. **Handle sequence representation** - Use the last token's hidden state as the sequence representation for classification.
3. **Apply proper regularization** - Integrate dropout to prevent overfitting during training.
4. **Generate log-probabilities** - Return properly normalized log-probabilities over all classes.

### Expected deliverables:

- Completed `forward()` method in `LlamaEmbeddingClassifier` class.
- Implementation that correctly extracts and processes sequence-level representations.
- Proper integration of dropout for training-time regularization.
- Output that produces valid log-probability distributions over classes.

## Classification approaches

This task implements two distinct classification strategies:

### 1. Zero-Shot Classification (`LlamaZeroShotClassifier`)
Uses the language model's natural text generation capabilities to score completion probabilities for different label names. This approach requires no additional training but may be less accurate.

**Key concept:** Measures how likely the model is to generate each class label as a continuation of the input text.

### 2. Embedding-Based Classification (`LlamaEmbeddingClassifier`)
Extracts learned representations from LLaMA and passes them through a trainable classification head. Supports both frozen (feature extraction) and fine-tuning modes.

**Key concept:** Uses LLaMA's contextual understanding as features for a traditional classification layer.

## Mathematical foundation

The embedding-based classifier transforms sequence representations using:

**Sequence representation extraction:**
<math>
<mrow>
<mi>h</mi>
<mo>=</mo>
<msub><mi>H</mi><mrow><mi>T</mi><mo>-</mo><mn>1</mn></mrow></msub>
</mrow>
</math>

**Regularized representation:**
<math>
<mrow>
<mover><mi>h</mi><mo>~</mo></mover>
<mo>=</mo>
<mi>Dropout</mi><mo>(</mo><mi>h</mi><mo>)</mo>
</mrow>
</math>

**Classification logits:**
<math>
<mrow>
<mi>z</mi>
<mo>=</mo>
<mi>W</mi>
<mover><mi>h</mi><mo>~</mo></mover>
<mo>+</mo>
<mi>b</mi>
</mrow>
</math>

**Log-probability output:**
<math>
<mrow>
<mi>log</mi>
<mi>P</mi>
<mo>(</mo>
<mi>y</mi>
<mo>|</mo>
<mi>x</mi>
<mo>)</mo>
<mo>=</mo>
<mi>log</mi>
<mi>softmax</mi>
<mo>(</mo>
<mi>z</mi>
<mo>)</mo>
</mrow>
</math>

Where:
- `H_{T-1}` is the hidden state of the final token
- `W` and `b` are the classifier head parameters
- `Dropout()` applies regularization during training

<div class="hint" title="Sequence Representation">

**Tip**: For text classification, the hidden state of the last token (`hidden_states[:, -1, :]`) serves as the sequence-level representation. This captures the model's understanding of the entire input after processing all tokens.

</div>

<div class="hint" title="Implementation Steps">

**Step-by-step approach**:

1. **Extract LLaMA representations**: Call `self.llama(input_ids)` to get hidden states for all tokens
2. **Get final token representation**: Select the last token's hidden state as the sequence representation
3. **Apply dropout**: Use `self.dropout()` for training-time regularization
4. **Generate classification logits**: Pass through `self.classifier_head` to get raw class scores
5. **Return log-probabilities**: Apply log-softmax to convert logits to normalized log-probabilities

</div>

<div class="hint" title="Training Modes">

**Tip**: The classifier supports two training modes controlled by `config.option`:
- **'pretrain' mode**: LLaMA parameters are frozen (`requires_grad = False`), only the classification head is trained
- **'finetune' mode**: All parameters are trainable (`requires_grad = True`), allowing end-to-end optimization

Choose based on your dataset size and computational resources.

</div>

## Implementation guidance

**Critical implementation note:**
You may reorganize functions within each class, but **never rename variables corresponding to LLaMA-2 parameters**. Renaming these variables will prevent loading pretrained weights, breaking the entire model.

**Key implementation details:**
- Extract the correct hidden state dimensions from the LLaMA model
- Apply dropout only during training (handled automatically by PyTorch)
- Use `torch.nn.functional.log_softmax` for the final activation
- Ensure proper tensor dimensions throughout the forward pass

**Testing considerations:**
The classifier will be evaluated on text classification benchmarks, so accuracy depends on correctly implementing the representation extraction and classification head integration.
