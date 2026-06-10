### Learning objectives

By the end of this task you will be able to:
- **Navigate complex codebases** - Understand how large-scale machine learning projects are organized and how different components interact.
- **Identify implementation boundaries** - Distinguish between provided infrastructure code and components you need to implement yourself.
- **Understand transformer architecture mapping** - See how theoretical concepts (attention, normalization, embeddings) translate into concrete code structures.
- **Apply modular design principles** - Learn how complex models like LLaMA are broken down into manageable, testable components.

### Problem context

Modern transformer architectures like LLaMA involve sophisticated interactions between multiple components: attention mechanisms, normalization layers, positional encodings, and optimization algorithms. Understanding how these components are organized in code is crucial for successful implementation and debugging.

**Why understanding codebase structure matters:**
- **Implementation efficiency** - Knowing which files to modify and which to leave unchanged saves significant development time.
- **Debugging capability** - Understanding component interactions helps isolate issues when implementations don't work as expected.
- **Architecture comprehension** - Seeing how theoretical concepts map to code structures deepens your understanding of transformer design.
- **Professional development** - Large ML projects require navigating complex codebases, making this a valuable practical skill.

**What makes this challenging:**
- **Component interdependencies** - Changes in one component can affect others in non-obvious ways.
- **Scale complexity** - Transformer implementations involve many interacting pieces that must work together precisely.
- **Abstraction levels** - Code operates at multiple levels from mathematical operations to high-level model interfaces.
- **Parameter sensitivity** - Variable naming and initialization patterns are crucial for loading pretrained weights correctly.

## Implementation requirements

Your task is to implement specific components of the LLaMA-2 architecture while working with provided infrastructure code. Understanding the codebase structure is essential for successful implementation.

### Specific requirements:

1. **Study the provided codebase** - Understand the role of each file and how components interact.
2. **Identify implementation targets** - Know exactly which functions you need to implement and which are provided.
3. **Follow architectural patterns** - Implement your components to integrate seamlessly with the existing codebase.
4. **Maintain parameter compatibility** - Ensure your implementations can load pretrained weights correctly.
5. **Use provided testing infrastructure** - Leverage sanity checks and unit tests to verify your implementations.

### Expected deliverables:

- Clear understanding of which files you modify vs. which provide infrastructure.
- Knowledge of how transformer components map to code organization.
- Ability to implement required components following established patterns.
- Successful integration with provided testing and evaluation infrastructure.

## Codebase structure overview

The directory contains the following files: `base_llama.py`, `config.py`, `tokenizer.py`, `utils.py`, and `run_llama.py`.

### Infrastructure files (do not modify)

**`base_llama.py`**
- Contains the base class for the LLaMA model providing common functionality.
- Handles model loading, weight management, and high-level interfaces.
- **Your task**: Study this file to understand the model's overall structure.

**`tokenizer.py`** 
- Implements text tokenization for the LLaMA model using SentencePiece tokenizer.
- Handles conversion between text and token IDs.
- **Your task**: Understand how text gets converted to model inputs.

**`config.py`**
- Defines configuration parameters for different LLaMA model sizes.
- Specifies architecture details like layer counts, hidden dimensions, attention heads.
- **Your task**: Review configuration options to understand model architecture variants.

**`utils.py`**
- Contains utility functions for model operations, data loading, and evaluation.
- Provides helper functions used throughout the codebase.
- **Your task**: Familiarize yourself with available utility functions.

**`run_llama.py`**
- Main execution script for running LLaMA model operations.
- Orchestrates training, evaluation, and generation workflows.
- **Your task**: Use this as your primary interface for testing implementations.

### Components to implement

**`RoPE/task.py` (rotary positional embeddings)**
- **Purpose**: Implement Rotary Positional Embeddings, a key innovation in LLaMA architecture.
- **Resources**: 
  - Slide 22 of the [Transformers lecture](https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf#page=22)
  - Section 3 of [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Testing**: Unit test provided in `RoPE_test.py` for modular verification.

**`Llama/task.py` (core transformer implementation)**
- **Purpose**: Implement the core LLaMA-2 transformer architecture components.
- **Foundation**: Based on the [transformer](https://arxiv.org/pdf/1706.03762.pdf#page=3) architecture (Section 3 recommended reading).

#### Attention mechanism implementation

**Multi-Head Attention Overview:**
The attention mechanism maps queries and key-value pairs to outputs through weighted value combinations, where weights derive from query-key interactions.

**Mathematical foundation:**

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

**LLaMA-2 Innovation - Grouped-Query Attention:**
- Uses [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) for improved efficiency.
- Groups of query heads share the same key and value parameters.
- Reduces memory requirements while maintaining performance.

**Your implementation task:**
The codebase handles linear projections (step 1), vector splitting (step 2), and output concatenation (step 4). You need to implement the core attention computation using the equation above.

#### LlamaLayer (transformer layer implementation)

**Architecture Overview:**
A single transformer layer implementing the standard transformer pattern with LLaMA-specific modifications.

**Component sequence:**
1. **Input normalization** - Root Mean Square Layer Normalization (RMSNorm) of input.
2. **Self-attention** - Apply multi-head attention to normalized input.
3. **Residual connection** - Add original input to self-attention output.
4. **Output normalization** - RMSNorm on the self-attention output.
5. **Feed-forward network** - Apply position-wise FFN to normalized attention output.
6. **Final residual** - Add unnormalized attention output to FFN output.

#### Llama (full model implementation)

**Model Architecture:**
The complete LLaMA model processes input token IDs and produces next-token predictions plus contextualized representations.

**Component pipeline:**
1. **Token embeddings** - Convert input IDs to dense vectors (`tok_embeddings`).
2. **Transformer stack** - Process through `config.num_hidden_layers` LlamaLayer instances.
3. **Output projection** - Project hidden states to vocabulary logits for next-token prediction.
4. **Generation capability** - Temperature sampling for text continuation (no nucleus/top-k sampling).

**Expected outputs:**
- `logits`: Vocabulary scores predicting next tokens at each position.
- `hidden_states`: Final contextualized representations for each input token.

## Implementation checklist

Components requiring your implementation are marked with `#todo`. Each includes detailed instructions in the corresponding code blocks.

### Core transformer components:
- `RoPE.task.apply_rotary_emb` - Rotary positional embeddings implementation.
- `Llama.task.RMSNorm._norm` - Root Mean Square normalization.
- `Llama.task.Attention.compute_query_key_value_scores` - Attention score computation.
- `Llama.task.LlamaLayer.forward` - Single transformer layer forward pass.

### Training and classification components:
- `Optimizer.task.AdamW.step` - AdamW optimizer step function.
- `Classifier.task.LlamaEmbeddingClassifier.forward` - Classification head implementation.

**Critical implementation note:** You may reorganize functions within classes, but **never rename variables corresponding to LLaMA-2 parameters**. Renaming these variables will prevent loading pretrained weights, breaking the entire system.

### Testing and validation

**Sanity check integration test:**
Use `sanity_check.py` to verify your LLaMA implementation. This test loads reference embeddings and validates your implementation outputs against the expected results.

**`Optimizer/task.py` (AdamW implementation)**
- **Purpose**: Implement the AdamW optimizer for model training.
- **References**: 
  - [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

**Implementation specifications:**
- Uses "efficient" bias correction from Kingma & Ba (2014) section 2.
- Learning rate incorporated into weight decay update.
- No learning rate scheduling required.

**`Classifier/task.py` (fine-tuning pipeline)**
- **Purpose**: Implement classification pipeline using pretrained LLaMA model.

**Pipeline components:**
- Pretrained model loading and verification.
- Example sentence generation for implementation validation.
- Sentence encoding using LLaMA for contextualized representations.
- Classification head implementation for downstream tasks.
- Fine-tuning procedures for task-specific adaptation.

#### LlamaEmbeddingClassifier implementation

**Core functionality:**
- Extract sentence representations using LLaMA's final token hidden states.
- Apply dropout regularization to pooled outputs.
- Project through linear layer for final classification scores.

### References

[Vaswani et al. 2017] Attention is All You Need https://arxiv.org/abs/1706.03762

[Touvron et al. 2023] Llama 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/abs/2307.09288
