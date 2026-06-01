### Learning objectives

By completing this task, you will be able to:
- **Master parameter-efficient fine-tuning techniques** - Understand and implement LoRA, IA³, and Prompt Tuning for large language model adaptation.
- **Configure PEFT methods effectively** - Design appropriate configurations for different adaptation techniques based on model architecture and task requirements.
- **Compare adaptation strategies empirically** - Evaluate the trade-offs between parameter efficiency, performance, and computational cost across different PEFT methods.
- **Build production-ready fine-tuning pipelines** - Create complete training workflows that can be deployed for real-world model adaptation tasks.

### Problem context

Traditional fine-tuning requires updating all parameters of large language models, which is computationally expensive and often unnecessary. Parameter-Efficient Fine-Tuning (PEFT) methods achieve comparable performance while training only a small fraction of parameters, making model adaptation accessible and cost-effective.

**Why PEFT techniques matter:**
- **Resource efficiency** - Reduce memory usage and training time by orders of magnitude compared to full fine-tuning.
- **Performance preservation** - Achieve similar or better results than full fine-tuning with minimal parameter updates.
- **Modularity** - PEFT adapters can be saved, shared, and combined independently of the base model.
- **Practical deployment** - Enable fine-tuning on consumer hardware and in resource-constrained environments.

**What makes this challenging:**
- **Method selection complexity** - Different PEFT techniques have varying strengths for different tasks and model architectures.
- **Configuration sensitivity** - PEFT methods require careful hyperparameter tuning for optimal performance.
- **Architecture compatibility** - Target module selection must align with the specific transformer architecture being adapted.
- **Performance evaluation** - Comparing methods requires understanding both quantitative metrics and qualitative generation quality.

## Implementation requirements

Implement and compare three state-of-the-art parameter-efficient fine-tuning methods: LoRA, IA³, and Prompt Tuning. Each method represents a different approach to efficient model adaptation.

### Specific requirements:

1. **Configure LoRA (Low-Rank Adaptation)** - Implement rank-based matrix decomposition for attention layer adaptation.
2. **Set up IA³ (Infused Adapter by Inhibiting and Amplifying)** - Configure learned scaling vectors for selective activation modification.
3. **Implement Prompt Tuning** - Design learnable continuous prompts that guide model behavior.
4. **Execute comparative evaluation** - Train all methods and analyze their relative performance and efficiency.

### Expected deliverables:

- Completed configuration files for all three PEFT methods (`configs/lora.py`, `configs/ia3.py`, `configs/prompt_tuning.py`).
- Successfully trained models for each PEFT technique.
- Comparative analysis understanding the trade-offs between methods.
- Knowledge of when to apply each technique based on specific requirements.

## PEFT method overview

### 1. LoRA (Low-Rank Adaptation)

**Core concept:** Decomposes weight updates into low-rank matrices, dramatically reducing trainable parameters while maintaining expressiveness.

**Mathematical foundation:**
<math>
<mrow>
<mi>W</mi>
<mo>=</mo>
<msub><mi>W</mi><mn>0</mn></msub>
<mo>+</mo>
<mi>α</mi>
<mi>A</mi>
<mi>B</mi>
</mrow>
</math>

Where:
- **W₀** is the frozen pre-trained weight matrix
- **A** and **B** are trainable low-rank matrices
- **α** is a scaling factor
- **rank r** determines the capacity: A ∈ ℝᵈˣʳ, B ∈ ℝʳˣᵏ

**Key parameters:**
- **`r`**: Rank of adaptation matrices (typically 8-64)
- **`lora_alpha`**: Scaling factor (often 16-32)
- **`target_modules`**: Which layers to adapt (attention projections, feed-forward layers)

### 2. IA³ (Infused Adapter by Inhibiting and Amplifying)

**Core concept:** Learns scaling vectors that selectively amplify or suppress activations in key transformer components.

**Mathematical foundation:**
<math>
<mrow>
<mi>h</mi>
<mo>=</mo>
<mi>γ</mi>
<mo>⊙</mo>
<mi>f</mi>
<mo>(</mo><mi>x</mi><mo>)</mo>
</mrow>
</math>

Where:
- **γ** is a learned scaling vector
- **⊙** denotes element-wise multiplication
- **f(x)** is the original layer computation

**Key characteristics:**
- **Minimal parameters** - Only learns scaling vectors, not full weight matrices
- **Strategic placement** - Targets key information bottlenecks in attention and feed-forward layers
- **Interpretability** - Scaling factors provide insights into what model components are most important

### 3. Prompt Tuning

**Core concept:** Prepends learnable continuous prompt embeddings to input sequences, guiding model behavior without modifying any original parameters.

**Mathematical foundation:**
<math>
<mrow>
<mi>h</mi>
<mo>=</mo>
<mi>LM</mi>
<mo>(</mo>
<mo>[</mo><msub><mi>P</mi><mi>θ</mi></msub><mo>;</mo><mi>x</mi><mo>]</mo><mo>)</mo>
</mrow>
</math>

Where:
- **Pθ** are learnable prompt embeddings
- **x** is the input sequence
- **[;]** denotes concatenation
- **LM** is the frozen language model

**Key advantages:**
- **Ultra-lightweight** - Only prompt embeddings are trainable
- **Task-agnostic base model** - Can switch tasks by swapping prompt embeddings
- **Interpretability challenges** - Continuous prompts lack clear semantic meaning

## Configuration implementation

### LoRA Configuration (`configs/lora.py`):

```python
@property
def peft_config(self) -> PeftConfig:
    return LoraConfig(
        r=8,                    # Low rank dimension
        lora_alpha=16,          # Scaling factor
        target_modules=[        # Modules to adapt
            'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
            'up_proj', 'down_proj', 'gate_proj'      # Feed-forward
        ],
        lora_dropout=0.05,      # Regularization
        task_type=TaskType.CAUSAL_LM
    )
```

### IA³ Configuration (`configs/ia3.py`):

```python
@property
def peft_config(self) -> PeftConfig:
    return IA3Config(
        peft_type="IA3",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[        # Strategic scaling locations
            "k_proj", "v_proj", "o_proj",      # Attention components
            "up_proj", "down_proj", "gate_proj" # Feed-forward components
        ],
        inference_mode=False
    )
```

### Prompt Tuning Configuration (`configs/prompt_tuning.py`):

Implementation requires careful consideration of prompt initialization and virtual token count for optimal performance.

<div class="hint" title="Target Module Selection">

**Tip**: For transformer models, the most effective target modules are typically the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and feed-forward layers (`up_proj`, `down_proj`, `gate_proj`). These are the information processing bottlenecks where adaptation has the highest impact.

</div>

<div class="hint" title="Hyperparameter Trade-offs">

**Tip**: LoRA's rank `r` controls the adaptation capacity - higher ranks allow more expressiveness but require more parameters. Start with `r=8` and `lora_alpha=16` as these provide a good balance. For IA³, focus on target module selection as this determines which components can be adapted.

</div>

<div class="hint" title="Performance Expectations">

**Tip**: Expected evaluation losses are approximately: LoRA (~1.41), IA³ (~1.51), Prompt Tuning (~1.91). LoRA typically performs best due to its higher parameter budget, while Prompt Tuning is most parameter-efficient but may require more careful tuning for complex tasks.

</div>

## Comparative analysis framework

### Performance vs. efficiency trade-offs:

**LoRA:**
- **Parameters**: ~0.1-1% of base model
- **Performance**: Typically best among PEFT methods
- **Use case**: When you need maximum performance with moderate parameter efficiency

**IA³:**
- **Parameters**: ~0.01% of base model  
- **Performance**: Good performance with minimal parameters
- **Use case**: When extreme parameter efficiency is crucial

**Prompt Tuning:**
- **Parameters**: Fixed count regardless of model size
- **Performance**: Requires more tuning but very lightweight
- **Use case**: When you need task-specific behavior with minimal storage overhead

### Experimental workflow:

1. **Configuration implementation** - Complete all three PEFT configuration classes
2. **Training execution** - Run `run.sh` to train all methods with grid search
3. **Performance evaluation** - Use `run_eval.py` to assess each method's effectiveness
4. **Comparative analysis** - Compare results across methods considering both performance and efficiency
5. **Method selection** - Choose the best approach based on your specific requirements

## Technical implementation details

### PEFT integration with Hugging Face:

**Model wrapping:**
```python
peft_model = get_peft_model(base_model, peft_config)
```

**Parameter efficiency verification:**
```python
peft_model.print_trainable_parameters()
```

### Training infrastructure:

**Automated pipeline features:**
- Grid search over multiple hyperparameter combinations
- Weights & Biases integration for experiment tracking
- Automatic model checkpointing and evaluation
- Generated text monitoring during training

### Evaluation methodology:

**Quantitative metrics:**
- Training and validation loss curves
- Parameter count comparison across methods
- Training time and memory usage analysis

**Qualitative assessment:**
- Generated text quality for vocabulary definition tasks
- Consistency and accuracy of model outputs
- Comparison with base model behavior

This comprehensive PEFT implementation demonstrates the cutting edge of efficient model adaptation, showing how sophisticated techniques can achieve excellent performance while using minimal computational resources.