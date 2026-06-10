### Learning objectives

By completing this task, you will be able to:
- **Understand different LLM usage modes** - Learn the distinctions between text generation, zero-shot prompting, and fine-tuning approaches.
- **Apply text generation techniques** - Use temperature control to generate diverse and creative text completions.
- **Implement zero-shot classification** - Leverage pre-trained language models for classification without task-specific training.
- **Execute model fine-tuning** - Adapt pre-trained models to specific downstream tasks through supervised learning.

### Problem context

Large language models like LLaMA can be deployed in multiple ways depending on your specific use case, data availability, and computational constraints. Understanding when and how to use each approach is crucial for practical NLP applications.

**Why different deployment modes matter:**
- **Text generation** - Demonstrates the model's creative and completion capabilities for content creation and interactive applications.
- **Zero-shot prompting** - Enables immediate task performance without training data, useful for rapid prototyping and low-resource scenarios.
- **Fine-tuning** - Provides optimal performance on specific tasks by adapting the model's parameters to domain-specific data.
- **Practical deployment** - Real-world applications often require choosing the right balance between performance, cost, and development time.

**What makes this challenging:**
- **Mode selection complexity** - Different approaches have varying computational requirements, data needs, and performance characteristics.
- **Hyperparameter tuning** - Each mode requires different parameter settings for optimal performance.
- **Evaluation interpretation** - Understanding how to assess and compare results across different deployment strategies.
- **Resource management** - Balancing model performance with computational and memory constraints.

## Implementation requirements

Run and evaluate your complete LLaMA implementation across three different deployment modes to understand their capabilities and trade-offs.

### Specific requirements:

1. **Execute text generation mode** - Generate creative text completions with temperature control.
2. **Run zero-shot prompting** - Perform classification using natural language prompts without training.
3. **Conduct fine-tuning** - Train the model on labeled data for improved task-specific performance.
4. **Compare and analyze results** - Evaluate the strengths and limitations of each approach.

### Expected deliverables:

- Successfully execute all three modes without errors.
- Generate coherent and diverse text samples in generation mode.
- Achieve reasonable classification accuracy in both prompting and fine-tuning modes.
- Understand the trade-offs between different deployment strategies.

## Deployment modes

### 1. Text Generation Mode (`generate`)

Tests your model's ability to continue text from a given prefix with controllable creativity.

**Key features:**
- **Temperature control** - Low temperature (0.0) for deterministic, focused outputs; high temperature (1.0) for creative, diverse outputs
- **Prefix completion** - Demonstrates the model's understanding of context and style
- **Output comparison** - Shows how temperature affects generation quality and diversity

**Example usage:**
```bash
python main.py --option generate
```

**What to expect:** The model will complete the given movie review prefix with two different temperature settings, showing both conservative and creative completion styles.

### 2. Zero-Shot Prompting Mode (`prompt`)

Evaluates classification performance using the model's pre-trained knowledge without additional training.

**Key features:**
- **No training required** - Uses the model's existing knowledge to perform classification
- **Prompt engineering** - Relies on carefully crafted prompts to elicit correct predictions
- **Immediate deployment** - Can be applied to new tasks without collecting training data

**Example usage:**
```bash
python main.py --option prompt
```

**What to expect:** The model will classify movie reviews as positive or negative using only natural language prompts, demonstrating emergent classification abilities.

### 3. Fine-Tuning Mode (`finetune`)

Adapts the model to specific classification tasks through supervised learning on labeled data.

**Key features:**
- **Parameter updates** - Modifies model weights based on task-specific training data
- **Supervised learning** - Uses labeled examples to learn task-specific patterns
- **Optimal performance** - Typically achieves the best accuracy on the target task

**Example usage:**
```bash
python main.py --option finetune
```

**What to expect:** The model will train on sentiment analysis data, then evaluate performance on development and test sets, showing improved task-specific accuracy.

## Configuration and hyperparameters

### Core execution parameters:
- **`--option`**: Choose between `generate`, `prompt`, or `finetune`
- **`--pretrained-model-path`**: Path to your trained LLaMA checkpoint
- **`--use_gpu`**: Enable GPU acceleration for faster training/inference

### Dataset parameters:
- **`--train`**: Training data file path
- **`--dev`**: Development/validation data file path  
- **`--test`**: Test data file path
- **`--label-names`**: Label mapping configuration file

### Training hyperparameters:
- **`--epochs`**: Number of training epochs (default: 5)
- **`--batch_size`**: Training batch size (default: 8)
- **`--lr`**: Learning rate for fine-tuning (default: 2e-5)
- **`--hidden_dropout_prob`**: Dropout rate for regularization (default: 0.3)

### Generation parameters:
- **`--temperature`**: Controls randomness in text generation (0.0 = deterministic, 1.0 = creative)
- **`--max_sentence_len`**: Maximum sequence length for tokenization (default: 512)

<div class="hint" title="Getting Started">

**Tip**: Start with the generation mode to verify your basic implementation works, then proceed to prompting to test zero-shot capabilities, and finally use fine-tuning to achieve optimal performance. Each mode builds on the previous one's requirements.

</div>

<div class="hint" title="Temperature Effects">

**Understanding temperature**: In generation mode, you'll see two outputs:
- **Temperature 0.0**: Deterministic, focused, and conservative completions
- **Temperature 1.0**: Creative, diverse, and sometimes surprising completions

This demonstrates how the same model can be tuned for different creative vs. reliable use cases.

</div>

<div class="hint" title="Performance Expectations">

**Expected results**:
- **Generate**: Should produce fluent, contextually appropriate text completions
- **Prompt**: May achieve 60-75% accuracy on sentiment classification (depends on prompt quality)
- **Finetune**: Should achieve 80-90%+ accuracy with proper hyperparameter tuning

Fine-tuning typically provides the best performance but requires labeled training data and computational resources.

</div>

## Execution workflow

### Step-by-step execution:

1. **Verify implementation**: Ensure all previous LLaMA components (attention, embeddings, optimizer, etc.) are correctly implemented
2. **Run text generation**: Execute with `--option generate` to test basic text completion capabilities
3. **Test zero-shot prompting**: Execute with `--option prompt` to evaluate classification without training
4. **Perform fine-tuning**: Execute with `--option finetune` to train and evaluate on labeled data
5. **Analyze and compare**: Review outputs from all three modes to understand their strengths and limitations

### Quick execution (modify line 80 in main.py):
```python
if __name__ == "__main__":
    main("generate")  # Change to "prompt" or "finetune" as needed
```

## Success criteria

**Your implementation is successful when:**
- All three modes execute without runtime errors
- Generated text is coherent and relevant to the input prefix
- Zero-shot prompting produces reasonable classification decisions
- Fine-tuning improves performance over zero-shot baselines
- You understand the trade-offs between computational cost, data requirements, and performance across modes

This task represents the culmination of your LLaMA implementation, demonstrating how the same underlying architecture can be deployed for diverse NLP applications.
