### Learning objectives

By completing this task, you will be able to:
- **Master parameter-efficient fine-tuning** - Apply LoRA (Low-Rank Adaptation) to reduce trainable parameters while maintaining model performance.
- **Configure sentence transformer training** - Set up proper training arguments, evaluation strategies, and loss functions for bi-encoder models.
- **Implement contrastive learning** - Train models using triplet loss to learn robust similarity representations for retrieval tasks.
- **Integrate PEFT with modern ML frameworks** - Combine Hugging Face PEFT library with Sentence-Transformers for efficient fine-tuning workflows.

### Problem context

Training large language models for retrieval tasks requires significant computational resources and time. Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA enable effective adaptation of pre-trained models using only a small fraction of the original parameters, making advanced NLP capabilities accessible with limited resources.

**Why PEFT for retrieval matters:**
- **Resource efficiency** - LoRA reduces trainable parameters by 90%+ while maintaining comparable performance to full fine-tuning.
- **Contrastive learning requirements** - Retrieval models need specialized training with triplet loss to learn meaningful similarity metrics.
- **Deployment advantages** - Smaller adapter modules enable easier model versioning, sharing, and deployment in production environments.
- **Transfer learning benefits** - PEFT preserves pre-trained knowledge while adapting to specific domains and tasks.

**What makes this challenging:**
- **Hyperparameter sensitivity** - LoRA rank and alpha parameters significantly impact training dynamics and final performance.
- **Training pipeline complexity** - Coordinating PEFT configuration, loss functions, evaluation metrics, and training arguments requires careful orchestration.
- **Evaluation strategy design** - Effective triplet evaluation requires proper anchor-positive-negative sampling from held-out data.
- **Memory management** - Balancing batch sizes, gradient accumulation, and model precision for optimal GPU utilization.

## Task – Bi-encoder Fine-Tuning with LoRA 

Fine-tune a Sentence-Transformer model using LoRA adapters and triplet loss for improved retrieval performance in RAG systems.

### Implementation requirements

#### `get_peft_model`
Load a base Sentence-Transformer model and attach LoRA adapters for parameter-efficient fine-tuning.

**Core functionality:**
- Initialize the base model using `SentenceTransformer()` with `trust_remote_code=True` for security.
- Create a `LoraConfig` with appropriate rank, alpha, and target modules.
- Use `TaskType.FEATURE_EXTRACTION` for sentence embedding tasks.
- Add the LoRA adapter to the model using `add_adapter()` method.

#### `configure_training`
Set up a complete training pipeline with proper arguments, loss function, and evaluation strategy.

**Core functionality:**
- Configure `SentenceTransformerTrainingArguments` with training epochs, batch size, and evaluation settings.
- Align `eval_steps`, `save_steps`, and `logging_steps` for consistent monitoring.
- Create `TripletEvaluator` using anchor, positive, and negative examples from the evaluation split.
- Initialize `SentenceTransformerTrainer` with the model, training arguments, datasets, `TripletLoss`, and evaluator.
  

<div class="hint" title="LoRA Configuration">

**Choosing LoRA parameters**: The rank `r` controls the adaptation capacity - higher ranks allow more expressive adaptations but increase parameters. The alpha parameter controls the scaling of LoRA updates. Target modules should focus on attention layers (`attention.qkv_proj`, `attention.o_proj`) for maximum impact on similarity learning.

</div>

<div class="hint" title="Triplet Training Setup">

**Training pipeline essentials**: Use `TripletLoss` as your loss function and `TripletEvaluator` for evaluation. Ensure your evaluation strategy uses `"steps"` and that `eval_steps`, `save_steps`, and `logging_steps` are aligned for consistent monitoring. The evaluator needs anchor, positive, and negative examples from your eval split.

</div>

### Other Functions and Motivation
- **`get_datasets(path)`** - Load a disk-cached HF dataset and print its size.
- **`run_training(trainer)`** - Call `.train()` and save the final model automatically.

### Scripts
**`run.py`**  
- **Purpose**: Load your adapter-augmented model, datasets, and start training.  
- **Options**:  
  - `--data`: path to the processed dataset dir (default from `config.yaml`)  
  - `--epochs`: number of training epochs (default from `config.yaml`)  
- **Recommendation**: Try the defaults first:
  ```bash
  python run.py
  ```

**Important**: Please run the script and ensure it worked correctly before moving on to the next task.


### Notes
`CUDA_VISIBLE_DEVICES`: \
Set up the environment (e.g. `CUDA_VISIBLE_DEVICES`) \
Model will be saved locally (see `MODEL_OUT` in `config.yaml`)

`wandb`: \
Please log in to Weights & Biases (W&B) to track your training runs. \
You can find how to log in [here](https://docs.wandb.ai/guides/launch/walkthrough/#prerequisites).