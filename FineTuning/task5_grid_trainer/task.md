### Learning objectives

By completing this task, you will be able to:
- **Master systematic hyperparameter optimization** - Implement grid search methodologies for finding optimal training configurations.
- **Build robust training pipelines** - Create end-to-end training workflows with proper logging, evaluation, and model persistence.
- **Work with PEFT model training** - Integrate parameter-efficient fine-tuning techniques with Hugging Face Trainer framework.
- **Implement experiment tracking** - Use Weights & Biases for comprehensive experiment monitoring and comparison.

### Problem context

Finding optimal hyperparameters for fine-tuning is crucial for model performance, but manually testing different configurations is time-consuming and error-prone. Systematic hyperparameter optimization through grid search enables reproducible, comprehensive exploration of the parameter space.

**Why systematic training pipelines matter:**
- **Reproducible experiments** - Standardized training procedures ensure consistent results across different runs and researchers.
- **Optimal performance discovery** - Grid search systematically explores hyperparameter combinations to find the best configuration.
- **Resource efficiency** - Automated experiment management maximizes GPU utilization and minimizes manual intervention.
- **Scientific rigor** - Proper experiment tracking enables valid comparisons and meaningful conclusions about model performance.

**What makes this challenging:**
- **Combinatorial explosion** - Multiple hyperparameters with multiple values create exponentially large search spaces.
- **Resource management** - Grid search requires careful management of compute resources, storage, and experiment state.
- **Experiment coordination** - Managing multiple training runs requires robust state management and error handling.
- **Results analysis** - Comparing dozens or hundreds of experiments requires sophisticated analysis and visualization tools.

## Implementation requirements

Build a comprehensive training and hyperparameter optimization system that can execute both single training runs and systematic grid search experiments for PEFT models.

### Specific requirements:

1. **Implement `train` method** - Create a complete training pipeline with PEFT model setup, logging, and model persistence.
2. **Build `_get_params_grid` function** - Generate all hyperparameter combinations for systematic exploration.
3. **Complete `grid_search` method** - Execute multiple training runs with different hyperparameter configurations.
4. **Integrate experiment tracking** - Properly manage Weights & Biases logging for both single runs and grid search.

### Expected deliverables:

- Completed `train` method that handles both regular and grid search training modes.
- Working `_get_params_grid` function that generates all parameter combinations using `itertools.product`.
- Functional `grid_search` method that systematically explores hyperparameter space.
- Proper Weights & Biases integration for experiment tracking and comparison.

## Grid search methodology

### Hyperparameter space exploration

Grid search systematically tests all combinations of specified hyperparameters:

**Example parameter space:**
```python
grid_params = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "per_device_train_batch_size": [8, 16],
    "num_train_epochs": [3, 5],
    "warmup_ratio": [0.1, 0.2]
}
```

**Results in 24 total experiments:**
3 learning rates × 2 batch sizes × 2 epoch counts × 2 warmup ratios = 24 combinations

### Training pipeline architecture

**Single training run workflow:**
1. **PEFT model creation** - Apply parameter-efficient fine-tuning configuration to base model
2. **Trainer setup** - Configure Hugging Face Trainer with datasets, callbacks, and training arguments
3. **Training execution** - Run training with automatic evaluation and checkpointing
4. **Model persistence** - Save best model and training configuration for future use
5. **Experiment logging** - Track metrics, hyperparameters, and artifacts in Weights & Biases

**Grid search workflow:**
1. **Parameter combination generation** - Create all hyperparameter combinations using Cartesian product
2. **Experiment iteration** - Execute training for each parameter combination
3. **State management** - Update training arguments, output directories, and run names for each experiment
4. **Progress tracking** - Log completion status and maintain experiment metadata
5. **Results aggregation** - Collect results from all runs for comparative analysis

## Mathematical foundation

### Grid search formalization

For a hyperparameter space with parameters **p₁, p₂, ..., pₙ** and their respective value sets **V₁, V₂, ..., Vₙ**, grid search explores:

<math>
<mrow>
<mi>Θ</mi>
<mo>=</mo>
<msub><mi>V</mi><mn>1</mn></msub>
<mo>×</mo>
<msub><mi>V</mi><mn>2</mn></msub>
<mo>×</mo>
<mo>...</mo>
<mo>×</mo>
<msub><mi>V</mi><mi>n</mi></msub>
</mrow>
</math>

**Total experiments:**
<math>
<mrow>
<mo>|</mo><mi>Θ</mi><mo>|</mo>
<mo>=</mo>
<munderover>
<mo>∏</mo>
<mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow>
<mi>n</mi>
</munderover>
<mo>|</mo><msub><mi>V</mi><mi>i</mi></msub><mo>|</mo>
</mrow>
</math>

Where |Vᵢ| is the number of values for parameter i.

### PEFT parameter efficiency

Parameter-efficient fine-tuning dramatically reduces the optimization space:

**Traditional fine-tuning:** All **θ** parameters are trainable
**PEFT approach:** Only **θ_PEFT ⊂ θ** parameters are trainable, where |θ_PEFT| ≪ |θ|

**Typical parameter reduction:**
- **LoRA**: ~0.1-1% of original parameters
- **IA³**: ~0.01% of original parameters  
- **Prompt Tuning**: Fixed number regardless of model size

<div class="hint" title="Grid Search Implementation">

**Tip**: Use `itertools.product(*grid_params.values())` to generate all hyperparameter combinations. The `zip` function then pairs each combination with parameter names to create dictionaries that can update training arguments.

</div>

<div class="hint" title="PEFT Model Integration">

**Tip**: The `get_peft_model(self.base_model, self.peft_config)` call wraps your base model with parameter-efficient layers. Always call `print_trainable_parameters()` to verify that only the expected parameters are being trained - this is crucial for both memory efficiency and training effectiveness.

</div>

<div class="hint" title="Experiment State Management">

**Tip**: Each grid search run needs isolated state: separate output directories, unique run names, and independent Weights & Biases logging. Use the `_update_params_grid` method to ensure experiments don't interfere with each other and results are properly organized.

</div>

## Implementation approach

### Core training method structure:

```python
def train(self, is_grid: bool = False) -> PeftModel:
    # 1. Initialize experiment tracking (if not grid search)
    if not is_grid:
        self._init_wandb()
    
    # 2. Setup callbacks and model
    self._update_callbacks(self.training_args.output_dir)
    peft_model = get_peft_model(self.base_model, self.peft_config)
    
    # 3. Configure trainer
    trainer = Trainer(
        model=peft_model,
        args=self.training_args,
        train_dataset=self.train_dataset,
        eval_dataset=self.eval_dataset,
        data_collator=DataCollatorForLanguageModeling(self.base_tokenizer, mlm=False),
        callbacks=self.default_callbacks,
    )
    
    # 4. Execute training and save results
    trainer.train()
    trainer.save_model(Path(self.training_args.output_dir) / "best_model")
    
    return peft_model
```

### Grid search implementation:

```python
def _get_params_grid(self, grid_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    return [
        dict(zip(grid_params.keys(), values))
        for values in itertools.product(*grid_params.values())
    ]

def grid_search(self, grid_params: Dict[str, List[Any]]) -> None:
    param_combinations = self._get_params_grid(grid_params)
    
    for i, params in enumerate(param_combinations):
        self._update_params_grid(i, params)
        self.train(is_grid=True)
        wandb.log({"completed_runs": i + 1})
    
    wandb.finish()
```

## Supporting infrastructure

### Experiment tracking with Weights & Biases

**Automatic logging:**
- Training and validation loss curves
- Hyperparameter configurations
- Model performance metrics
- Generated text examples during training

**Grid search coordination:**
- Progress tracking across multiple runs
- Centralized experiment comparison
- Automated artifact collection

### Training callbacks

**LoggingCallback:**
- Provides detailed training progress logs
- Saves logs to structured text files for offline analysis

**ExampleGenerationCallback:**
- Generates sample outputs during training to monitor quality
- Helps detect overfitting or convergence issues early

### Best run identification

**BestRunSearcher utility:**
- Automatically identifies optimal hyperparameter configurations
- Compares runs based on specified evaluation metrics
- Enables easy deployment of best-performing models

## Experimental design considerations

### Grid search strategy:

**Parameter selection priorities:**
1. **Learning rate** - Most critical for convergence and final performance
2. **Batch size** - Affects both memory usage and gradient quality
3. **Training epochs** - Balance between underfitting and overfitting
4. **Warmup ratio** - Important for training stability with large learning rates

### Resource planning:

**Computational requirements:**
- **Time estimation**: Total time = (single run time) × (number of combinations)
- **Memory management**: Ensure each run fits within available GPU memory
- **Storage planning**: Each run generates checkpoints, logs, and model artifacts

### Results analysis:

**Performance metrics:**
- **Validation loss** - Primary optimization objective
- **Training stability** - Monitor for convergence issues
- **Generation quality** - Evaluate sample outputs for task-specific quality

This systematic approach to hyperparameter optimization ensures you find the best possible configuration for your specific fine-tuning task while maintaining rigorous experimental standards.
