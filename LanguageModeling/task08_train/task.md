
### Learning objectives

By completing this task, you will be able to:
- **Understand neural network training loops** - Learn the fundamental components of training: forward pass, loss computation, backpropagation, and parameter updates.
- **Implement training and validation monitoring** - Build a complete training system with proper validation tracking and convergence monitoring.
- **Apply optimization techniques** - Work with PyTorch optimizers and understand how gradient-based learning updates model parameters.
- **Monitor training dynamics** - Interpret loss curves, detect convergence, and understand the relationship between training and validation performance.

### Problem context

Training neural networks is both an art and a science, requiring careful orchestration of multiple components you've built in previous tasks. A proper training loop is far more than just iterating through data - it's about creating a robust system that learns effectively while providing insight into the model's behavior.

**Why proper training procedures matter:**
- **Learning dynamics** - Neural networks require careful tuning of learning rates, batch sizes, and training schedules to converge effectively.
- **Overfitting detection** - Monitoring validation performance helps identify when models start memorizing rather than generalizing.
- **Debugging and diagnostics** - Training curves, generated samples, and loss plots provide crucial insights into model behavior.
- **Reproducibility** - Systematic training procedures ensure consistent results across different runs and environments.

**What makes this challenging:**
- **Integration complexity** - Must coordinate model, loss function, optimizer, data loading, and evaluation components seamlessly.
- **Memory management** - Efficient batch processing and gradient computation without running out of memory.
- **Training stability** - Handling numerical instabilities, exploding gradients, and convergence issues.
- **Monitoring and visualization** - Generating meaningful diagnostics without slowing down training significantly.

## Implementation requirements

Build a comprehensive training system that brings together all the components from previous tasks: RNN model, loss function, text preprocessing, and evaluation. This represents the culmination of your language modeling pipeline.

### Specific requirements:

1. **Implement complete training loop** - Integrate forward pass, loss computation, backpropagation, and parameter updates.
2. **Handle batch processing** - Efficiently process training data in batches with proper shuffling and sampling.
3. **Monitor validation performance** - Track development set performance to detect overfitting and measure generalization.
4. **Generate training diagnostics** - Create loss plots and sample generations to monitor model behavior during training.
5. **Ensure convergence** - Implement proper training procedures that lead to stable model convergence.

### Expected deliverables:

- Completed `train` method that successfully trains the RNN language model.
- Training system that handles both training and validation data properly.
- Implementation that generates loss plots showing convergence behavior.
- Code that produces coherent text samples demonstrating model learning.
- Training procedure that achieves stable convergence as evidenced by decreasing loss curves.

### Method specifications

#### **`train`**

Execute a complete neural network training procedure that integrates all components of your language modeling system into a robust, monitored training loop.

**Training Procedure Overview:**

The training process follows the standard deep learning workflow while incorporating language-specific monitoring and evaluation:

1. **Batch sampling** - Randomly sample training batches to ensure diverse gradient updates.
2. **Forward pass** - Process batches through your RNN model to get predictions.
3. **Loss computation** - Calculate masked cross-entropy loss using your custom loss function.
4. **Backward pass** - Compute gradients via backpropagation through time.
5. **Parameter updates** - Apply optimizer to update model weights.
6. **Validation monitoring** - Periodically evaluate on development data to track generalization.
7. **Diagnostic generation** - Create loss plots and text samples to monitor training progress.

**Parameters:**
- `model`: RNN language model instance to be trained.
- `opt`: PyTorch optimizer (e.g., Adam, SGD) for parameter updates.
- `loss_fn`: Custom cross-entropy loss function with masking.
- `train_lines`: List of training text strings.
- `dev_lines`: List of validation text strings.
- `device`: Computation device ('cpu' or 'cuda').
- `gen_conf`: Configuration dictionary for text generation.
- `batch_size`: Number of sequences per training batch.
- `draw_every`: Frequency (in epochs) for generating plots and samples.
- `score_dev_every`: Frequency (in epochs) for validation evaluation.
- `n_epochs`: Total number of training epochs.

**Returns:**
- `train_history`: List of (epoch, loss) tuples tracking training progress.
- `dev_history`: List of (epoch, loss) tuples tracking validation progress.
- `model`: The trained model with updated parameters.

<div class="hint" title="Training Loop Structure">

**Tip**: The core training loop should: 
1) set model to train mode;
2) sample random batch;
3) zero gradients;
4) forward pass;
5) compute loss;
6) backward pass;
7) optimizer step. 

Don't forget to call `model.train()` before training and `model.eval()` during validation.

</div>

<div class="hint" title="Batch Sampling Strategy">

**Tip**: Use `random.sample(train_lines, batch_size)` to get random batches. This ensures each epoch sees different combinations of training data, improving generalization. Make sure to handle cases where `batch_size` might be larger than the number of training lines.

</div>

<div class="hint" title="Memory Management">

**Tip**: After each training step, consider calling `torch.cuda.empty_cache()` if using GPU to prevent memory accumulation. Also, detach loss values before logging to avoid keeping the computation graph in memory: `train_history.append((epoch, loss.item()))`.

</div>

<div class="hint" title="Convergence Monitoring">

**Tip**: Good convergence signs include: decreasing training loss, stable (not increasing) validation loss, and improving quality of generated text samples. If validation loss starts increasing while training loss decreases, you may be overfitting.

</div>

**Usage Example:**
```python
from torch.optim import Adam

# Set up training configuration
model = RNNLanguageModel(tokens)
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training data
train_lines = ["hello world", "deep learning", "neural networks"]
dev_lines = ["machine learning", "artificial intelligence"]

# Generator configuration for sampling
gen_conf = {"prefix": "hello", "mode": "sample", "max_len": 20}

# Execute training
train_hist, dev_hist, trained_model = TrainProcedure.train(
    model=model, opt=optimizer, loss_fn=loss_function,
    train_lines=train_lines, dev_lines=dev_lines,
    device=device, gen_conf=gen_conf,
    batch_size=8, draw_every=5, score_dev_every=10, n_epochs=50
)
```

**Expected Training Output:**
```text
Epoch #1: Train loss = 3.247
Epoch #5: Train loss = 2.891, generating samples...
Epoch #10: Train loss = 2.445, Dev loss = 2.523
Epoch #15: Train loss = 2.187, generating samples...
...
Training completed! Final train loss = 1.856, Final dev loss = 1.923
```

**Important Notes:**
- **Convergence verification**: Check that both training and validation losses decrease over time.
- **Generated samples**: Monitor the quality of generated text - it should become more coherent as training progresses.
- **Loss plots**: The generated `train_test_plot.png` should show smooth, decreasing curves for both training and validation loss.
- **Overfitting detection**: If validation loss starts increasing while training loss continues decreasing, consider early stopping.

After completing your implementation, run `run.py` to train the model and verify successful convergence. You can also run `task.py` to see a demonstration of the training process and examine the generated loss plots in `train_test_plot.png`.