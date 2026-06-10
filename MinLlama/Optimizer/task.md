### Learning objectives

By completing this task, you will be able to:
- **Understand adaptive optimization algorithms** - Learn how AdamW combines momentum with adaptive learning rates for efficient training.
- **Implement gradient-based parameter updates** - Apply mathematical optimization formulas to update neural network parameters.
- **Work with optimizer state management** - Maintain moving averages and step counters across training iterations.
- **Handle bias correction in optimization** - Understand why and how bias correction improves early training dynamics.

### Problem context

Training large language models like LLaMA requires sophisticated optimization algorithms that can handle the challenges of deep learning at scale. Simple gradient descent often fails due to varying gradient magnitudes across parameters and training instability.

**Why AdamW matters:**
- **Adaptive learning rates** - Different parameters get different effective learning rates based on their gradient history.
- **Momentum integration** - Combines the benefits of momentum (faster convergence) with adaptive rates.
- **Weight decay decoupling** - AdamW decouples weight decay from gradient-based updates, improving generalization.
- **State-of-the-art performance** - Used in most modern large language models including GPT, LLaMA, and others.

**What makes this challenging:**
- **Mathematical precision** - Small errors in the update formulas can prevent model convergence.
- **State management complexity** - Must correctly maintain exponential moving averages across parameters and steps.
- **Bias correction implementation** - Requires careful handling of early training steps to avoid biased estimates.
- **Memory efficiency** - Optimizer state can double memory usage, requiring efficient implementation.

## Implementation requirements

Implement the AdamW optimizer algorithm that will be used to train your LLaMA model. AdamW is a variant of the Adam optimizer with decoupled weight decay that provides better generalization performance.

### Specific requirements:

1. **Implement the `step()` method** - Complete the parameter update logic using the AdamW algorithm.
2. **Handle exponential moving averages** - Correctly update momentum and RMS (root mean square) averages.
3. **Apply bias correction** - Implement the bias correction mechanism for early training steps.
4. **Integrate weight decay** - Apply decoupled weight decay as specified in the AdamW paper.

### Expected deliverables:

- Completed `step()` method in the `AdamW` class.
- Implementation that maintains numerical stability during training.
- Code that properly handles optimizer state initialization and updates.
- Correct integration of all AdamW components (momentum, adaptive rates, weight decay, bias correction).

## Mathematical foundation

AdamW uses exponential moving averages to adapt learning rates per parameter. The algorithm maintains two key statistics:

**Momentum estimate (first moment):**
<math>
<mrow>
<msub><mi>m</mi><mi>t</mi></msub>
<mo>=</mo>
<msub><mi>β</mi><mn>1</mn></msub>
<msub><mi>m</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub>
<mo>+</mo>
<mrow><mo>(</mo><mn>1</mn><mo>-</mo><msub><mi>β</mi><mn>1</mn></msub><mo>)</mo></mrow>
<msub><mi>g</mi><mi>t</mi></msub>
</mrow>
</math>

**RMS estimate (second moment):**
<math>
<mrow>
<msub><mi>v</mi><mi>t</mi></msub>
<mo>=</mo>
<msub><mi>β</mi><mn>2</mn></msub>
<msub><mi>v</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub>
<mo>+</mo>
<mrow><mo>(</mo><mn>1</mn><mo>-</mo><msub><mi>β</mi><mn>2</mn></msub><mo>)</mo></mrow>
<msubsup><mi>g</mi><mi>t</mi><mn>2</mn></msubsup>
</mrow>
</math>

**Parameter update with bias correction:**
<math>
<mrow>
<msub><mi>θ</mi><mi>t</mi></msub>
<mo>=</mo>
<msub><mi>θ</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub>
<mo>-</mo>
<mi>α</mi>
<mfrac>
<msqrt><mn>1</mn><mo>-</mo><msubsup><mi>β</mi><mn>2</mn><mi>t</mi></msubsup></msqrt>
<mn>1</mn><mo>-</mo><msubsup><mi>β</mi><mn>1</mn><mi>t</mi></msubsup>
</mfrac>
<mfrac>
<msub><mi>m</mi><mi>t</mi></msub>
<msqrt><msub><mi>v</mi><mi>t</mi></msub></msqrt><mo>+</mo><mi>ε</mi>
</mfrac>
<mo>-</mo>
<mi>α</mi>
<mi>λ</mi>
<msub><mi>θ</mi><mrow><mi>t</mi><mo>-</mo><mn>1</mn></mrow></msub>
</mrow>
</math>

Where:
- `m_t`, `v_t` are the momentum and RMS estimates
- `β₁`, `β₂` are the exponential decay rates (typically 0.9, 0.999)
- `g_t` is the current gradient
- `α` is the learning rate
- `λ` is the weight decay coefficient
- `ε` is a small constant for numerical stability

<div class="hint" title="State Initialization">

**Tip**: The optimizer maintains state for each parameter including step count, momentum estimate (`exp_avg`), and RMS estimate (`exp_avg_sq`). Initialize these to zero when first encountered, and remember to increment the step counter.

</div>

<div class="hint" title="AdamW Implementation Steps">

**Implementation approach**:

1. **Update moving averages**: Apply exponential moving averages to gradients and squared gradients
2. **Apply bias correction**: Compute the bias-corrected learning rate using the step number
3. **Compute parameter update**: Divide momentum by (RMS + epsilon) and multiply by corrected learning rate
4. **Apply weight decay**: Subtract weight decay term directly from parameters (decoupled from gradients)

The key insight is that AdamW applies weight decay directly to parameters, separate from the gradient-based update.

</div>

<div class="hint" title="Numerical Stability">

**Tip**: When computing bias correction, use `torch.tensor([value], device=p.device)` to ensure the correction factor is on the same device as your parameters. Also, make sure to apply weight decay to the original parameters, not the gradient-updated ones.

</div>

## Implementation guidance

**Critical implementation note:**
You may reorganize functions within each class, but **never rename variables corresponding to LLaMA-2 parameters**. Renaming these variables will prevent loading pretrained weights, breaking the entire model.

**Key components to implement:**
- Exponential moving average updates for both first and second moments
- Bias correction computation using step count and beta values  
- Proper parameter updates combining adaptive rates with weight decay
- State management and initialization

**Testing your implementation:**
The optimizer will be used during model training, so correctness is essential for convergence. Pay special attention to the mathematical formulas and ensure all operations are performed in the correct order.
