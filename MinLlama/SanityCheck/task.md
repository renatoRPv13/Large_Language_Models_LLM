### Learning objectives

By completing this task, you will be able to:
- **Validate your implementation** - Verify that all components of your Llama model work correctly together.
- **Debug integration issues** - Identify and resolve problems in the complete model pipeline.
- **Understand model behavior** - Observe how your implemented components produce text generation results.

### Problem context

After implementing individual components of the Llama architecture (embeddings, attention, feed-forward networks, RoPE), it's crucial to verify that everything works together correctly. This sanity check ensures your implementation can successfully generate coherent text.

**Why this validation matters:**
- **Integration testing** - Individual components might work in isolation but fail when combined.
- **Configuration verification** - Ensures all hyperparameters and dimensions are correctly aligned.
- **Performance baseline** - Establishes whether your implementation produces reasonable outputs.

**What makes this challenging:**
- **Component interactions** - Subtle bugs may only appear when all components work together.
- **Tensor dimension mismatches** - Small errors in reshaping can cause runtime failures.
- **Numerical stability** - Accumulated errors across layers can lead to unexpected behavior.

## Task

Verify the correctness of your complete Llama implementation by running the provided test script.

### Specific requirements:
1. Run the `main.py` file to test your complete implementation
2. Ensure the model can generate text without errors
3. Verify that the output is coherent and relevant to the input prompt
4. Check that all implemented components work together seamlessly

### Expected deliverables:
- Successfully execute `main.py` without runtime errors.
- Generate meaningful text output from your Llama model.
- Confirm that your implementation passes all integration tests.

<div class="hint" title="Debugging Common Issues">

**Tip**: If you encounter dimension mismatch errors, check that your attention mechanism correctly handles the sequence length and hidden dimensions. Pay special attention to the reshape operations in your RoPE implementation and ensure they match the expected tensor shapes.

</div>
