### Learning objectives

By completing this lesson on MinLlama implementation, you will be able to:
- **Build modern transformer architectures** - Implement core components of the LLaMA-2 model including self-attention, feedforward layers, and positional embeddings.
- **Understand large language model design** - Grasp the architectural innovations that make models like LLaMA efficient and effective for diverse language tasks.
- **Apply transfer learning principles** - Use pretrained language models for downstream tasks through fine-tuning and prompt-based approaches.
- **Compare architectural approaches** - Understand the evolution from RNNs to transformers and the specific design choices in modern LLMs.
- **Implement practical NLP systems** - Build working classification systems using transformer architectures for real-world text processing tasks.

### Lesson learning objectives

By completing this lesson, you will be able to:
- **Transformer architecture mastery** - Understand and implement the core components of the LLaMA-2 architecture, including multi-head attention mechanisms, rotary position embeddings, and SwiGLU activations.
- **Large-scale model understanding** - Learn how modern language models are structured to handle billions of parameters efficiently while maintaining computational tractability.
- **Fine-tuning and adaptation** - Apply pretrained language models to specific tasks through both prompt-based approaches and task-specific fine-tuning methods.
- **Performance evaluation** - Compare different approaches (zero-shot prompting vs. fine-tuning) and understand their trade-offs in terms of performance and computational requirements.

### The problem: implementing production-grade language models

Modern large language models like LLaMA, GPT, and PaLM represent the state-of-the-art in natural language processing, powering everything from chatbots to code generation systems. However, these models are often treated as "black boxes" - their internal workings remain opaque to many practitioners.

**Why this problem matters:**
- **Industry relevance** - Understanding transformer architectures is essential for working with modern NLP systems and building custom language models.
- **Architectural evolution** - LLaMA represents significant innovations over earlier transformers, including improved efficiency and performance optimizations.
- **Transfer learning mastery** - Learning to adapt pretrained models for specific tasks is a core skill in modern machine learning workflows.
- **Scale and efficiency** - Understanding how to implement models that can scale to billions of parameters while remaining computationally tractable.

**What makes this challenging:**
- **Architectural complexity** - Modern transformers involve multiple interacting components: attention mechanisms, normalization schemes, positional encodings, and activation functions.
- **Implementation precision** - Small errors in implementation can lead to dramatically different model behavior or training instability.
- **Scale considerations** - Designing architectures that work efficiently at both small scales (for experimentation) and large scales (for production).
- **Fine-tuning strategies** - Understanding when and how to adapt pretrained models for specific downstream tasks.

![Llama scheme gif](Llama_scheme.gif)

### Topics covered

You will learn about these essential transformer architecture techniques:

**Core architectural components:**
* **Multi-head self-attention** - The fundamental mechanism that allows transformers to process sequences efficiently and capture long-range dependencies.
* **Rotary position embeddings (RoPE)** - LLaMA's innovative approach to encoding positional information that improves performance on longer sequences.
* **SwiGLU activation functions** - The gated activation function used in LLaMA's feedforward networks for improved model capacity.
* **RMSNorm normalization** - An efficient alternative to LayerNorm that reduces computational overhead while maintaining training stability.

**Model implementation and optimization:**
* **Attention mechanism variants** - Understanding how different attention patterns (causal masking, key-value caching) enable different model behaviors.
* **Parameter initialization** - Proper weight initialization strategies for stable transformer training at scale.
* **Memory efficiency** - Techniques for implementing transformers that can handle large context windows without excessive memory usage.

**Transfer learning applications:**
* **Zero-shot prompting** - Using pretrained models for tasks without any task-specific training through careful prompt engineering.
* **Fine-tuning strategies** - Adapting pretrained language models to specific classification tasks through targeted parameter updates.
* **Evaluation methodologies** - Comparing different adaptation strategies and understanding their trade-offs.

This lesson bridges the gap between theoretical understanding and practical implementation, giving you hands-on experience with the architectures powering today's most capable language models.

## Implementation overview

The code you need to implement can be found in the upcoming tasks. You are responsible for writing core components of LLaMA-2, one of the leading open-source language models, developing a deep understanding of modern transformer architectures.

**Technical setup:**
- **Model scale**: 8-layer, 42-million-parameter language model.
- **Pretrained weights**: Loaded from `stories42M.pt`, pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset.
- **Training data**: Machine-generated children's stories (enabling training without GPU requirements).
- **Computational requirements**: Designed for CPU training, with GPU acceleration recommended for faster iterations.

**Evaluation scenarios:**

Once you have implemented the core components, you will test your model in three progressively sophisticated settings:

1. **Text generation evaluation** - Generate completions starting with: `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`. The output should demonstrate coherent grammar and structure, though content may be whimsical due to the children's stories training data.

2. **Zero-shot classification** - Apply prompt-based sentiment analysis to SST-5 and CFIMDB datasets without any task-specific training. This will demonstrate the model's general language understanding but likely yield performance close to random guessing.

3. **Fine-tuned classification** - Implement a classification head in `classifier.py` and perform task-specific fine-tuning of your LLaMA-2 implementation. This approach should yield significantly stronger classification results, demonstrating the power of transfer learning.

### Important notes

* **Code structure guidance** - A detailed description of the code structure is provided in the next task, [HelperCode](course://MinLlama/HelperCode/task.md), including guidance on which parts you need to implement.
* **Preview mode** - Remember that you can switch to `Preview` mode (in the upper-right corner) when reading markdown files to see the rendered version.
* **Library restrictions** - You are only allowed to use libraries installed via `pip install -r requirements.txt` and the ones preloaded at the start of the course. External libraries (e.g., `transformers`) are not allowed.
* **Development environment** - To speed up iterations, you are encouraged to use Google Colab or a personal GPU-enabled machine (e.g., a MacBook).

### Course progression

This MinLlama implementation serves as a bridge between the foundational language modeling concepts from earlier lessons and the advanced techniques used in production language models. You'll apply many of the same principles you learned with RNNs and character-level models, but at the scale and sophistication of modern transformer architectures.

The hands-on implementation approach ensures you understand not just what these models do, but how they achieve their remarkable capabilities through careful architectural design and engineering.

### Acknowledgement

This exercise has been adapted and slightly modified from an assignment in Carnegie Mellon University's CS11-711 Advanced NLP course.

This code is based on `llama2.c` developed by Andrej Karpathy, with portions originating from the [`transformers`](https://github.com/huggingface/transformers) library (licensed under the [MIT License](https://github.com/jetbrains-academy/llm-agent-course/blob/main/LICENSE)).
