> **Expected time for completion: 10 days (~30 hours)**

### Lesson Learning Objectives

By completing this lesson, you will be able to:
- **Build statistical language models** - Implement n-gram models and understand probabilistic text modeling
- **Apply smoothing techniques** - Handle data sparsity using Laplace smoothing and other methods
- **Implement neural language models** - Create RNN-based models using PyTorch for text generation
- **Evaluate model performance** - Use perplexity and cross-entropy loss to assess language model quality
- **Generate coherent text** - Apply various text generation strategies from greedy to sampling-based approaches
- **Understand model training** - Implement the complete pipeline from data preprocessing to model evaluation

### The Problem: Statistical Modeling of Human Language

Language modeling tackles one of the most fundamental challenges in AI: **predicting the next word in a sequence**. This seemingly simple task underlies virtually all modern NLP applications.

**Why language modeling matters:**
- **Foundation of modern AI**: Powers GPT, BERT, and all transformer-based models
- **Universal applicability**: Enables text generation, machine translation, autocomplete, and conversational AI
- **Probability framework**: Provides mathematical foundation for understanding text as sequences of probabilistic events
- **Research gateway**: Understanding language modeling opens doors to cutting-edge NLP research

**What makes this challenging:**
- **Massive vocabulary**: Natural languages contain hundreds of thousands of unique words
- **Long-range dependencies**: Words separated by many positions can be strongly related
- **Context sensitivity**: The same word can have different meanings depending on context
- **Data sparsity**: Most possible word combinations never appear in training data
- **Computational complexity**: Processing and storing probabilities for all possible sequences

**Why start with fundamentals:**
Before diving into transformers and attention mechanisms, understanding classical approaches like n-grams and RNNs provides essential intuition about the core challenges and solutions in language modeling.

### Topics Covered

You will learn about these essential language modeling techniques and concepts:

**Statistical Language Models:**
* **N-grams** - Build probabilistic models that predict next words based on previous n-1 words
* **Laplace Smoothing** - Handle unseen word combinations using additive smoothing techniques
* **Perplexity** - Measure and compare language model quality using this fundamental evaluation metric

**Neural Language Models:**
* **Simple RNN (with PyTorch)** - Implement recurrent neural networks for sequential text processing
* **Cross-Entropy Loss** - Understand the mathematical foundation for training language models
* **Standard Training Procedure** - Master the complete pipeline from data preparation to model evaluation

**Text Generation and Tools:**
* **Text Generation Techniques** - Explore different strategies for sampling from language models
* **Text Tools** - Build practical utilities for text processing and model evaluation

This progression takes you from classical statistical methods through modern neural approaches, providing both theoretical understanding and practical implementation skills. You'll see how each technique addresses specific challenges in language modeling, building toward the neural architectures that power today's large language models.

### Materials

#### Reminders
1. [Torch tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html)
2. [Numpy tutorial](https://numpy.org/doc/stable/user/quickstart.html)

#### Lesson related
1. [Lesson on Language Modeling](https://lena-voita.github.io/nlp_course/language_modeling.html)
    * It's highly recommended to read this lesson thoroughly before moving to exercises.
2. [RNN for Text Classification](https://lena-voita.github.io/nlp_course/text_classification.html#nn_training)
3. [Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

#### Additional
1. [BPE Tokenization](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#bpe)
    * Understanding subword tokenization used in modern language models
2. [Text Generation Tutorial](https://readmedium.com/all-you-need-to-know-about-llm-text-generation-03b138e0ed19)
    * Techniques discussed: Beam Search, Sampling, Top-k Sampling, Top-p Sampling, Temperature Sampling
    * Comprehensive guide on text generation with interesting references
3. [Original LSTM Paper by Schmidhuber](https://www.bioinf.jku.at/publications/older/2604.pdf)
    * A must-read for every NLP enthusiast - foundational work on modern RNNs
4. [GPU Computing Fundamentals](https://jb.gg/g0f6xt)
    * Essential background for understanding why neural language models require GPU acceleration

### Technical Requirements

**Hardware**: You'll need access to a GPU for training the RNN model. The statistical models (n-grams) can run on CPU, but neural network training requires GPU acceleration for reasonable performance.

**Custom Libraries**: This lesson uses custom utility libraries to handle common tasks like data loading and visualization. Feel free to explore these libraries to understand the implementation details.

**PyTorch**: You'll work extensively with PyTorch for implementing neural language models. This provides hands-on experience with the same framework used in cutting-edge research.

### Learning Path

This lesson bridges classical and modern approaches:
1. **Start statistical** - N-grams provide intuitive understanding of language modeling
2. **Add neural networks** - RNNs introduce the power of learned representations
3. **Master evaluation** - Perplexity and loss functions guide model development
4. **Apply practically** - Text generation brings models to life

By the end, you'll have the foundation needed to understand and implement more advanced architectures like transformers, having built the core concepts from first principles. 