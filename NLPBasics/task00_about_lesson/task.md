 **Expected time for the lesson completion: 7 days (~20 hours)**

### Course Learning Goals

By the end of this course on Large Language Models, you will be able to:
- Understand the fundamentals of natural language processing and how it forms the foundation for modern LLMs
- Build and evaluate text classification systems using traditional NLP techniques
- Implement and optimize retrieval-augmented generation (RAG) systems
- Develop custom language models from scratch, understanding the underlying architecture
- Apply advanced optimization techniques for training and fine-tuning language models
- Create practical AI agents that can interact with external tools and APIs

### About the Lesson

Welcome to the course on Large Language Models! 🙌

This foundational lesson introduces you to Natural Language Processing (NLP), the cornerstone technology that enables machines to understand and process human language. Before diving into complex transformer architectures and large language models, it's essential to master these fundamental concepts and techniques that still power many production systems today.

### Lesson Learning Objectives

By completing this lesson, you will be able to:
- **Preprocess text data** using standard NLP techniques including cleaning, normalization, and tokenization
- **Transform text into numerical representations** using various embedding techniques (TF-IDF, Word2Vec, etc.)
- **Implement and compare classical machine learning approaches** for text classification (kNN, Naive Bayes, Logistic Regression)
- **Evaluate model performance** using appropriate metrics for classification tasks
- **Understand the pipeline** from raw text to trained classifier, establishing the foundation for more advanced NLP systems

### The Problem: Movie Review Sentiment Classification

In this lesson, you'll tackle one of the most fundamental problems in NLP: **sentiment analysis**. Specifically, you'll build systems to automatically determine whether a movie review expresses positive or negative sentiment.

**Why this problem matters:**
- It's a concrete, well-defined binary classification task perfect for learning core NLP concepts
- Sentiment analysis has real-world applications in business intelligence, social media monitoring, and customer feedback analysis
- The techniques you'll learn here form the building blocks for more complex NLP tasks
- It provides a clear evaluation framework to compare different approaches

**What makes this challenging:**
- Natural language is ambiguous and context-dependent
- The same words can express different sentiments in different contexts
- Reviews may contain sarcasm, mixed opinions, or nuanced language
- Raw text needs to be transformed into numerical representations that machines can process

### Topics covered

You will learn about these essential NLP and machine learning techniques:

**Text Processing Fundamentals:**
* **Data Preprocessing** - Clean and normalize raw text data, handle common issues like punctuation, case sensitivity, and special characters
* **Tokenization** - Break text into meaningful units (words, subwords) that can be processed by algorithms

**Text Representation:**
* **Embeddings** - Transform text into numerical vectors that capture semantic meaning, including TF-IDF and Word2Vec approaches

**Classification Algorithms:**
* **kNN (k-Nearest Neighbors)** - A simple yet effective instance-based learning algorithm that classifies text based on similarity to training examples
* **Naive Bayes** - A probabilistic classifier particularly well-suited for text classification, leveraging word frequency patterns
* **Logistic Regression** - A linear classifier that learns to weigh different features to make classification decisions

Each technique builds upon the previous ones, creating a complete pipeline from raw text to trained classifier. You'll implement these from scratch and compare their strengths and weaknesses on the movie review dataset.

### Materials

**Note:** You are free, and actually encouraged, to look for the materials on your own. We provide some links to help you get started.

**Note 2:** In each lesson, we'll provide 3 types of materials:
1. `reminders` - links to the materials we expect you to know. No worries if you don't know them, just be ready to spend some more time on the lesson.
2. `lesson related` - links to the materials that are directly related to the lesson. It's highly recommended to read them.
3. `additional` - links to the materials that are not directly related to the lesson, but can be interesting for you.

#### Reminders
1. kNN model
    * [k-Nearest-Neighbors](https://medium.com/@rndayala/k-nearest-neighbors-a76d0831bab0)
2. Logistic regression
    * [Logistic Regression | Toward Data Science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
3. Naive-bayes
    * [Naive Bayes Classifier](https://lena-voita.github.io/nlp_course/text_classification.html#naive_bayes)
4. Data Processing 
    * [Data Cleaning in Pandas | YouTube](https://jb.gg/bpf90p) (some of the topics discussed are applied in the lesson)

#### Lesson related
1. Lesson on Word Embeddings
    * [Word Embeddings | NLP Course Notes](https://lena-voita.github.io/nlp_course/word_embeddings.html#main_content)
2. Tokenization
    * [NLP | How tokenizing text, sentence, words works](https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/)
3. Gensim documentation
    * [docs](https://radimrehurek.com/gensim/auto_examples/index.html)

#### Additional
1. Semantic search with Embeddings
    * [The Power of Embeddings for Semantic Search | Medium](https://medium.com/@umairali.khan/the-power-of-embeddings-for-semantic-search-8883f3fe8ba2)
2. A comprehensive lesson on Text Classification
    * [Text Classification | NLP Course notes](https://lena-voita.github.io/nlp_course/text_classification.html)

### Other Information

In this lesson, we'll use custom library [`tools_basics`](https://github.com/jetbrains-academy/llm-agent-course-utils/tree/main/tools_basics) to implement some basic methods (mainly for data handling and visualization). \
We want our students to be able concentrate on the main tasks of this course. Of course, you can look inside the library and see how it works.

**Important (!):** For each task, please do run task.py file after passing the tests. Some tasks save information essential for the next tasks.