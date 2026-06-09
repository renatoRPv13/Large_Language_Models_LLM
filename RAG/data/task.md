### Learning objectives

By completing this task, you will be able to:
- **Understand RAG dataset design** - Learn how to structure data for training retrieval-augmented generation systems.
- **Work with contrastive learning data** - Master the use of positive, hard negative, and random negative samples for robust model training.
- **Analyze multi-source data integration** - Understand how WordNet definitions and generated content complement each other in RAG systems.
- **Design effective training signals** - Recognize how different types of negative examples improve model discrimination abilities.

### Problem context

Retrieval-Augmented Generation requires carefully curated datasets that train models to distinguish between relevant and irrelevant information. Unlike simple text generation, RAG systems must learn to identify the most useful knowledge from large collections of potentially related documents.

**Why RAG dataset design matters:**
- **Retrieval quality** - Good training data teaches models to find truly relevant information, not just superficially similar text.
- **Generation accuracy** - Contrastive examples help models avoid hallucination by learning to reject plausible but incorrect information.
- **Real-world robustness** - Training with hard negatives prepares models for challenging retrieval scenarios with many similar but incorrect options.

**What makes this challenging:**
- **Hard negative creation** - Generating plausible but incorrect examples requires sophisticated understanding of semantic similarity.
- **Training signal balance** - Too easy negatives make training ineffective; too hard negatives can confuse the model.
- **Multi-source integration** - Combining generated and curated data requires careful quality control and consistency management.

## RAG Definition Generator Dataset

Here you can find the way to generate a dataset for training a Retrieval-Augmented Generation (RAG) which will be used in our lesson. \
Take a look at the data (`dataset.json`) and move on to the next task.

**Note:** This is just an overview of the dataset generation process. \
You don't need to run the code.

---

## 📁 Directory Structure

* `dataset.json`: The main dataset with positive, hard, and negative samples
* `instruction.md`: Prompt templates and OpenAI annotation instructions
* `main.py`: Script to generate the dataset using `words.csv` and OpenAI API
* `wordnet_definitions.json`: Auxiliary WordNet definitions for retrieval

---

## 🗂️ Data Files

- **`dataset.json`**
  - **`word`**: target word
  - **`part_of_speech`**: noun | verb | adjective
  - **`definition`**: ground-truth sense
  - **`close_definition`** & **`sentence`**: hard negatives for contrastive learning
  - **`random_definition`**: negative sample

- **`dataset_without_random_definition.json`**: same as `dataset.json`, excluding `random_definition` for setups without explicit negatives.
  - Note: this is an intermediate step, you won't see the actual file in the repository.

- **`wordnet_definitions.json`**: list of WordNet-sourced definitions for richer retrieval contexts.

- **`words.csv`**: seed list of words. Used by `main.py` and OpenAI API to generate `dataset.json`.

---

## 🔧 Training Signals

- **Positive**: `definition`
- **Hard Negatives**: `close_definition`, `sentence`
- **Negative**: `random_definition`

These signals guide the RAG retriever and generator to distinguish correct definitions from distractors during training.

---

## 💡 Key Insights

<div class="hint" title="Contrastive Learning Design">

**Understanding the training signals**: The dataset uses three types of negative examples to teach different aspects of retrieval quality:
- **Hard negatives** (`close_definition`) - Similar but incorrect definitions that test semantic understanding
- **Contextual negatives** (`sentence`) - Related content that isn't definitional, testing task understanding  
- **Random negatives** (`random_definition`) - Completely unrelated content for basic discrimination

</div>

<div class="hint" title="Multi-Source Data Strategy">

**Why combine generated and curated data**: OpenAI-generated definitions provide consistency and task alignment, while WordNet definitions add diversity and coverage. This combination creates robust training data that generalizes well to real-world retrieval scenarios.

</div>

<div class="hint" title="Dataset Structure Impact">

**How structure affects RAG training**: The explicit separation of positive/negative signals enables contrastive learning approaches where the model learns to rank correct information higher than plausible distractors. This is crucial for RAG systems that must choose between multiple potentially relevant documents.

</div>

