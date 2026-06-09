### Learning objectives

By completing this task, you will be able to:
- **Master triplet dataset construction** - Transform raw JSON data into structured training triplets for contrastive learning in retrieval systems.
- **Implement robust data preprocessing** - Handle missing fields, whitespace normalization, and data validation in production ML pipelines.
- **Design contrastive learning signals** - Create anchor-positive-negative triplets that teach models to distinguish relevant from irrelevant information.
- **Work with Hugging Face Datasets** - Leverage modern ML libraries for efficient dataset creation, concatenation, and train/eval splitting.

### Problem context

Retrieval-Augmented Generation systems require sophisticated training data that teaches bi-encoder models to identify semantically relevant information from large corpora. Unlike simple classification tasks, RAG models must learn nuanced similarity relationships through contrastive learning with carefully constructed positive and negative examples.

**Why triplet dataset construction matters:**
- **Contrastive learning foundation** - Triplet loss requires explicit anchor-positive-negative relationships to learn robust similarity metrics.
- **Hard negative mining** - Different negative types (unrelated, contextual, near-miss) teach models to handle various types of irrelevant information.
- **Training efficiency** - Well-structured datasets enable faster convergence and better generalization in retrieval models.
- **Production readiness** - Robust preprocessing handles real-world data quality issues and missing information.

**What makes this challenging:**
- **Data quality management** - Raw JSON often contains missing fields, inconsistent formatting, and quality variations requiring careful validation.
- **Negative sampling strategy** - Balancing different types of negatives (random, hard, contextual) to create effective training signals without confusing the model.
- **Memory efficiency** - Processing large datasets requires efficient in-memory structures and batch operations for scalability.
- **Reproducible splitting** - Ensuring consistent train/eval splits across experiments while maintaining proper randomization.

## Task – Triplet Dataset Builder

Transform raw JSON word entries into structured Hugging Face datasets with `(anchor, positive, negative)` triplets for contrastive learning in RAG systems.

### Implementation requirements

#### `TripletDatasetBuilder.build_triplets`
Build an in-memory triplet `Dataset` with columns `anchor`, `positive`, `negative`.

**Core functionality:**
- Process raw JSON entries into triplet format where:
  - **Anchor**: The target word being defined.
  - **Positive**: The correct definition for the word.
  - **Negative**: The incorrect example (from specified field: `close_definition`, `sentence`, or `random_definition`).
- Skip entries missing the requested negative field to maintain data integrity.
- Strip whitespace from all text fields to ensure consistent formatting.
- Return a Hugging Face `Dataset` using `Dataset.from_dict()`.

#### `TripletDatasetBuilder.concat_and_split`
Merge multiple triplet datasets and create reproducible train/evaluation splits.

**Core functionality:**
- Concatenate multiple `Dataset` objects using `concatenate_datasets()`.
- Apply shuffling with fixed seed for reproducible experiments.
- Split into training and evaluation sets using `train_test_split()`.
- Return `DatasetDict` with `train` and `eval` splits where `train_frac` determines the training proportion.  


<div class="hint" title="Triplet Dataset Implementation">

**Building effective triplets**: When processing raw JSON entries, remember that the quality of your negative examples determines training effectiveness. The `negative_field` parameter lets you experiment with different contrastive signals:
- `"close_definition"` - Near-miss definitions that test fine-grained semantic understanding.
- `"sentence"` - Usage examples that aren't definitional, testing task comprehension.
- `"random_definition"` - Unrelated content for basic positive/negative discrimination.

</div>

<div class="hint" title="Dataset Concatenation and Splitting">

**Merging and splitting best practices**: Use `concatenate_datasets()` to merge multiple triplet datasets, then apply `train_test_split(test_size=1-train_frac, seed=seed, shuffle=True)` for reproducible evaluation. The fixed seed ensures consistent splits across experiments while shuffle=True prevents any ordering bias from affecting training.

</div>

### Other Functions and Motivation
- **`TripletDatasetBuilder` class**  
  Helps you centralize dataset-building logic so scripts stay minimal.
- TripletDatasetBuilder.load_raw
  Read a JSON file into Python safely.

### Notes
Outputs are written to `Data Overview/cache/rag_biencoder/` as an on‑disk Dataset **and** to JSON‑Lines for inspection.

### Scripts
**`run.py`**
- **Purpose**: Glue everything together — load raw JSON, build three types of triplets, merge/split, and write both a HF dataset and JSON-Lines.  
- **Options**:  
  - `--data`: path to raw JSON (default from `config.yaml`)  
  - `--out-dir`: where to save the processed dataset (default from `config.yaml`)  
- **Recommendation**: Run with no flags first to use the defaults:
  ```bash
  python run.py
  ```

**Important**: Please run the script and ensure it worked correctly before moving on to the next task.