### Learning objectives

By completing this task, you will be able to:
- **Implement efficient vector similarity search** - Build in-memory vector stores using normalized embeddings and cosine similarity for fast semantic search.
- **Master embedding normalization techniques** - Apply L2 normalization to enable efficient cosine similarity computation through dot products.
- **Design scalable retrieval systems** - Create modular vector storage components that support persistence, caching, and high-level search interfaces.
- **Optimize search algorithms** - Use numpy operations like `argpartition` and `argsort` for efficient top-k retrieval from large embedding collections.

### Problem context

Vector similarity search forms the core of modern retrieval systems, enabling semantic matching between queries and large document collections. Unlike keyword-based search, vector stores capture semantic meaning through learned embeddings, allowing systems to find relevant information even when exact words don't match.

**Why vector storage optimization matters:**
- **Search speed requirements** - Production RAG systems must retrieve relevant documents from millions of candidates in milliseconds.
- **Memory efficiency** - Storing and searching large embedding collections requires careful optimization of data structures and algorithms.
- **Similarity computation accuracy** - Proper normalization ensures that cosine similarity reflects true semantic relationships between texts.
- **System scalability** - Well-designed vector stores support incremental updates, persistence, and distributed deployment patterns.

**What makes this challenging:**
- **Numerical stability** - Proper handling of zero vectors and normalization edge cases prevents runtime errors and maintains search quality.
- **Algorithm efficiency** - Naive similarity computation scales quadratically; efficient implementations use optimized numpy operations and approximate search.
- **Memory management** - Large embedding matrices require careful consideration of data types, storage formats, and memory access patterns.
- **Abstraction design** - Creating clean APIs that hide implementation complexity while supporting diverse search use cases and deployment scenarios.

## Task – In-Memory Vector Store

Implement an efficient in-memory vector storage system with cosine similarity search for semantic retrieval in RAG applications.

### Implementation requirements

#### `SimpleVectorStore._normalize_embeddings`
Normalize embedding vectors to unit length for efficient cosine similarity computation.

**Core functionality:**
- Compute row-wise L2 norms using `np.linalg.norm(axis=1, keepdims=True)`.
- Handle zero vectors by replacing zero norms with 1.0 to avoid division errors.
- Normalize embeddings in-place by dividing by their norms.

#### `SimpleVectorStore._encode_and_normalize`
Encode input text and return a normalized embedding vector.

**Core functionality:**
- Use `model.encode()` with `convert_to_numpy=True` to get embeddings.
- Apply L2 normalization to the resulting vector.
- Handle edge case where norm is zero by returning the original vector.

#### `SimpleVectorStore.search`
Perform efficient top-k similarity search using normalized vector operations.

**Core functionality:**
- Handle edge case where `k` exceeds the number of stored vectors.
- Encode and normalize the query using `_encode_and_normalize()`.
- Compute cosine similarities using dot product (`@` operator) with normalized vectors.
- Use `np.argpartition(-scores, k-1)[:k]` for efficient top-k selection.
- Apply `np.argsort(-scores[idx])` to get final ranking order.

#### `Searcher`
High-level interface that combines text storage with vector search capabilities.

**Core functionality:**
- Initialize with text data and model, automatically creating embeddings.
- Provide `search(query, k)` method returning `List[(text, score)]` tuples.
- Support persistence through `save()` and `load()` methods using pickle serialization.
- Handle model loading and embedding generation with progress indicators.

<div class="hint" title="Vector Normalization">

**Why normalization matters**: Normalizing embeddings to unit length allows cosine similarity computation using simple dot products (`@` operator). This transforms the computationally expensive cosine similarity formula into an efficient linear algebra operation. Remember to handle zero vectors by setting their norm to 1.0 before division.

</div>

<div class="hint" title="Efficient Top-K Search">

**Optimizing similarity search**: Use `np.argpartition(-scores, k-1)[:k]` to efficiently find the top-k elements without fully sorting the entire array. This is O(n) instead of O(n log n). Then apply `np.argsort(-scores[idx])` only on the selected k elements to get the final ranking.

</div>


### Other Functions and Motivation
- **`load_definitions(path)`** - Read a JSON list of strings—your knowledge base.
- **`load_model(path)`** - Instantiate a `SentenceTransformer` from disk for search.

### Scripts
**`run.py`**  
- **Purpose**: Either load a cached store or build a new one, then run a query.  
- **Options**:  
  - `--model`: path to your fine-tuned model (default)  
  - `--definitions`: JSON file of definitions (default)  
  - `--query`: the word to search for  
  - `--top-k`: how many results to show  
  - `--save` / `--reuse`: cache control  
- **Recommendation**: Play with the script parameters
  ```bash
  python run.py --save
  ```
  *Note:* When launching with `--save`, it will create a new vector store and save it. Please don't forget to save it before moving on to the next task.

**Important**: Please run the script and ensure it worked correctly before moving on to the next task.
