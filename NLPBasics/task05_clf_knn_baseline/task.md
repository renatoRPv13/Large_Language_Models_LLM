
### Learning Objectives

By completing this task, you will be able to:
- **Understand instance-based learning** - Learn how kNN makes predictions based on similarity to training examples
- **Implement cosine similarity** - Calculate semantic similarity between text vectors using cosine distance
- **Apply majority voting** - Combine predictions from multiple neighbors to make final classification decisions
- **Evaluate classifier performance** - Measure accuracy and understand baseline model performance
- **Handle similarity-based retrieval** - Efficiently find nearest neighbors in high-dimensional embedding spaces

### Problem Context

**k-Nearest Neighbors (kNN)** is one of the simplest yet surprisingly effective machine learning algorithms. It makes predictions by finding the most similar examples in the training data and taking a majority vote among their labels.

**Why kNN works well for text classification:**
- **Intuitive approach**: Similar texts often have similar sentiments or categories
- **Non-parametric**: No assumptions about data distribution, adapts to local patterns
- **Interpretable**: You can inspect which training examples influenced each prediction
- **Effective baseline**: Often provides competitive performance with minimal complexity
- **Works with embeddings**: Leverages semantic similarity captured in word vectors

**The kNN algorithm for sentiment classification:**
1. **Query representation**: Convert new review to embedding vector
2. **Similarity calculation**: Compute cosine similarity with all training examples
3. **Neighbor selection**: Find k most similar training reviews
4. **Majority vote**: Predict the most common label among neighbors
5. **Confidence**: Proportion of neighbors supporting the prediction

**Technical challenges:**
- **Computational complexity**: O(n) similarity calculations for each prediction
- **Similarity metric choice**: Cosine similarity works well for text but other metrics exist
- **k parameter selection**: Too small (overfitting) vs. too large (oversmoothing)
- **Curse of dimensionality**: High-dimensional spaces can make distance metrics less meaningful
- **Data truncation**: Large datasets require sampling for computational efficiency

**Why this serves as a baseline:**
kNN provides an excellent baseline because it's simple to understand and implement, yet often competitive with more complex methods. Understanding its performance helps evaluate whether sophisticated models provide meaningful improvements.

## Implementation Requirements

Build a complete `KNNClassifier` that implements the k-nearest neighbors algorithm for text classification using semantic embeddings.

### Specific Requirements:

**1. Similarity Computation:**
- `cos_sim(a, b)` - Calculate cosine similarity between vectors
  - Implement cosine similarity formula: `(a · b) / (||a|| × ||b||)`
  - Handle edge cases: zero vectors, identical vectors, numerical precision
  - Return value between -1 and 1 (higher values indicate greater similarity)
  - Use efficient numpy operations for vector computations

**2. Nearest Neighbor Retrieval:**
- `find_nearest(query, k)` - Find k most similar training examples
  - Convert query text to embedding using provided `get_phrase_embedding` function
  - Compute cosine similarities between query and all training vectors
  - Return k training examples with highest similarity scores
  - Return both text content and corresponding labels for neighbors
  - Handle case where k > number of training examples

**3. Classification and Evaluation:**
- `get_accuracy(test_texts, test_labels, k)` - Evaluate classifier performance
  - For each test example, find k nearest neighbors
  - Predict label using majority vote among neighbors (mode of neighbor labels)
  - Handle ties in voting (e.g., use random selection or prefer positive class)
  - Calculate overall accuracy as proportion of correct predictions
  - Return single accuracy score between 0 and 1

### Expected Deliverables:
- Completed `KNNClassifier` class with all three core methods
- Efficient implementation suitable for reasonably-sized datasets
- Proper error handling for edge cases (empty inputs, invalid k values)
- Accurate majority voting that handles ties consistently
- Performance evaluation that matches standard accuracy metrics

### Examples

```python
>>> from your_code import KNNClassifier as knn
>>> train_data = pd.DataFrame({'text': ['I love Python', 'Python is great', 'I dislike Java'], 'label': [1, 1, 0]})
>>> test_data = pd.DataFrame({'text': ['I love programming', 'I dislike Python'], 'label': [1, 0]})
>>> get_embedding = lambda text: np.random.rand(25)  # Mock embedding function
>>> classifier = knn(train_data, test_data, get_embedding)

>>> sim = classifier.cos_sim(np.array([1, 0, 1]), np.array([0, 1, 0]))
>>> sim
0.0

>>> nearest_texts, nearest_labels = classifier.find_nearest("I love programming", k=2)
>>> nearest_texts
array(['I love Python', 'Python is great'], dtype=object)
>>> nearest_labels
array([1, 1])

>>> accuracy = classifier.get_accuracy(test_data['text'], test_data['label'], k=2)
>>> accuracy
0.5
```


## Notes

1. **Computational complexity**: kNN is O(n) for each prediction, making it slow for large datasets. We truncate the training data to maintain reasonable performance.

2. **k parameter selection**: Start with k=5 as a reasonable default. Odd values help avoid ties in binary classification.

3. **Distance vs. similarity**: We use cosine similarity (higher values = more similar) rather than cosine distance (lower values = more similar).

4. **Baseline importance**: kNN often provides surprisingly competitive performance and serves as an excellent baseline for more complex models.

<div class="hint" title="Cosine Similarity Implementation">

**Tip**: Use `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` for cosine similarity. Handle zero vectors by checking if either norm is zero before division.

</div>

<div class="hint" title="Efficient Neighbor Finding">

**Tip**: Instead of sorting all similarities, use `np.argpartition(-similarities, k)[:k]` to find the k largest values efficiently, then sort just those k elements.

</div>

<div class="hint" title="Majority Vote Implementation">

**Tip**: Use `scipy.stats.mode()` or `collections.Counter` to find the most common label among neighbors. Handle ties by choosing randomly or preferring the positive class.

</div>

<div class="hint" title="Vectorized Operations">

**Tip**: When computing similarities between a query and all training vectors, use matrix operations: `np.dot(query, training_vectors.T)` to compute all dot products at once.

</div>
