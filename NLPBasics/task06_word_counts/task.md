
### Learning Objectives

By completing this task, you will be able to:
- **Understand bag-of-words representation** - Learn how texts can be converted to numerical vectors using word frequency counts
- **Implement vocabulary construction** - Build vocabularies from training data and handle frequency-based feature selection
- **Create sparse feature vectors** - Convert variable-length texts into fixed-dimension vectors suitable for machine learning
- **Apply feature engineering principles** - Understand the trade-offs between vocabulary size, computational efficiency, and information retention
- **Build foundation for advanced methods** - Prepare for more sophisticated approaches like TF-IDF and neural embeddings

### Problem Context

**Bag of Words (BoW)** is one of the most fundamental text representation methods in NLP. Despite its simplicity, it remains effective for many classification tasks and provides the foundation for more advanced techniques.

**Why Bag of Words matters:**
- **Simplicity**: Easy to understand, implement, and debug compared to complex embedding methods
- **Interpretability**: Feature weights directly correspond to word importance, making models explainable
- **Computational efficiency**: Fast to compute and requires minimal memory compared to dense embeddings
- **Baseline performance**: Often provides competitive results, especially with smaller datasets
- **Foundation for advanced methods**: TF-IDF, n-grams, and even transformer attention build on BoW principles

**The representation challenge:**
- Machine learning algorithms require fixed-size numerical inputs, but texts vary in length
- BoW solves this by creating a vocabulary of important words and counting occurrences
- Each document becomes a vector where position i contains the count of vocabulary word i
- This transforms the variable-length text problem into a standard fixed-feature ML problem

**Technical trade-offs:**
- **Word order loss**: "dog bites man" vs "man bites dog" have identical BoW representations
- **Sparsity**: Most documents contain only a small fraction of vocabulary words
- **Vocabulary size**: Larger vocabularies capture more information but increase computational cost
- **Frequency selection**: Using top-k most frequent words balances coverage with efficiency
- **Out-of-vocabulary words**: Test-time words not in training vocabulary are ignored

**Why this approach works:**
For sentiment analysis, the presence/absence and frequency of certain words ("excellent", "terrible", "boring") often matter more than their precise ordering, making BoW surprisingly effective.

## Implementation Requirements

Build a comprehensive `WordCounts` class that implements the complete Bag of Words pipeline from raw text to numerical feature matrices.

### Specific Requirements:

**1. Word Frequency Analysis:**
- `get_words_count(texts)` - Count word occurrences across corpus
  - Split each text into individual words (handle tokenization consistently)
  - Count total occurrences of each unique word across all texts
  - Return `collections.Counter` object for efficient frequency queries
  - Handle edge cases (empty texts, punctuation, case sensitivity)

**2. Vocabulary Construction:**
- `get_bow_vocabulary(texts, word_counts, k)` - Build frequency-based vocabulary
  - Select the k most frequent words from the corpus
  - Return ordered list of vocabulary words (most frequent first)
  - Handle ties in frequency consistently
  - Store vocabulary for consistent feature mapping

**3. Index Mapping Creation:**
- `get_bow_to_id_mapping()` - Create word-to-index dictionary
  - Map each vocabulary word to unique integer index (0 to k-1)
  - Maintain consistent ordering across all conversions
  - Return dictionary for efficient word-to-position lookup

**4. Single Text Conversion:**
- `text_to_bow(text)` - Convert individual text to BoW vector
  - Split text into words and count occurrences of vocabulary words
  - Return fixed-length numpy array (length = vocabulary size)
  - Ignore out-of-vocabulary words (words not in training vocabulary)
  - Handle empty texts and edge cases gracefully

**5. Batch Matrix Computation:**
- `compute_bow_matrix(texts)` - Convert multiple texts to feature matrix
  - Apply `text_to_bow` to each input text
  - Return 2D numpy array (n_texts × vocabulary_size)
  - Ensure consistent dimensionality across all inputs
  - Optimize for efficiency with large text collections

### Expected Deliverables:
- Completed `WordCounts` class with all five core methods
- Consistent vocabulary handling between training and test data
- Efficient sparse representation suitable for machine learning algorithms
- Proper handling of edge cases and out-of-vocabulary words
- Feature matrices ready for classification algorithms

### Examples

```python
>>> from your_code import WordCounts as wc
>>> train_data = pd.DataFrame({'text': ['I love Python', 'Python is great', 'I dislike Java']})
>>> test_data = pd.DataFrame({'text': ['I love programming', 'I dislike Python']})
>>> word_counter = wc(train_data, test_data)

>>> word_count = word_counter.get_words_count(train_data['text'].values)
>>> word_count
Counter({'I': 2, 'love': 1, 'Python': 2, 'is': 1, 'great': 1, 'dislike': 1, 'Java': 1})

>>> bow_vocab = word_counter.get_bow_vocabulary(train_data['text'].values, word_count, k=5)
>>> bow_vocab
['I', 'Python', 'love', 'is', 'great']

>>> bow_to_id = word_counter.get_bow_to_id_mapping()
>>> bow_to_id
{'I': 0, 'Python': 1, 'love': 2, 'is': 3, 'great': 4}

>>> bow_vector = word_counter.text_to_bow("I love Python")
>>> bow_vector
array([1., 1., 1., 0., 0.])

>>> bow_matrix = word_counter.compute_bow_matrix(test_data['text'].values)
>>> bow_matrix
array([[1., 0., 1., 0., 0.],  # "I love programming"
       [1., 1., 0., 0., 0.]]) # "I dislike Python"
```


## Notes

1. **Alternative text vectorization**: Bag of Words is just one approach. TF-IDF, n-grams, and word embeddings are popular alternatives with different trade-offs.

2. **Vocabulary size considerations**: Typical BoW vocabularies range from 1,000 to 10,000 words. Larger vocabularies capture more information but increase computational cost and sparsity.

3. **Preprocessing impact**: The quality of your tokenization and text cleaning directly affects BoW performance. Consistent preprocessing between training and test data is crucial.

4. **Sparsity**: BoW vectors are typically very sparse (mostly zeros). Consider using `scipy.sparse` matrices for memory efficiency with large vocabularies.

<div class="hint" title="Efficient Word Counting">

**Tip**: Use `collections.Counter` for word counting. You can combine multiple counters with `counter1 + counter2` and get most common words with `counter.most_common(k)`.

</div>

<div class="hint" title="Vocabulary Selection">

**Tip**: When selecting top-k words, consider excluding very common words ("the", "and") and very rare words. The `Counter.most_common(k)` method returns (word, count) tuples sorted by frequency.

</div>

<div class="hint" title="Vectorization Efficiency">

**Tip**: For `text_to_bow`, split the text once, then iterate through words to build the count vector. Use the word-to-index mapping for O(1) lookups instead of searching the vocabulary list.

</div>

<div class="hint" title="Matrix Construction">

**Tip**: For `compute_bow_matrix`, you can use list comprehension with `text_to_bow`: `np.array([self.text_to_bow(text) for text in texts])`. This creates the matrix efficiently.

</div>