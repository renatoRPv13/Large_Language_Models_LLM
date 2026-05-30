
### Learning Objectives

By completing this task, you will be able to:
- **Understand word embedding fundamentals** - Learn how words are represented as dense numerical vectors that capture semantic meaning
- **Work with pre-trained embeddings** - Use GloVe vectors trained on large text corpora to represent vocabulary
- **Handle vocabulary limitations** - Deal with out-of-vocabulary words using zero vectors and fallback strategies
- **Convert variable-length text to fixed vectors** - Transform sentences and phrases into consistent vector representations
- **Apply vector averaging techniques** - Combine multiple word vectors into single document representations

### Problem Context

**Word embeddings** are the bridge between human language and machine learning algorithms. They transform discrete words into continuous vector spaces where semantic relationships are preserved through geometric proximity.

**Why embeddings matter:**
- Traditional "bag-of-words" approaches lose semantic meaning and relationships between words
- Pre-trained embeddings like GloVe capture rich semantic knowledge from massive text corpora
- Vector representations enable mathematical operations on text (similarity, analogy, clustering)
- Embeddings form the foundation for all modern NLP models, including transformers

**The variable-length text challenge:**
- Machine learning algorithms require fixed-size inputs, but sentences have varying lengths
- Simple concatenation creates unwieldy, sparse feature spaces
- **Vector averaging** provides a simple but effective solution for creating sentence-level representations
- This approach, while basic, often performs surprisingly well for classification tasks

**Technical challenges we'll address:**
- **Vocabulary coverage**: Not all words appear in pre-trained embeddings
- **Handling unknowns**: Developing strategies for out-of-vocabulary words
- **Aggregation methods**: Converting word-level vectors to document-level representations
- **Efficiency**: Working with large embedding matrices and processing batches of text

We'll use **Gensim**, a robust NLP library that provides easy access to various pre-trained embedding models including GloVe (Global Vectors for Word Representation).


## Implementation Requirements

Build a comprehensive `GloVeEmbeddings` class that handles word-to-vector conversion and phrase-level aggregation.

### Specific Requirements:

**1. Word-Level Vector Retrieval:**
- `get_word_vectors(words)` - Extract embeddings for individual words
  - Return numpy array of word vectors for input word list
  - Handle out-of-vocabulary words by returning zero vectors of appropriate dimension
  - Maintain consistent vector dimensions across all outputs
  - Preserve input order in output array

**2. Phrase-Level Embedding Generation:**
- `get_phrase_embedding(phrase)` - Convert text to single vector representation
  - Split phrase into tokens and retrieve individual word vectors
  - Average available word vectors to create phrase representation
  - Ignore out-of-vocabulary words in averaging (don't include zeros in mean)
  - Return zero vector if no words are found in vocabulary
  - Handle edge cases (empty strings, punctuation-only text)

**3. Batch Processing with Token Limits:**
- `compute_phrase_vectors(phrases, max_tokens=None)` - Efficient batch processing
  - Process multiple phrases simultaneously for efficiency
  - Apply optional token limiting (truncate if phrase exceeds max_tokens)
  - Return 2D numpy array where each row is a phrase vector
  - Maintain consistent output dimensions across all phrases

### Expected Deliverables:
- Completed `GloVeEmbeddings` class with robust error handling
- Efficient vector operations using numpy for batch processing
- Proper handling of edge cases (empty inputs, unknown words, long phrases)
- Consistent output format suitable for downstream machine learning tasks

### Examples

```python
>>> from your_code import GloVeEmbeddings as ge
>>> emb = ge('glove-twitter-25')
>>> words = ['hello', 'world', 'unknownword']
>>> emb.get_word_vectors(words)
array([[ 0.345,  0.567,  ...],   # Sample output for known word "hello"
       [ 0.789,  0.123,  ...],   # Sample output for known word "world"
       [ 0.,     0.,     ...]])  # Zero vector for unknown word

>>> phrase = "I love Python"
>>> emb.get_phrase_embedding(phrase)
array([ 0.234,  0.789,  ...])  # Averaged word vectors for "I", "love", and "Python"

>>> phrases = ["I love Python", "AI is amazing"]
>>> emb.compute_phrase_vectors(phrases, max_tokens=2)
array([[ 0.567,  0.890,  ...],   # Sample output for truncated phrase "I love"
       [ 0.345,  0.234,  ...]])  # Sample output for truncated phrase "AI is"]
```


## Notes

1. **Model downloading**: The GloVe model will be downloaded automatically on first use and saved to `./.gensim_data` directory. This may take several minutes depending on your connection.

2. **Environment setup**: To override this behavior, set the `GENSIM_DATA_DIR` environment variable to your preferred directory.

3. **Pre-trained vs. custom embeddings**: While you can enhance pre-trained vectors with domain-specific data, gensim doesn't provide this functionality directly since context vectors aren't included in downloaded models.

<div class="hint" title="macOS certificate issues">
If you encounter SSL/certificate errors on macOS, try:
<code>
   pip install -U certifi
   /Applications/Python\ 3.X/Install\ Certificates.command
</code>
</div>
<div class="hint" title="Handling Unknown Words">

**Tip**: When averaging word vectors, keep track of how many known words you're averaging over. Use `np.mean()` only on the vectors for known words, and return a zero vector if no words are found in the vocabulary.

</div>

<div class="hint" title="Efficient Vector Operations">

**Tip**: Use numpy operations for efficiency. When getting multiple word vectors, you can stack them into a 2D array. For averaging, `np.mean(axis=0)` will average across words while preserving the embedding dimension.

</div>

<div class="hint" title="Token Limiting Strategy">

**Tip**: For `max_tokens` parameter, consider splitting the phrase into tokens first, then slice the list: `tokens[:max_tokens]`. This ensures you process only the first N tokens of each phrase consistently.

</div>

<div class="hint" title="Gensim Model Loading">

**Tip**: Load your GloVe model using `gensim.downloader.load('glove-twitter-25')`. The model object has a `get_vector()` method to retrieve individual word vectors and an `__contains__` method to check if words exist in vocabulary.

</div>