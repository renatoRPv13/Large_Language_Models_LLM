
### Learning Objectives

By completing this task, you will be able to:
- **Understand tokenization fundamentals** - Learn why simple string splitting is insufficient for real-world text data
- **Implement robust text preprocessing** - Handle HTML artifacts, punctuation, and case normalization
- **Apply NLTK tokenization tools** - Use industry-standard library functions for text segmentation
- **Process datasets efficiently** - Apply preprocessing functions to entire DataFrames using pandas operations
- **Recognize preprocessing trade-offs** - Understand what information is lost and preserved during tokenization

### Problem Context

**Tokenization** is the foundational step that transforms raw text into structured units that machine learning algorithms can process. However, real-world text data is messy and requires sophisticated preprocessing.

**Why tokenization matters:**
- Raw text contains punctuation, HTML tags, and inconsistent formatting that confuses algorithms
- Simple `str.split()` fails with contractions, punctuation, and special characters
- Proper tokenization significantly impacts the performance of downstream NLP models
- Consistent preprocessing ensures reproducible results across different text sources

**Challenges with our movie review data:**
- **HTML artifacts**: Reviews contain `<br />` tags from web scraping
- **Emphasis patterns**: Capitalized words like "AMAZING" or "TERRIBLE" express sentiment
- **Named entities**: Movie titles, actor names, and proper nouns need consistent handling
- **Punctuation attachment**: Words like "great!" or "don't" require intelligent splitting

We'll use **NLTK (Natural Language Toolkit)**, a comprehensive library that provides robust solutions for these tokenization challenges.

## Implementation Requirements

Create a robust `Tokenizer` class that handles the preprocessing pipeline for movie review text data.

### Specific Requirements:

**1. Core Tokenization Method:**
- `tokenize(text)` - Convert raw text into clean token list
  - Convert to lowercase for consistency
  - Remove HTML `<br />` tags 
  - Split into individual tokens using appropriate NLTK functions
  - Handle punctuation and contractions properly

**2. Text Preprocessing Pipeline:**
- `preprocess_text(text)` - Complete preprocessing workflow
  - Apply tokenization to input text
  - Rejoin tokens into clean, normalized string
  - Maintain readability while ensuring consistency

**3. Batch Processing Method:**
- `apply_preprocess(dataframe)` - Efficient DataFrame processing
  - Process entire 'text' column using vectorized operations
  - Return DataFrame with cleaned text ready for feature extraction
  - Handle potential errors gracefully (empty texts, special characters)

### Expected Deliverables:
- Completed `Tokenizer` class with all three methods
- Properly cleaned text that maintains semantic meaning
- Efficient processing suitable for large datasets
- Consistent output format across all input variations

### Examples

```python
>>> import pandas as pd
>>> from your_code import Tokenizer
>>> tokenizer = Tokenizer()

# Tokenizing a simple text
>>> tokenizer.tokenize('I LOVE this!<br />But it could be better.')
['i', 'love', 'this', 'but', 'it', 'could', 'be', 'better']

# Preprocessing text by tokenizing and joining
>>> tokenizer.preprocess_text('I LOVE this!<br />But it could be better.')
'i love this but it could be better'

# Applying preprocessing to a DataFrame
>>> data = pd.DataFrame({'text': ['I LOVE this!<br />But it could be better.', 'Great movie!']})
>>> tokenizer.apply_preprocess(data)
                             text
0  i love this but it could be better
1                        great movie
```

## Notes

1. **NLTK setup**: Make sure to download required NLTK data using `nltk.download('punkt')` if you encounter tokenization errors.
2. **Trade-offs to consider**: Converting to lowercase helps with consistency but loses information about emphasis and proper nouns.
3. **HTML handling**: The `<br />` tags are artifacts from web scraping and should be removed completely.

<div class="hint" title="NLTK Tokenization">

**Tip**: Use `nltk.word_tokenize()` for robust tokenization that handles punctuation and contractions better than simple splitting. Don't forget to import nltk and ensure the punkt tokenizer is downloaded.

</div>

<div class="hint" title="String Replacement">

**Tip**: Python's `str.replace()` method is efficient for removing HTML tags. Consider the sequence of operations: should you remove HTML tags before or after tokenization?

</div>

<div class="hint" title="DataFrame Processing">

**Tip**: When using `pd.Series.apply()`, remember that it applies the function to each individual element in the series. Make sure your `preprocess_text` function can handle single string inputs.

</div>
