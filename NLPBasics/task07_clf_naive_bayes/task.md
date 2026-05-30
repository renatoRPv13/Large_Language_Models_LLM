
### Learning Objectives

By completing this task, you will be able to:
- **Understand probabilistic classification** - Learn how Naive Bayes uses probability distributions to make predictions
- **Implement conditional independence assumptions** - Apply the "naive" assumption that features are independent given the class
- **Calculate marginal and conditional probabilities** - Estimate probabilities from training data using frequency counts
- **Apply smoothing techniques** - Handle zero probabilities using additive smoothing (Laplace smoothing)
- **Interpret model predictions** - Analyze which features (words) are most discriminative for each class

### Problem Context

**Naive Bayes** is one of the most elegant and effective algorithms for text classification. Despite its "naive" assumption that words are independent, it performs remarkably well in practice and provides interpretable results.

**Why Naive Bayes excels at text classification:**
- **Probabilistic foundation**: Makes predictions based on learned probability distributions from training data
- **Handles sparse data**: Works well with high-dimensional, sparse bag-of-words features
- **Computationally efficient**: Training and prediction are fast, making it suitable for large datasets
- **Interpretable**: You can easily see which words contribute most to positive vs. negative predictions
- **Robust to irrelevant features**: Performs well even when many features are noise

**The mathematical foundation:**
Naive Bayes applies Bayes' theorem with the "naive" assumption of feature independence:
- **Bayes' theorem**: <math><mi>P</mi><mo>(</mo><mi>class</mi><mo>|</mo><mi>features</mi><mo>)</mo><mo>∝</mo><mi>P</mi><mo>(</mo><mi>features</mi><mo>|</mo><mi>class</mi><mo>)</mo><mo>×</mo><mi>P</mi><mo>(</mo><mi>class</mi><mo>)</mo></math>
- **Naive assumption**: <math><mi>P</mi><mo>(</mo><mi>features</mi><mo>|</mo><mi>class</mi><mo>)</mo><mo>=</mo><mo>∏</mo><mi>P</mi><mo>(</mo><msub><mi>feature</mi><mi>i</mi></msub><mo>|</mo><mi>class</mi><mo>)</mo></math>
- **Classification rule**: Choose class with highest posterior probability

**Key components:**
- **Prior probabilities**: <math><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math> - how common each class is in training data
- **Conditional probabilities**: <math><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math> - how likely each word is given the class
- **Smoothing**: Add small <math><mi>δ</mi></math> to avoid zero probabilities for unseen word-class combinations
- **Log probabilities**: Use log-space to avoid numerical underflow with many small probabilities

**Model interpretability:**
After training, you can analyze which words are most discriminative by examining the ratio <math><mfrac><mrow><mi>P</mi><mo>(</mo><mi>word</mi><mo>|</mo><mi>positive</mi><mo>)</mo></mrow><mrow><mi>P</mi><mo>(</mo><mi>word</mi><mo>|</mo><mi>negative</mi><mo>)</mo></mrow></mfrac></math>. Words with high ratios strongly indicate positive sentiment, while words with low ratios indicate negative sentiment.

**Why this approach works:**
For sentiment analysis, the presence of words like "excellent", "amazing", "terrible", or "boring" provides strong signals regardless of their exact order or context, making the independence assumption reasonable.

## Implementation Requirements

Build a complete `BinaryNaiveBayes` classifier that implements the probabilistic learning and prediction pipeline.

### Specific Requirements:

**1. Model Training:**
- `fit(X, y, delta=1.0)` - Learn probability distributions from training data
  - Calculate prior probabilities: <math><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math> for each class <math><mi>k</mi><mo>∈</mo><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>}</mo></math>
  - Calculate conditional probabilities: <math><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math> for each feature i and class k
  - Apply additive smoothing with parameter <math><mi>δ</mi></math> to handle zero counts
  - Store learned parameters for prediction phase
  - Handle edge cases (empty classes, all-zero feature vectors)

**2. Probability Score Computation:**
- `predict_scores(X)` - Compute log-probability scores for classification
  - Calculate <math><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>|</mo><mi>X</mi><mo>)</mo></math> for each class using Bayes' theorem
  - Use log-space arithmetic to avoid numerical underflow
  - Apply naive independence assumption: <math><mi>log</mi><mi>P</mi><mo>(</mo><mi>X</mi><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo><mo>=</mo><mo>∑</mo><mi>log</mi><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math>
  - Return matrix of shape [n_samples, 2] with log-scores for both classes
  - Ensure numerical stability with appropriate handling of zero probabilities

**3. Classification Prediction:**
- `predict(X)` - Make binary class predictions
  - Use `predict_scores` to get log-probability scores for both classes
  - Select class with higher log-probability (argmax)
  - Return array of predicted class labels {0, 1}
  - Handle ties consistently (e.g., prefer positive class)

### Expected Deliverables:
- Completed `BinaryNaiveBayes` class with all three core methods
- Proper implementation of additive smoothing to handle unseen words
- Numerically stable log-probability computations
- Efficient matrix operations for batch prediction
- Model that can identify most discriminative words for interpretability analysis

### Mathematical Details:
- **Prior**: <math><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>count of class k</mi><mo>+</mo><mi>δ</mi></mrow><mrow><mi>total samples</mi><mo>+</mo><mn>2</mn><mi>δ</mi></mrow></mfrac></math>
- **Conditional**: <math><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>count of feature i in class k</mi><mo>+</mo><mi>δ</mi></mrow><mrow><mi>total feature counts in class k</mi><mo>+</mo><mi>vocabulary_size</mi><mo>×</mo><mi>δ</mi></mrow></mfrac></math>
- **Prediction**: <math><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>|</mo><mi>X</mi><mo>)</mo><mo>=</mo><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo><mo>+</mo><mo>∑</mo><msub><mi>x</mi><mi>i</mi></msub><mo>×</mo><mi>log</mi><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>=</mo><mi>k</mi><mo>)</mo></math>

### Examples

```python
>>> from your_code import BinaryNaiveBayes as bnb
>>> X_train = np.array([[1, 0, 2], [0, 1, 0], [1, 1, 1]])  # Example BoW feature vectors
>>> y_train = np.array([0, 1, 0])  # Binary labels
>>> classifier = bnb()
>>> classifier.fit(X_train, y_train)

>>> X_test = np.array([[1, 0, 1], [0, 1, 0]])  # Test BoW feature vectors
>>> scores = classifier.predict_scores(X_test)
>>> scores
array([[-1.234, -2.345],  # Log-scores for class 0 and 1
       [-1.789, -1.123]])

>>> predictions = classifier.predict(X_test)
>>> predictions
array([0, 1])  # Predicted classes for each sample
```


## Notes

1. **Mathematical foundation**: If you need to refresh your understanding of Naive Bayes mathematics, read [this excellent explanation](https://lena-voita.github.io/nlp_course/text_classification.html#naive_bayes) before implementing.

2. **Smoothing importance**: The <math><mi>δ</mi></math> parameter prevents zero probabilities which would make <math><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>|</mo><mi>X</mi><mo>)</mo><mo>=</mo><mo>-</mo><mi>∞</mi></math>. Start with <math><mi>δ</mi><mo>=</mo><mn>1.0</mn></math> (Laplace smoothing).

3. **Log-space arithmetic**: Always work in log space to avoid numerical underflow when multiplying many small probabilities.

4. **Feature interpretability**: After training, you can compute word importance ratios <math><mfrac><mrow><mi>P</mi><mo>(</mo><mi>word</mi><mo>|</mo><mi>negative</mi><mo>)</mo></mrow><mrow><mi>P</mi><mo>(</mo><mi>word</mi><mo>|</mo><mi>positive</mi><mo>)</mo></mrow></mfrac></math> to find the top-25 most discriminative words for each class.

<div class="hint" title="Probability Estimation">

**Tip**: For conditional probabilities, count how many times each word appears in each class, then normalize by total word counts in that class. Don't forget to add smoothing: `(word_count_in_class + delta) / (total_words_in_class + vocab_size * delta)`.

</div>

<div class="hint" title="Log-Space Computations">

**Tip**: Instead of multiplying probabilities, add log-probabilities: <math><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>|</mo><mi>X</mi><mo>)</mo><mo>=</mo><mi>log</mi><mi>P</mi><mo>(</mo><mi>y</mi><mo>)</mo><mo>+</mo><mo>∑</mo><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>×</mo><mi>log</mi><mi>P</mi><mo>(</mo><msub><mi>x</mi><mi>i</mi></msub><mo>|</mo><mi>y</mi><mo>)</mo><mo>)</mo></math> where <math><msub><mi>x</mi><mi>i</mi></msub></math> is the count of word i in the document.

</div>

<div class="hint" title="Efficient Matrix Operations">

**Tip**: Use numpy broadcasting for batch predictions. Store log-probabilities as matrices and use `np.dot()` to compute scores for all samples at once: `X @ log_probs.T`.

</div>

<div class="hint" title="Handling Smoothing">

**Tip**: When computing conditional probabilities, the denominator should be `total_word_count_in_class + vocabulary_size * delta` to ensure probabilities sum to 1 after smoothing.

</div>