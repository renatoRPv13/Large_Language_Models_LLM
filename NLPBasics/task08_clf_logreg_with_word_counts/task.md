### Learning Objectives

By completing this task, you will be able to:
- **Understand linear classification** - Learn how logistic regression creates decision boundaries using linear combinations of features
- **Apply feature scaling techniques** - Use StandardScaler to normalize bag-of-words features for optimal model performance
- **Implement logistic regression pipeline** - Build complete training and evaluation workflow with proper data preprocessing
- **Evaluate classification performance** - Calculate accuracy metrics and visualize model performance using ROC curves
- **Interpret ROC analysis** - Understand Area Under Curve (AUC) as a measure of classification quality

### Problem Context

**Logistic Regression** is a fundamental linear classifier that's particularly well-suited for text classification tasks. Despite its simplicity, it often provides competitive performance and serves as a strong baseline for more complex models.

**Why logistic regression works well for text:**
- **Linear decision boundaries**: Creates interpretable decision rules based on weighted combinations of word features
- **Probabilistic output**: Provides calibrated probabilities, not just binary classifications
- **Handles sparse data**: Works efficiently with high-dimensional, sparse bag-of-words representations
- **Fast training**: Converges quickly even on large datasets with many features
- **Regularization support**: Built-in L1/L2 regularization prevents overfitting with many features

**The mathematical foundation:**
Logistic regression models the probability of positive class membership using the sigmoid function:
- **Sigmoid transformation**: <math><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>|</mo><mi>X</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>-</mo><mo>(</mo><msub><mi>β</mi><mn>0</mn></msub><mo>+</mo><msub><mi>β</mi><mn>1</mn></msub><msub><mi>x</mi><mn>1</mn></msub><mo>+</mo><mo>...</mo><mo>+</mo><msub><mi>β</mi><mi>p</mi></msub><msub><mi>x</mi><mi>p</mi></msub><mo>)</mo></mrow></msup></mrow></mfrac></math>
- **Linear combination**: <math><mi>z</mi><mo>=</mo><msub><mi>β</mi><mn>0</mn></msub><mo>+</mo><mo>∑</mo><msub><mi>β</mi><mi>i</mi></msub><msub><mi>x</mi><mi>i</mi></msub></math> represents the log-odds of positive classification
- **Decision boundary**: Classification threshold typically set at P = 0.5 (z = 0)

**Feature scaling importance:**
Bag-of-words features have vastly different scales (rare vs. common words), making feature scaling crucial:
- **StandardScaler**: Transforms features to have zero mean and unit variance
- **Improved convergence**: Scaled features help optimization algorithms converge faster
- **Fair feature weighting**: Prevents high-frequency words from dominating due to scale alone

**ROC curve analysis:**
Receiver Operating Characteristic curves provide comprehensive performance evaluation:
- **True Positive Rate vs. False Positive Rate**: Shows classifier performance across all thresholds
- **Area Under Curve (AUC)**: Single metric summarizing overall classification quality (0.5 = random, 1.0 = perfect)
- **Threshold selection**: Helps choose optimal classification threshold for specific use cases

## Implementation Requirements

Build a complete `MyLogisticRegression` class that implements feature scaling, model training, and comprehensive evaluation with ROC analysis.

### Specific Requirements:

**1. Feature Scaling:**
- `_scale_features(X_train, X_test)` - Normalize features for optimal training
  - Apply `StandardScaler` to ensure zero mean and unit variance
  - Fit scaler on training data only to prevent data leakage
  - Transform both training and test data using same scaling parameters
  - Return scaled versions of both datasets
  - Handle edge cases (constant features, very small variance)

**2. Model Training and Evaluation:**
- `eval_model(X_train, y_train, X_test, y_test)` - Complete ML pipeline
  - Scale features using `_scale_features` method
  - Train `LogisticRegression` model on scaled training data
  - Calculate accuracy on both training and test sets
  - Generate probability predictions for ROC analysis
  - Plot ROC curves for both training and test data
  - Compute and display AUC (Area Under Curve) values
  - Return trained model for further analysis

### Expected Deliverables:
- Completed `MyLogisticRegression` class with both core methods
- Proper feature scaling that prevents data leakage
- Comprehensive evaluation including accuracy and ROC analysis
- Clear visualization showing model performance on both datasets
- Trained model ready for inference on new data

### Performance Expectations:
- Training accuracy should typically be higher than test accuracy
- AUC values above 0.7 indicate reasonable classification performance
- ROC curves should show clear separation from the diagonal (random performance line)
- Significant gap between training and test performance may indicate overfitting

### Examples

```python
>>> from your_code import MyLogisticRegression as mlr
>>> X_train = np.array([[1, 0], [0, 1], [1, 1]])  # Example BoW feature vectors
>>> y_train = np.array([0, 1, 1])  # Binary labels
>>> X_test = np.array([[0, 0], [1, 0]])  # Test BoW feature vectors
>>> y_test = np.array([0, 1])

>>> model = mlr()
>>> model.eval_model(X_train, y_train, X_test, y_test)
train accuracy: 1.000
test accuracy: 0.500
```

## Notes

1. **Feature scaling importance**: Logistic regression is sensitive to feature scales. StandardScaler ensures all features contribute equally to the optimization process.

2. **ROC curve interpretation**: The ROC curve plots True Positive Rate vs. False Positive Rate. A curve closer to the top-left corner indicates better performance.

3. **AUC significance**: AUC = 0.5 means random performance, AUC = 1.0 means perfect classification. Values above 0.7 are generally considered good for text classification.

4. **Overfitting detection**: Large gaps between training and test accuracy/AUC indicate overfitting. Consider regularization if this occurs.

<div class="hint" title="Feature Scaling Implementation">

**Tip**: Use `sklearn.preprocessing.StandardScaler`. Fit it on training data with `scaler.fit(X_train)`, then transform both datasets: `X_train_scaled = scaler.transform(X_train)` and `X_test_scaled = scaler.transform(X_test)`.

</div>

<div class="hint" title="ROC Curve Plotting">

**Tip**: Use `sklearn.metrics.roc_curve` and `roc_auc_score`. Get probability predictions with `model.predict_proba(X)[:, 1]`, then plot with `plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')`.

</div>

<div class="hint" title="Model Training">

**Tip**: Use `sklearn.linear_model.LogisticRegression` with default parameters. For bag-of-words data, consider setting `max_iter=1000` to ensure convergence with sparse features.

</div>

<div class="hint" title="Evaluation Metrics">

**Tip**: Calculate accuracy with `sklearn.metrics.accuracy_score(y_true, y_pred)`. For probability predictions needed for ROC, use `model.predict_proba(X)` instead of `model.predict(X)`.

</div>

