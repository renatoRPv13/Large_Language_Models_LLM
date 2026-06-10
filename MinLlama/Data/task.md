### Learning objectives

By completing this task, you will be able to:
- **Understand evaluation methodologies** - Learn how different datasets test various aspects of language model performance and robustness.
- **Work with sentiment analysis benchmarks** - Gain experience with standard NLP evaluation datasets and their characteristics.
- **Apply zero-shot vs. fine-tuning approaches** - Compare different methods for adapting language models to downstream tasks.
- **Interpret model performance** - Understand how to analyze results across different evaluation scenarios and datasets.

### Problem context

Evaluating large language models requires carefully designed datasets that test different aspects of language understanding. Simple accuracy metrics on a single dataset don't capture the full picture of a model's capabilities, especially for complex tasks like sentiment analysis.

**Why comprehensive evaluation matters:**
- **Real-world performance** - Models must handle diverse text styles, domains, and edge cases encountered in practical applications.
- **Robustness assessment** - Understanding how models perform when faced with subtle variations or adversarial inputs.
- **Transfer learning validation** - Comparing zero-shot capabilities against task-specific fine-tuning to understand adaptation trade-offs.
- **Benchmarking standards** - Using established datasets allows comparison with other models and approaches in the literature.

**What makes this challenging:**
- **Dataset diversity** - Different datasets test different aspects of language understanding, requiring nuanced interpretation.
- **Evaluation methodology** - Choosing appropriate metrics and understanding their limitations for different types of language tasks.
- **Performance interpretation** - Understanding why models succeed or fail on specific examples and what this reveals about their capabilities.
- **Baseline establishment** - Knowing what constitutes good performance for different types of models and training approaches.

## Implementation requirements

Your task is to evaluate your LLaMA implementation using two complementary sentiment analysis datasets that test different aspects of model performance. Understanding these datasets is crucial for proper model evaluation and interpretation.

### Specific requirements:

1. **Understand dataset characteristics** - Learn the structure, labeling schemes, and intended use cases for each evaluation dataset.
2. **Apply multiple evaluation approaches** - Test both zero-shot prompting and fine-tuned classification methods.
3. **Interpret performance differences** - Understand why models perform differently across datasets and approaches.
4. **Handle evaluation edge cases** - Work with datasets that have missing labels or non-standard formatting.
5. **Compare model capabilities** - Analyze the trade-offs between different model adaptation strategies.

### Expected deliverables:

- Successful evaluation of your LLaMA model on both provided datasets.
- Understanding of why different evaluation approaches yield different results.
- Ability to interpret model performance in the context of dataset characteristics.
- Proper handling of evaluation data formatting and missing labels.

## Dataset specifications

### SST-5 (Stanford Sentiment Treebank)

**Dataset characteristics:**
- **Purpose**: Fine-grained sentiment analysis evaluation.
- **Domain**: Movie reviews from critics and audiences.
- **Label scheme**: Five-class sentiment classification (very negative, negative, neutral, positive, very positive).
- **Complexity**: Requires understanding of nuanced sentiment expressions and contextual meaning.

**Why SST-5 matters:**
- **Fine-grained evaluation**: Tests model ability to distinguish subtle sentiment differences beyond simple positive/negative.
- **Established benchmark**: Widely used in NLP research for comparing sentiment analysis approaches.
- **Challenging examples**: Contains complex sentences where sentiment depends on context and linguistic nuance.

**Evaluation approach:**
- Use this dataset to test both zero-shot prompting and fine-tuned classification performance.
- Compare how well your model handles neutral sentiment vs. clearly positive/negative examples.

### CFIMDB (Counterfactual IMDB)

**Dataset characteristics:**
- **Purpose**: Robustness evaluation through counterfactual analysis.
- **Domain**: Modified IMDB movie reviews with controlled alterations.
- **Structure**: Original reviews paired with counterfactual versions where key phrases are altered.
- **Challenge**: Tests model sensitivity to specific textual changes that should affect sentiment.

**Why CFIMDB matters:**
- **Robustness testing**: Evaluates whether models rely on appropriate textual cues for sentiment determination.
- **Counterfactual reasoning**: Tests ability to distinguish subtle but meaningful changes in text.
- **Generalization assessment**: Reveals whether models have learned robust sentiment patterns or superficial correlations.

**Important note about evaluation:**
The provided `Data/cfimdb-test.txt` file contains placeholder negative (-1) labels rather than true gold labels. This is intentional for the educational setting. When evaluating against this dataset, expect lower accuracy scores - this is normal and doesn't indicate problems with your implementation.

### Dataset usage guidelines

**For zero-shot evaluation:**
- Design prompts that clearly specify the sentiment classification task.
- Test different prompt formulations to understand sensitivity to prompt design.
- Use these results to establish baseline model capabilities.

**For fine-tuning evaluation:**
- Use training portions of datasets to adapt your model to specific tasks.
- Compare fine-tuned performance against zero-shot baselines.
- Analyze which types of examples benefit most from task-specific training.

**Performance interpretation:**
These datasets provide complementary views of model performance. SST-5 tests fine-grained sentiment understanding, while CFIMDB evaluates robustness to controlled textual variations. Together, they offer a comprehensive assessment of your LLaMA implementation's sentiment analysis capabilities.
