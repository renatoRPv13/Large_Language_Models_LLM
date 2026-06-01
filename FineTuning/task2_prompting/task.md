### Learning objectives

By completing this task, you will be able to:
- **Master prompt engineering fundamentals** - Understand how different prompt structures affect model behavior and output quality.
- **Design effective prompt templates** - Create reusable prompt patterns for various NLP tasks and model types.
- **Implement few-shot learning techniques** - Use in-context examples to guide model behavior without parameter updates.
- **Compare prompting strategies** - Evaluate the effectiveness of different prompt designs for the same underlying task.

### Problem context

Prompt engineering is the art and science of crafting inputs that elicit desired behaviors from large language models. The way you structure a prompt can dramatically affect the quality, consistency, and usefulness of model outputs, even with the same underlying model.

**Why prompt engineering matters:**
- **Zero-training adaptation** - Achieve task-specific behavior without any model training or fine-tuning.
- **Cost efficiency** - Avoid expensive fine-tuning by leveraging models' existing capabilities through clever prompting.
- **Rapid prototyping** - Quickly test ideas and iterate on model behavior without waiting for training cycles.
- **Foundation for fine-tuning** - Good prompts serve as starting points for more sophisticated adaptation techniques.

**What makes this challenging:**
- **Prompt sensitivity** - Small changes in wording can lead to dramatically different outputs.
- **Model-specific optimization** - Different models respond better to different prompting styles and structures.
- **Task complexity scaling** - Simple prompts work for basic tasks, but complex reasoning requires sophisticated prompt design.
- **Evaluation subjectivity** - Assessing prompt effectiveness often requires human judgment and domain expertise.

## Implementation requirements

Design and implement four distinct prompt templates that demonstrate different approaches to eliciting word definitions and examples from language models.

### Specific requirements:

1. **Implement four prompt templates** - Create basic, instruction, few-shot, and structured prompting approaches.
2. **Build dynamic prompt construction** - Implement the `build_prompt` method to generate prompts with variable content.
3. **Handle few-shot examples** - Integrate multiple examples into prompts for in-context learning.
4. **Compare prompt effectiveness** - Test different templates and analyze their relative performance.

### Expected deliverables:

- Completed `PROMPT_TEMPLATES` dictionary with four distinct prompting strategies.
- Working `build_prompt` method that handles all template types and few-shot examples.
- Understanding of how different prompt structures affect model behavior.
- Ability to evaluate and compare prompt effectiveness for definition generation tasks.

## Prompt engineering strategies

### 1. Basic Prompting
**Approach**: Direct, conversational request for information.
**Example**: `"Define the word and give an example: '{word}'"`
**Use case**: Simple tasks with straightforward instructions.

### 2. Instruction Prompting  
**Approach**: Role-based prompting that establishes context and authority.
**Example**: `"You are a dictionary writer. Provide a definition and example for: {word}"`
**Use case**: Tasks requiring specific persona or expertise framing.

### 3. Few-Shot Prompting
**Approach**: Provides examples within the prompt to demonstrate desired output format.
**Structure**: Multiple word-definition-example triplets followed by the target word.
**Use case**: Complex tasks where output format needs to be precisely specified.

### 4. Structured Prompting
**Approach**: Leverages the model's text completion training by providing partial structure.
**Example**: `"Here is a definition and an example for the word.\nword: {word}\ndefinition: "`
**Use case**: Tasks that align with the model's pre-training objective of text completion.

## Implementation approach

### Template design principles:

**Clarity**: Each template should clearly communicate the desired task
**Consistency**: Output format should be predictable across different inputs  
**Alignment**: Templates should match the model's training paradigm when possible
**Flexibility**: Allow for easy adaptation to different words and contexts

### Few-shot implementation details:

The few-shot template requires special handling to incorporate examples:

```python
def build_prompt(self, word, examples=None, n_shots=3):
    if self.template_type == "few_shot" and examples:
        shot_examples = "\n".join(
            f"word: {ex['word']}\ndefinition: {ex['definition']}\nexample: {ex['example']}"
            for ex in random.sample(examples, min(n_shots, len(examples)))
        )
        return self.template.format(word=word, examples=shot_examples)
    return self.template.format(word=word)
```

<div class="hint" title="Prompt Structure Design">

**Tip**: Consider how each template type aligns with different aspects of language model training. Basic prompts work well for instruction-tuned models, while structured prompts leverage the completion-based pre-training that most models receive.

</div>

<div class="hint" title="Few-Shot Example Selection">

**Tip**: When implementing few-shot prompting, random sampling of examples provides variety, but you might also consider selecting examples based on similarity to the target word or ensuring diverse example types for better generalization.

</div>

<div class="hint" title="Template Evaluation">

**Tip**: Run the `run.py` script to test all templates with the word "quantum" and observe how different prompting strategies affect output quality, consistency, and format adherence. Pay attention to which templates produce more accurate definitions and better examples.

</div>

## Experimental evaluation

### Testing methodology:

1. **Run comparative tests**: Use `run.py` to generate outputs for all four template types
2. **Analyze output quality**: Compare definition accuracy, example relevance, and format consistency
3. **Evaluate robustness**: Test with different words to see which templates generalize better
4. **Consider model limitations**: Remember that GPT-2 has limitations compared to larger, more recent models

### Expected observations:

- **Template effectiveness varies**: Different prompting strategies work better for different types of words and complexity levels
- **Model limitations**: GPT-2 may struggle with advanced vocabulary or complex definitions regardless of prompt quality
- **Format consistency**: Some templates produce more consistent output formatting than others
- **Context utilization**: Few-shot prompts may show improved performance when relevant examples are provided

This task demonstrates how prompt design is both an art and a science, requiring experimentation and evaluation to determine what works best for your specific use case and model.