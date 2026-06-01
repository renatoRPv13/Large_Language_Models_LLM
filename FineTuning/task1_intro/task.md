### Lesson learning objectives

By completing this lesson, you will be able to:
- **Master prompt engineering techniques** - Design effective prompts for zero-shot, few-shot, and chain-of-thought reasoning across diverse tasks
- **Implement parameter-efficient fine-tuning** - Apply LoRA, IA³, and prompt tuning to adapt large models with minimal computational overhead
- **Work with Hugging Face ecosystem** - Leverage datasets, transformers, and PEFT libraries for rapid ML development and experimentation
- **Build end-to-end fine-tuning pipelines** - Create complete training workflows with proper evaluation, logging, and hyperparameter optimization
- **Evaluate and compare model adaptations** - Assess the effectiveness of different fine-tuning approaches using rigorous experimental methodology

### The problem: Adapting Foundation Models to Specialized Tasks

Large language models like GPT-4, LLaMA, and others demonstrate remarkable capabilities on general tasks, but they often fall short on domain-specific problems or require specialized behavior that wasn't captured during pre-training.

**Why this problem matters:**
- **Domain specialization** - Foundation models may lack expertise in specific fields like medical diagnosis, legal analysis, or technical documentation
- **Cost and efficiency** - Running massive foundation models for every task is computationally expensive and often overkill
- **Control and customization** - Organizations need models that behave consistently according to their specific requirements and style guidelines
- **Data privacy** - Many applications require keeping sensitive data on-premises rather than sending it to external APIs

**What makes this challenging:**
- **Computational constraints** - Full fine-tuning of large models requires enormous GPU resources and memory
- **Data efficiency** - Many specialized tasks have limited labeled data available for training
- **Prompt instability** - Small changes in prompts can lead to dramatically different outputs, making systems unreliable
- **Evaluation complexity** - Measuring model performance on open-ended generation tasks requires sophisticated evaluation frameworks

### Topics covered

You will learn about these essential fine-tuning and adaptation techniques:

**Prompt Engineering:**
* **Zero-shot prompting** - Elicit desired behavior from models without any examples, using carefully crafted instructions
* **Few-shot prompting** - Provide examples within the prompt to guide model behavior and improve task performance
* **Chain-of-thought reasoning** - Structure prompts to encourage step-by-step reasoning for complex problem solving

**Parameter-Efficient Fine-Tuning (PEFT):**
* **LoRA (Low-Rank Adaptation)** - Adapt models by learning low-rank matrices that modify attention weights with minimal parameter overhead
* **IA³ (Infused Adapter by Inhibiting and Amplifying)** - Scale intermediate activations to modify model behavior without changing core parameters
* **Prompt Tuning** - Learn continuous prompt embeddings that guide model behavior while keeping all original parameters frozen

**Development Infrastructure:**
* **Hugging Face Ecosystem** - Master the industry-standard toolkit for working with transformers, datasets, and model training
* **Experiment tracking** - Use Weights & Biases to monitor training progress, compare experiments, and reproduce results
* **Evaluation frameworks** - Implement robust evaluation pipelines that measure both automatic metrics and human-aligned quality

These techniques represent the cutting edge of efficient model adaptation, allowing you to achieve state-of-the-art performance on specialized tasks while using a fraction of the computational resources required for full fine-tuning. By the end of this lesson, you'll have the skills to adapt any foundation model to your specific use case, whether that's creating a domain-specific chatbot, building a specialized code generation tool, or developing a custom content generation system.

### Materials

#### Reminders
1. GPU Refresher ([link](https://hsf-training.github.io/hsf-training-ml-gpu-webpage/aio/index.html))

#### Lesson Related
1. Prompt Engineering ([page](https://www.promptingguide.ai/)
    * Introduction (all chapters)
    * Prompting Techniques: Zero-shot, Few-shot, Chain-of-thought
1. Hugging Face ([NLP Course](https://huggingface.co/learn/nlp-course/chapter0/1?fw=pt))
    * [Transofrmers](https://huggingface.co/learn/nlp-course/chapter2/1?fw=pt)
    * [Datasets](https://huggingface.co/learn/nlp-course/chapter5/1?fw=pt)
    * [Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/1?fw=pt)
    * [PEFT](https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt)
1. PEFT Techniques
    * [Lecture](https://jb.gg/nkv7x6)
    * Shorter version ([link](https://jb.gg/lqkjx4))
1. WandB ([page](https://wandb.ai/site))
    * [Quickstart](https://docs.wandb.ai/quickstart)
    * [Integration with HF](https://docs.wandb.ai/guides/integrations/huggingface)

#### Additional
1. Prompt Engineering is overhyped
    * Prompts are unstable ([paper](https://arxiv.org/pdf/2102.09690))
    * Best prompts are complete nonsense ([paper](https://arxiv.org/abs/2010.15980))
1. PEFT Techniques (papers)
    * Prompt Tuning ([paper](https://arxiv.org/abs/2104.08691))
    * LoRA ([paper](https://arxiv.org/abs/2106.09685))
1. Prompt Engineering ([page](https://www.promptingguide.ai/))
    * Craft your prompt: [guide](https://www.promptingguide.ai/guides/optimizing-prompts)
1. LLM Security
    * Guide on prompt hacking: [guide](https://learnprompting.org/docs/prompt_hacking/)

### Other Information

**Important:** For this lesson, you'll need to update the environment. \
Use the `run` button to install the dependencies (recommended). \
Or, run the following command in the terminal:
```bash
pip install --force-reinstall -r FineTuning/requirements.txt
```
<br>

In this lesson, we'll use some custom libraries. You're welcome to look inside. \
On top of that, in this lesson you'll need a GPU with at least 8GB of memory.

### Configuration files

* `requirements.txt`: Specific for this lesson, located in `FineTuning/requirements.txt`.
* `conf.yaml`: Located in `FineTuning/conf.yaml`.