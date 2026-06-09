### Learning objectives

By completing this task, you will be able to:
- **Integrate retrieval with generation** - Combine vector search results with large language model generation for knowledge-grounded text production.
- **Master RAG prompt engineering** - Design effective prompts that incorporate retrieved context while maintaining generation quality and coherence.
- **Implement inference API integration** - Work with Hugging Face Inference API for scalable text generation in production RAG systems.
- **Evaluate RAG system performance** - Compare baseline and retrieval-augmented generation to measure the impact of external knowledge integration.

### Problem context

The final stage of RAG systems combines retrieved relevant information with large language model generation to produce accurate, contextually grounded responses. This integration requires careful prompt engineering to effectively incorporate retrieved context while maintaining the model's natural generation capabilities.

**Why RAG inference integration matters:**
- **Knowledge grounding** - Retrieved context provides factual anchors that reduce hallucination and improve response accuracy.
- **Dynamic knowledge access** - RAG enables models to incorporate up-to-date information without requiring expensive retraining or parameter updates.
- **Transparency and verification** - Users can trace generated responses back to specific retrieved sources, improving trust and accountability.
- **Scalable deployment** - Inference API integration enables production-ready RAG systems that can handle varying loads and model updates.

**What makes this challenging:**
- **Context integration complexity** - Effectively blending retrieved information with generation requires sophisticated prompt design and formatting strategies.
- **Quality vs. relevance trade-offs** - Too much retrieved context can confuse generation; too little reduces the benefits of external knowledge.
- **API reliability and latency** - Production systems must handle inference API limitations, rate limits, and variable response times gracefully.
- **Prompt engineering optimization** - Finding the right balance of instruction clarity, context formatting, and generation constraints requires systematic experimentation.

## Task – Prompt-Level Inference 

Implement end-to-end RAG inference by combining vector retrieval with large language model generation through the Hugging Face Inference API.

### Implementation requirements

#### `InferenceClient.__init__`
Initialize a wrapper around Hugging Face Inference API for streamlined text generation.

**Core functionality:**
- Store default `model_id` for consistent model usage across generation calls.
- Initialize `HFInferenceClient` with provider and API key configuration.
- Support flexible provider selection (e.g., Novita) for cost-effective inference.

#### `InferenceClient.generate`
Execute text generation using the chat completions API and return clean response content.

**Core functionality:**
- Format input prompt as messages array with user role.
- Call `chat.completions.create()` with model, max_tokens, and temperature parameters.
- Extract and return the content string from `choices[0].message.content`.
- Handle API response structure according to OpenAI-compatible format.

#### `build_prompt`
Construct prompts for both baseline and RAG-augmented generation modes.

**Core functionality:**
- **Baseline mode**: Simple prompt with header and target word only.
- **RAG mode**: Include retrieved definitions as bullet-point context before the generation request.
- Use consistent `PROMPT_HEADER` to establish the model's role as a dictionary assistant.
- Format retrieved context clearly to guide the model's attention to relevant information.


<div class="hint" title="Prompt Engineering for RAG">

**Effective context integration**: When building RAG prompts, structure retrieved definitions as clear bullet points before the generation request. Use consistent formatting with `PROMPT_HEADER` to establish the model's role, then provide retrieved context, and finally request the definition. This structure helps the model distinguish between reference material and generation tasks.

</div>

<div class="hint" title="API Integration Best Practices">

**Chat completions format**: The modern HF Inference API uses OpenAI-compatible chat format. Structure your request as `messages = [{"role": "user", "content": prompt}]` and extract the response from `completion.choices[0].message.content`. This format is more robust than legacy text generation endpoints.

</div>

### Other Functions and Motivation
- `compare_generations` - A simple wrapper to compare the two generation modes.
- **`PROMPT_HEADER`** - Frames the assistant as a dictionary—ensures consistency.  
  

### Scripts
**`run.py`**  
- **Purpose**: Load your vector store (unless `--no-retrieval`), spin up the inference client, and compare baseline vs RAG outputs.  
- **Options**:  
  - `--query`: word to define  
  - `--model-id`: HF model to call  
  - `--max-tokens`: generation length  
  - `--no-retrieval`: skip search  
- **Recommendation**: Export your token, then run with defaults:
  ```bash
  export HUGGINGFACEHUB_API_TOKEN=your_token_here
  python run.py
  ```

**Important**: Please run the script and ensure it worked correctly.

### Notes
You **must** set `HUGGINGFACEHUB_API_TOKEN` in your env; otherwise inference calls will fail at runtime. \
[Instruction how to obtain a token](https://huggingface.co/docs/hub/security-tokens) (read token should be enough) \
*Note:* Don't forget to restart the IDE! Otherwise the environment variable won't be visible.