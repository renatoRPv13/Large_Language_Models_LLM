
### Lesson learning objectives

By completing this lesson, you will be able to:
- **Understand the complete RAG pipeline** - Master the end-to-end flow: *raw data → (bi-encoder) retrieve → (LLM) generate*.
- **Design contrastive learning datasets** - Create triplet datasets with positive, hard negative, and random negative examples for robust retrieval training.
- **Work with modern ML libraries** - Leverage [🤗 Datasets](https://huggingface.co/docs/datasets) and [🤗 Sentence-Transformers](https://www.sbert.net) for efficient development.
- **Build in-memory vector stores** - Implement efficient storage and retrieval systems for semantic search applications.
- **Integrate retrieval with generation** - Combine bi-encoder retrieval with large language model generation for knowledge-grounded text production.

### The problem: Knowledge-Grounded Text Generation

Large language models excel at generating fluent text but often struggle with factual accuracy and up-to-date information. They can hallucinate plausible-sounding facts and cannot access information beyond their training cutoff, limiting their utility for knowledge-intensive applications.

**Why this problem matters:**
- **Factual accuracy requirements** - Many applications need verifiably correct information, not just fluent text.
- **Dynamic knowledge needs** - Real-world applications require access to frequently updated information and domain-specific knowledge bases.
- **Transparency and trust** - Users need to understand and verify the sources of generated information for critical applications.
- **Cost efficiency** - RAG enables powerful applications without requiring massive model retraining for new information.

**What makes this challenging:**
- **Retrieval quality** - Finding truly relevant information from large corpora requires sophisticated similarity understanding.
- **Integration complexity** - Seamlessly combining retrieved knowledge with generation requires careful prompt engineering and model coordination.
- **Evaluation difficulty** - Assessing both retrieval accuracy and generation quality requires multi-faceted evaluation frameworks.
- **Scalability concerns** - Real-world systems must handle millions of documents with sub-second response times.
**Motivation behind the lesson:**
> **Retrieval-Augmented Generation (RAG)** is a powerful technique that combines the strengths of retrieval and generation models.
> It allows us to retrieve relevant information from a large corpus and use it to generate coherent and contextually relevant text.
> This technique is particularly useful in scenarios where we need to generate text based on specific information or context, such as in chatbots, question-answering systems, and content generation applications.
> It's widely used in various modern AI applications.

---

### Topics covered

You will learn about these essential RAG system components:

**Retrieval Architecture:**
* **Bi-encoder models** - Learn how dense retrieval systems encode queries and documents into shared embedding spaces for efficient similarity search.
* **Contrastive learning** - Understand how triplet loss trains models to distinguish relevant from irrelevant information using positive and negative examples.
* **Vector similarity metrics** - Master cosine similarity and other distance measures for semantic search applications.

**Data Engineering for RAG:**
* **Triplet dataset construction** - Transform raw JSON data into structured training triplets with anchors, positives, and hard negatives.
* **Training signal design** - Create effective positive/negative pairs that teach models robust discrimination between relevant and irrelevant content.
* **Multi-source data integration** - Combine generated content with curated knowledge bases like WordNet for comprehensive coverage.

**System Integration:**
* **In-memory vector stores** - Implement efficient storage and retrieval systems for real-time semantic search applications.
* **Retrieval-generation pipelines** - Build end-to-end systems that combine dense retrieval with large language model generation.
* **Performance optimization** - Balance retrieval accuracy with computational efficiency for production-ready applications.

The lesson demonstrates how sophisticated RAG systems solve the fundamental challenge of grounding language model generation in verifiable, retrievable knowledge. In Retrieval-Augmented Generation, a bi-encoder must learn to **distinguish** similar from dissimilar text pairs. You'll transform a raw "dictionary-like" JSON into a Hugging Face `Dataset` of `(anchor, positive, negative)` triplets — exactly what you need for **triplet-loss** training.

By the end of this lesson, you will have:
- Created RAG training data from scratch.
- Trained a bi-encoder to retrieve relevant definitions.
- Created a simple in-memory vector store to store and retrieve relevant definitions.
- Built a simple inference pipeline to generate definitions using the bi-encoder and a large language model (LLM).


## Materials

### Reminders
- [Text Embeddings](https://lena-voita.github.io/nlp_course/word_embeddings.html)
- [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Embedding normalization](https://medium.com/%40tellmetiger/embedding-normalization-dummy-learn-2ac8d816e776)
- [Tutorial](https://huggingface.co/docs/datasets/tutorial) on HF Datasets

### Lesson-Related
- [Retriever-Reranker Tutorial](https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data)
- [Lecture about RAG](https://jb.gg/9sw9ko) at Stanford
- [Triplet Loss Intro](https://medium.com/analytics-vidhya/triplet-loss-b9da35be21b8)
- [Sentence-Transformers Tutorial](https://huggingface.co/blog/how-to-train-sentence-transformers) -- Training & Fine-tuning

### Deep Dives
- [RAG docs](https://huggingface.co/docs/transformers/en/model_doc/rag) at HF
- [Triplet loss in-depth](https://www.v7labs.com/blog/triplet-loss)
- [RAG tutorial](https://jb.gg/ub865k) by freeCodeCamp
- [FAISS](https://github.com/facebookresearch/faiss) -- more efficient way to store embeddings 
- [RAG with LangChain](https://medium.com/%40akriti.upadhyay/implementing-rag-with-langchain-and-hugging-face-28e3ea66c5f7)
