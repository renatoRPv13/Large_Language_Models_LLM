### Lesson: Anki Cards Generator
> Expected time for completion: 10 hours

### In this lesson you will
- Understand the end-to-end goal: turn a list of words into high‑quality study flashcards for [Anki](https://apps.ankiweb.net/) using LLMs
- Use text, audio, and image representations from LLM to construct flashcards

### Motivation
Language learning is most effective with consistent exposure and spaced repetition. In this lesson, you’ll assemble a lightweight app that transforms raw vocabulary into structured, memorable flashcards and helps you review them efficiently. You’ll also practice integrating LLM outputs into a practical product workflow.

### About the Lesson
This lesson contains three tasks that build on each other:
- Intro (this task): Read the overview, set up your environment, and skim the code layout so you know where things live
- Flashcard Generator: Implement the flashcard generator function in Python. You’ll return dictionaries with fields like word, translation, definition, example sentence, and a mnemonic. You can optionally support CEFR levels (A1–C2) and multiple languages
- Image and Audio Components: Add image and audio to the flashcard generator functionality, using LLM and Google Text-to-Speech (gTTS) package
- Anki Integration: Implement a simple pipeline to upload generated flashcards in Anki

By the end, you will have:
- A reusable function to generate flashcards programmatically
- A working local pipeline for creating and reviewing cards with spaced repetition in Anki
- A clear pattern for plugging LLM functionality into backend APIs

---

### Materials
 - [CEFR levels overview](https://www.coe.int/en/web/common-european-framework-reference-languages).
 - [SM‑2 algorithm (spaced repetition)](https://www.supermemo.com/en/archives1990-2015/english/ol/sm2).
 - [Anki documentation](https://docs.ankiweb.net/).
 - [Anki Connect](https://ankiweb.net/shared/info/2055492159).
