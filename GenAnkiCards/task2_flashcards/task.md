### General Information

This task focuses on implementing a `generate_flashcards(words, language, level)` function, 
which creates flashcards for learning some language vocabulary in English.

### Task Description

You need to implement a flashcard generator, let's say for Greek vocabulary. 
The generator should create flashcards with Greek translations of English words with their 
definitions, example sentences, and mnemonic aids.

#### **`GreekFlashcardGenerator` Class**

- **Purpose**: Generates flashcards for learning Greek vocabulary with various aids.
- **Functionality**:
  - Creates flashcards with English words, their definitions, example sentences, mnemonics, and Greek translations
  - Supports different CEFR levels: A1, A2, B1, B2, C1, C2
  - Provides helpful mnemonics to aid in memorization

#### **`generate_flashcard` Method**

- **Functionality**:
  - Takes an English word as input.
  - Returns a dictionary containing the word, its translation to Greek, and also definition, example sentences in Greek with English translation, and a mnemonic
  - The flashcard should follow this structure:
    ```json
    {
      "word": "apple",
      "language": "Greek",
      "level": "A1",
      "translation": "μήλο",
      "definition": "A common round fruit with red, yellow, or green skin and crisp flesh.",
      "example_sentence": "Μου αρέσει να τρώω ένα μήλο κάθε μέρα.",
      "example_translation": "I like to eat an apple every day.",
      "mnemonic": "Think of the Greek letter 'μ' (mu) as looking like a bite taken out of an apple."
    }
    ```

#### **`generate_flashcards` Method**

- **Functionality**:
  - Takes a list of English words as input
  - Returns a list of flashcard dictionaries, one for each word

### Task
 - Get content from the `response` variable, based on Openrouter API documentation (which uses OpenAI-compatible interface) or with help of AI Assistant

### Note
- You can use predefined English vocabulary or generate examples using your knowledge of Greek or other language