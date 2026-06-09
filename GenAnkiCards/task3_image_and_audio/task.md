### General Information

In this task you will add image and audio generation to the existing flashcards.

#### Data Model
A flashcard includes the following fields:
- `id`: integer (unique identifier)
- `word`: string (English word)
- `language`: string (target language, e.g., "Greek")
- `level`: string or null (CEFR level like A1–C2)
- `translation`: string (target language translation)
- `definition`: string (short meaning in English)
- `example_sentence`: string (example in target language)
- `example_translation`: string or `null` (English translation of the example)
- `mnemonic`: string (helpful memory aid)
- `image_path`: string or `null` (URL/path to an image representing the word)
- `audio_path`: string or `null` (URL/path to an audio representing the example_sentence)

#### Functionality

1. Function `store_flashcards()` — store flashcards into database.
 - Takes a list of English words `["word1", "word2", ...]`, 
  then generates flashcards for given English words in Greek language
 - Returns list of stored flashcards with their IDs


2. Function `generate_image()` — generates and stores an image for a given word
 - Takes ImageRequest object containing: `word` (string), `language` (string), and optional `flashcard_id` (integer)
 - Checks if image already exists for the word/language combination
 - Generates new image using Openrouter Images API with Google Gemini 2.5 Flash model if no existing image found
 - Saves image to local filesystem and updates flashcard's `image_path` if `flashcard_id` provided

3. Function `generate_audio()` — generates audio for example sentences
 - Takes audio request with all the information needed to generate audio
 - Uses Google Text-to-Speech (gTTS) package to generate audio
 - Saves audio to local filesystem and updates flashcard's `audio_path`

### Task
 - Implement the LLM call in `image_generator.py` to satisfy the API using Openrouter with Google Gemini 2.5 Flash model
 - Implement audio generation call in `audio_generator.py`
