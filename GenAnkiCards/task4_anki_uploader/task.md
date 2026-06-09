
### General Information

In this task you will integrate your locally generated flashcards (from the [previous task](course://GenAnkiCards/task3_image_and_audio)) with Anki using the AnkiConnect API.
You will upload notes that include text, image, and audio, and mark uploaded cards in your local SQLite database.

#### Prerequisites
- Complete the previous task so the SQLite DB and media files exist
- Anki Desktop running with the AnkiConnect add‑on (code: `2055492159`), default port: `8765`

#### Data Source
Flashcards are read from the database defined in `GenAnkiCards/task3_image_and_audio/dataclasses.py` (table `FlashcardDB`).
Relevant fields used for building a note:
- `word`, `translation`, `definition`, `example_sentence`, `example_translation`, `mnemonic`
- `image_path` (relative like `./images/…`), audio_path (relative like `./audio/…`)

#### What the Uploader Does
- Loads not‑yet‑uploaded flashcards from the DB
- Creates an Anki deck if it doesn’t exist
- Uploads media files to Anki
- Adds notes of the selected model (default: Basic) with:
  - Front: translation (+ example sentence if available)
  - Back: front content + word, example translation, definition, mnemonic, and media refs
- Marks successfully uploaded records as `uploaded_to_anki = True`

### Functionality (CLI)
The uploader is implemented in `task.py` and exposes a CLI:

- `--deck <name>`: target deck (default: "GreekCustom")
- `--model <name>`: Anki note type (model), default "Basic"
- `--limit <N>`: upload only the first N cards selected
- `--where "field='Value'"`: filter by language or level (e.g., `"language='Greek'"`)
- `--dry-run`: show prepared content without calling Anki
- `--ankiconnect <url>`: AnkiConnect URL (default http://127.0.0.1:8765)

Example usages:
- Dry run preview: `python task.py --dry-run --limit 3`
- Upload Greek A2 cards: `python task.py --deck GreekA2 --where "language='Greek'" --limit 20`

### Task
 - Read `task.py` to understand how notes are built from DB records
 - Ensure Anki is running with AnkiConnect enabled
 - *Implement adding information to the Anki note*
 - Then upload a few cards to a test deck and verify in Anki that:
  - Media appears (image on Back, audio plays)
  - Front/Back fields contain the described content
 - Confirm that uploaded records have `uploaded_to_anki=True` in the DB

### Hints
- Media paths are resolved from the [Image & Audio Components](course://GenAnkiCards/task3_image_and_audio) task folder; keep the project structure intact
- If Anki isn’t running or AnkiConnect is missing, requests will fail—start Anki first