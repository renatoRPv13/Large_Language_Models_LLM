import os
import uuid

from fastapi import Depends, HTTPException
from sqlmodel import Session, select

import sys
import os

# Adicionar o caminho raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora as importações devem funcionar
from GenAnkiCards.task3_image_and_audio.schemas import logger, AudioRequest, get_session, FlashcardDB, slugify

# Audio directory setup
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "./audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Dummy audio paths
DUMMY_AUDIO_FS_PATH = os.path.join(AUDIO_DIR, "dummy.mp3")
DUMMY_AUDIO_URL = "./audio/dummy.mp3"

# Audio generation with gTTS
try:
    from gtts import gTTS
    import io

    _gtts_available = True
except ImportError:
    _gtts_available = False
    logger.warning("gTTS package not available - audio generation will use dummy files")


def generate_audio(req: AudioRequest, session: Session = Depends(get_session)):
    logger.info(
        f"Starting audio generation for word: '{req.word}', language: '{req.language}', flashcard_id: {req.flashcard_id}")

    # Check if the audio already exists in a database for this word/language combination
    existing_card_with_audio = None
    if req.flashcard_id:
        # If flashcard_id is provided, check if this specific card already has audio
        card = session.get(FlashcardDB, req.flashcard_id)
        if card and card.audio_path:
            logger.info(f"Flashcard {req.flashcard_id} already has audio: {card.audio_path}")
            return {"audio_path": card.audio_path, "text": f"Existing audio for '{req.word}'"}
    else:
        # Check if any card with the same word/language combination has audio
        stmt = select(FlashcardDB).where(
            FlashcardDB.word == req.word,
            FlashcardDB.language == req.language,
            FlashcardDB.audio_path.isnot(None)
        ).limit(1)
        existing_card_with_audio = session.exec(stmt).first()

        if existing_card_with_audio:
            logger.info(
                f"Found existing audio for word '{req.word}' in language '{req.language}': {existing_card_with_audio.audio_path}")
            return {"audio_path": existing_card_with_audio.audio_path, "text": f"Existing audio for '{req.word}'"}

    # No existing audio found, proceed with generation
    # Get the flashcard to access example_sentence
    card = None
    if req.flashcard_id:
        card = session.get(FlashcardDB, req.flashcard_id)

    # Use example_sentence if available, otherwise use the word itself
    text_to_speak = card.example_sentence if card and card.example_sentence else req.word
    logger.info(f"Text to convert to speech: '{text_to_speak}'")

    safe_word = slugify(req.word, "audio")
    unique_id = uuid.uuid4().hex[:12]

    filename = f"{safe_word}_{unique_id}.mp3"
    audio_fs_path = os.path.join(AUDIO_DIR, filename)

    try:
        if not _gtts_available:
            logger.error("gTTS package is not available")
            raise RuntimeError("gTTS package is not configured")

        # Map language codes for gTTS
        language_map = {
            "Greek": "el", "Spanish": "es", "French": "fr",
            "German": "de", "Italian": "it", "Portuguese": "pt",
            "Russian": "ru", "Japanese": "ja", "Chinese": "zh",
            "Korean": "ko", "English": "en"
        }

        gtts_lang = language_map.get(req.language, "en")  # default to English
        logger.info(f"Using gTTS language code: {gtts_lang}")

        logger.info("Generating audio with gTTS...")

        # CORREÇÃO: Criar o objeto gTTS corretamente
        tts = gTTS(text=text_to_speak, lang=gtts_lang, slow=False)

        tts.save(audio_fs_path)
        logger.info(f"Audio saved to local file system: {audio_fs_path}")

        audio_url_path = f"./audio/{filename}"

    except Exception as e:
        logger.exception(f"Audio generation failed, falling back to dummy audio. Error: {e}")
        audio_fs_path = DUMMY_AUDIO_FS_PATH
        audio_url_path = DUMMY_AUDIO_URL

    # Persist to DB if flashcard_id provided: store a relative web path in audio_path
    if req.flashcard_id is not None:
        card = session.get(FlashcardDB, req.flashcard_id)
        if not card:
            raise HTTPException(status_code=404, detail="Flashcard not found for audio update")
        card.audio_path = audio_url_path  # store relative path like ./audio/...
        session.add(card)
        session.commit()
        session.refresh(card)

    logger.info(f"Generated audio URL: {audio_url_path}")

    logger.info(f"Audio generation completed. File: {audio_fs_path}, URL: {audio_url_path}")
    return {"audio_path": audio_url_path, "text": text_to_speak}



# import os
# import uuid
#
# from fastapi import Depends, HTTPException
# from sqlmodel import Session, select
#
# import sys
# import os
#
# # Adicionar o caminho raiz do projeto ao sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# # Agora as importações devem funcionar
# from GenAnkiCards.task3_image_and_audio.schemas import logger, AudioRequest, get_session, FlashcardDB, slugify
#
# # Audio directory setup
# AUDIO_DIR = os.path.join(os.path.dirname(__file__), "./audio")
# os.makedirs(AUDIO_DIR, exist_ok=True)
#
# # Dummy audio paths
# DUMMY_AUDIO_FS_PATH = os.path.join(AUDIO_DIR, "dummy.mp3")
# DUMMY_AUDIO_URL = "./audio/dummy.mp3"
#
# # Audio generation with gTTS
# try:
#     from gtts import gTTS
#     import io
#     _gtts_available = True
# except ImportError:
#     _gtts_available = False
#     logger.warning("gTTS package not available - audio generation will use dummy files")
#
#
# def generate_audio(req: AudioRequest, session: Session = Depends(get_session)):
#     logger.info(f"Starting audio generation for word: '{req.word}', language: '{req.language}', flashcard_id: {req.flashcard_id}")
#
#     # Check if the audio already exists in a database for this word/language combination
#     existing_card_with_audio = None
#     if req.flashcard_id:
#         # If flashcard_id is provided, check if this specific card already has audio
#         card = session.get(FlashcardDB, req.flashcard_id)
#         if card and card.audio_path:
#             logger.info(f"Flashcard {req.flashcard_id} already has audio: {card.audio_path}")
#             return {"audio_path": card.audio_path, "text": f"Existing audio for '{req.word}'"}
#     else:
#         # Check if any card with the same word/language combination has audio
#         stmt = select(FlashcardDB).where(
#             FlashcardDB.word == req.word,
#             FlashcardDB.language == req.language,
#             FlashcardDB.audio_path.isnot(None)
#         ).limit(1)
#         existing_card_with_audio = session.exec(stmt).first()
#
#         if existing_card_with_audio:
#             logger.info(f"Found existing audio for word '{req.word}' in language '{req.language}': {existing_card_with_audio.audio_path}")
#             return {"audio_path": existing_card_with_audio.audio_path, "text": f"Existing audio for '{req.word}'"}
#
#     # No existing audio found, proceed with generation
#     # Get the flashcard to access example_sentence
#     card = None
#     if req.flashcard_id:
#         card = session.get(FlashcardDB, req.flashcard_id)
#
#     # Use example_sentence if available, otherwise use the word itself
#     text_to_speak = card.example_sentence if card and card.example_sentence else req.word
#     logger.info(f"Text to convert to speech: '{text_to_speak}'")
#
#     safe_word = slugify(req.word, "audio")
#     unique_id = uuid.uuid4().hex[:12]
#
#     filename = f"{safe_word}_{unique_id}.mp3"
#     audio_fs_path = os.path.join(AUDIO_DIR, filename)
#
#     try:
#         if not _gtts_available:
#             logger.error("gTTS package is not available")
#             raise RuntimeError("gTTS package is not configured")
#
#         # Map language codes for gTTS
#         language_map = {
#             "Greek": "el", "Spanish": "es", "French": "fr",
#             "German": "de", "Italian": "it", "Portuguese": "pt",
#             "Russian": "ru", "Japanese": "ja", "Chinese": "zh",
#             "Korean": "ko", "English": "en"
#         }
#
#         gtts_lang = language_map.get(req.language, "en")  # default to English
#         logger.info(f"Using gTTS language code: {gtts_lang}")
#
#         logger.info("Generating audio with gTTS...")
#         tts = None # TODO: implement gTTS call for audio generation
#         tts.save(audio_fs_path)
#         logger.info(f"Audio saved to local file system: {audio_fs_path}")
#
#         audio_url_path = f"./audio/{filename}"
#
#     except Exception as e:
#         logger.exception(f"Audio generation failed, falling back to dummy audio. Error: {e}")
#         audio_fs_path = DUMMY_AUDIO_FS_PATH
#         audio_url_path = DUMMY_AUDIO_URL
#
#     # Persist to DB if flashcard_id provided: store a relative web path in audio_path
#     if req.flashcard_id is not None:
#         card = session.get(FlashcardDB, req.flashcard_id)
#         if not card:
#             raise HTTPException(status_code=404, detail="Flashcard not found for audio update")
#         card.audio_path = audio_url_path  # store relative path like ./audio/...
#         session.add(card)
#         session.commit()
#         session.refresh(card)
#
#     logger.info(f"Generated audio URL: {audio_url_path}")
#
#     logger.info(f"Audio generation completed. File: {audio_fs_path}, URL: {audio_url_path}")
#     return {"audio_path": audio_url_path, "text": text_to_speak}
