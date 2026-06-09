import sys
from typing import List, Optional
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Session, select

import sys
import os

# Adicionar o caminho raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora as importações devem funcionar
#from GenAnkiCards.task3_image_and_audio.dataclasses import logger, AudioRequest, get_session, FlashcardDB, slugify

from GenAnkiCards.task3_image_and_audio.audio_generator import generate_audio
from GenAnkiCards.task3_image_and_audio.schemas import FlashcardDB, FlashcardOut, Flashcard, logger, ImageRequest, \
    get_session, init_db, engine, AudioRequest
from GenAnkiCards.task3_image_and_audio.image_generator import generate_image

# Add the project root to a Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from GenAnkiCards.task2_flashcards.task import generate_flashcards


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield


def _serialize_card(card: FlashcardDB) -> FlashcardOut:
    data = card.model_dump()
    return FlashcardOut(**data)


def get_card_by_id(card_id: int, session: Session = Depends(get_session)) -> Optional[FlashcardOut]:
    card = session.get(FlashcardDB, card_id)
    if not card:
        return None
    return _serialize_card(card)


def create_flashcard(card: Flashcard, session: Session = Depends(get_session)) -> FlashcardOut:
    try:
        db_card = FlashcardDB(**card.model_dump())
        session.add(db_card)
        session.commit()
        session.refresh(db_card)
        return _serialize_card(db_card)
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create flashcard: {e}")
        raise HTTPException(status_code=500, detail="Failed to create flashcard")


def create_multiple_flashcards(cards: List[Flashcard], session: Session = Depends(get_session)) -> List[FlashcardOut]:
    try:
        db_cards = [FlashcardDB(**card.model_dump()) for card in cards]
        session.add_all(db_cards)
        session.commit()
        for card in db_cards:
            session.refresh(card)
        return [_serialize_card(card) for card in db_cards]
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to create multiple flashcards: {e}")
        raise HTTPException(status_code=500, detail="Failed to create multiple flashcards")


def store_flashcards(words: List[Flashcard]):
    init_db()

    flashcards = generate_flashcards(words, "Greek")
    print(f"Generated {len(flashcards)} flashcards:")

    # Store flashcards in database
    with Session(engine) as session:
        stored_flashcards = create_multiple_flashcards(flashcards, session)

        print(f"\nSuccessfully stored {len(stored_flashcards)} flashcards in database:")
        for card in stored_flashcards:
            print(f"ID: {card.id}, Word: {card.word}, Translation: {card.translation}")


def add_images():
    # Generate images for existing flashcards
    with Session(engine) as session:
        # Get all flashcards without images
        stmt = select(FlashcardDB).where(FlashcardDB.image_path.is_(None))
        cards = session.exec(stmt).all()

        for card in cards:
            try:
                # Create image request
                req = ImageRequest(
                    word=card.word,
                    language=card.language,
                    definition=card.definition,
                    flashcard_id=card.id
                )

                # Generate image
                result = generate_image(req, session)
                print(f"Generated image for {card.word}: {result['image_path']}")

            except Exception as e:
                print(f"Failed to generate image for {card.word}: {e}")


def add_audio():
    # Generate audio for existing flashcards
    with Session(engine) as session:
        # Get all flashcards without audio
        stmt = select(FlashcardDB).where(FlashcardDB.audio_path.is_(None))
        cards = session.exec(stmt).all()

        for card in cards:
            try:
                # Create audio request
                req = AudioRequest(
                    word=card.word,
                    language=card.language,
                    definition=card.definition,
                    flashcard_id=card.id
                )

                # Generate audio
                result = generate_audio(req, session)
                print(f"Generated audio for {card.word}: {result['audio_path']}")

            except Exception as e:
                print(f"Failed to generate audio for {card.word}: {e}")


if __name__ == "__main__":
    # Generate and store flashcards
    words = ["knock", "boring", "advance", "minced meat"]
    store_flashcards(words)

    add_images()
    add_audio()
