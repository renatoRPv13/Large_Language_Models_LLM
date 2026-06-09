#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload flashcards from the local database (task3) to Anki via AnkiConnect.
Faça o upload dos flashcards do banco de dados local (tarefa 3) para o Anki através do AnkiConnect.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to Python path # Adicione a raiz do projeto ao caminho do Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import click
import requests
from sqlmodel import Session, select
from tqdm import tqdm

from GenAnkiCards.task3_image_and_audio.schemas import (
    FlashcardDB,
    engine,
    DB_PATH,
)

# Configure logging # Configurar registro
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("task4_anki_uploader")


class AnkiConnect:
    def __init__(self, endpoint: str = "http://127.0.0.1:8765"):
        self.endpoint = endpoint.rstrip("/")

    def _request(self, action: str, params: Optional[dict] = None):
        payload = {"action": action, "version": 6, "params": params or {}}
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            error_content = data.get("error")
            if error_content:
                # Robustly handle duplicate errors, which are non-fatal for this script.
                # Lidar de forma robusta com erros duplicados que não sejam fatais para este script.
                if action == "addNotes" and "duplicate" in str(error_content):
                    logger.warning("AnkiConnect: Algumas ou todas as notas eram duplicadas e foram ignoradas..")
                    return data.get("result") or []  # Ensure we return a list
                
                raise RuntimeError(f"AnkiConnect error in {action}: {error_content}")
            
            return data.get("result")

        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to AnkiConnect at {self.endpoint}. Is Anki running with AnkiConnect installed?")
            raise RuntimeError(f"Connection to AnkiConnect failed: {e}") from e


    def create_deck(self, deck_name: str):
        return self._request("createDeck", {"deck": deck_name})

    def store_media_file(self, filename: str, path: str):
        return self._request("storeMediaFile", {"filename": filename, "path": path})

    def add_notes(self, notes: List[dict]):
        return self._request("addNotes", {"notes": notes})

    def get_model_field_names(self, model_name: str) -> List[str]:
        return self._request("modelFieldNames", {"modelName": model_name})


TASK3_DIR = Path(__file__).resolve().parent.parent / "task3_image_and_audio"

def resolve_media_fs_path(rel_path: Optional[str]) -> Optional[str]:
    if not rel_path:
        return None
    rp = rel_path.lstrip("./")
    abs_path = TASK3_DIR / rp
    return str(abs_path) if abs_path.exists() else None

class NoteBuild:
    def __init__(self, front: str, back: str, audio_filename: Optional[str] = None, audio_fs_path: Optional[str] = None):
        self.front = front
        self.back = back
        self.audio_filename = audio_filename
        self.audio_fs_path = audio_fs_path

def build_note_from_record(card: FlashcardDB) -> NoteBuild:
    audio_fs_path = resolve_media_fs_path(card.audio_path)
    audio_filename = Path(audio_fs_path).name if audio_fs_path else None
    audio_tag = f"[sound:{audio_filename}]" if audio_filename else ""

    front = card.word
    if not front or not front.strip():
        raise ValueError(f"Card ID {card.id} has no valid content for the 'Front' field.")
        
    back = f"{card.translation}<br>{audio_tag}"

    return NoteBuild(
        front=front,
        back=back,
        audio_filename=audio_filename,
        audio_fs_path=audio_fs_path,
    )

@click.command()
@click.option("--deck", "deck_name", type=str, default="GreekCustom", help="Anki deck name")
@click.option("--model", "model_name", type=str, default="Básico", help="Anki note type (model) name")
def main(deck_name: str, model_name: str):
    logger.info(f"DB path: {DB_PATH}")
    logger.info(f"Deck: {deck_name} | Model: {model_name}")

    anki = AnkiConnect()
    try:
        field_names = anki.get_model_field_names(model_name)
        logger.info(f"Fields for model '{model_name}': {field_names}")
        if len(field_names) < 2:
            raise RuntimeError(f"Model '{model_name}' must have at least two fields.")
        front_field, back_field = field_names[0], field_names[1]
    except Exception as e:
        logger.error(f"Could not get model fields from Anki. Is Anki running and AnkiConnect installed? Error: {e}")
        return

    with Session(engine) as session:
        stmt = select(FlashcardDB).where(FlashcardDB.uploaded_to_anki == False)
        cards = list(session.exec(stmt))

    if not cards:
        logger.info("No new flashcards to upload.")
        return

    anki.create_deck(deck_name)
    
    note_builds = [build_note_from_record(c) for c in cards]

    for nb in tqdm(note_builds, desc="Uploading media"):
        if nb.audio_fs_path:
            anki.store_media_file(filename=nb.audio_filename, path=nb.audio_fs_path)

    notes_payload = [
        {
            "deckName": deck_name,
            "modelName": model_name,
            "fields": {front_field: nb.front, back_field: nb.back},
            "options": {"allowDuplicate": False},
            "tags": ["gen-anki-cards-v2"],
        }
        for nb in note_builds
    ]

    result_nids = anki.add_notes(notes_payload)
    added_count = sum(1 for nid in result_nids if nid)
    logger.info(f"Notas processadas com sucesso. Novas notas adicionadas.: {added_count}")

    with Session(engine) as session:
        for card, nid in zip(cards, result_nids):
            if nid:
                db_card = session.get(FlashcardDB, card.id)
                if db_card:
                    db_card.uploaded_to_anki = True
        session.commit()
    logger.info(f"Marked {added_count} cartões conforme carregados no banco de dados.")

if __name__ == "__main__":
    main()
