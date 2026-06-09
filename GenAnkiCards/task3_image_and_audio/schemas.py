import logging
import os
import re

from typing import List, Optional
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, Session, create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Database setup ----------
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "flashcards.sqlite")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})

def init_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session

# ---- filename hygiene ---- # ---- higiene de nomes de arquivos ----
def slugify(s: str, modality: str) -> str:
    s = (s or modality).strip()
    s = s.lower()
    # keep letters, digits, dash/underscore; collapse runs to a single underscore
    # manter letras, dígitos, hífen/sublinhado; condensar sequências em um único sublinhado
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s.strip("_")[:60] or modality


class FlashcardDB(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    word: str
    language: str
    level: Optional[str] = None
    translation: str
    definition: str
    example_sentence: str
    example_translation: Optional[str] = None
    mnemonic: str

    # Image path (local file path instead of URL) # Caminho da imagem (caminho do arquivo local em vez de URL)
    image_path: Optional[str] = None
    # Audio path (local file path instead of URL) # Caminho do áudio (caminho do arquivo local em vez de URL)
    audio_path: Optional[str] = None

    # Whether this flashcard has been uploaded to Anki # Se este cartão de memorização foi carregado no Anki
    uploaded_to_anki: Optional[bool] = Field(default=False)


class Flashcard(BaseModel):
    word: str
    language: str
    level: Optional[str]
    translation: str
    definition: str
    example_sentence: str
    example_translation: Optional[str]
    mnemonic: str
    image_path: Optional[str] = None
    audio_path: Optional[str] = None

    class Config:
        from_attributes = True


class FlashcardOut(Flashcard):
    id: int


class GenerateRequest(BaseModel):
    words: List[str]
    language: str
    level: Optional[str] = None


class ImageRequest(BaseModel):
    word: str
    language: str
    definition: str
    flashcard_id: Optional[int] = None


class AudioRequest(BaseModel):
    word: str
    language: str
    definition: str
    flashcard_id: Optional[int] = None
