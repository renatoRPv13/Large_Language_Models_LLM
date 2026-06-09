from pydantic import BaseModel
from typing import Optional

class Flashcard(BaseModel):
    word: str
    language: str
    level: Optional[str]
    translation: str
    definition: str
    example_sentence: str
    example_translation: Optional[str]
    mnemonic: str
