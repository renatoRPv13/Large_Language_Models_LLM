#!/usr/bin/env python3
"""
lexicographer_generator.py

Read a CSV containing `word` and `part_of_speech`, query the OpenAI API to
generate a precise definition, a close-but-wrong definition (hard negative),
and an example sentence (hard negative) for each word, then write the results
as a JSON array.

Configuration is via the globals below and the `OPENAI_API_KEY` environment
variable.

Requirements:
  pip install pandas openai tenacity python-dotenv (optional)

Author: ChatGPT (OpenAI)
License: MIT
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI, OpenAIError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Global configuration – edit these to suit your environment
# ---------------------------------------------------------------------------
SOURCE_CSV_PATH: Path = Path("words.csv")
OUTPUT_JSON_PATH: Path = Path("dataset.json")
PROMPT_TEMPLATE: str = """
You are a helpful lexicographer.
You have to provide a definition, a close definition, and an example sentence for the given word.

1. Definition (concise, dictionary style).
2. Close definition (wrong but semantically similar; hard negative).
3. Example sentence using the word (hard negative).

Respond only as valid JSON with *exactly* these keys:
{{
  "definition": "...",
  "close_definition": "...",
  "sentence": "..."
}}

# Example
Word: "bank"
Part of speech: "noun"
Output:
{{
  "definition": "A financial institution that accepts deposits from the public.",
  "close_definition": "A financial institution responsible for managing the country's currency.",
  "sentence": "I went to the bank to withdraw some cash."
}}

# Task
Word: "{word}"
Part of speech: "{part_of_speech}"
Output:
"""
OPENAI_MODEL: str = "gpt-4.1-mini"  # or any model capable of JSON-mode output
OPENAI_API_TIMEOUT: int = 60       # seconds
MAX_RETRIES: int = 5               # total attempts per word
# ---------------------------------------------------------------------------

# Initialise OpenAI client
_client = OpenAI(timeout=OPENAI_API_TIMEOUT)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LexicographerEntry:
    word: str
    part_of_speech: str
    definition: str
    close_definition: str
    sentence: str

    @classmethod
    def from_api_response(
        cls, word: str, part_of_speech: str, response_json: dict
    ) -> "LexicographerEntry":
        """
        Validate that the response contains all required keys and return an
        instance of LexicographerEntry.
        """
        required_keys = {"definition", "close_definition", "sentence"}
        missing = required_keys - response_json.keys()
        if missing:
            raise ValueError(f"API response missing keys: {', '.join(missing)}")

        return cls(
            word=word,
            part_of_speech=part_of_speech,
            definition=response_json["definition"].strip(),
            close_definition=response_json["close_definition"].strip(),
            sentence=response_json["sentence"].strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# OpenAI call with automatic retries
# ---------------------------------------------------------------------------


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type((OpenAIError, ValueError, json.JSONDecodeError)),
    reraise=True,
)
def _generate_entry(word: str, part_of_speech: str) -> LexicographerEntry:
    prompt = PROMPT_TEMPLATE.format(word=word, part_of_speech=part_of_speech)

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    content: str = response.choices[0].message.content.strip()  # type: ignore
    logging.debug("Raw response for %s: %s", word, content)

    parsed = json.loads(content)
    return LexicographerEntry.from_api_response(word, part_of_speech, parsed)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    # Read CSV
    logging.info("Reading CSV from %s", SOURCE_CSV_PATH)
    df = pd.read_csv(SOURCE_CSV_PATH)

    # Ensure required columns
    for col in ("word", "part_of_speech"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column")

    entries: List[LexicographerEntry] = []

    # Generate data
    progress_bar = tqdm(total=len(df), desc="Generating entries", unit="entry")
    for row in df.itertuples(index=False):
        word = str(row.word).strip()
        pos = str(row.part_of_speech).strip()

        try:
            entry = _generate_entry(word, pos)
            entries.append(entry)
            logging.info("✔ Generated entry for '%s'", word)
        except Exception as exc:
            logging.error("✖ Failed to generate entry for '%s': %s", word, exc)
        finally:
            progress_bar.update(1)
    progress_bar.close()

    # Ensure output directory exists
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entries], f, ensure_ascii=False, indent=2)

    logging.info("Wrote %d entries to %s", len(entries), OUTPUT_JSON_PATH)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
