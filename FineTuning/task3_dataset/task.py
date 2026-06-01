from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

import json
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import Tuple
from datasets import load_from_disk


class DatasetHandler:
    """Handler for processing and managing datasets"""
    TEMPLATE = f"""\
    word: {{word}}
    definition: {{definition}}
    example: {{example}}
    """.replace("    ", "").strip()

    def __init__(self, tokenizer_name: str, num_proc: int = 1) -> None:
        """Initialize the handler with the tokenizer name
        :@param tokenizer_name: Hugging Face tokenizer name
        :@param num_proc: Number of processes to use for tokenization
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_proc = num_proc

    def convert_to_hf(self, json_path: str) -> Dataset:
        """Convert JSON dataset to Hugging Face Dataset format"""
        with open(json_path) as f:
            data = json.load(f)

        return Dataset.from_list(data)

    def _add_text_column(self, dataset: Dataset) -> Dataset:
        """Add a text column to the dataset

        Format: Use the TEMPLATE to combine word, definition, and example
        """
        dataset = dataset.map(
            lambda x: {"text": self.TEMPLATE.format(
                word=x["word"],
                definition=x["definition"],
                example=x["example"]
            )},
            num_proc=self.num_proc
        )
        return dataset

    def process(self, dataset: Dataset, text_col: str = "text") -> Dataset:
        """Add text column and tokenize along it

        Note: No truncation is applied since the text is short, no padding is needed since we'll use collate_fn
        """
        dataset = self._add_text_column(dataset)

        # Tokenize the texts and leave only attention mask and input ids
        def tokenize_function(examples):
            tokens = self.tokenizer(
                examples[text_col],
                truncation=False,
                padding=False,
                return_attention_mask=True
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            }

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=dataset.column_names  # Remove original columns
        )

        return dataset

    def train_test_split(self, dataset: Dataset, test_size: float = 0.15, random_state: int = 42) -> Tuple[
        Dataset, Dataset]:
        """Split dataset into train and test sets"""
        split_data = dataset.train_test_split(test_size=test_size, seed=random_state)
        return split_data['train'], split_data['test']

    def save(self, train: Dataset, test: Dataset, save_dir: str) -> None:
        """Save train and test datasets to disk"""
        train.save_to_disk(f"{save_dir}/train")
        test.save_to_disk(f"{save_dir}/test")

    @staticmethod
    def load(load_dir: str) -> Tuple[Dataset, Dataset]:
        """Load train and test datasets from disk"""
        return load_from_disk(f"{load_dir}/train"), load_from_disk(f"{load_dir}/test")


# Exemplo de uso
def main():
    # Criar handler com um tokenizer pequeno para teste
    handler = DatasetHandler("distilbert-base-uncased", num_proc=1)

    # Criar dados de exemplo
    sample_data = [
        {
            "word": "python",
            "definition": "a programming language",
            "example": "Python is great for data science."
        },
        {
            "word": "machine learning",
            "definition": "subset of AI",
            "example": "Machine learning models learn from data."
        }
    ]

    # Salvar como JSON temporário para teste
    with open("temp_sample.json", "w") as f:
        json.dump(sample_data, f)

    # Converter para HF dataset
    dataset = handler.convert_to_hf("temp_sample.json")
    print(f"Dataset criado com {len(dataset)} exemplos")

    # Processar (adicionar texto e tokenizar)
    processed_dataset = handler.process(dataset)
    print(f"Colunas após processamento: {processed_dataset.column_names}")
    print(f"input_ids shape: {len(processed_dataset[0]['input_ids'])} tokens")

    # Split em treino/teste
    train, test = handler.train_test_split(processed_dataset)
    print(f"Treino: {len(train)} amostras")
    print(f"Teste: {len(test)} amostras")

    # Salvar
    handler.save(train, test, "./saved_dataset")

    # Carregar
    train_loaded, test_loaded = handler.load("./saved_dataset")
    print(f"Dataset carregado: treino={len(train_loaded)}, teste={len(test_loaded)}")


if __name__ == '__main__':
    main()