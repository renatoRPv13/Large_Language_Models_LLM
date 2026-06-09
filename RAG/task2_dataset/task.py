import json
from pathlib import Path
from typing import List, Dict, Union

from datasets import Dataset, DatasetDict, concatenate_datasets
from custom_helpers import get_config


DEFAULT_CONFIG = {
    "TRAIN_FRAC": 0.8,
    "SEED": 42,
}

def load_config() -> Dict:
    """Load project configuration if available, otherwise use safe defaults.

    The original code expected config.yaml to exist at RAG/config.yaml.
    If that file is missing, importing this module fails immediately.
    This function makes configuration loading more robust by checking
    likely locations and falling back to defaults.
    Carregar a configuração do projeto, se disponível; caso contrário, usar os valores padrão seguros.

    O código original esperava que o arquivo config.yaml existisse em RAG/config.yaml.
    Se esse arquivo estiver faltando, a importação deste módulo falhará imediatamente.
    Esta função torna o carregamento da configuração mais robusto, verificando
    os locais prováveis ​​e recorrendo aos valores padrão.
    """
    current_file = Path(__file__).resolve()

    candidate_paths = [
        current_file.parents[1] / "config.yaml",  # RAG/config.yaml
        current_file.parents[1] / "conf.yaml",    # RAG/conf.yaml
        current_file.parents[2] / "config.yaml",  # project root/config.yaml
        current_file.parents[2] / "conf.yaml",    # project root/conf.yaml
    ]

    for path in candidate_paths:
        if path.exists():
            config = get_config(str(path))
            return {
                **DEFAULT_CONFIG,
                **dict(config),
            }

    return DEFAULT_CONFIG.copy()


CONFIG = load_config()


class TripletDatasetBuilder:
    """Utilities to create `(anchor, positive, negative)` triplets."""

    @staticmethod
    def load_raw(path: Union[str, Path]) -> List[Dict]:
        """Load raw dataset from JSON file.
        Args:
            path: Path to the dataset file.
        Returns:
            List of dictionary items from the dataset.
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON root is not a list.
        Carrega o conjunto de dados brutos de um arquivo JSON.
        Argumentos:
            caminho: Caminho para o arquivo do conjunto de dados.
        Retorno:
            Lista de itens do dicionário do conjunto de dados.
        Exceções:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se a raiz JSON não for uma lista.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected dataset JSON root to be a list, got {type(data).__name__}")

        return data

    @staticmethod
    def build_triplets(raw: List[Dict], negative_field: str) -> Dataset:
        """Create a dataset with (anchor, positive, negative) triplets.

        Args:
            raw: Raw dataset as a list of dictionaries.
            negative_field: Field name to use for negative examples.

        Returns:
            Dataset with three columns: anchor, positive, negative.
        """
        anchors, positives, negatives = [], [], []

        required_fields = {"word", "definition", "sentence", negative_field}

        for row in raw:
            if not isinstance(row, dict):
                continue

            if not required_fields.issubset(row):
                continue

            anchor = f"word: {row['word']}\ndefinition: {row['definition']}"
            positive = row["sentence"]
            negative = row[negative_field]

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return Dataset.from_dict({
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        })

    @staticmethod
    def concat_and_split(
        datasets: List[Dataset],
        train_frac: float = CONFIG["TRAIN_FRAC"],
        seed: int = CONFIG["SEED"],
    ) -> DatasetDict:
        """Concatenate multiple datasets and split into train/eval.
        Args:
            datasets: List of datasets to concatenate.
            train_frac: Fraction of data to use for training.
            seed: Random seed for reproducibility.
        Returns:
            DatasetDict with 'train' and 'eval' splits.
        Raises:
            ValueError: If datasets is empty or train_frac is invalid.
        """
        if not datasets:
            raise ValueError("At least one dataset is required.")

        if not 0 < train_frac < 1:
            raise ValueError(f"train_frac must be between 0 and 1, got {train_frac}")

        merged = concatenate_datasets(datasets).shuffle(seed=seed)

        if len(merged) < 2:
            raise ValueError("At least two rows are required to create train/eval splits.")

        split = merged.train_test_split(train_size=train_frac, seed=seed)

        return DatasetDict({
            "train": split["train"],
            "eval": split["test"],
        })





# import json
# from pathlib import Path
# from typing import List, Dict, Union
# import os
#
# from datasets import Dataset, DatasetDict, concatenate_datasets
# from custom_helpers import get_config
#
# CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))
#
# class TripletDatasetBuilder:
#     """Utilities to create `(anchor, positive, negative)` triplets."""
#
#     @staticmethod
#     def load_raw(path: Union[str, Path]) -> List[Dict]:
#         """Load raw dataset from JSON file.
#
#         Args:
#             path: Path to the dataset file
#
#         Returns:
#             List of dictionary items from the dataset
#
#         Raises:
#             FileNotFoundError: If the file doesn't exist
#         """
#         path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"Dataset file not found: {path}")
#         with path.open() as f:
#             return json.load(f)
#
#     @staticmethod
#     def build_triplets(raw: List[Dict], negative_field: str) -> Dataset:
#         """Create a dataset with (anchor, positive, negative) triplets.
#
#         Args:
#             raw: Raw dataset as a list of dictionaries
#             negative_field: Field name to use for negative examples
#
#         Returns:
#             Dataset with three columns: anchor, positive, negative
#         """
#         anchors, positives, negatives = [], [], []
#
#         for row in raw:
#             if negative_field not in row:
#                 continue
#
#             anchor = f"word: {row['word']}\ndefinition: {row['definition']}"
#             positive = row["sentence"]
#             negative = row[negative_field]
#
#             anchors.append(anchor)
#             positives.append(positive)
#             negatives.append(negative)
#
#         return Dataset.from_dict({
#             "anchor": anchors,
#             "positive": positives,
#             "negative": negatives,
#         })
#
#     @staticmethod
#     def concat_and_split(
#         datasets: List[Dataset],
#         train_frac: float = CONFIG["TRAIN_FRAC"],
#         seed: int = CONFIG["SEED"]
#     ) -> DatasetDict:
#         """Concatenate multiple datasets and split into train/eval.
#
#         Args:
#             datasets: List of datasets to concatenate
#             train_frac: Fraction of data to use for training
#             seed: Random seed for reproducibility
#
#         Returns:
#             DatasetDict with 'train' and 'eval' splits
#         """
#         merged = concatenate_datasets(datasets).shuffle(seed=seed)
#         split = merged.train_test_split(train_size=train_frac, seed=seed)
#
#         return DatasetDict({
#             "train": split["train"],
#             "eval": split["test"],
#         })


# """Bi-encoder training."""
# import json
# from pathlib import Path
# from typing import List, Dict, Union
# import os
#
# from datasets import Dataset, DatasetDict, concatenate_datasets
# from custom_helpers import get_config
#
# CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))
#
#
# class TripletDatasetBuilder:
#     """Utilities to create `(anchor, positive, negative)` triplets."""
#
#     @staticmethod
#     def load_raw(path: Union[str, Path]) -> List[Dict]:
#         """Load raw dataset from JSON file.
#
#         Args:
#             path: Path to the dataset file
#
#         Returns:
#             List of dictionary items from the dataset
#
#         Raises:
#             FileNotFoundError: If the file doesn't exist
#         """
#         path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"Dataset file not found: {path}")
#         with path.open() as f:
#             return json.load(f)
#
#     @staticmethod
#     def build_triplets(raw: List[Dict], negative_field: str) -> Dataset:
#         """Create a dataset with (anchor, positive, negative) triplets.
#
#         Args:
#             raw: Raw dataset as a list of dictionaries
#             negative_field: Field name to use for negative examples
#
#         Returns:
#             Dataset with three columns: anchor, positive, negative
#         """
#         anchors, positives, negatives = [], [], []
#         for row in raw:
#             if negative_field not  in row:
#                 continue
#             # Anchor: combine word and definition
#             anchor = f"word: {row['word']}\ndefinition: {row['definition']}"
#
#             # Positive: example sentence
#             positive = row['sentence']
#
#             # Negative: use the specified field with guard
#             if negative_field in row:
#                 negative = row[negative_field]
#             else:
#                 # If negative_field not present, use a fallback
#                 negative = row.get('close_definition', '')
#
#             anchors.append(anchor)
#             positives.append(positive)
#             negatives.append(negative)
#
#         # Create HF dataset with fields anchor, positive, negative
#         return Dataset.from_dict({
#             "anchor": anchors,
#             "positive": positives,
#             "negative": negatives
#         })
#
#     @staticmethod
#     def concat_and_split(
#             datasets: List[Dataset],
#             train_frac: float = CONFIG["TRAIN_FRAC"],
#             seed: int = CONFIG["SEED"]
#     ) -> DatasetDict:
#         """Concatenate multiple datasets and split into train/eval.
#
#         Args:
#             datasets: List of datasets to concatenate
#             train_frac: Fraction of data to use for training
#             seed: Random seed for reproducibility
#
#         Returns:
#             DatasetDict with 'train' and 'eval' splits
#         """
#         # Merge datasets
#         merged = concatenate_datasets(datasets)
#
#         # Shuffle and split on train and test
#         merged = merged.shuffle(seed=seed)
#         split = merged.train_test_split(train_size=train_frac, seed=seed)
#
#         # Return the result as DatasetDict
#         return DatasetDict({
#             "train": split["train"],
#             "eval": split["test"]
#         })
#
#
# # Exemplo de uso
# def main():
#     # Caminho para o dataset gerado pelo lexicographer
#     dataset_path = Path("dataset.json")
#
#     # Caminho alternativo
#     if not dataset_path.exists():
#         dataset_path = Path("../data/dataset.json")
#
#     if not dataset_path.exists():
#         print(f"❌ Dataset not found: {dataset_path}")
#         print("Run lexicographer_generator.py first to generate dataset.json")
#         return
#
#     # Carregar dados brutos
#     raw_data = TripletDatasetBuilder.load_raw(dataset_path)
#     print(f"✅ Loaded {len(raw_data)} entries from {dataset_path}")
#
#     # Construir triplets usando close_definition como negative
#     triplets = TripletDatasetBuilder.build_triplets(raw_data, negative_field="close_definition")
#     print(f"✅ Created {len(triplets)} triplets")
#     print(f"✅ Columns: {triplets.column_names}")
#
#     # Mostrar exemplo
#     print("\n📝 Exemplo de triplet:")
#     print(f"Anchor: {triplets[0]['anchor'][:100]}...")
#     print(f"Positive: {triplets[0]['positive']}")
#     print(f"Negative: {triplets[0]['negative']}")
#
#     # Concatenar e dividir (se houver múltiplos datasets)
#     dataset_dict = TripletDatasetBuilder.concat_and_split([triplets])
#     print(f"\n✅ Train size: {len(dataset_dict['train'])}")
#     print(f"✅ Eval size: {len(dataset_dict['eval'])}")
#
#     # Opcional: Salvar os datasets
#     # dataset_dict.save_to_disk("./triplet_dataset")
#
#
# if __name__ == "__main__":
#     main()





# """Triplet dataset builder for RAG bi‑encoder training."""
# import json
# from pathlib import Path
# from typing import List, Dict, Union
# import os
#
# from datasets import Dataset, DatasetDict, concatenate_datasets
# from custom_helpers import get_config
#
# CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))
#
#
# class TripletDatasetBuilder:
#     """Utilities to create `(anchor, positive, negative)` triplets."""
#
#     @staticmethod
#     def load_raw(path: Union[str, Path]) -> List[Dict]:
#         """Load raw dataset from JSON file.
#
#         Args:
#             path: Path to the dataset file
#
#         Returns:
#             List of dictionary items from the dataset
#
#         Raises:
#             FileNotFoundError: If the file doesn't exist
#         """
#         path = Path(path)
#         if not path.exists():
#             raise FileNotFoundError(f"Dataset file not found: {path}")
#         with path.open() as f:
#             return json.load(f)
#
#     @staticmethod
#     def build_triplets(raw: List[Dict], negative_field: str) -> Dataset:
#         """Create a dataset with (anchor, positive, negative) triplets.
#
#         Args:
#             raw: Raw dataset as a list of dictionaries
#             negative_field: Field name to use for negative examples
#
#         Returns:
#             Dataset with three columns: anchor, positive, negative
#         """
#         anchors, positives, negatives = [], [], []
#         for row in raw:
#             # TODO: update the lists above. Don't forget about the guard in case negative_field is not present in the row
#
#         return # TODO: create HF dataset with fields anchor, positive, negative
#
#     @staticmethod
#     def concat_and_split(
#         datasets: List[Dataset],
#         train_frac: float = CONFIG["TRAIN_FRAC"],
#         seed: int = CONFIG["SEED"]
#     ) -> DatasetDict:
#         """Concatenate multiple datasets and split into train/eval.
#
#         Args:
#             datasets: List of datasets to concatenate
#             train_frac: Fraction of data to use for training
#             seed: Random seed for reproducibility
#
#         Returns:
#             DatasetDict with 'train' and 'eval' splits
#         """
#         merged = # TODO: merge datasets
#         split = # TODO: split on train and test (note: don't forget to shuffle)
#         return # TODO: return the result