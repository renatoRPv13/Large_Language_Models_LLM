"""Bi‑encoder fine‑tuning with LoRA."""
"Ajuste fino de bi-encoders com LoRa."
from pathlib import Path
from typing import Union
import os

from datasets import DatasetDict, load_from_disk, disable_caching
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import TripletLoss
from sentence_transformers.evaluation import TripletEvaluator
from peft import LoraConfig, TaskType

from custom_helpers import get_config

# Carregar configuração
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
CONFIG = get_config(config_path)

# Garantir que as chaves existem
if "RAG_DATASET_DIR" not in CONFIG:
    CONFIG["RAG_DATASET_DIR"] = "./rag_dataset"
if "MODEL_OUT" not in CONFIG:
    CONFIG["MODEL_OUT"] = "./models/bi-encoder-lora"
if "MODEL_BASE" not in CONFIG:
    CONFIG["MODEL_BASE"] = "sentence-transformers/all-MiniLM-L6-v2"
if "EPOCHS" not in CONFIG:
    CONFIG["EPOCHS"] = 3
if "BATCH_SIZE" not in CONFIG:
    CONFIG["BATCH_SIZE"] = 16
if "EVAL_STEPS" not in CONFIG:
    CONFIG["EVAL_STEPS"] = 100

disable_caching()

print("✅ Configuração carregada:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

def get_peft_model(
    base_model: str = CONFIG["MODEL_BASE"],
    r: int = 16,
    alpha: int = 32,
) -> SentenceTransformer:
    """Initialize a PEFT model with LoRA adapters.
    Args:
        base_model: Base model identifier
        r: LoRA rank parameter
        alpha: LoRA alpha parameter
    Returns:
        SentenceTransformer model with LoRA adapters
    Inicializa um modelo PEFT com adaptadores LoRA.
    Argumentos:
        base_model: Identificador do modelo base
        r: Parâmetro de classificação LoRA
        alpha: Parâmetro alfa LoRA
    Retorna:
        Modelo SentenceTransformer com adaptadores LoRA
    """
    model = SentenceTransformer(
        base_model,
        trust_remote_code=True,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
        ),
    )

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["query", "key", "value"],
    )
    model.add_adapter(lora_cfg)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"🧠  Modelo inicializado com {trainable_params:,} parâmetros treináveis")
    if total_params > 0:
        print(
            f"📊 Parâmetros treináveis: {trainable_params:,} / {total_params:,} "
            f"({trainable_params / total_params:.2%})"
        )

    return model

def get_datasets(path: Union[str, Path]) -> DatasetDict:
    """Load datasets from disk.
    Args:
        path: Path to the datasets directory
    Returns:
        DatasetDict with 'train' and 'eval' splits
    Carregar conjuntos de dados do disco.
    Argumentos:
        caminho: Caminho para o diretório de conjuntos de dados
    Retorno:
        Dicionário de conjuntos de dados com divisões 'treino' e 'avaliação'
    """
    dataset = load_from_disk(Path(path))
    print("📚 Conjuntos de dados carregados:") # Loaded datasets
    print(f"  - Treinamento: {len(dataset['train']):,} examples") # Training
    print(f"  - Avaliação: {len(dataset['eval']):,} examples") # Evaluation
    return dataset

def configure_training(
    model: SentenceTransformer,
    datasets: DatasetDict,
    epochs: int = CONFIG["EPOCHS"],
    batch_size: int = CONFIG["BATCH_SIZE"],
    eval_steps: int = CONFIG["EVAL_STEPS"],
) -> SentenceTransformerTrainer:
    """Configure the sentence transformer training.
    Args:
        model: SentenceTransformer model
        datasets: DatasetDict with train and eval splits
        epochs: Number of training epochs
        batch_size: Batch size for training
        eval_steps: Steps between evaluations
    Returns:
        Configured SentenceTransformerTrainer
    Configure o treinamento do SentenceTransformer.
    Argumentos:
        model: Modelo SentenceTransformer
        datasets: DatasetDict com divisões para treino e avaliação
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do lote para treinamento
        eval_steps: Passos entre avaliações
    Retorna:
        SentenceTransformerTrainer configurado
    """
    args = SentenceTransformerTrainingArguments(
        output_dir=CONFIG["MODEL_OUT"],
        load_best_model_at_end=False,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        fp16=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
    )
    evaluator = TripletEvaluator(
        anchors=datasets["eval"]["anchor"],
        positives=datasets["eval"]["positive"],
        negatives=datasets["eval"]["negative"],
        name="eval",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        loss=TripletLoss(model=model),
        evaluator=evaluator,
    )

    return trainer


def run_training(trainer: SentenceTransformerTrainer) -> None:
    """Run the training process.
    Args:
        trainer: Configured SentenceTransformerTrainer
    Execute o processo de treinamento.
    Argumentos:
        trainer: SentenceTransformerTrainer configurado"
    """
    print("🚀 Início do treinamento...") # Starting training
    trainer.train()

    Path(CONFIG["MODEL_OUT"]).mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(CONFIG["MODEL_OUT"])
    print(f"✅ Modelo salvo em {CONFIG['MODEL_OUT']}") # Model saved to






#
# """Bi‑encoder fine‑tuning with LoRA."""
# from pathlib import Path
# from typing import Union
# import os
#
# from datasets import DatasetDict, load_from_disk, disable_caching
# from sentence_transformers import (
#     SentenceTransformer,
#     SentenceTransformerTrainer,
#     SentenceTransformerTrainingArguments,
#     SentenceTransformerModelCardData,
# )
# from sentence_transformers.losses import TripletLoss
# from sentence_transformers.evaluation import TripletEvaluator
# from peft import LoraConfig, TaskType
#
# from custom_helpers import get_config
#
# # Carregar configuração
# config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
# CONFIG = get_config(config_path)
#
# # Garantir que as chaves existem
# if "RAG_DATASET_DIR" not in CONFIG:
#     CONFIG["RAG_DATASET_DIR"] = "./rag_dataset"
# if "MODEL_OUT" not in CONFIG:
#     CONFIG["MODEL_OUT"] = "./models/bi-encoder-lora"
# if "MODEL_BASE" not in CONFIG:
#     CONFIG["MODEL_BASE"] = "sentence-transformers/all-MiniLM-L6-v2"
# if "EPOCHS" not in CONFIG:
#     CONFIG["EPOCHS"] = 3
# if "BATCH_SIZE" not in CONFIG:
#     CONFIG["BATCH_SIZE"] = 16
# if "EVAL_STEPS" not in CONFIG:
#     CONFIG["EVAL_STEPS"] = 100
#
# disable_caching()
#
# print("✅ Configuração carregada:")
# for key, value in CONFIG.items():
#     print(f"  {key}: {value}")
#
# def get_peft_model(
#     base_model: str = CONFIG["MODEL_BASE"],
#     r: int = 16,
#     alpha: int = 32,
# ) -> SentenceTransformer:
#     """Initialize a PEFT model with LoRA adapters.
#
#     Args:
#         base_model: Base model identifier
#         r: LoRA rank parameter
#         alpha: LoRA alpha parameter
#
#     Returns:
#         SentenceTransformer model with LoRA adapters
#     """
#     model = SentenceTransformer(
#         base_model,
#         trust_remote_code=True,
#         model_card_data=SentenceTransformerModelCardData(
#             language="en",
#             license="apache-2.0",
#         ),
#     )
#
#     lora_cfg = LoraConfig(
#         r=r,
#         lora_alpha=alpha,
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.FEATURE_EXTRACTION,
#         target_modules=["query", "key", "value"],
#     )
#
#     model.add_adapter(lora_cfg)
#
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_params = sum(p.numel() for p in model.parameters())
#
#     print(f"🧠 Model initialized with {trainable_params:,} trainable parameters")
#     if total_params > 0:
#         print(
#             f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} "
#             f"({trainable_params / total_params:.2%})"
#         )
#
#     return model
#
# def get_datasets(path: Union[str, Path]) -> DatasetDict:
#     """Load datasets from disk.
#
#     Args:
#         path: Path to the datasets directory
#
#     Returns:
#         DatasetDict with 'train' and 'eval' splits
#     """
#     dataset = load_from_disk(Path(path))
#     print("📚 Loaded datasets:")
#     print(f"  - Training: {len(dataset['train']):,} examples")
#     print(f"  - Evaluation: {len(dataset['eval']):,} examples")
#     return dataset
#
#
# def configure_training(
#     model: SentenceTransformer,
#     datasets: DatasetDict,
#     epochs: int = CONFIG["EPOCHS"],
#     batch_size: int = CONFIG["BATCH_SIZE"],
#     eval_steps: int = CONFIG["EVAL_STEPS"],
# ) -> SentenceTransformerTrainer:
#     """Configure the sentence transformer training.
#
#     Args:
#         model: SentenceTransformer model
#         datasets: DatasetDict with train and eval splits
#         epochs: Number of training epochs
#         batch_size: Batch size for training
#         eval_steps: Steps between evaluations
#
#     Returns:
#         Configured SentenceTransformerTrainer
#     """
#     args = SentenceTransformerTrainingArguments(
#         output_dir=CONFIG["MODEL_OUT"],
#         load_best_model_at_end=False,
#         gradient_accumulation_steps=4,
#         warmup_ratio=0.05,
#         learning_rate=1e-4,
#         lr_scheduler_type="cosine",
#         fp16=True,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_steps=10,
#     )
#
#     evaluator = TripletEvaluator(
#         anchors=datasets["eval"]["anchor"],
#         positives=datasets["eval"]["positive"],
#         negatives=datasets["eval"]["negative"],
#         name="eval",
#     )
#
#     trainer = SentenceTransformerTrainer(
#         model=model,
#         args=args,
#         train_dataset=datasets["train"],
#         eval_dataset=datasets["eval"],
#         loss=TripletLoss(model=model),
#         evaluator=evaluator,
#     )
#
#     return trainer
#
#
# def run_training(trainer: SentenceTransformerTrainer) -> None:
#     """Run the training process.
#
#     Args:
#         trainer: Configured SentenceTransformerTrainer
#     """
#     print("🚀 Starting training...")
#     trainer.train()
#
#     Path(CONFIG["MODEL_OUT"]).mkdir(parents=True, exist_ok=True)
#
#     trainer.model.save_pretrained(CONFIG["MODEL_OUT"])
#     print(f"✅ Model saved to {CONFIG['MODEL_OUT']}")

