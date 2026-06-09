"""Run bi-encoder fine-tuning with LoRA."""
"Execute o ajuste fino do bi-encoder com LoRa."
import os
import sys
import click
from pathlib import Path

# Adicionar o diretório RAG ao sys.path para encontrar custom_helpers
rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

# Adicionar o diretório raiz do projeto para importações absolutas
project_root = os.path.abspath(os.path.join(rag_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from custom_helpers import get_config
# Usar importação absoluta para evitar ambiguidade
from RAG.task3_training.task import (
    get_peft_model,
    get_datasets,
    configure_training,
    run_training
)

# Carregar configuração a partir da pasta RAG
config_path = os.path.join(rag_path, "config.yaml")
CONFIG = get_config(config_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@click.command()
@click.option(
    "--data",
    default=CONFIG.get("RAG_DATASET_DIR", "./rag_dataset"),
    help="Path to the processed dataset directory"
)
@click.option(
    "--epochs",
    default=CONFIG.get("EPOCHS", 3),
    help="Number of training epochs"
)
def main(data: str, epochs: int) -> None:
    """Run bi-encoder fine-tuning with LoRA."""
    "Execute o ajuste fino do bi-encoder com LoRa."
    # Construir o caminho completo para os dados a partir da pasta RAG
    data_path = Path(rag_path) / data
    if not data_path.exists():
        click.echo(f"❌ Dataset path does not exist: {data_path}")
        return

    click.echo(f"🔄 Initializing PEFT model...")
    model = get_peft_model()

    click.echo(f"📚 Loading datasets from {data_path}...")
    datasets = get_datasets(data_path)

    click.echo(f"🛠️ Configuring training (epochs={epochs})...")
    trainer = configure_training(model, datasets, epochs=epochs)

    click.echo(f"🚀 Starting training...")
    run_training(trainer)


if __name__ == "__main__":
    main()

# pip install xformers