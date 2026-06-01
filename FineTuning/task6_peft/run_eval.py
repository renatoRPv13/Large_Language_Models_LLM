"""run_eval.py – Evaluation Pipeline for PEFT Model Fine-Tuning
This script does not perform training. It simply loads the best model checkpoint
(for the chosen PEFT method), prints a nicely formatted validation results dataframe,
and shows generated outputs on a few sample words.

run_eval.py – Pipeline de Avaliação para Ajuste Fino do Modelo PEFT
Este script não realiza treinamento. Ele simplesmente carrega o melhor checkpoint do modelo
(para o método PEFT escolhido), imprime um dataframe de resultados de validação bem formatado
e exibe as saídas geradas em algumas palavras de exemplo.
"""

from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import random
import warnings
from typing import Tuple

import torch
import click
import pandas as pd
from tabulate import tabulate  # For nicely formatted tables # Para tabelas bem formatadas
from transformers import TrainingArguments
from peft import PeftConfig

from FineTuning.task3_dataset.task import DatasetHandler
from FineTuning.task4_helpers.task import Helpers
from FineTuning.task5_grid_trainer.best_run_searcher import BestRunSearcher
from FineTuning.task6_peft.run_train import load_components, choose_config

from configs import PeftIA3Config, PeftLoRAConfig, PeftPromptTuningConfig, BaseConfig

# Environment settings and warning suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)


@click.command()
@click.option("--method", type=click.Choice(["lora", "prompt_tuning", "ia3"]), required=True)
def main(method: str):
    # Choose configuration based on method # Escolha a configuração com base no método
    conf_path = "../conf.yaml"
    config = choose_config(method, conf_path)
    
    # Load core components: model, tokenizer, datasets, and sample words
    # Carregar componentes principais: modelo, tokenizador, conjuntos de dados e palavras de exemplo
    base_model, base_tokenizer, _, test_dataset, sample = load_components(config)
    base_model.use_position_ids = False  # Disable position IDs to avoid warnings # Desative os IDs de posição para evitar avisos
    
    # Search for the best checkpoint in the output directory
    # Procure o melhor ponto de verificação no diretório de saída
    searcher = BestRunSearcher(search_dir=config.training_config["output_dir"])
    best_checkpoint, results = searcher.find_best_run()
    
    click.echo(f"\nBest checkpoint: {best_checkpoint}")
    click.echo("\nValidation results ranking:")
    
    # Nicely print results as a table if it is a DataFrame
    # Exiba os resultados de forma organizada como uma tabela, caso seja um DataFrame.
    if isinstance(results, pd.DataFrame):
        table = tabulate(results, headers='keys', tablefmt='psql', showindex=False)
        click.echo(table)
    else:
        click.echo(results)
    
    # Load the best model from the checkpoint  # Carregar o melhor modelo do ponto de verificação
    best_model = BestRunSearcher.load_peft_model(
        base_model=base_model,
        checkpoint_dir=best_checkpoint
    )
    
    # Generate sample outputs using the best model # Gere exemplos de saída usando o melhor modelo
    click.echo("\nSample outputs from best model:")
    params = {"temperature": 0.7, "max_new_tokens": 20}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for word_data in sample:
        word = word_data["word"]
        output = Helpers.get_example_and_definition(
            model=best_model,
            tokenizer=base_tokenizer,
            word=word,
            params=params,
            device=device
        )
        click.echo(f"\nWord: {word}")
        click.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()