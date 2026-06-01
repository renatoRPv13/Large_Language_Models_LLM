import json
import logging
import os
import torch
import wandb
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
from transformers import TrainerCallback
from datasets import Dataset
from pathlib import Path


class LoggingCallback(TrainerCallback):
    """
    A callback to log training metrics to a file and console.
    Attributes:
        logger (logging.Logger): Logger instance for saving logs.
        log_path (str): Path to the log file.
    Uma função de retorno de chamada para registrar métricas de treinamento em um arquivo e no console.
    Atributos:
        logger (logging.Logger): Instância do logger para salvar os registros.
        log_path (str): Caminho para o arquivo de registro.
    """
    def __init__(self, log_path: str) -> None:
        """
        Initializes the LoggingCallback.
        Args:
            log_path (str): Path to the log file.
        Inicializa a função de retorno de chamada de registro (LoggingCallback).

        Argumentos:
            log_path (str): Caminho para o arquivo de log.
        """
        self.logger = logging.getLogger(__name__)
        self.log_path = log_path
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def on_log(self, args, state, control, logs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Logs training metrics to a file and console.
        Args:
            args: Training arguments.
            state: Training state.
            control: Control object for callbacks.
            logs (Optional[Dict[str, Any]]): Dictionary of logs.
            **kwargs: Additional arguments.
        Registra as métricas de treinamento em um arquivo e no console.
        Argumentos:
            args: Argumentos de treinamento.
            state: Estado do treinamento.
            control: Objeto de controle para callbacks.
            logs (Opcional[Dict[str, Any]]): Dicionário de logs.
            **kwargs: Argumentos adicionais.
        """
        if state.is_world_process_zero and logs:
            self.logger.info(json.dumps(logs))
            print(f"Step {state.global_step}: {logs}")


class ExampleGenerationCallback(TrainerCallback):
    """
    A callback to generate example outputs during evaluation.
    Attributes:
        test_samples (Dataset): Dataset for generating examples.
        tokenizer (Callable): Tokenizer for the model.
        generation_fn (Callable): Function to generate examples.
          Function syntax: generation_fn(model, tokenizer, word, device)
        log_dir (str): Directory to save logs.
        num_samples (int): Number of samples to generate during evaluation.
    Uma função de retorno de chamada para gerar exemplos de saída durante a avaliação.
    Atributos:
        test_samples (Dataset): Conjunto de dados para gerar exemplos.
        tokenizer (Callable): Tokenizador para o modelo.
        generation_fn (Callable): Função para gerar exemplos.
    Sintaxe da função: generation_fn(model, tokenizer, word, device)
        log_dir (str): Diretório para salvar os logs.
        num_samples (int): Número de exemplos a serem gerados durante a avaliação.
    """

    def __init__(
        self,
        test_samples: Dataset,
        tokenizer: Callable,
        generation_fn: Callable,
        log_dir: str,
        num_samples: int = 3,
    ) -> None:
        """
        Initializes the ExampleGenerationCallback.
        Args:
            test_samples (Dataset): Dataset for generating examples.
            tokenizer (Callable): Tokenizer for the model.
            generation_fn (Callable): Function to generate examples.
            log_dir (str): Directory to save logs.
            num_samples (int): Number of samples to generate during evaluation.
        Inicializa a função de retorno de chamada `ExampleGenerationCallback`.
        Argumentos:
            `test_samples` (Conjunto de dados): Conjunto de dados para gerar exemplos.
            `tokenizer` (Chamável): Tokenizador para o modelo.
            generation_fn` (Chamável): Função para gerar exemplos.
            `log_dir` (str): Diretório para salvar os logs.
            `num_samples` (int): Número de exemplos a serem gerados durante a avaliação.
        """
        self.test_samples = test_samples
        self.tokenizer = tokenizer
        self.generation_fn = generation_fn
        self.log_dir = log_dir
        self.num_samples = num_samples
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    def on_evaluate(self, args, state, control, **kwargs) -> None:
        """
        Generates examples during evaluation and logs them.
        Args:
            args: Training arguments.
            state: Training state.
            control: Control object for callbacks.
            **kwargs: Additional arguments.
        Gera exemplos durante a avaliação e os registra.
        Argumentos:
            args: Argumentos de treinamento.
            state: Estado do treinamento.
            control: Objeto de controle para callbacks.
            **kwargs: Argumentos adicionais.
        """
        model = kwargs.pop("model", None)
        if model is None:
            logging.warning("Model not found. Skipping example generation.")
            return

        # Select random samples # Selecionar amostras aleatórias
        indices = torch.randperm(len(self.test_samples))[: self.num_samples]
        results = []

        for idx in indices:
            sample = self.test_samples[int(idx)]
            generated = self.generation_fn(
                model, self.tokenizer, sample["word"], device=model.device
            )
            results.append({
                "word": sample["word"],
                "true_definition": sample["definition"],
                "gen_definition": generated["definition"],
                "true_example": sample["example"],
                "gen_example": generated["example"],
            })

        # Save and log # Salvar e registrar
        file_path = Path(self.log_dir) / f"generations_step_{state.global_step}.json"
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2)

        wandb.log({"examples": wandb.Table(dataframe=pd.DataFrame(results))})