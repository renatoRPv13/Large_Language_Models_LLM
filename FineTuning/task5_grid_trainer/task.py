from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2)

import os
import json
# import logging
import itertools
# import pandas as pd
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union

import torch
import wandb
# import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback
)
from peft import PeftConfig, get_peft_model, PeftModel
from datasets import Dataset

from FineTuning.task5_grid_trainer.callbacks import LoggingCallback, ExampleGenerationCallback
from FineTuning.task4_helpers.task import Helpers

generate_definition_and_example = Helpers.get_example_and_definition
convert_to_serializable = Helpers.convert_to_serializable


class GridTrainer:
    """
    Uma classe para lidar com treinamento, avaliação e ajuste de hiperparâmetros para modelos PEFT.
    Attributes:
        training_args (Argumentos de Treinamento): Configuração para o treinamento.
        base_model (Modelo Pré-Treinado): O modelo base a ser ajustado.
        base_tokenizer (Tokenizador Pré-Treinado): Tokenizador para o modelo base.
        train_dataset (Conjunto de Dados): Conjunto de dados para treinamento.
        eval_dataset (Conjunto de Dados): Conjunto de dados para avaliação.
        peft_config (Configuração do PEFT): Configuração para o PEFT.
        test_samples (Opcional [Conjunto de Dados]): Conjunto de dados opcional para gerar exemplos.
    """

    def __init__(
            self,
            training_args: TrainingArguments,
            base_model: PreTrainedModel,
            base_tokenizer: PreTrainedTokenizer,
            train_dataset: Dataset,
            eval_dataset: Dataset,
            peft_config: PeftConfig,
            test_samples: Optional[Dataset] = None,
    ) -> None:
        self.training_args = training_args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.peft_config = peft_config
        self.test_samples = test_samples

        # Initialize default callbacks # Inicializar funções de retorno de chamada padrão
        self.default_callbacks = self._create_default_callbacks()
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

    def _create_default_callbacks(self) -> List[TrainerCallback]:
        """Crie funções de retorno de chamada padrão com caminhos iniciais."""
        return [
            ExampleGenerationCallback(
                self.test_samples,
                self.base_tokenizer,
                partial(generate_definition_and_example, params={"max_new_tokens": 20}),
                log_dir=self.training_args.output_dir
            ),
            LoggingCallback(log_path=os.path.join(self.training_args.output_dir, "training_logs.txt"))
        ]

    def _update_callbacks(self, output_dir: str) -> None:
        """Atualizar caminhos de retorno de chamada com o diretório de saída atual."""
        for callback in self.default_callbacks:
            if isinstance(callback, ExampleGenerationCallback):
                callback.log_dir = output_dir
            elif isinstance(callback, LoggingCallback):
                callback.log_path = os.path.join(output_dir, "training_logs.txt")

    def _init_wandb(self) -> None:
        """Inicializar o registro de pesos e vieses."""
        wandb.init(
            project="peft-training",
            name=self.training_args.run_name,
            config={
                "learning_rate": self.training_args.learning_rate,
                "batch_size": self.training_args.per_device_train_batch_size,
                "epochs": self.training_args.num_train_epochs,
                "peft_config": convert_to_serializable(self.peft_config.to_dict()),
            }
        )

    def _save_file(self, relative_path: str, data: Union[str, bytes]) -> None:
        """Salvar dados em arquivo no diretório de saída atual."""
        file_path = Path(self.training_args.output_dir) / relative_path
        with open(file_path, "w" if isinstance(data, str) else "wb") as f:
            f.write(data)

    def train(self, is_grid: bool = False) -> PeftModel:
        """Executar uma única sessão de treinamento."""
        if not is_grid:
            self._init_wandb()

        # Update callback paths # Atualizar caminhos de retorno de chamada
        self._update_callbacks(self.training_args.output_dir)

        # Configurar modelo e treinador
        peft_model = get_peft_model(self.base_model, self.peft_config)  # TODO: get peft model
        peft_model.print_trainable_parameters()

        # Coletor de dados para modelagem de linguagem
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.base_tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        trainer = Trainer(
            model=peft_model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=self.default_callbacks  # TODO: init Trainer
        )
        # Executar treinamento
        trainer.train()

        # Salvar artefatos
        trainer.save_model(Path(self.training_args.output_dir) / "best_model")
        self._save_file("training_args.json",
                        json.dumps(convert_to_serializable(self.training_args.to_dict()), indent=2))

        if not is_grid:
            wandb.finish()

        return peft_model

    def _update_params_grid(self, idx: int, params: Dict[str, Any]) -> None:
        """Atualizar o estado dos argumentos de treinamento para a busca em grade.
        Updates:
        1. Diretório de saída
        2. Nome da execução
        3. Sobrescrever diretório de saída
        4. Configuração do WandB
        5. Callbacks
        6. Argumentos de treinamento
        """
        # Ensure run_name is always a string and has a base value
        # Garanta que run_name seja sempre uma string e tenha um valor base
        if self.training_args.run_name is None:
            self.training_args.run_name = f"peft_run"
        
        # Clean up previous run_name suffix if it exists
        # Limpar o sufixo run_name anterior, se existir.
        if idx > 0 and self.training_args.run_name.endswith(f"_run_{idx - 1}"):
            self.training_args.run_name = self.training_args.run_name[:-len(f"_run_{idx - 1}")]

        # Update output directory # Atualizar diretório de saída
        run_dir = Path(self.training_args.output_dir) / f"run_{idx}"
        self.training_args.output_dir = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True) #Update output directory
                                                    #Update run name

        # Update run name # Atualizar nome da execução
        self.training_args.run_name = f"{self.training_args.run_name}_run_{idx}"
        self.training_args.overwrite_output_dir = True

        # Update training arguments # Atualizar argumentos de treinamento
        self.training_args = TrainingArguments(
            **{**self.training_args.to_dict(), **params}
        )
        # Inicialize o WandB para esta execução.
        self._init_wandb()
        wandb.config.update(params, allow_val_change=True)

        # Salvar parâmetros
        self._save_file("hyperparams.json", json.dumps(convert_to_serializable(params), indent=2))
        self._save_file(
            "wandb_config.json",
            json.dumps(convert_to_serializable(wandb.config.as_dict()), indent=2)
        )
        # Atualizar callbacks
        self._update_callbacks(self.training_args.output_dir)

    def _get_params_grid(self, grid_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Gere combinações de parâmetros para busca em grade."""
        # Obtenha todas as combinações de parâmetros.
        keys = grid_params.keys()
        values = grid_params.values()

        # Gere todas as combinações usando itertools.product
        combinations = list(itertools.product(*values))

        # Converter em lista de dicionários
        param_combinations = [dict(zip(keys, combo)) for combo in combinations]

        return param_combinations

    def grid_search(self, grid_params: Dict[str, List[Any]]) -> None:
        """Executar busca em grade de hiperparâmetros."""
        # Generate parameter combinations
        param_combinations = self._get_params_grid(grid_params)
        if wandb.run is not None:
            wandb.log({"total_runs": len(param_combinations)})

        for i, params in enumerate(param_combinations):
            print(f"\n=== Running grid search {i + 1}/{len(param_combinations)} ===")
            print(f"Parameters: {params}")

            # Update grid parameters # Atualizar parâmetros da grade
            self._update_params_grid(i, params)

            # Run training # Treinamento de corrida
            self.train(is_grid=True)

            wandb.log({"completed_runs": i + 1})

            # Clear GPU cache if using CUDA
            # Limpar o cache da GPU se estiver usando CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        wandb.finish()

        # estudar isso
    # def _update_params_grid(self, idx: int, params: Dict[str, Any]) -> None:
    #     """Atualizar o estado dos argumentos de treinamento para a busca em grade.
    #
    #     Updates:
    #     1. Diretório de saída
    #     2. Nome da execução
    #     3. Sobrescrever diretório de saída
    #     4. Configuração do WandB
    #     5. Callbacks
    #     6. Argumentos de treinamento
    #     """
    #     base_output_dir = getattr(self, "_grid_base_output_dir", None)
    #     if base_output_dir is None:
    #         base_output_dir = self.training_args.output_dir
    #         self._grid_base_output_dir = base_output_dir
    #
    #     base_run_name = getattr(self, "_grid_base_run_name", None)
    #     if base_run_name is None:
    #         base_run_name = self.training_args.run_name or Path(base_output_dir).name or "grid_search"
    #         self._grid_base_run_name = base_run_name
    #
    #     # Atualizar diretório de saída
    #     run_dir = Path(base_output_dir) / f"run_{idx}"
    #     run_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Atualizar argumentos de treinamento
    #     updated_args = {
    #         **self.training_args.to_dict(),
    #         **params,
    #         "output_dir": str(run_dir),
    #         "run_name": f"{base_run_name}_run_{idx}",
    #         "overwrite_output_dir": True,
    #     }
    #
    #     self.training_args = TrainingArguments(**updated_args)
    #
    #     # Inicialize o WandB para esta execução.
    #     self._init_wandb()
    #     wandb.config.update(params, allow_val_change=True)
    #
    #     # Salvar parâmetros
    #     self._save_file("hyperparams.json", json.dumps(convert_to_serializable(params), indent=2))
    #     self._save_file(
    #         "wandb_config.json",
    #         json.dumps(convert_to_serializable(wandb.config.as_dict()), indent=2)
    #     )
    #
    #     # Atualizar callbacks
    #     self._update_callbacks(self.training_args.output_dir)

   # if self.training_args.run_name.endswith(f"_run_{idx - 1}"):
        # está com valor None.
        # Como None não é uma string, ele não possui o método .endswith().
        """
        No TrainingArguments, o campo run_name é opcional. Se você não definiu explicitamente run_name ao criar os argumentos de treino, ele pode ficar como None.
        Durante o grid_search, o método _update_params_grid() tenta montar nomes 
        como
        """