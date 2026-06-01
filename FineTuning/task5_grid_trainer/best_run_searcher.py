# best_run_searcher.py
import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from peft import PeftModel, PeftConfig
from transformers import PreTrainedModel


class BestRunSearcher:
    """
    Finds and loads the best-performing model from grid search results.
    Attributes:
        search_dir (Path): Directory containing grid search results.
    Encontra e carrega o modelo com melhor desempenho a partir dos resultados da pesquisa em grade.
    Atributos:
        search_dir (Caminho): Diretório que contém os resultados da pesquisa em grade.
    """

    def __init__(self, search_dir: str) -> None:
        """
        Initializes the BestRunSearcher.
        Args:
            search_dir (str): Directory containing grid search results.
        Inicializa o BestRunSearcher.
        Argumentos:
            search_dir (str): Diretório contendo os resultados da pesquisa em grade.
        """
        self.search_dir = Path(search_dir)

    def find_best_run(self) -> Tuple[Path, pd.DataFrame]:
        """
        Finds the best-performing run from grid search results.
        Returns:
            Tuple[Path, pd.DataFrame]: 
                - Path to the best checkpoint directory
                - DataFrame of all results with hyperparameters and eval losses
        Encontra a execução com melhor desempenho a partir dos resultados da busca em grade.
        Retorna:
            Tupla[Caminho, pd.DataFrame]:
            - Caminho para o diretório do melhor checkpoint
            - DataFrame com todos os resultados, incluindo hiperparâmetros e perdas de avaliação
        """
        results = []

        for run_dir in self._iter_run_dirs():
            try:
                checkpoint_dir, eval_loss = self._find_best_checkpoint(run_dir)
                params = self._load_hyperparams(run_dir)

                results.append({
                    "run_path": str(run_dir),
                    "checkpoint_path": str(checkpoint_dir),
                    "eval_loss": eval_loss,
                    **params
                })

            except Exception as e:
                #print(f"Skipping run {run_dir.name}: {str(e)}")
                print(f"Ignorando execução {run_dir.name}: {str(e)}")
                continue

        if not results:
            raise ValueError("No valid runs found in grid search results")

        results = pd.DataFrame(results)
        results.sort_values("eval_loss", inplace=True, ascending=True)
        best_checkpoint = Path(results.iloc[0]["checkpoint_path"])
        return best_checkpoint, results

    def _iter_run_dirs(self):
        """Retorna diretórios de execução válidos dentro do diretório de pesquisa."""
        """Retorna diretórios de execução válidos dentro do diretório de pesquisa."""
        return (d for d in self.search_dir.iterdir() if d.is_dir())

    def _find_best_checkpoint(self, run_dir: Path) -> Tuple[Path, float]:
        """
        Encontra o melhor ponto de verificação em um diretório de execução.

        Args:
            run_dir (Caminho): Caminho para o diretório de execução.

        Returns:
            Tupla[Caminho, float]: Caminho para o melhor ponto de verificação e sua perda de avaliação.
        """
        checkpoints = [d for d in run_dir.glob("checkpoint-*") if d.is_dir()]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {run_dir}")

        best_checkpoint, best_loss = None, float('inf')

        for checkpoint in checkpoints:
            try:
                with open(checkpoint / "trainer_state.json") as f:
                    loss = json.load(f).get("best_metric", float('inf'))

                if loss is not None and loss < best_loss:
                    best_checkpoint, best_loss = checkpoint, loss

            except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                #print(f"Ignorando o ponto de verificação {checkpoint.name}: {str(e)}")
                print(f"Ignorando o ponto de verificação {checkpoint.name}: {str(e)}")
                continue

        return best_checkpoint, best_loss

    def _load_hyperparams(self, run_dir: Path) -> Dict:
        """
        Carrega hiperparâmetros de um diretório de execução.
        Args:
            run_dir (Caminho): Caminho para o diretório de execução.
        Returns:
            Dicionário: Dicionário de hiperparâmetros.
        """
        with open(run_dir / "hyperparams.json") as f:
            return json.load(f)

    @staticmethod
    def load_peft_model(base_model: PreTrainedModel, checkpoint_dir: Path) -> PeftModel:
        """
            Carrega um modelo PEFT a partir de um diretório de pontos de verificação.
        Args:
            base_model (ModeloPré-Treinado): Modelo base original.
            checkpoint_dir (Caminho): Caminho para o diretório de checkpoint.
        Returns:
            PeftModel: Modelo PEFT carregado.
            PeftModel: Loaded PEFT model.
        """
        return PeftModel.from_pretrained(
            base_model,
            str(checkpoint_dir),
            is_trainable=False,
            config=PeftConfig.from_pretrained(str(checkpoint_dir))
        )