"""
Module with implementation of APIBase class (Base class for API classes).]
Módulo com implementação da classe APIBase (classe base para classes de API).
"""
from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

import numpy as np
import torch

from torch.nn import functional as F
from LanguageModeling.task01_ngrams.task import BOS, EOS, UNK
from LanguageModeling.task01_ngrams.download_data import get_train_test


class MyDefaultDict(dict):
    """Custom default dict that prints a message when a key is not found (for debugging purposes)"""
    """Dicionário padrão personalizado que imprime uma mensagem quando uma chave não é encontrada (para fins de depuração)"""

    def __missing__(self, key: str) -> str:
        """Get the value of the missing key and print a message.

        :param key: Missing key.
        :return: Value associated with the missing key.

        Obtém o valor da chave ausente e exibe uma mensagem.
        :param key: Chave ausente.
        :return: Valor associado à chave ausente.
        """
        # Custom behavior when key is not found # Comportamento personalizado quando a chave não for encontrada
        # print(f"Token '{key}' not found in the vocabulary. Replacing with '{UNK}' token.")
        print(f"Token '{key}' não encontrado no vocabulário. Substituindo pelo token '{UNK}'.")
        self[key] = self[UNK]
        return self[key]


def get_token_to_id_mapping_and_tokens() -> tuple[dict, list]:
    """
    Get char-level tokenization

    :return: Tuple with token_to_id mapping dictionary and tokens list.
    Obter tokenização em nível de caractere
    :return: Tupla com dicionário de mapeamento token_para_id e lista de tokens.
    """
    train_lines, test_lines = get_train_test()
    tokens = set()
    [tokens.update(set(line)) for line in train_lines]
    tokens = list(tokens)
    # add BOS (same as UNK) and EOS tokens # Adicionar tokens BOS (igual a UNK) e EOS
    tokens = [BOS, EOS] + tokens
    token_to_id = {token: i for i, token in enumerate(tokens)}

    token_to_id = MyDefaultDict(token_to_id)

    return token_to_id, tokens


class TextTools:
    """
    A class that provides useful tools for text processing

    TOKEN_TO_ID: a dictionary that maps tokens to their integer ids
    Uma classe que fornece ferramentas úteis para processamento de texto.
    TOKEN_TO_ID: um dicionário que mapeia tokens para seus IDs inteiros.
    """
    TOKEN_TO_ID, TOKENS = get_token_to_id_mapping_and_tokens()

    @classmethod
    def to_matrix(cls, lines: list, max_len: int = None, pad: int = None, dtype=np.int64) -> torch.Tensor:
        """Casts a list of lines into torch-digestable matrix

        :param lines: List of lines.
        :param max_len: Max length of the lines.
        :param pad: Padding value.
        :param dtype: Data type.
        :return: Tensor matrix representation of the lines

        Converte uma lista de linhas em uma matriz processável pelo PyTorch.

        :param lines: Lista de linhas.
        :param max_len: Comprimento máximo das linhas.
        :param pad: Valor de preenchimento.
        :param dtype: Tipo de dados.
        :return: Representação matricial tensorial das linhas.
        """
        lines = list(map(str.lower, lines))
        max_len = max_len or max(map(len, lines))
        pad = pad if pad is not None else cls.TOKEN_TO_ID[EOS]
        lines_matrix = np.full([len(lines), max_len], pad, dtype=dtype)

        for i in range(len(lines)):
            line_ix = list(map(cls.TOKEN_TO_ID.__getitem__, lines[i][:max_len]))
            # Converter cada caractere da linha em seu ID de token
            line_ix = list(map(cls.TOKEN_TO_ID.__getitem__, lines[i][:max_len]))
            # Preencha a linha correspondente da matriz de linhas
            lines_matrix[i, :len(line_ix)] = line_ix

        return torch.tensor(lines_matrix)

    @classmethod
    def compute_mask(cls, input_idx: torch.Tensor, eos_idx: int = None) -> torch.Tensor:
        """ compute a boolean mask that equals "1" until first EOS (including that EOS)

        :param input_idx: Input indices.
        :param eos_idx: Index of the EOS token.
        :return: Boolean mask tensor.

        Calcula uma máscara booleana que seja igual a "1" até o primeiro EOS (incluindo esse EOS).
        :param input_idx: Índices de entrada.
        :param eos_idx: Índice do token EOS.
        :return: Tensor da máscara booleana.
        """
        eos_idx = eos_idx or cls.TOKEN_TO_ID[EOS]
        return F.pad(torch.cumsum(input_idx == eos_idx, dim=-1)[..., :-1] < 1, pad=(1, 0, 0, 0), value=True)


def main():
    token_to_id, tokens = get_token_to_id_mapping_and_tokens()
    print(f"Tamanho do vocabulário: {len(tokens)} tokens")
    sample_text = "Olá mundo"
    matrix = TextTools.to_matrix([sample_text])
    print(f"Texto '{sample_text}' como formato de matriz: {matrix.shape}")
    mask = TextTools.compute_mask(matrix)
    print(f"Formato da máscara calculada: {mask.shape}")


if __name__ == '__main__':
    main()