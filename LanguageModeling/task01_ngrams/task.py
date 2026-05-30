"""
Module with implementation of APIBase class (Base class for API classes).
Módulo com implementação da classe APIBase (classe base para classes de API).
"""
from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

from collections import defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Dict, Tuple, DefaultDict, Iterable

BOS, EOS = "?", "?"  # one-letter tokens (not the best practice) # Tokens de uma letra (não é a melhor prática)
UNK = BOS


class BaseLanguageModel(ABC):
    """
    Abstract base class for language models.
    Classe base abstrata para modelos de linguagem.
    """
    BOS = BOS
    EOS = EOS
    UNK = UNK

    @abstractmethod
    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get possible next tokens and their probabilities given a prefix.
        :param prefix: string with space-separated prefix tokens
        :return: Dictionary {token : probability} for all tokens with positive probabilities
        Obtém os próximos tokens possíveis e suas probabilidades, dado um prefixo.
        :param prefix: string com os tokens do prefixo separados por espaços
        :return: Dicionário {token : probabilidade} para todos os tokens com probabilidades positivas
        """
        pass

    @abstractmethod
    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of the next token given a prefix.
        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :return: Probability of next_token given prefix, a value between 0 and 1
        Obtém a probabilidade do próximo token dado um prefixo.
        :param prefix: string com os tokens do prefixo separados por espaços
        :param next_token: o próximo token para o qual a probabilidade deve ser prevista
        :return: Probabilidade de next_token dado o prefixo, um valor entre 0 e 1
        """
        pass


class NGramLanguageModel(BaseLanguageModel):
    """
    Class for a simple n-gram language model.
    Classe para um modelo de linguagem n-gram simples.
    """

    @staticmethod
    def count_ngrams(lines: Iterable[str], n: int) -> DefaultDict[Tuple[str, ...], Counter[str]]:
        """
        Count n-grams occurrences in the lines.

        :param lines: an iterable of strings with space-separated tokens
        :param n: n-gram size
        :return: Dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}

        When building counts, please consider the following two edge cases:
        - if prefix is shorter than (n - 1) tokens, it should be padded with UNK. For n=3,
        empty prefix: "" -> (UNK, UNK)
        short prefix: "the" -> (UNK, the)
        long prefix: "the new approach" -> (new, approach)
        - you should add a special token, EOS, at the end of each sequence
        "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
        count the probability of this token just like all others.
        Contar ocorrências de n-gramas nas linhas.
        :param linhas: um iterável de strings com tokens separados por espaço
        :param n: tamanho do n-grama
        :return: Dicionário { tupla(prefixo_tokens): {próximo_token_1: contagem_1, próximo_token_2: contagem_2}}
        Ao construir as contagens, considere os dois casos extremos a seguir:
        - se o prefixo for menor que (n - 1) tokens, ele deve ser preenchido com UNK. Para n=3,
        prefixo vazio: "" -> (UNK, UNK)
        prefixo curto: "o" -> (UNK, o)
        prefixo longo: "a nova abordagem" -> (nova, abordagem)
        - você deve adicionar um token especial, EOS, ao final de cada sequência
        "... com redes neurais profundas ." -> (..., com, redes, neurais, profundas, ., EOS)
        conte a probabilidade deste token da mesma forma que todos os outros.
        """
        counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)

        for line in lines:
            # Tokenize the line # Tokenizar a linha
            tokens = line.split()

            # Add EOS token at the end # Adicione o token EOS no final
            tokens.append(EOS)

            # Add padding at the beginning with UNK tokens # Adicionar preenchimento no início com tokens UNK
            padded_tokens = [UNK] * (n - 1) + tokens

            # Slide window of size n over the padded tokens # Deslize a janela de tamanho n sobre os tokens acolchoados
            for i in range(len(tokens) + n - 1):
                if i + n <= len(padded_tokens):
                    # Prefix is first (n-1) tokens, next token is the nth token # O prefixo corresponde aos primeiros (n-1) tokens, o próximo token corresponde ao n-ésimo token
                    prefix = tuple(padded_tokens[i:i + n - 1])
                    next_token = padded_tokens[i + n - 1]
                    counts[prefix][next_token] += 1

        return counts

    def __init__(self, lines: Iterable[str], n: int) -> None:
        """
        Initialize the n-gram language model.

        :param n: n-gram size
        :param lines: an iterable of strings with space-separated tokens
        Inicializa o modelo de linguagem n-gram.
        :param n: tamanho do n-gram
        :param lines: um iterável de strings com tokens separados por espaço
        """
        assert n >= 1
        self.n: int = n

        counts = self.count_ngrams(lines, self.n)

        # compute token probabilities given counts # calcular probabilidades de tokens dados os counts
        self.probs: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)

        # populate self.probs with actual probabilities # Preencha self.probs com probabilidades reais
        for pref, cnt in counts.items():
            total = sum(cnt.values())
            for token, count in cnt.items():
                # Probability = count / total (relative frequency) # Probabilidade = contagem / total (frequência relativa)
                self.probs[pref][token] = count / total

    def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
        """
        Get possible next tokens and their probabilities given a prefix.

        :param prefix: string with space-separated prefix tokens
        :return: Dictionary {token : probability} for all tokens with positive probabilities

        Steps:
        1. Complete or truncate the prefix if needed (note: for completion add UNK tokens in the beginning)
        2. Use the distribution of tokens given prefix (use self.probs)
        Obtenha os próximos tokens possíveis e suas probabilidades dado um prefixo.
        :param prefix: string com os tokens do prefixo separados por espaços
        :return: Dicionário {token : probabilidade} para todos os tokens com probabilidades positivas
        Passos:
        1. Complete ou trunque o prefixo, se necessário (nota: para completar, adicione tokens UNK no início)
        2. Use a distribuição de tokens dado o prefixo (use self.probs)
        """
        prefix_tokens = prefix.split()

        # Pad or truncate to get exactly (n-1) tokens # Preencha ou trunque para obter exatamente (n-1) tokens
        if len(prefix_tokens) > self.n - 1:
            # Take only the last (n-1) tokens # Selecione apenas os últimos (n-1) tokens
            relevant_tokens = prefix_tokens[-(self.n - 1):]
        else:
            # Pad with UNK at the beginning # Comece com UNK
            relevant_tokens = [UNK] * (self.n - 1 - len(prefix_tokens)) + prefix_tokens

        prefix_tuple = tuple(relevant_tokens)

        # Return the distribution or empty dict if prefix not found
        # Retorna a distribuição ou um dicionário vazio se o prefixo não for encontrado
        return dict(self.probs.get(prefix_tuple, {}))

    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """
        Get probability of the next token given a prefix.

        :param prefix: string with space-separated prefix tokens
        :param next_token: the next token to predict probability for
        :return: Probability of next_token given prefix, a value between 0 and 1
        Obtém a probabilidade do próximo token dado um prefixo.
        :param prefix: string com os tokens do prefixo separados por espaços
        :param next_token: o próximo token para o qual a probabilidade deve ser prevista
        :return: Probabilidade de next_token dado o prefixo, um valor entre 0 e 1
        """
        return self.get_possible_next_tokens(prefix).get(next_token, 0)


def main():
    sample_lines = ["this is a sample sentence", "another example for testing"]
    model = NGramLanguageModel(sample_lines, n=2)
    print("Modelo de exemplo criado com bigramas")# Sample model created with bigrams
    print("Possíveis próximos tokens após 'este'':", model.get_possible_next_tokens("this")) # Possible next tokens after 'this


if __name__ == '__main__':
    main()




# """
# Module with implementation of APIBase class (Base class for API classes).
# """
# from custom_helpers import add_root_to_pythonpath
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# from collections import defaultdict, Counter
# from abc import ABC, abstractmethod
# from typing import Dict, Tuple, DefaultDict, Iterable
#
# BOS, EOS = "→", "←"  # one-letter tokens (not the best practice)
# UNK = BOS
# class BaseLanguageModel(ABC):
#     """
#     Abstract base class for language models.
#     """
#     BOS = BOS
#     EOS = EOS
#     UNK = UNK
#
#     @abstractmethod
#     def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
#         """
#         Get possible next tokens and their probabilities given a prefix.
#
#         :param prefix: string with space-separated prefix tokens
#         :return: Dictionary {token : probability} for all tokens with positive probabilities
#         """
#         pass
#
#     @abstractmethod
#     def get_next_token_prob(self, prefix: str, next_token: str) -> float:
#         """
#         Get probability of the next token given a prefix.
#
#         :param prefix: string with space-separated prefix tokens
#         :param next_token: the next token to predict probability for
#         :return: Probability of next_token given prefix, a value between 0 and 1
#         """
#         pass
#
#
# class NGramLanguageModel(BaseLanguageModel):
#     """
#     Class for a simple n-gram language model.
#     """
#     @staticmethod
#     def count_ngrams(lines: Iterable[str], n: int) -> DefaultDict[Tuple[str, ...], Counter[str]]:
#         """
#         Count n-grams occurrences in the lines.
#
#         :param lines: an iterable of strings with space-separated tokens
#         :param n: n-gram size
#         :return: Dictionary { tuple(prefix_tokens): {next_token_1: count_1, next_token_2: count_2}}
#
#         When building counts, please consider the following two edge cases:
#         - if prefix is shorter than (n - 1) tokens, it should be padded with UNK. For n=3,
#         empty prefix: "" -> (UNK, UNK)
#         short prefix: "the" -> (UNK, the)
#         long prefix: "the new approach" -> (new, approach)
#         - you should add a special token, EOS, at the end of each sequence
#         "... with deep neural networks ." -> (..., with, deep, neural, networks, ., EOS)
#         count the probability of this token just like all others.
#         """
#         counts: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
#
#         # TODO: fill in the counts
#
#         return counts
#
#     def __init__(self, lines: Iterable[str], n: int) -> None:
#         """
#         Initialize the n-gram language model.
#
#         :param n: n-gram size
#         :param lines: an iterable of strings with space-separated tokens
#         """
#         assert n >= 1
#         self.n: int = n
#
#         counts = self.count_ngrams(lines, self.n)
#
#         # compute token probabilities given counts
#         self.probs: DefaultDict[Tuple[str, ...], Counter[str]] = defaultdict(Counter)
#
#         # populate self.probs with actual probabilities
#         for pref, cnt in counts.items():
#             total = sum(cnt.values())
#             for token, count in cnt.items():
#                 # TODO: fill in the probs for prefix, token. Formula: token_i / (sum_j token_j), where token_i is frequence of token_i after given prefix
#
#     def get_possible_next_tokens(self, prefix: str) -> Dict[str, float]:
#         """
#         Get possible next tokens and their probabilities given a prefix.
#
#         :param prefix: string with space-separated prefix tokens
#         :return: Dictionary {token : probability} for all tokens with positive probabilities
#
#         Steps:
#         1. Complete or truncate the prefix if needed (note: for completion add UNK tokens in the beginning)
#         2. The the distribution of tokens given prefix (use self.probs)
#         """
#         prefix_tokens = prefix.split()
#         pass # TODO
#
#     def get_next_token_prob(self, prefix: str, next_token: str) -> float:
#         """
#         Get probability of the next token given a prefix.
#
#         :param prefix: string with space-separated prefix tokens
#         :param next_token: the next token to predict probability for
#         :return: Probability of next_token given prefix, a value between 0 and 1
#         """
#         return self.get_possible_next_tokens(prefix).get(next_token, 0)
#
#
# def main():
#     sample_lines = ["this is a sample sentence", "another example for testing"]
#     model = NGramLanguageModel(sample_lines, n=2)
#     print("Sample model created with bigrams")
#     print("Possible next tokens after 'this':", model.get_possible_next_tokens("this"))
#
#
# if __name__ == '__main__':
#     main()