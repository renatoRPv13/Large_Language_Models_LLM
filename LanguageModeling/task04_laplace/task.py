"""
Module containing LaplaceLanguageModel class - an extension of NGramLanguageModel.
Module containing LaplaceLanguageModel class - an extension of NGramLanguageModel.
"""
from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

from LanguageModeling.task01_ngrams.task import NGramLanguageModel, EOS, UNK

class LaplaceLanguageModel(NGramLanguageModel): 
    """
    Uma versão estendida do NGramLanguageModel que implementa o suavização de Laplace.
    An extended version of NGramLanguageModel that implements Laplace smoothing.
    """
    def __init__(self, lines: List[str], n: int, delta: float = 1.0):
        """
        Initialize the LaplaceLanguageModel object with the given parameters.
        :param lines: List of text lines.
        :param n: Size of n-grams.
        :param delta: Smoothing parameter.
        Note: You don't need to store all the tokens in self.probs. Store only tokens that have been seen after the given prefix.
        Inicialize o objeto LaplaceLanguageModel com os parâmetros fornecidos.
        :param lines: Lista de linhas de texto.
        :param n: Tamanho dos n-gramas.
        :param delta: Parâmetro de suavização.
        Observação: Não é necessário armazenar todos os tokens em self.probs. Armazene apenas os tokens que foram vistos após o prefixo fornecido.
        """
        self.n = n
        self.dalta = delta # Armazene o delta para suavização
        counts = self.count_ngrams(lines, self.n)
        # Criar vocabulário a partir de todos os tokens vistos nos dados de treinamento
        self.vocab = set(token for token_counts in counts.values() for token in token_counts)
        for  token_counts in counts.values():
            for token in token_counts:
                self.vocab.add(token)

        self.probs = defaultdict(lambda: defaultdict(float))
        # Inicialize self.probs com suavização de Laplace
        for pref, cnt in counts.items():
            total_seen = sum(cnt.values())
            # Number of unique next tokens seen for this prefix# Número de tokens únicos "next" observados para este prefixo
            seen_tokens = len(cnt)
            # Total vocabulary size (including unseen tokens) # Tamanho total do vocabulário (incluindo tokens não vistos)
            vocab_size = len(self.vocab)

            # Apply Laplace smoothing: (count + delta) / (total_seen + delta * vocab_size)
            # Aplicar suavização de Laplace: (contagem + delta) / (total_visto + delta * tamanho_do_vocabulário)
            for token, count in cnt.items():
                self.probs[pref][token] = (count + delta) / (total_seen + delta * vocab_size)
            
    def get_possible_next_tokens(self, prefix: Tuple[str, ...]) -> Dict[str, float]:
        """
        Returns possible next tokens and their probabilities given a prefix.
        :param prefix: Prefix tuple.
        :return: Dictionary with tokens and their probabilities.
        Note: Missing tokens should have uniform probability among all missing tokens.
        Note 2: It's a design choice not to store all the tokens in self.probs, therefore,
        we need to calculate missing tokens probabilities on the fly.
        Retorna os próximos tokens possíveis e suas probabilidades, dado um prefixo.
        :param prefix: Tupla de prefixo.
        :return: Dicionário com os tokens e suas probabilidades.
        Nota: A probabilidade de tokens ausentes deve ser uniforme entre todos os tokens ausentes.
        Nota 2: É uma escolha de projeto não armazenar todos os tokens em `self.probs`, portanto,
        precisamos calcular as probabilidades dos tokens ausentes dinamicamente.
        """
        token_probs = super().get_possible_next_tokens(prefix)
        missing_prob_total = 1.0 - sum(token_probs.values())
        missing_prob = missing_prob_total / max(1, len(self.vocab) - len(token_probs))
        return {token: token_probs.get(token, missing_prob) for token in self.vocab}
    
    def get_next_token_prob(self, prefix: Tuple[str, ...], next_token: str) -> float:
        """
        Returns the probability of a specific next token given a prefix.
        :param prefix: Prefix tuple.
        :param next_token: Next token.
        :return: Probability of the next token.
        Note: we're choosing to assign the missing probability even to the tokens we haven't seen in the training data.
        Retorna a probabilidade de um token específico ser o próximo token, dado um prefixo.
        :param prefix: Tupla de prefixos.
        :param next_token: Próximo token.
        :return: Probabilidade do próximo token.
        Observação: optamos por atribuir a probabilidade de tokens ausentes mesmo aos tokens que não foram vistos nos dados de treinamento.
        """
        token_probs = super().get_possible_next_tokens(prefix)
        if next_token in token_probs:
            return token_probs[next_token]
        else:
            missing_prob_total = 1.0 - sum(token_probs.values())
            missing_prob_total = max(0, missing_prob_total)  # prevent rounding errors
            return missing_prob_total / max(1, len(self.vocab) - len(token_probs))

def main():
    sample_lines = ["this is a sample sentence", "another example for testing"]
    model = LaplaceLanguageModel(sample_lines, n=2, delta=0.5)
    print("Modelo suavizado por Laplace criado com delta=0.5")
    print("Probabilidade de 'is' após 'this':", model.get_next_token_prob("this", "is"))
    print("Probabilidade de token desconhecido após 'this':", model.get_next_token_prob("this", "unknown"))

if __name__ == '__main__':
    main()