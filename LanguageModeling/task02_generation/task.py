"""
Module with implementation of Generator class for sequence token generation.
Módulo com implementação da classe Generator para geração de tokens de sequência.
"""
from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

from LanguageModeling.task01_ngrams.task import BaseLanguageModel, UNK, EOS
from typing import List
import numpy as np
from functools import partial


class Generator:
    """
    Class for generating token sequences based on a language model.
    Classe para gerar sequências de tokens com base em um modelo de linguagem.
    """
    def __init__(self, model: BaseLanguageModel, token_level: str = 'word') -> None:
        """
        Initialize the Generator object.

        :param model: an instance of BaseLanguageModel
        :param token_level: either 'word' or 'char'

        Inicialize o objeto Generator.
        :param model: uma instância de BaseLanguageModel
        :param token_level: 'word' ou 'char'
        """
        assert token_level in ['word', 'char'], f"Unknown token_level: {token_level}"
        self.model = model

        if token_level == 'word':
            self.tokenize = lambda x: x.split()
            self.detokenize = lambda x: ' '.join(x)
        elif token_level == 'char':
            self.tokenize = list
            self.detokenize = lambda x: ''.join(x)

    def _get_tokens_probs_safe(self, prefix: str) -> dict[str, float]:
        """
        Get tokens and their probabilities for a given prefix.
        In case of an empty list of tokens, return EOS with probability 1.

        :param prefix: Prefix of the sequence.
        :return: Dictionary with tokens and their probabilities.

        Obtém os tokens e suas probabilidades para um prefixo dado.
        Caso a lista de tokens esteja vazia, retorna o fim da sequência (EOS) com probabilidade 1.
        :param prefix: Prefixo da sequência.
        :return: Dicionário com os tokens e suas probabilidades.
        """
        tokens_probs = self.model.get_possible_next_tokens(prefix)
        if not tokens_probs:
            tokens_probs[EOS] = 1
        return tokens_probs

    def get_next_token_sample(self, prefix: str, temperature: float = 1.0) -> str:
        """
        Return next token after prefix.

        :param prefix: Prefix of the sequence.
        :param temperature: samples proportionally to lm probabilities ^ (1 / temperature)
            if temperature == 0, always takes most likely token (greedy). Break ties arbitrarily.
        :return: Predicted next token.

        Retorna o próximo token após o prefixo.

        :param prefixo: Prefixo da sequência.
        :param temperatura: amostra proporcionalmente às probabilidades de lm ^ (1 / temperatura)
        se temperatura == 0, sempre pega o token mais provável (guloso). Em caso de empate, escolha o token que não for mais provável.
        :return: Próximo token previsto
        """
        probs = self._get_tokens_probs_safe(prefix)

        # Temperature == 0 case: greedy selection # Caso Temperatura == 0: seleção gulosa
        if temperature == 0:
            # Return the token with highest probability # Retorna o token com maior probabilidade
            return max(probs.items(), key=lambda x: x[1])[0]

        # General case: sample with temperature # Caso geral: amostra com temperatura
        tokens = list(probs.keys())
        probabilities = np.array(list(probs.values()))

        # Apply temperature scaling # Aplicar escala de temperatura
        if temperature != 1.0:
            probabilities = probabilities ** (1.0 / temperature)

        # Normalize probabilities # Normalizar probabilidades
        probabilities = probabilities / probabilities.sum()

        # Sample next token # Exemplo do próximo token
        return np.random.choice(tokens, p=probabilities)

    def get_next_token_nucleus(self, prefix: str, nucleus: float = 0.9) -> str:
        """
        Generate a sequence with nucleus sampling.

        :param prefix: Prefix of the sequence.
        :param nucleus: N from the formulae above, N \in [0, 1]
        :return: Predicted next token.
        :note: make sure that nucleus always contains at least one word, even if p(w*) > nucleus

        Steps:
        1. Get sorted_probs_ids (indices of tokens sorted by probabilities in descending order) and sorted_probs (sorted probabilities)
        2. Calculate cumulative probabilities cum_probs and create a mask for the tokens that should be included in the nucleus
            Note: if no token should be included, include the most probable one (that is mask[0] is always True)
        3. Normalize the probabilities of the selected tokens and sample the next token
        4. Return the sampled token

        Gere uma sequência com amostragem de núcleo.
        :param prefix: Prefixo da sequência.
        :param nucleus: N da fórmula acima, N ∈ [0, 1]
        :return: Próximo token previsto.
        :note: certifique-se de que o núcleo sempre contenha pelo menos uma palavra, mesmo se p(w*) > nucleus
    Passos:
    1. Obtenha sorted_probs_ids (índices dos tokens ordenados por probabilidades em ordem decrescente) e sorted_probs (probabilidades ordenadas)
    2. Calcule as probabilidades cumulativas cum_probs e crie uma máscara para os tokens que devem ser incluídos no núcleo
    Nota: se nenhum token deve ser incluído, inclua o mais provável (ou seja, mask[0] é sempre True)
    3. Normalize as probabilidades dos tokens selecionados e amostre o próximo token
    4. Retorne o token amostrado
        """
        token_probs = self._get_tokens_probs_safe(prefix)
        tokens, probs = zip(*token_probs.items())
        assert np.isclose(sum(probs), 1), f"Sum of probabilities is not close to 1: {sum(probs)}"

        # Step 1: Sort tokens by probabilities in descending order
        # Etapa 1: Ordene os tokens por probabilidades em ordem decrescente
        sorted_indices = np.argsort(probs)[::-1]
        sorted_tokens = [tokens[i] for i in sorted_indices]
        sorted_probs = np.array([probs[i] for i in sorted_indices])

        # Step 2: Calculate cumulative probabilities and create mask
        # Etapa 2: Calcular probabilidades cumulativas e criar máscara
        cum_probs = np.cumsum(sorted_probs)

        # Find tokens where cumulative probability <= nucleus
        # Always include at least the first token
        # Encontrar tokens onde a probabilidade cumulativa é menor ou igual ao núcleo
        # Sempre incluir pelo menos o primeiro token
        mask = cum_probs <= nucleus
        if not np.any(mask):  # If no token satisfies the condition, include the first one # Se nenhum token satisfizer a condição, inclua o primeiro.
            mask[0] = True
        elif mask[0] is False:  # Ensure first token is always included # Garantir que o primeiro token esteja sempre incluído
            mask[0] = True

        # Get the tokens to sample from # Obtenha os tokens para amostrar
        selected_tokens = [sorted_tokens[i] for i in range(len(mask)) if mask[i]]
        selected_probs = np.array([sorted_probs[i] for i in range(len(mask)) if mask[i]])

        # Step 3: Normalize probabilities and sample Etapa 3: Normalizar probabilidades e amostra
        selected_probs = selected_probs / selected_probs.sum()
        next_token = np.random.choice(selected_tokens, p=selected_probs)

        return next_token

    def generate_sequence(self, prefix: str = UNK, mode: str = 'sample', max_len: int = 100, **kwargs) -> List[str]:
        """
        Generate a sequence of tokens.

        :param prefix: Prefix of the sequence.
        :param mode: either 'sample' or 'nucleus'
        :param **kwargs: additional arguments for the chosen mode
            temperature: for mode='sample'
            nucleus, max_len: for mode='nucleus'
        :returns: A list of tokens (including EOS)

        Gera uma sequência de tokens.
        :param prefix: Prefixo da sequência.
        :param mode: 'sample' ou 'nucleus'
        :param **kwargs: argumentos adicionais para o modo escolhido
        temperature: para mode='sample'
        nucleus, max_len: para mode='nucleus'
        :returns: Uma lista de tokens (incluindo EOS)
        """
        assert mode in ['sample', 'nucleus'], f"Unknown mode: {mode}"
        if mode == 'sample':
            temperature = kwargs.get('temperature', 0.5)
        elif mode == 'nucleus':
            nucleus = kwargs.get('nucleus', 0.9)

        sequence = self.tokenize(prefix)

        if mode == 'sample':
            sample_fn = partial(self.get_next_token_sample, temperature=temperature)
        elif mode == 'nucleus':
            sample_fn = partial(self.get_next_token_nucleus, nucleus=nucleus)

        # Generate tokens until EOS or max_len is reached
        # Gere tokens até que o fim da operação (EOS) ou o comprimento máximo (max_len) seja atingido.
        for _ in range(max_len):
            # Convert current sequence to string prefix # Converter a sequência atual em prefixo de string
            current_prefix = self.detokenize(sequence)

            # Generate next token # Gerar próximo token
            next_token = sample_fn(current_prefix)

            # Add to sequence # Adicionar à sequência
            sequence.append(next_token)

            # Stop if we generated EOS # Pare se gerarmos EOS
            if next_token == EOS:
                break

        return sequence


def main():
    from LanguageModeling.task01_ngrams.task import NGramLanguageModel
    sample_lines = ["Esta é uma frase de exemplo.", "outro exemplo para teste"]
    model = NGramLanguageModel(sample_lines, n=2)
    generator = Generator(model)
    generated = generator.generate_sequence(max_len=10)
    print("Sequência gerada:", generator.detokenize(generated))

    # with prefix 'another'
    generated = generator.generate_sequence(prefix='another', max_len=10)
    print("Sequência gerada com prefixo 'outro':", generator.detokenize(generated))


if __name__ == '__main__':
    main()




# """
# Module with implementation of Generator class for sequence token generation.
# """
# from custom_helpers import add_root_to_pythonpath
# from sympy.physics.units import current
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# from LanguageModeling.task01_ngrams.task import BaseLanguageModel, UNK, EOS
# from typing import List
# import numpy as np
# from functools import partial
#
#
# class Generator:
#     """
#     Classe para gerar sequências de tokens com base em um modelo de linguagem.
#     """
#     def __init__(self, model: BaseLanguageModel, token_level: str = 'word') -> None:
#         """
#         Initialize the Generator object.
#
#         :param model: an instance of BaseLanguageModel
#         :param token_level: either 'word' or 'char'
#         """
#         assert token_level in ['word', 'char'], f"Unknown token_level: {token_level}"
#         self.model = model
#
#         if token_level == 'word':
#             self.tokenize = lambda x: x.split()
#             self.detokenize = lambda x: ' '.join(x)
#         elif token_level == 'char':
#             self.tokenize = list
#             self.detokenize = lambda x: ''.join(x)
#
#     def _get_tokens_probs_safe(self, prefix: str) -> dict[str, float]:
#         """
#         Get tokens and their probabilities for a given prefix.
#         In case of an empty list of tokens, return EOS with probability 1.
#
#         :param prefix: Prefix of the sequence.
#         :return: Dictionary with tokens and their probabilities.
#         """
#         tokens_probs = self.model.get_possible_next_tokens(prefix)
#         if not tokens_probs:
#             tokens_probs[EOS] = 1
#         return tokens_probs
#
#     def get_next_token_sample(self, prefix: str, temperature: float = 1.0) -> str:
#         """
#         Return next token after prefix.
#
#         :param prefix: Prefix of the sequence.
#         :param temperature: samples proportionally to lm probabilities ^ (1 / temperature)
#             if temperature == 0, always takes most likely token (greedy). Break ties arbitrarily.
#         :return: Predicted next token.
#         """
#         probs = self._get_tokens_probs_safe(prefix)
#         # Caso Temperatura == 0: seleção gulosa
#         if temperature == 0:
#                 # Retorna o token com maior probabilidade
#             return  max(probs.items(), key=lambda x: x[1])[0]
#
#         # Caso geral: amostra com temperatura
#         tokens = list(probs.keys())
#         probabilities = np.array(list(probs.values()))
#
#         # Aplicar escala de temperatura
#         if (temperature!= 1.0):
#
#         # Normalizar probabilidades
#                 probabilities = probabilities / probabilities.sum()
#
#         # Exemplo do próximo token
#         return  np.random.choice(tokens, p=probabilities)
#
#
#
#     def get_next_token_nucleus(self, prefix: str, nucleus: float = 0.9) -> str:
#         """
#         Generate a sequence with nucleus sampling.
#
#         :param prefix: Prefix of the sequence.
#         :param nucleus: N from the formulae above, N \in [0, 1]
#         :return: Predicted next token.
#         :note: make sure that nucleus always contains at least one word, even if p(w*) > nucleus
#
#         Steps:
#         1. Get sorted_probs_ids (indices of tokens sorted by probabilities in descending order) and sorted_probs (sorted probabilities)
#         2. Calculate cumulative probabilities cum_probs and create a mask for the tokens that should be included in the nucleus
#             Note: if no token should be included, include the most probable one (that is mask[0] is always True)
#         3. Normalize the probabilities of the selected tokens and sample the next token
#         4. Return the sampled token
#         """
#         token_probs = self._get_tokens_probs_safe(prefix)
#         tokens, probs = zip(*token_probs.items())
#         assert np.isclose(sum(probs), 1), f"Sum of probabilities is not close to 1: {sum(probs)}"
#
#         # Etapa 1: Ordene os tokens por probabilidades em ordem decrescente
#         sorted_indices = np.argsort(probs)[::-1]
#         sorted_tokens = [tokens[i] for i in sorted_indices]
#         sorted_probs = np.array([probs[i] for i in sorted_indices])
#
#         # Etapa 2: Calcular probabilidades cumulativas e criar máscara
#         cum_probs = np.cumsum(sorted_probs)
#
#         # Encontrar tokens onde a probabilidade cumulativa é menor ou igual ao núcleo
#         # Sempre incluir pelo menos o primeiro token
#         musk = cum_probs <= nucleus
#         if not np.any(musk): # Se nenhum token satisfizer a condição, inclua o primeiro.
#             musk[0] =True
#         elif musk[0] is False: # Ensure first token is always included
#             musk[0] = False
#
#         # Obtenha os tokens para amostrar
#         selected_tokens = [sorted_tokens[i] for i in range(len(sorted_tokens)) if musk[i]]
#         selected_probs = np.array(sorted_probs[i] for i in range(len(musk)) if musk[i])
#
#         # Etapa 3: Normalizar probabilidades e amostra
#         selected_probs = selected_probs / sorted_probs.sum()
#         next_token = np.random.choice(selected_tokens, p=selected_probs)
#
#         return next_token
#
#     def generate_sequence(self, prefix: str = UNK, mode: str = 'sample', max_len: int = 100, **kwargs) -> List[str]:
#         """
#         Generate a sequence of tokens.
#
#         :param prefix: Prefix of the sequence.
#         :param mode: either 'sample' or 'nucleus'
#         :param **kwargs: additional arguments for the chosen mode
#             temperature: for mode='sample'
#             nucleus, max_len: for mode='nucleus'
#         :returns: A list of tokens (including EOS)
#         """
#         assert mode in ['sample', 'nucleus'], f"Unknown mode: {mode}"
#         if mode == 'sample':
#             temperature = kwargs.get('temperature', 0.5)
#         elif mode == 'nucleus':
#             nucleus = kwargs.get('nucleus', 0.9)
#
#         sequence = self.tokenize(prefix)
#         if mode == 'sample':
#             sample_fn = partial(self.get_next_token_sample, temperature=temperature)
#         elif mode == 'nucleus':
#             sample_fn = partial(self.get_next_token_nucleus, nucleus=nucleus)
#
#         # Gere tokens até que o fim da operação (EOS) ou o comprimento máximo (max_len) seja atingido.
#         for _ in range(max_len):
#             # Converter a sequência atual em prefixo de string
#             current_prefix = self.detokenize(sequence)
#              # Gerar próximo token
#             next_tokens =  sample_fn(current_prefix)
#             # Adicionar à sequência
#             sequence.append(next_tokens)
#             # Pare se gerarmos EOS
#             #if next_tokens == EOS:
#             if next_tokens[-1] == EOS:
#                 breakpoint()
#
#         return sequence
#
#
# def main():
#     from LanguageModeling.task01_ngrams.task import NGramLanguageModel
#     sample_lines = ["this is a sample sentence", "another example for testing"]
#     model = NGramLanguageModel(sample_lines, n=2)
#     generator = Generator(model)
#     generated = generator.generate_sequence(max_len=10)
#     print("Generated sequence:", generator.detokenize(generated))
#
#     # with prefix 'another'
#     generated = generator.generate_sequence(prefix='another', max_len=10)
#     print("Generated sequence with prefix 'another':", generator.detokenize(generated))
#
#
# if __name__ == '__main__':
#     main()