from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up = 2, verbose = True)

import numpy as np
from typing import List

from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS

class Evaluator:
    """
    Classe para avaliação da perplexidade do modelo de linguagem.
    Class for evaluating language model perplexity.
    """
    @staticmethod
    def perplexity(model: BaseLanguageModel, lines: List[str], min_logprob: float = np.log(10**-50.)) -> float:
        # :param nucleus: N from the formulae above, N \in [0, 1]
        """
        Calculate the perplexity of the language model on a given corpus.
        :param model: BaseLanguageModel object representing the language model.
        :param lines: List of strings with space-separated tokens.
        :param min_logprob: Minimum log probability threshold.
        :param nucleus: N from the formulae above, N \\in [0, 1]
        Probability of the next token will be taken into account as max(min_logprob, log(P(token|prefix)) )
        :return: Calculated perplexity value.
        Note: do not forget to compute P(w_first | empty) and P(eos | full_sequence)
        PLEASE USE model.get_next_token_prob and NOT model.get_possible_next_tokens

        Calcula a perplexidade do modelo de linguagem em um corpus fornecido.
        :param model: Objeto BaseLanguageModel representando o modelo de linguagem.
        :param lines: Lista de strings com tokens separados por espaço.
        :param min_logprob: Limiar mínimo de probabilidade logarítmica.
        A probabilidade do próximo token será considerada como max(min_logprob, log(P(token|prefix)) )
        :return: Valor da perplexidade calculada.
        Nota: não se esqueça de calcular P(w_first | empty) e P(eos | full_sequence)
        POR FAVOR, UTILIZE model.get_next_token_prob e NÃO model.get_possible_next_tokens
        """
        total_logprob, total_num_tokens = 0, 0
        for line in lines:
            tokens = line.split()
            tokens.append(EOS) # Adicione o token EOS no final
            pref = ''
            total_num_tokens += len(tokens)
            # Percorra cada token na sequência
            for token in tokens:
                prob = model.get_next_token_prob(pref, token)
                # Calcule o logaritmo e aplique o limite min_logprob
                if prob == 0:
                    # Use uma probabilidade muito pequena em vez de zero
                    # Isso é equivalente à suavização por adição de um
                    prob = 1e-10
                logprob = np.log(prob)
                logprob = max(logprob, min_logprob)

                # Adicionar à probabilidade logarítmica total
                total_logprob += logprob

                # Atualizar prefixo para a próxima iteração
                if pref == '':
                    pref = token
                else:
                    pref= pref + ' ' + token

        # Calcular a perplexidade: exp(-probabilidade logarítmica média)
        return np.exp(-total_logprob / total_num_tokens)

def main():
    from LanguageModeling.task01_ngrams.task import NGramLanguageModel
    #sample_lines = ["esta é uma frase de exemplo", "outro exemplo para teste"]
    #test_lines = ["este é um teste"]
    sample_lines = ["this is a sample sentence", "another example for testing"]
    test_lines = ["this is a test"]
    model = NGramLanguageModel(sample_lines, n=2)
    perplexity = Evaluator.perplexity(model, test_lines)
    print(f"Perplexidade do modelo em dados de teste: {perplexity:.2f}")

if __name__ == '__main__':
    main()