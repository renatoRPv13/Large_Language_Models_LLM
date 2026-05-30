"""
Module with implementation of RNNLanguageModel class (Recurrent Neural Network Language Model).
Módulo com implementação da classe RNNLanguageModel (Modelo de Linguagem de Rede Neural Recorrente).
"""

from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

import torch
from torch import nn
import torch.nn.functional as F
from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS, UNK
from LanguageModeling.task05_text_tools.task import TextTools as tt


class RNNLanguageModel(nn.Module, BaseLanguageModel):
    def __init__(self, tokens: list, emb_size: int = 16, hid_size: int = 256):
        """
        Build a recurrent language model.
        You are free to choose anything you want, but the recommended architecture is
        - token embeddings
        - one or more LSTM/GRU layers with hid size
        - linear layer to predict logits
        :param tokens: List of tokens.
        :param emb_size: Size of the embedding layer.
        :param hid_size: Size of the hidden layer.
        :note: if you use nn.RNN/GRU/LSTM, make sure you specify batch_first=True
         With batch_first, your model operates with tensors of shape [batch_size, sequence_length, num_units]
         Also, please read the docs carefully: they don't just return what you want them to return :)

         Construa um modelo de linguagem recorrente.
        Você tem liberdade para escolher o que quiser, mas a arquitetura recomendada é:
        - incorporações de tokens
        - uma ou mais camadas LSTM/GRU com tamanho de camada oculta (hid size)
        - camada linear para prever logits
        :param tokens: Lista de tokens.
        :param emb_size: Tamanho da camada de incorporação.
        :param hid_size: Tamanho da camada oculta.
        :note: se você usar nn.RNN/GRU/LSTM, certifique-se de especificar batch_first=True
        Com batch_first, seu modelo opera com tensores de formato [batch_size, sequence_length, num_units]
        Além disso, leia a documentação com atenção: ela não retorna exatamente o que você espera :)
        """
        super().__init__()  # initialize base class to track sub-layers, trainable variables, etc.

        n_tokens = len(tokens)
        self.tokens = tokens

        # Important: You can experiment with the architecture
        # Create the layers of the RNN model
        # Importante: Você pode experimentar com a arquitetura
        # Crie as camadas do modelo RNN
        self.embedding = nn.Embedding(n_tokens, emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hid_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.linear = nn.Linear(hid_size, n_tokens)
        self.dropout = nn.Dropout(0.2)

    def __call__(self, input_ix: torch.Tensor) -> torch.Tensor:
        """
        compute language model logits given input tokens
        :param input_ix: batch of sequences with token indices, tensor: int32[batch_size, sequence_length]
        :returns: pre-softmax linear outputs of language model [batch_size, sequence_length, n_tokens]
        these outputs will be used as logits to compute P(x_t | x_0, ..., x_{t - 1})
        Calcula os logits do modelo de linguagem dados os tokens de entrada
        :param input_ix: lote de sequências com índices de tokens, tensor: int32[batch_size, sequence_length]
        :returns: saídas lineares pré-softmax do modelo de linguagem [batch_size, sequence_length, n_tokens]
        essas saídas serão usadas como logits para calcular P(x_t | x_0, ..., x_{t - 1})
        """
        # Get embeddings: [batch_size, sequence_length, emb_size]
        # Obter embeddings: [batch_size, sequence_length, emb_size]
        embeddings = self.embedding(input_ix)
        embeddings = self.dropout(embeddings)

        # Pass through LSTM: outputs [batch_size, sequence_length, hid_size],
        # (hidden, cell) states are also returned but we don't need them
        # Passar pelo LSTM: saídas [batch_size, sequence_length, hid_size],
        # Os estados (hidden, cell) também são retornados, mas não precisamos deles
        lstm_outputs, _ = self.lstm(embeddings)
        lstm_outputs = self.dropout(lstm_outputs)

        # Project to vocabulary size: [batch_size, sequence_length, n_tokens]
        # Projetar para o tamanho do vocabulário: [batch_size, sequence_length, n_tokens]
        logits = self.linear(lstm_outputs)

        return logits

    def get_possible_next_tokens(self, prefix: str) -> dict:
        """
        :returns: probabilities of next token, dict {token : prob} for all tokens
        Note: Use torch.no_grad
        Steps:
        1. Convert to matrix
        2. Get models output
        3. Apply softmax and return the result
        Retorna as probabilidades do próximo token, um dicionário `{token : prob}` para todos os tokens.
        Observação: Use `torch.no_grad`.
        Passos:
        1. Converter para matriz
        2. Obter a saída do modelo
        3. Aplicar softmax e retornar o resultado
        """
        # NÃO use torch.no_grad() se esta função for chamada durante treinamento
        # Ou crie uma versão separada para treinamento
        with torch.no_grad():  # Mantenha isso apenas para inferência
            input_matrix = tt.to_matrix([prefix])
            logits = self.__call__(input_matrix)
            last_logits = logits[0, -1, :]
            probs = F.softmax(last_logits, dim=0)
            return {self.tokens[i]: probs[i].item() for i in range(len(self.tokens))}

    # def get_possible_next_tokens(self, prefix: str) -> dict:
    #     """
    #     :returns: probabilities of next token, dict {token : prob} for all tokens
    #
    #     Note: Use torch.no_grad
    #     Steps:
    #     1. Convert to matrix
    #     2. Get models output
    #     3. Apply softmax and return the result
    #     """
    #     with torch.no_grad():
    #         # Convert prefix string to matrix of token IDs
    #         # prefix is a string, we need to convert it to a tensor
    #         input_matrix = tt.to_matrix([prefix])  # Shape: [1, len(prefix)]
    #
    #         # Get model outputs (logits)
    #         logits = self.__call__(input_matrix)  # Shape: [1, sequence_length, n_tokens]
    #
    #         # Take the last token's logits (the prediction for next token)
    #         last_logits = logits[0, -1, :]  # Shape: [n_tokens]
    #
    #         # Apply softmax to get probabilities
    #         probs = F.softmax(last_logits, dim=0)  # Shape: [n_tokens]
    #
    #         # Convert to dictionary mapping token to probability
    #         result = {self.tokens[i]: probs[i].item() for i in range(len(self.tokens))}
    #
    #     return result

    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """ :returns: probability of next_token given prefix, float """
        """ :returns: probabilidade do próximo token dado o prefixo, float """
        return self.get_possible_next_tokens(prefix).get(next_token, 0.0)


def main():
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    tokens = tt.TOKENS
    model = RNNLanguageModel(tokens, emb_size=8, hid_size=32)
    #print(f"Created RNN model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Modelo RNN criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    #sample_text = "Hello"
    sample_text = "Olá"
    #print(f"Top 3 next tokens for '{sample_text}':",
    print(f"Os 3 próximos tokens principais para '{sample_text}':",
          sorted(model.get_possible_next_tokens(sample_text).items(), key=lambda x: x[1], reverse=True)[:3])


if __name__ == '__main__':
    main()



# """
# Módulo com implementação da classe RNNLanguageModel (Modelo de Linguagem de Rede Neural Recorrente)
# Module with implementation of RNNLanguageModel class (Recurrent Neural Network Language Model).
# """
# from unittest import result
#
# from custom_helpers import add_root_to_pythonpath
# from numpy.random.mtrand import logistic
# from regex import F
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
# from torch import nn, no_grad
# from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS, UNK
# from LanguageModeling.task05_text_tools.task import TextTools as tt
#
# class RNNLanguageModel(nn.Module, BaseLanguageModel):
#     def __init__(self, tokens: list, emb_size: int = 16, hid_size: int = 256):
#         """
#         Build a recurrent language model.
#         You are free to choose anything you want, but the recommended architecture is
#         - token embeddings
#         - one or more LSTM/GRU layers with hid size
#         - linear layer to predict logits
#
#         :param tokens: List of tokens.
#         :param emb_size: Size of the embedding layer.
#         :param hid_size: Size of the hidden layer.
#         :note: if you use nn.RNN/GRU/LSTM, make sure you specify batch_first=True
#          With batch_first, your model operates with tensors of shape [batch_size, sequence_length, num_units]
#          Also, please read the docs carefully: they don't just return what you want them to return :)
#         """
#         super().__init__() # initialize base class to track sub-layers, trainable variables, etc.
#
#         n_tokens = len(tokens)
#         self.tokens = tokens
#
#         # Important: You can experiment with the architecture
#
#         # Importante: Você pode experimentar com a arquitetura
#         # Crie as camadas do modelo RNN
#         self.embedding = nn.Embedding(n_tokens, emb_size)
#         self.lstm = nn.LSTM(
#             input_size=emb_size,
#             hidden_size= hid_size,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.2
#           )
#         self.linear = nn.Linear(hid_size, n_tokens)
#         self.dropout = nn.Dropout(p=0.5)
#
#
#     def __call__(self, input_ix: torch.Tensor) -> torch.Tensor:
#         """
#         compute language model logits given input tokens
#         :param input_ix: batch of sequences with token indices, tensor: int32[batch_size, sequence_length]
#         :returns: pre-softmax linear outputs of language model [batch_size, sequence_length, n_tokens]
#             these outputs will be used as logits to compute P(x_t | x_0, ..., x_{t - 1})
#         """
#         # Obter embeddings: [batch_size, sequence_length, emb_size]
#         embeddings = self.embedding(input_ix)
#         embeddings = self.dropout(embeddings)
#
#         # Passar pelo LSTM: saídas [batch_size, sequence_length, hid_size],
#         # Os estados (hidden, cell) também são retornados, mas não precisamos deles
#         lstm_outputs, _ = self.lstm(embeddings)
#         lstm_outputs = self.dropout(lstm_outputs)
#
#         # Projetar para o tamanho do vocabulário: [batch_size, sequence_length, n_tokens]
#         logits = self.linear(lstm_outputs)
#         return  logits
#
#     def get_possible_next_tokens(self, prefix: str) -> dict:
#         """
#         :returns: probabilities of next token, dict {token : prob} for all tokens
#
#         Note: Use torch.no_grad
#         Steps:
#         1. Convert to matrix
#         2. Get models output
#         3. Apply softmax and return the result
#         """
#         with torch.no_grad():
#             # Converter string de prefixo em matriz de IDs de token
#             # prefixo é uma string, precisamos convertê-la em um tensor
#             input_matrix = tt.to_matrix(prefix) # Formato: [1, len(prefixo)]
#
#             # Obter resultados do modelo (logits)
#             logits = self.__call__(input_matrix)# Formato: [1, comprimento_da_sequência, n_tokens]
#
#             # Pegue os logits do último token (a previsão para o próximo token)
#             last_logits = logits[0,-1, :] # Formato: [n_tokens]
#
#             # Aplique softmax para obter probabilidades
#             #probs = F.softmax(last_logits, dim=-1)
#             probs = F.softmax(last_logits, dim=0)
#
#             # Converter para dicionário mapeando token para probabilidade
#             result = {self.tokens[i]: probs[i].item() for i in range(len(probs))}
#         return result
#
#
#     def get_next_token_prob(self, prefix: str, next_token: str) -> float:
#         """ :returns: probability of next_token given prefix, float """
#         return self.get_possible_next_tokens(prefix).get(next_token, 0.0)
#
#
# def main():
#     from LanguageModeling.task05_text_tools.task import TextTools as tt
#     tokens = tt.TOKENS
#     model = RNNLanguageModel(tokens, emb_size=8, hid_size=32)
#     print(f"Created RNN model with {sum(p.numel() for p in model.parameters())} parameters")
#     sample_text = "Hello"
#     print(f"Top 3 next tokens for '{sample_text}':",
#           sorted(model.get_possible_next_tokens(sample_text).items(), key=lambda x: x[1], reverse=True)[:3])
#
#
# if __name__ == '__main__':
#     main()