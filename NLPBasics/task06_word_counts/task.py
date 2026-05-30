# SETUP
import sys
import os
from collections import Counter
from numpy import ndarray
from pandas.core.frame import DataFrame
from typing import Dict

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
import pandas as pd
from collections import Counter

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config, save_as_numpy
from tools_basics.data_handler import DataHandler

# IMPORTANTE: Importar o Tokenizer da task02
from task02_tokenization.task import Tokenizer

class WordCounts:
    """Class to compute word counts and Bag of Words (BoW) representations."""
    "Classe para calcular contagens de palavras e representações de Bag of Words (BoW)."

    def __init__(self, train_df: DataFrame, test_df: DataFrame) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = Tokenizer()  # Agora Tokenizer está definido
        self.bow_vocabulary = None
        self.bow_to_id = None
        self._set_bow()

    def _set_bow(self) -> None:
        texts = self.train_df.text.values
        word_count = self.get_words_count(texts)
        self.bow_vocabulary = self.get_bow_vocabulary(texts, word_count)
        self.bow_to_id = self.get_bow_to_id_mapping()

    def get_words_count(self, texts: ndarray) -> Counter:
        """Compute word counts from the given texts.""" "Calcule a contagem de palavras nos textos fornecidos."
        words_count = Counter()
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            words_count.update(tokens)
        return words_count

    def get_bow_vocabulary(self, texts: list[str], word_count: Counter, k: int = 10000) -> list[str]:
        """Get the Bag of Words vocabulary with up to k most frequent tokens."""
        "Obtenha o vocabulário do Bag of Words com até k palavras mais frequentes."
        most_common = word_count.most_common(k)
        self.bow_vocabulary = [word for word, count in most_common]
        return self.bow_vocabulary

    def get_bow_to_id_mapping(self) -> Dict[str, int]:
        """Create a mapping {token -> its index in the vocabulary}."""
        """Crie um mapeamento {token -> seu índice no vocabulário}."""
        self.bow_to_id = {token: idx for idx, token in enumerate(self.bow_vocabulary)}
        return self.bow_to_id

    def text_to_bow(self, text: str) -> ndarray:
        """Convert a text string to an array of token counts based on the BoW vocabulary."""
        "Converte uma string de texto em uma matriz de contagens de tokens com base no vocabulário BoW."
        bow = np.zeros(len(self.bow_vocabulary), dtype=np.int32)
        tokens = self.tokenizer.tokenize(text)

        for token in tokens:
            if token in self.bow_to_id:
                idx = self.bow_to_id[token]
                bow[idx] += 1

        return bow

    def compute_bow_matrix(self, texts: ndarray) -> ndarray:
        """Compute the BoW matrix for a list of texts."""
        "Calcule a matriz BoW para uma lista de textos."
        return np.stack(list(map(self.text_to_bow, texts)))

def test(X_train_bow: np.ndarray, X_test_bow: np.ndarray, word_count: WordCounts) -> None:
    """Validate the shapes of the BoW matrices."""
    "Validar os formatos das matrizes BoW."
    k_max = len(set(' '.join(word_count.train_df.text.values).split()))
    assert X_train_bow.shape == (len(word_count.train_df.text.values), min(10000, k_max))
    assert X_test_bow.shape == (len(word_count.test_df.text.values), min(10000, k_max))
    assert len(word_count.bow_vocabulary) <= min(10000, k_max)
    print("Validação da matriz BoW passada.")

#def print_sample(word_counts: WordCounts, df: pd.DataFrame) -> None:
#     """Print the BoW representation of a sample text."""
  #   sample_text = df.text.iloc[0][:100]
 #    print("Texto de exemplo:", sample_text) # Sample text
  #   print("Caixa de palavras:", word_counts.text_to_bow(sample_text)) # Bag of words
#
def print_sample(word_counts: WordCounts, df: pd.DataFrame) -> None:
    """Imprimir a representação BoW de um texto de amostra."""
    sample_text = df.text.iloc[0][:100]
    print("Texto de exemplo:", sample_text)  # Sample text
    bow_vector = word_counts.text_to_bow(sample_text)
    print("Lista de palavras vetor (entradas não zero):")
    non_zero = [(word_counts.bow_vocabulary[i], bow_vector[i])
                for i in range(len(bow_vector)) if bow_vector[i] > 0]
    print(non_zero[:20])

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()
    word_counts = WordCounts(train_df, test_df)
    print_sample(word_counts, train_df)

    X_train_bow = word_counts.compute_bow_matrix(train_df.text.values)
    X_test_bow = word_counts.compute_bow_matrix(test_df.text.values)

    test(X_train_bow, X_test_bow, word_counts)
    save_as_numpy(X_train_bow, conf.path.train_bow)
    save_as_numpy(X_test_bow, conf.path.test_bow)


if __name__ == '__main__':
    main()





# # SETUP
# import sys
# import os
# from collections import Counter
# from numpy import ndarray
# from pandas.core.frame import DataFrame
# from typing import Dict
#
# CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
# sys.path.insert(0, ROOT_DIRECTORY)
#
# # IMPORTS
# import numpy as np
# import pandas as pd
# import nltk
# nltk.download('punkt')
# from collections import Counter
# #python -c "import nltk; nltk.download('punkt')"
#
# # noinspection PyUnresolvedReferences
# from envvars import LessonEnv
# from tools_basics.helpers import get_config, save_as_numpy
# from tools_basics.data_handler import DataHandler
#
#
# class WordCounts:
#     """Class to compute word counts and Bag of Words (BoW) representations."""
#
#     def __init__(self, train_df: DataFrame, test_df: DataFrame) -> None:
#         self.train_df = train_df
#         self.test_df = test_df
#         #self.tokenizer = Tokenizer()
#         self.tokenizer = Tokenizer()  # Cria a instância
#         self.bow_vocabulary = None
#         self.bow_to_id = None
#         self._set_bow()
#
#     def _set_bow(self) -> None:
#         texts = self.train_df.text.values
#         word_count = self.get_words_count(texts)
#         self.bow_vocabulary = self.get_bow_vocabulary(texts, word_count)
#         self.bow_to_id = self.get_bow_to_id_mapping()
#
#
#     def get_words_count(self, texts: ndarray) -> Counter:
#         """Compute word counts from the given texts."""
#         words_count = Counter()
#         for text in texts:
#             # Tokenize each text
#             tokens = self.tokenizer.tokenize(text)
#             # Update counter with tokens
#             words_count.update(tokens)
#         return words_count
#
#
#     def get_bow_vocabulary(self, texts: list[str], word_count: Counter, k: int = 10000) -> list[str]:
#         """Get the Bag of Words vocabulary with up to k most frequent tokens."""
#         most_common = word_count.most_common(k)
#         # self.bow_vocabulary =  = [word for word, count in word_count.most_common(k) ]
#         self.bow_vocabulary = [word for word, count in most_common ]
#         return self.bow_vocabulary
#
#     def get_bow_to_id_mapping(self) -> Dict[str, int]:
#         """Create a mapping {token -> its index in the vocabulary}."""
#         self.bow_to_id = {token: idx for idx, token in enumerate(self.bow_vocabulary)}
#         return self.bow_to_id
#
#     def text_to_bow(self, text: str) -> ndarray:
#         """Convert a text string to an array of token counts based on the BoW vocabulary.
#
#         Steps:
#         1. Tokenize the input text.
#         2. For each token, get its index in the BoW vocabulary.
#         3. Create a BoW vector with the token counts. (e.g. if word 'apply' with id 3 occurs 7 times, bow[3] = 7)
#             Note: the length of BoW vector should be equal to the length of the BoW vocabulary.
#         """
#         bow = np.zeros(len(self.bow_vocabulary),dtype=np.int32)
#         tokens = self.tokenizer.tokenize(text)
#         for token in tokens:
#             if token in self.bow_to_id:
#                 idx = self.bow_to_id[token]
#                 bow[idx] += 1
#         return bow
#
#     def compute_bow_matrix(self, texts: ndarray) -> ndarray:
#         """Compute the BoW matrix for a list of texts."""
#         return np.stack(list(map(self.text_to_bow, texts)))
#
# def test(X_train_bow: np.ndarray, X_test_bow: np.ndarray, word_count: WordCounts) -> None:
#     """Validate the shapes of the BoW matrices."""
#     k_max = len(set(' '.join(word_count.train_df.text.values).split()))
#     assert X_train_bow.shape == (len(word_count.train_df.text.values), min(10000, k_max))
#     assert X_test_bow.shape == (len(word_count.test_df.text.values), min(10000, k_max))
#     assert len(word_count.bow_vocabulary) <= min(10000, k_max)
#     print("Validação da matriz BoW aprovada.") # BoW matrix validation passed
#
# def print_sample(word_counts: WordCounts, df: pd.DataFrame) -> None:
#     """Print the BoW representation of a sample text."""
#     sample_text = df.text.iloc[0][:100]
#     print("Texto de exemplo:", sample_text) # Sample text
#     print("Caixa de palavras:", word_counts.text_to_bow(sample_text)) # Bag of words
#
# def main() -> None:
#     conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
#     dh = DataHandler(conf)
#     train_df, test_df = dh.get_data()
#
#     word_counts = WordCounts(train_df, test_df)
#     print_sample(word_counts, train_df)
#
#     # Compute BoW matrices for training and test datasets
#     X_train_bow = word_counts.compute_bow_matrix(train_df.text.values)
#     X_test_bow = word_counts.compute_bow_matrix(test_df.text.values)
#
#     test(X_train_bow, X_test_bow, word_counts)
#
#     save_as_numpy(X_train_bow, conf.path.train_bow)
#     save_as_numpy(X_test_bow, conf.path.test_bow)
#
# if __name__ == '__main__':
#     main()
