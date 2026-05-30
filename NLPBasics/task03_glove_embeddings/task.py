# SETUP
import sys
import os
CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
# custom gensim data directory
os.environ['GENSIM_DATA_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), './.gensim_data')
sys.path.insert(0, ROOT_DIRECTORY)
# IMPORTS
import os
import numpy as np
import gensim.downloader as api
# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config, save_as_numpy
from tools_basics.data_handler import DataHandler
# noinspection PyUnresolvedReferences
from task02_tokenization.task import Tokenizer


class GloVeEmbeddings:
    """Class to load and work with GloVe embeddings."""
    "Classe para carregar e trabalhar com embeddings GloVe."
    def __init__(self, model_name: str = 'glove-twitter-25') -> None:
        """Initialize the class and load the GloVe model.""" "Inicialize a classe e carregue o modelo GloVe."
        self.model = api.load(model_name)
        self.tokenizer = Tokenizer()
        self._unknown_emb = np.zeros(self.model.vector_size)

    def get_word_vectors(self, words: list):
        """Get the vector representations for a list of words.
        Note: In case a word is not in the vocabulary, return a zero vector."""
        """Obtenha as representações vetoriais para uma lista de palavras.
            Observação: caso uma palavra não esteja no vocabulário, retorne um vetor nulo."""
        word_vectors = []
        for word in words:
            try:
                # Try to get the vector for the word # Tente obter o vetor da palavra
                vec = self.model.get_vector(word)
            except KeyError:
                # If word not in vocabulary, return zero vector
                # Se a palavra não estiver no vocabulário, retorne um vetor zero.
                vec = self._unknown_emb.copy()
            word_vectors.append(vec)
        return np.array(word_vectors)

    def get_phrase_embedding(self, phrase: str) -> np.ndarray:
        """
        Convert phrase to a vector by aggregating word embeddings.
        - Lowercase the phrase
        - Tokenize the phrase
        - Average the word vectors for all words in the tokenized phrase
        - Skip words not in the model's vocabulary
        - If all words are missing from the vocabulary, return zeros
        """
        """Converter frase em vetor agregando embeddings de palavras.
            - Converter a frase para minúsculas
            - Tokenizar a frase
            - Calcular a média dos vetores de palavras para todas as palavras na frase tokenizada
            - Ignorar palavras que não estão no vocabulário do modelo
            - Se todas as palavras estiverem ausentes do vocabulário, retornar zeros"""
        # Lowercase the phrase # Escreva a frase em minúsculas
        phrase = phrase.lower()

        # Tokenize the phrase # Tokenizar a frase
        tokens = self.tokenizer.tokenize(phrase)

        # Collect vectors for words that are in the vocabulary
        # Coletar vetores para palavras que estão no vocabulário
        vectors = []
        for token in tokens:
            try:
                vec = self.model.get_vector(token)
                vectors.append(vec)
            except KeyError:
                # Skip words not in vocabulary
                continue

        # If all words are missing from the vocabulary, return zeros
        # Se todas as palavras estiverem faltando no vocabulário, retorne zeros
        if len(vectors) == 0:
            return self._unknown_emb.copy()

        # Average the word vectors # Calcule a média dos vetores de palavras
        return np.mean(vectors, axis=0)

    def compute_phrase_vectors(self, phrases: list[str], max_tokens: int | None = 30) -> np.ndarray:
        """Truncate and compute vectors for a list of phrases.
        Args:
            phrases (list[str]): List of phrases to compute embeddings for.
            max_tokens (int): Maximum number of tokens to consider in each phrase. (If None, consider all tokens)
        Trunca e calcula vetores para uma lista de frases.
        Argumentos:
            frases (lista[str]): Lista de frases para as quais calcular os embeddings.
            tokens_máximos (int): Número máximo de tokens a serem considerados em cada frase. (Se None, considera todos os tokens)
        """
        phrase_vectors = []

        for phrase in phrases:
            # Lowercase the phrase # Escreva a frase em minúsculas
            phrase = phrase.lower()
            # Tokenize the phrase # Tokenizar a frase
            tokens = self.tokenizer.tokenize(phrase)
            # Truncate tokens if max_tokens is specified # Truncar tokens se max_tokens for especificado
            if max_tokens is not None and len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
            # Collect vectors for words in vocabulary # Coletar vetores para palavras no vocabulário
            vectors = []
            for token in tokens:
                try:
                    vec = self.model.get_vector(token)
                    vectors.append(vec)
                except KeyError:
                    # Skip words not in vocabulary # Ignore palavras que não estão no vocabulário
                    continue

            # If no valid vectors, use zero vector # Se não houver vetores válidos, use o vetor zero.
            if len(vectors) == 0:
                phrase_vectors.append(self._unknown_emb.copy())
            else:
                # Average the vectors # Calcule a média dos vetores
                phrase_vectors.append(np.mean(vectors, axis=0))

        return np.array(phrase_vectors)

N_WORDS = 1000
N_PHRASES = 1000
def extract_and_save_data(train_df, test_df, glove, conf):
    """Extract data for the following tasks:
    1. Words visualization (word embeddings, words)
    2. Chosen phrases visualization (phrase embeddings, phrases)
    3. Train and test data embeddings (train_embeddings, test_embeddings, target)
    Then, save
     Extraia os dados para as seguintes tarefas:
    1. Visualização de palavras (embeddings de palavras, palavras)
    2. Visualização de frases selecionadas (embeddings de frases, frases)
    3. Embeddings de dados de treino e teste (embeddings_treino, embeddings_teste, alvo)
    Em seguida, salve.
    """
    words = glove.model.index_to_key[:N_WORDS]
    word_vectors = glove.get_word_vectors(words)

    # CORREÇÃO: Verificar se o DataFrame tem pelo menos N_PHRASES amostras
    n_samples = min(N_PHRASES, len(train_df))
    phrases = train_df.sample(n_samples, random_state=42).text.tolist()
    phrase_vectors = glove.compute_phrase_vectors(phrases)
    train_embeddings = glove.compute_phrase_vectors(train_df.text.tolist(), max_tokens=None)
    train_target = train_df.label.values
    test_embeddings = glove.compute_phrase_vectors(test_df.text.tolist(), max_tokens=None)
    test_target = test_df.label.values

    # Save
    for data, path in (
            (words, conf.path.words),
            (word_vectors, conf.path.word_embs),
            (phrases, conf.path.phrases),
            (phrase_vectors, conf.path.phrase_embs),
            (train_embeddings, conf.path.train_emb),
            (train_target, conf.path.train_target),
            (test_embeddings, conf.path.test_emb),
            (test_target, conf.path.test_target),
    ):
        save_as_numpy(data, path)

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()

    glove = GloVeEmbeddings()  # ~1.5min for 100d

    extract_and_save_data(train_df, test_df, glove, conf)

if __name__ == '__main__':
    main()