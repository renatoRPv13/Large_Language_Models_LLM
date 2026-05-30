# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
from sklearn.manifold import TSNE
# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.visualizer import Visualizer as Vis


class EmbeddingReducer:
    """Class to reduce word embeddings using t-SNE.""" "Classe para reduzir embeddings de palavras usando t-SNE."
    def reduce(self, word_vectors: np.ndarray) -> np.ndarray:
        """Performs dimensionality reduction using t-SNE and normalize the word vectors."""
        # Apply t-SNE to reduce dimensions to 2 # Aplicar t-SNE para reduzir as dimensões para 2
        word_tsne = TSNE(n_components=2, random_state=42).fit_transform(word_vectors) # TODO: reduce
        word_tsne = (word_tsne - word_tsne.min()) / (word_tsne.max() - word_tsne.min()) # TODO: normalize (see the picture from the description)
        return word_tsne # TODO: normalizar (veja a imagem na descrição)

def reduce_and_draw(vectors: np.array, tokens: np.array) -> None:
    """Reduce the dimensionality of vectors and visualize them."""
    "Reduzir a dimensionalidade dos vetores e visualizá-los."
    reducer = EmbeddingReducer()
    reduced_vectors = reducer.reduce(vectors)
    Vis.draw_vectors(reduced_vectors[:, 0], reduced_vectors[:, 1], token=tokens)

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)

    # words # palavras
    word_vectors = np.load(conf.path.word_embs)
    words = np.load(conf.path.words)

    reduce_and_draw(word_vectors, words)

    # chosen phrases # frase escolhida
    phrase_vectors = np.load(conf.path.phrase_embs)
    phrases = np.load(conf.path.phrases)

    reduce_and_draw(phrase_vectors, phrases)

if __name__ == '__main__':
    main()