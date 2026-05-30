# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.data_handler import DataHandler
# noinspection PyUnresolvedReferences
from task03_glove_embeddings.task import GloVeEmbeddings

class KNNClassifier:
    """A k-Nearest Neighbors (kNN) classifier based on cosine similarity."""
    "Um classificador k-Nearest Neighbors (kNN) baseado na similaridade de cosseno."

    def __init__(self, train_df, test_df, get_phrase_embedding_fn):
        self.train_df = train_df
        self.test_df = test_df
        self.get_phrase_embedding = get_phrase_embedding_fn
        self.X_train = np.array([self.get_phrase_embedding(phrase) for phrase in self.train_df.text])
        self.y_train = self.train_df.label.values

    @staticmethod
    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute the cosine similarity between two vectors."""
        "Calcule a similaridade de cosseno entre dois vetores."
        # Check if any vector is close to zero # Verificar se algum vetor está próximo de zero
        if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
            return 0.0

        # Compute cosine similarity: # Calcular a similaridade de cosseno: (a·b) / (||a|| * ||b||)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        return dot_product / (norm_a * norm_b)

    def find_nearest(self, query: str, k: int = 10) -> tuple:
        """
        Given a text query, return k most similar lines from the training data.
        Similarity is measured using cosine similarity between phrase embeddings.
        Steps:
        1. Compute cosine similarity between the query and all training samples.
        2. Find top_k_sorted_ids: indices of the k most similar training samples to the query (sorted by similarity).
            * Hint: use np.argpartition and np.argsort
        3. Return the text and labels of the top k most similar training samples.

            Dado um texto de consulta, retorne as k linhas mais semelhantes dos dados de treinamento.
        A similaridade é medida usando a similaridade de cosseno entre os embeddings de frases.
        Passos:
        1. Calcule a similaridade de cosseno entre a consulta e todas as amostras de treinamento.
        2. Encontre top_k_sorted_ids: índices das k amostras de treinamento mais semelhantes à consulta (ordenadas por similaridade).
        * Dica: use np.argpartition e np.argsort
        3. Retorne o texto e os rótulos das k amostras de treinamento mais semelhantes.
        """
        k = min(k, len(self.train_df))  # Ensure k is not larger than the training set size
        query_vec = self.get_phrase_embedding(query)

        # Compute cosine similarity with all training samples
        # Calcular a similaridade de cosseno com todas as amostras de treinamento
        similarities = np.array([self.cos_sim(query_vec, train_vec) for train_vec in self.X_train])

        # Get top k indices using argpartition (partial sort for efficiency)
        # Obtenha os k principais índices usando argpartition (ordenação parcial para maior eficiência)
        if len(similarities) >= k:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            # Sort these indices by similarity in descending order
            # Ordene esses índices por similaridade em ordem decrescente
            top_k_sorted_ids = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        else:
            top_k_sorted_ids = np.argsort(similarities)[::-1]

        return self.train_df.iloc[top_k_sorted_ids].text.values, self.y_train[top_k_sorted_ids]

    def get_accuracy(self, x_test_phrases: list, y_test: np.ndarray, k: int) -> float:
        """Compute the accuracy of the kNN classifier on the test dataset.
        Steps:
            1. For each test phrase, find the k most similar training samples.
            2. Extract the prediction, that is the most common label among the k nearest (e.g. [0, 1, 1, 0, 1] -> 1).
                Note: we assume k is odd to avoid ties.
                Hint: you can do it through np.bincount or np.sum

            Calcule a acurácia do classificador kNN no conjunto de dados de teste.
        Passos:
            1. Para cada frase de teste, encontre as k amostras de treinamento mais semelhantes.
            2. Extraia a predição, ou seja, o rótulo mais comum entre os k mais próximos (por exemplo, [0, 1, 1, 0, 1] -> 1).
            Observação: assumimos que k é ímpar para evitar empates.
        Dica: você pode fazer isso usando np.bincount ou np.sum.
        """
        correct = 0
        for i, phrase in tqdm(enumerate(x_test_phrases), total=len(x_test_phrases)):
            # Find k nearest neighbors # Encontrar os k vizinhos mais próximos
            _, neighbor_labels = self.find_nearest(phrase, k=k)

            # Extract prediction: most common label among neighbors
            # Using np.bincount to count occurrences of each label
            # Extrair previsão: rótulo mais comum entre os vizinhos
            # Usando np.bincount para contar as ocorrências de cada rótulo
            pred = np.argmax(np.bincount(neighbor_labels))

            # Alternative using np.sum (if labels are 0 and 1):
            # pred = 1 if np.sum(neighbor_labels) > k/2 else 0
            # Alternativa usando np.sum (se os rótulos forem 0 e 1):
            # pred = 1 se np.sum(rótulos_vizinhos) > k/2 senão 0

            correct += pred == y_test[i]
        return correct / len(y_test)

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()
    glove = GloVeEmbeddings()  # ~1.5min for 100d
    knn_classifier = KNNClassifier(train_df, test_df, glove.get_phrase_embedding)
    # Sample and truncate test data for faster execution
    # Amostrar e truncar dados de teste para uma execução mais rápida
    n_samples = min(100, len(test_df))  # Garantir que não ultrapasse o tamanho do test_df
    trunc_test_df = test_df.sample(n_samples)
    x_test_phrases_trunc = trunc_test_df.text.values
    y_test_trunc = trunc_test_df.label.values

    accuracy = knn_classifier.get_accuracy(x_test_phrases_trunc, y_test_trunc, k=3)
    print(f"Precisão do classificador k-NN: {accuracy:.4f}")


if __name__ == '__main__':
    main()
