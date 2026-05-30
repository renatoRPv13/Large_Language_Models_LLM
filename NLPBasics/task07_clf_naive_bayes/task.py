# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Tuple

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config

class BinaryNaiveBayes:
    """A binary Naive Bayes classifier using bag-of-words features."""
    "Um classificador Naive Bayes binário usando características de saco de palavras."
    delta = 1.0  # Smoothing parameter

    def fit(self, X, y):
        """
        Ajuste um classificador de Naive Bayes para duas classes.
        :param X: [batch_size, vocab_size] of bag-of-words features
        :param y: [batch_size] of binary targets {0, 1}
        """
        # Calcular probabilidades marginais p(y=k) para k=0,1
        n_samples = len(y)
        p_positive = np.mean(y)
        self.p_y = np.array([1 - p_positive, p_positive])

        # Contar ocorrências de palavras para cada classe
        # Você pode querer levar o delta em conta neste palco
        word_counts_negative = X[y == 0].sum(axis=0) + self.delta
        word_counts_positive = X[y == 1].sum(axis=0) + self.delta

        # Estimate p(x | y=k) for k=0,1
        self.p_x_given_positive = word_counts_positive / word_counts_positive.sum()
        self.p_x_given_negative = word_counts_negative / word_counts_negative.sum()
        return self

        """
        # Contar ocorrências de palavras para cada classe
        # Você pode querer levar o delta em conta neste palco
        word_counts_negative = # TODO
        word_counts_positive = # TODO

        # Estimate p(x | y=k) for k=0,1
        self.p_x_given_positive = # TODO
        self.p_x_given_negative = # TODO

        return self
        """

    def predict_scores(self, X):
        """
        :param X: [batch_size, vocab_size] of bag-of-words features
        :returns: a matrix of scores [batch_size, 2] for each class

        The sores should be computed as follows:
        1. Recall that you have p(x | y=k) for k=0,1 and p(y=k) for k=0,1.
        2. Compute the log-probabilities for each class:
            log p(y=k) + sum_i (count_i * log p(x_i | y=k)) where i is the index of the word in the vocabulary, count_i is the number of occurrences of the word in the text.
        Note: it's better that you write vectorized code here (without loops)
        :param X: [batch_size, vocabulário_size] de recursos de bag-of-words
        :retornos: uma matriz de pontuações [batch_size, 2] para cada classe
        Os escores devem ser calculados da seguinte forma:
        1. 1. Lembre-se que você tem p(x | y=k) para k=0,1 e p(y=k) para k=0,1.
        2. 2. Compute as probabilidades de log para cada classe:
            log p(y=k) + sum_i (count_i * log p(x_i | y=k)) onde i é o índice da palavra no vocabulário, count_i é o número de ocorrências da palavra no texto.
        Nota: é melhor que você escreva código vetorizado aqui (sem loops)
        """

        # Log probabilities for each class # Logaritmos das probabilidades para cada classe
        log_p_y = np.log(self.p_y)

        # Log likelihood for each class (vectorized) # Logaritmo da verossimilhança para cada classe (vetorizada)
        log_p_x_given_negative = X * np.log(self.p_x_given_negative + 1e-10)
        log_p_x_given_positve = X * np.log(self.p_x_given_positive + 1e-10)

        # Sum over vocabulary (axis=1) # Soma sobre o vocabulário (eixo=1)
        score_negative = log_p_y[0] + np.sum(log_p_x_given_negative, axis=1)
        score_positive = log_p_y[1] + np.sum(log_p_x_given_positve, axis=1)
        return np.vstack([score_negative, score_positive]).T

    def predict(self, X):
        """Predict the class for each sample.""" "Preveja a classe para cada amostra."
        return np.argmax(self.predict_scores(X), axis=1)

def load_bow_train_test(conf: OmegaConf) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the train and test data from the specified paths.
    Return:
        X_train_bow, X_test_bow, y_train, y_test
    Carrega os dados de treino e teste dos caminhos especificados.
    Retorno:
        X_train_bow, X_test_bow, y_train, y_test
    """
    X_train_bow = np.load(conf.path.train_bow)
    X_test_bow = np.load(conf.path.test_bow)
    y_train = np.load(conf.path.train_target)
    y_test = np.load(conf.path.test_target)
    return X_train_bow, X_test_bow, y_train, y_test


def visualize_auc(naive_model: BinaryNaiveBayes, X_train_bow: np.ndarray, X_test_bow: np.ndarray, y_train: np.ndarray,
                  y_test: np.ndarray) -> None:
    """Visualizar a curva ROC AUC para o trem e testar conjuntos de dados."""
    for name, X, y, model in [
        ('train', X_train_bow, y_train, naive_model),
        ('test', X_test_bow, y_test, naive_model)
    ]:
        proba = model.predict_scores(X)[:, 1] - model.predict_scores(X)[:, 0]  # p(y=1) - p(y=0)
        auc = roc_auc_score(y, proba)
        plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

    plt.plot([0, 1], [0, 1], '--', color='black')
    plt.legend(fontsize='large')
    plt.grid()
    plt.show()


def eval_test(naive_model: BinaryNaiveBayes, X_test_bow: np.ndarray, y_test: np.ndarray) -> None:
    """Evaluate the model on the test dataset."""
    "Avalie o modelo no conjunto de dados de teste."
    test_accuracy = np.mean(naive_model.predict(X_test_bow) == y_test)
    print(f"Modelo accuracy: {test_accuracy:.3f}")  # Model accuracy
    assert test_accuracy > 0.75, "Accuracy is too low. Check the model implementation." "A precisão está muito baixa. Verifique a implementação do modelo."
    print(
        "Avaliação do modelo de Naive Bayes concluída com sucesso.")  # Naive Bayes model evaluation completed successfully


def train_and_evaluate(X_train_bow: np.ndarray, X_test_bow: np.ndarray, y_train: np.ndarray,
                       y_test: np.ndarray) -> None:
    """Train a Naive Bayes classifier and evaluate its performance."""
    "Treine um classificador Naive Bayes e avalie seu desempenho."
    naive_model = BinaryNaiveBayes().fit(X_train_bow, y_train)
    visualize_auc(naive_model, X_train_bow, X_test_bow, y_train, y_test)
    eval_test(naive_model, X_test_bow, y_test)


def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)

    X_train_bow, X_test_bow, y_train, y_test = load_bow_train_test(conf)
    train_and_evaluate(X_train_bow, X_test_bow, y_train, y_test)


if __name__ == '__main__':
    main()
