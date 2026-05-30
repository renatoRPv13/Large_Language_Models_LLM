# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
from omegaconf import OmegaConf
from typing import Tuple

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
# noinspection PyUnresolvedReferences
from task08_clf_logreg_with_word_counts.task import MyLogisticRegression


def load_emb_train_test(conf: OmegaConf) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the train and test data from the specified paths.
    Carregue os dados de treino e de teste a partir dos caminhos especificados.

    Return:
        X_train_emb, X_test_emb, y_train, y_test
    """
    X_train_emb = np.load(conf.path.train_emb)
    X_test_emb = np.load(conf.path.test_emb)
    y_train = np.load(conf.path.train_target)
    y_test = np.load(conf.path.test_target)
    return X_train_emb, X_test_emb, y_train, y_test

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    
    X_train_bow, X_test_bow, y_train, y_test = load_emb_train_test(conf)
    logreg_model = MyLogisticRegression()
    logreg_model.eval_model(X_train_bow, y_train, X_test_bow, y_test)

if __name__ == '__main__':
    main()
