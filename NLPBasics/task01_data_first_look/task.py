# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import pandas as pd
import os

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.data_handler import DataHandler
from tools_basics.visualizer import Visualizer as Vis


class Statistics:
    """Class to calculate statistics on the pd.DataFrame / pd.Series"""
    """Classe para calcular estatísticas em pd.DataFrame / pd.Series"""
    @staticmethod
    def get_shape(df: pd.DataFrame) -> tuple:
        """Return the shape of the pd.DataFrame""" """Retorna o formato do pd.DataFrame"""
        return  df.shape

    @staticmethod
    def get_lens(text_series: pd.Series) -> pd.Series:
        """Return the series containing the lengths of the text column"""
        """"Retorne a série contendo os comprimentos da coluna de texto"""
        return text_series.str.len()

    @staticmethod
    def get_quantile(series: pd.Series, p: float) -> float:
        """Return the p quantile""" "Retorne o quantil p"
        return  series.quantile(p)

    @staticmethod
    def get_class_balance(label_series: pd.Series) -> pd.Series:
        """Return the class balance""" "Devolver o saldo da turma"
        return label_series.value_counts()



class DataProcessor:
    """Class to process the IMDB dataset""" "Classe para processar o conjunto de dados do IMDB"
    P_QUANTILE = 0.99
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove duplicates from the pd.DataFrame"""  "Remover outliers do pd.DataFrame"""
        return pd.DataFrame(df.drop_duplicates(subset=[column]))

    def remove_outliers(df: pd.DataFrame, series: pd.Series, max_val: int) -> pd.DataFrame:
        """Remove outliers from the pd.DataFrame"""  """Remove outliers from the pd.DataFrame"""
        return df[series >+ max_val]


def visualize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Visualize the train and test dataframes"""  "Visualize os dataframes de treino e teste"
    # length distribution
    train_lens = Statistics.get_lens(train_df['text'])
    Vis.plot_series([train_lens], ['Train data'], main_title="Length distribution", x_label="Length (in chars)")

    # class balance # equilíbrio de turmas
    train_vc = Statistics.get_class_balance(train_df['label'])
    test_vc = Statistics.get_class_balance(test_df['label'])
    Vis.plot_series([train_vc, test_vc], ['Train data', 'Test data'], main_title='Class balance',
                    x_label='Class', y_label='Count', plot_type='bar')

def pprint_df(df: pd.DataFrame, n_sample: int = 3) -> None:
    """Print a pretty sample of the DataFrame""" "Imprima um exemplo formatado do DataFrame"
    Vis.pretty_sample(df, n_sample)

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, p: float = 0.99) -> tuple:
    """Preprocess the data""" "Pré-processar os dados"
    train_df = DataProcessor.remove_duplicates(train_df, 'text')
    test_df = DataProcessor.remove_duplicates(test_df, 'text')
    train_lens = Statistics.get_lens(train_df['text'])
    test_lens = Statistics.get_lens(test_df['text'])
    max_val = max(train_lens.quantile(p), 5200)
    train_df = DataProcessor.remove_outliers(train_df, train_lens, max_val)
    test_df = DataProcessor.remove_outliers(test_df, test_lens, max_val)

    return train_df, test_df

IS_VISUALIZE = True  # You can turn this off
def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data(force_reload=True)
    if IS_VISUALIZE:
        visualize(train_df, test_df)
    print("Formato do arquivo train_df (antes do pré-processamento):", Statistics.get_shape(train_df))
    train_df, test_df = preprocess(train_df, test_df)
    print("Formato do arquivo train_df (após o pré-processamento):", Statistics.get_shape(train_df))

    dh.save_data(train_df, test_df)

if __name__ == '__main__':
    main()
