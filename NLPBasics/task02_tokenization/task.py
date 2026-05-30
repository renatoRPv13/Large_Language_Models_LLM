# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import pandas as pd
from nltk.tokenize import WordPunctTokenizer

# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config
from tools_basics.data_handler import DataHandler

class Tokenizer:
    """Class to tokenize the text data."""  "Classe para tokenizar os dados de texto."
    def __init__(self):
        self.tokenizer = WordPunctTokenizer()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenizes the input text.
        1. Lowercase the text
        2. Remove '<br />' tags
        3. Tokenize the text using WordPunctTokenizer
        Tokeniza o texto de entrada.
        1. Converta o texto para minúsculas
        2. Remova as tags '<br />'
        3. Tokenize o texto usando o WordPunctTokenizer
        """
        # 1. Lowercase the text   # 1. Converta o texto para minúsculas
        text = text.lower()
        # 2. Remove '<br />' tags  # 2. Remova as tags '<br />'
        text = text.replace('<br />', ' ')
        # 3. Tokenize the text using WordPunctTokenizer  # 3. Tokenize o texto usando o WordPunctTokenizer
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """Process the text (tokenize and join).""" "Processar o texto (tokenizar e unir)."""
        tokens = self.tokenize(text)
        return ' '.join(tokens)

    def apply_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies preprocessing of the 'text' column of the DataFrame.
        Use `pd.Series.apply`
        Aplica o pré-processamento à coluna 'text' do DataFrame.
        Use `pd.Series.apply`
        """
        df['text'] = df['text'].apply(self.preprocess_text)
        #df['text'] = self.preprocess_text(df['text'])
        return df

def get_sample_texts(df: pd.DataFrame, n: int = 3, random_state: int = 42) -> list:
    """Return a sample texts from the DataFrame.""" "Retorna um exemplo de texto do DataFrame."
    n_samples = min(n, len(df))
    return df.sample(n_samples, random_state=random_state)['text'].tolist()
# def get_sample_texts(df: pd.DataFrame, n: int = 3, random_state: int = 42) -> list:
#     """Return a sample texts from the DataFrame."""
#     return df.sample(n, random_state=random_state)['text'].tolist()

def print_texts(texts: list, title: str, n_truncate: int = 100) -> None:
    """Print the sample texts.""" "Imprima os textos de exemplo."
    print(title)
    for text in texts:
        print(text[:n_truncate])
    print()

def run_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame, verbose: bool = True) -> tuple:
    """Run preprocessing on the train and test DataFrames."""
    "Execute o pré-processamento nos DataFrames de treino e teste."
    if verbose:
        sample_before = get_sample_texts(train_df)
        print_texts(sample_before, "Textos de exemplo antes da tokenização:")
    
    tokenizer = Tokenizer()
    train_df = tokenizer.apply_preprocess(train_df)
    test_df = tokenizer.apply_preprocess(test_df)
    
    if verbose:
        sample_after = get_sample_texts(train_df)
        print_texts(sample_after, "Textos de exemplo após a tokenização:")
    
    return train_df, test_df

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    dh = DataHandler(conf)
    train_df, test_df = dh.get_data()
    train_df, test_df = run_preprocessing(train_df, test_df)
    
    # Save the tokenized data # Salvar os dados tokenizados
    dh.save_data(train_df, test_df)

if __name__ == '__main__':
    main()
