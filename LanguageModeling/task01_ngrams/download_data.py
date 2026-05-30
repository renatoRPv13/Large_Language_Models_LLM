from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import wget
import pandas as pd
import nltk
import os
import tarfile
from sklearn.model_selection import train_test_split
from LanguageModeling.task01_ngrams.task import BOS, EOS



def get_abs_path_in_files_folder(file_name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), file_name))

DATA_RAW_PATH = get_abs_path_in_files_folder("arxivData.json")
TRAIN_PATH = get_abs_path_in_files_folder("arxivLinesTrain.txt")
TEST_PATH = get_abs_path_in_files_folder("arxivLinesTest.txt")
TEST_SPLIT = 0.25
MAX_LENGTH = 512
def extract_arxiv():
    if os.path.exists(DATA_RAW_PATH):
       #print("Data has already been downloaded.")
        print("Os dados já foram baixados.")
        return
    
    DATA_COMPRESSED_PATH = get_abs_path_in_files_folder("arxivData.json.tar.gz")
    with tarfile.open(DATA_COMPRESSED_PATH) as tar:
        tar.extractall(path=os.path.dirname(DATA_RAW_PATH))


def process_data():
    data = pd.read_json(DATA_RAW_PATH)
    new_line_to_space = lambda x: x.replace('\n', ' ')
    lines = data.apply(
        lambda row: new_line_to_space(row['title']) + ' ; ' + new_line_to_space(row['summary']),
        axis=1
        ).apply(
            lambda line: line[:MAX_LENGTH].lower()
        ).apply(
            lambda line: BOS + line + EOS
        ).tolist()
        

    tokenizer = nltk.tokenize.WordPunctTokenizer()
    lines = [tokenizer.tokenize(line.lower()) for line in lines]
    lines = [' '.join(line) for line in lines]
    return lines

def split_data(lines):
    train, test = train_test_split(lines, test_size=TEST_SPLIT, random_state=42)
    with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train))
    with open(TEST_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test))
    return train, test

def get_train_test() -> tuple[list, list]:
    """
    Get train and test data as lists of strings.
    :returns: train, test
    Obtenha os dados de treino e teste como listas de strings.
    :returns: treino, teste
    """
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        #print("Data has not been downloaded yet. Downloading now.")
        print("Os dados ainda não foram baixados. Baixando agora.")
        main()
    
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train = f.readlines()
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test = f.readlines()
    return train, test

def main():
    extract_arxiv()
    lines = process_data()
    train, test = split_data(lines)
    print(f"Tamanho do treino: {len(train)} linha")
    print(f"Tamanho do teste: {len(test)} linha")

if __name__ == "__main__":
    main()
    