# Taken from llama code and modified by Andrej Karpathy.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import struct
import argparse
from typing import List

# Adicionar o caminho raiz do projeto ao sys.path (ANTES das importações)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora as importações do MinLlama funcionam
from MinLlama.HelperCode.config import download_data

from sentencepiece import SentencePieceProcessor

# as we run from the other directory
TOKENIZER_MODEL = "../HelperCode/tokenizer.model"


class Tokenizer:
    def __init__(self, max_len=None, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        url = "https://raw.githubusercontent.com/jetbrains-academy/llm-agent-course/main/MinLlama/HelperCode/tokenizer.model"
        download_data(url, model_path)
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path
        self.max_len = max_len

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        # Overwrite the default of pad_id=-1, which is problematic.
        self.pad_id: int = self.sp_model.piece_to_id("<0x00>")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if self.max_len is not None and len(t) > self.max_len:
            t = t[:self.max_len]
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):
        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ')  # sentencepiece uses this character as whitespace
            b = t.encode('utf-8')  # bytes of this token, utf-8 encoded
            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        tokenizer_bin = self.model_path.replace('.model', '.bin')
        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)
    t.export()



# # Taken from llama code and modified by Andrej Karpathy.
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
#
# import os
# import struct
# import argparse
# from typing import List
#
# from sentencepiece import SentencePieceProcessor
#
# from MinLlama.HelperCode.config import download_data
#
# import sys
# import os
#
# # Adicionar o caminho raiz do projeto ao sys.path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# # Agora as importações devem funcionar
# # from MinLlama.Classifier.task import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
# # from MinLlama.Llama.task import Llama, load_pretrained
# # from MinLlama.Optimizer.task import AdamW
# # from MinLlama.HelperCode.tokenizer import Tokenizer
#
# # as we run from the other directory # enquanto executamos a partir do outro diretório
# TOKENIZER_MODEL = "../HelperCode/tokenizer.model" # the llama sentencepiece tokenizer model
# # o modelo de tokenizador de frases Llama
# class Tokenizer:
#     def __init__(self, max_len=None, tokenizer_model=None):
#         model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
#         url = "https://raw.githubusercontent.com/jetbrains-academy/llm-agent-course/main/MinLlama/HelperCode/tokenizer.model"
#         download_data(url, model_path)
#         assert os.path.isfile(model_path), model_path
#         self.sp_model = SentencePieceProcessor(model_file=model_path)
#         self.model_path = model_path
#         self.max_len = max_len
#
#         # BOS / EOS token IDs # IDs de token BOS/EOS
#         self.n_words: int = self.sp_model.vocab_size()
#         self.bos_id: int = self.sp_model.bos_id()
#         self.eos_id: int = self.sp_model.eos_id()
#         # Overwrite the default of pad_id=-1, which is problematic.
#         # Sobrescrever o valor padrão de pad_id=-1, que é problemático.
#         self.pad_id: int = self.sp_model.piece_to_id("<0x00>")
#         #print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
#         assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()
#
#     def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
#         assert type(s) is str
#         t = self.sp_model.encode(s)
#         if self.max_len is not None and len(t) > self.max_len:
#             t = t[:self.max_len]
#         if bos:
#             t = [self.bos_id] + t
#         if eos:
#             t = t + [self.eos_id]
#         return t
#
#     def decode(self, t: List[int]) -> str:
#         return self.sp_model.decode(t)
#
#     def export(self):
#
#         # get all the tokens (postprocessed) and their scores as floats
#         # Obtenha todos os tokens (pós-processados) e suas pontuações como números de ponto flutuante
#         tokens, scores = [], []
#         for i in range(self.n_words):
#
#             # decode the token and light postprocessing # decodificar o token e realizar pós-processamento leve
#             t = self.sp_model.id_to_piece(i)
#             s = self.sp_model.get_score(i)
#             if i == self.bos_id:
#                 t = '\n<s>\n'
#             elif i == self.eos_id:
#                 t = '\n</s>\n'
#             t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace #sentencepiece usa este caractere como espaço em branco
#             b = t.encode('utf-8') # bytes of this token, utf-8 encoded # bytes deste token, codificados em utf-8
#
#             tokens.append(b)
#             scores.append(s)
#
#         # record the max token length # registre o comprimento máximo do token
#         max_token_length = max(len(t) for t in tokens)
#
#         # write to a binary file
#         # the tokenizer.bin file is the same as .model file, but .bin
#         # Escrever em um arquivo binário
#         # O arquivo tokenizer.bin é o mesmo que o arquivo .model, mas com a extensão .bin
#         tokenizer_bin = self.model_path.replace('.model', '.bin')
#         with open(tokenizer_bin, 'wb') as f:
#             f.write(struct.pack("I", max_token_length))
#             for bytes, score in zip(tokens, scores):
#                 f.write(struct.pack("fI", score, len(bytes)))
#                 f.write(bytes)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer ")
#     args = parser.parse_args()
#
#     t = Tokenizer(args.tokenizer_model)
#     t.export()
