import sys
import os

# Adicione a raiz do projeto ao caminho do Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F

from MinLlama.HelperCode.config import LlamaConfig
from MinLlama.Llama.task import load_pretrained
from MinLlama.HelperCode.tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
    def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
        super(LlamaZeroShotClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # A classificação Zero-shot não requer a atualização dos parâmetros da lhama.
        for param in self.llama.parameters():
            param.requires_grad = False
        assert len(label_names) == self.num_labels
        self.tokenizer = tokenizer
        self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


    def forward(self, input_ids):
        # Calcular a probabilidade de conclusão de cada string de rótulo
        logits, _ = self.llama(input_ids)
        log_probabilities = F.log_softmax(logits, dim=-1)
        label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
        for i, label_token_ids in enumerate(self.label_name_ids):
            total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
            label_probabilities[:, i] = total_log_prob[:, 0]
        return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
    def __init__(self, config):
        super(LlamaEmbeddingClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.llama = load_pretrained(config.pretrained_model_path)
        # If we use pretrain mode, we freeze Llama parameters.
        # Se usarmos o modo de pré-treinamento, congelamos os parâmetros da Lhama.
        for param in self.llama.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

    def forward(self, input_ids):
        """
        Passagem direta para classificação baseada em embeddings.

        Etapas:
        1) Obter os estados ocultos da Llama (última camada)
        2) Extrair o estado oculto do token final (representação da sequência)
        3) Aplicar dropout para regularização
        4) Passar pelo classificador linear para obter os logits
        5) Aplicar log-softmax para retornar as probabilidades logarítmicas
        Argumentos:
            input_ids: torch.LongTensor com formato (batch_size, seq_len)
        Retorna:
            log_probs: torch.FloatTensor com formato (batch_size, num_labels)
        Probabilidades logarítmicas para cada classe
        """
        """
        Forward pass for embedding-based classification.
        Steps:
        1) Get hidden states from Llama (last layer)
        2) Extract hidden state of the final token (sequence representation)
        3) Apply dropout for regularization
        4) Pass through linear classifier head to get logits
        5) Apply log-softmax to return log probabilities

        Args:
            input_ids: torch.LongTensor of shape (batch_size, seq_len)

        Returns:
            log_probs: torch.FloatTensor of shape (batch_size, num_labels)
                      Log probabilities for each class
        """
        # Etapa 1: Obter estados ocultos do Llama
        # Muitas implementações do Llama retornam (último_estado_oculto, valores_chave_passados)
        outputs = self.llama(input_ids)

        # Extrair estados ocultos - lidar com diferentes formatos de retorno
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # Opção A: a saída é um objeto com o atributo .hidden_states
            hidden_states = outputs.hidden_states[-1]  # Última camada
        elif isinstance(outputs, tuple) and len(outputs) >= 2:
            # Opção B: a saída é uma tupla (logits, hidden_states, ...)
            # Assumindo que o segundo elemento seja hidden_states, se disponível; caso contrário, o primeiro (logits)
            hidden_states = outputs[1] if (len(outputs) > 1 and outputs[1] is not None) else outputs[0]
        else:
            # Opção C: alternativa - assume que a saída é hidden_states diretamente
            hidden_states = outputs

        # Etapa 2: Obtenha o estado oculto do ÚLTIMO token (posição final)
        # Formato: (tamanho_do_lote, comprimento_da_sequência, dimensão_oculta) -> (tamanho_do_lote, dimensão_oculta)
        last_token_hidden = hidden_states[:, -1, :]

        # Etapa 3: Aplicar desistência
        dropped_hidden = self.dropout(last_token_hidden)

       # Etapa 4: Passe pelo cabeçalho do classificador para obter os logits.
        logits = self.classifier_head(dropped_hidden) # (tamanho_do_lote, número_de_rótulos)

        # Etapa 5: Aplique a função log-softmax para obter as probabilidades logarítmicas
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs
