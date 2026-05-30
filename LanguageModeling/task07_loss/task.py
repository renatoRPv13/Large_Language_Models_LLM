"""
Module for computing loss functions for language models.
Módulo para calcular funções de perda para modelos de linguagem.
"""
from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from LanguageModeling.task05_text_tools.task import TextTools as tt

class CrossEntropyLoss(nn.Module):
    """
    Module for computing cross-entropy loss function.
    Módulo para calcular a função de perda de entropia cruzada.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss over non-eos tokens.
        :param logits: Float32 tensor of shape [batch, n_tokens, seq_len]
        :param input_idx: Int32 tensor of shape [batch, seq_len] with answers
        :returns: Scalar float32 loss
        Calcula a perda de entropia cruzada sobre tokens não-eos.
        :param logits: Tensor float32 de formato [batch, n_tokens, seq_len]
        :param input_idx: Tensor int32 de formato [batch, seq_len] com respostas
        :returns: Perda escalar float32
        """
        device = logits.device
        batch_size, n_tokens, seq_len = logits.shape

        # Get mask for valid tokens (until first EOS)
        # Obter máscara para tokens válidos (até o primeiro EOS)
        masks = tt.compute_mask(input_idx).to(device)  # [batch, seq_len]

        # Reshape logits from [batch, n_tokens, seq_len] to [batch * seq_len, n_tokens]
        # Remodelar logits de [batch, n_tokens, seq_len] para [batch * seq_len, n_tokens]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, n_tokens)

        # Reshape targets and mask to 1D # Redimensionar alvos e máscara para 1D
        targets_flat = input_idx.reshape(-1)
        masks_flat = masks.reshape(-1)

        # Create mask to exclude the first token (BOS) of each sequence
        # Criar máscara para excluir o primeiro token (BOS) de cada sequência
        first_token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        first_token_mask[:, 0] = True
        first_token_mask_flat = first_token_mask.reshape(-1)

        # Combine masks: exclude first token AND only include tokens before first EOS
        # Combinar máscaras: excluir o primeiro token E incluir apenas os tokens anteriores ao primeiro EOS
        final_mask = masks_flat & (~first_token_mask_flat)

        # Select valid indices # Selecionar índices válidos
        valid_indices = final_mask.bool()

        if valid_indices.sum() > 0:
            valid_logits = logits_flat[valid_indices]
            valid_targets = targets_flat[valid_indices]
            loss = F.cross_entropy(valid_logits, valid_targets)
        else:
            loss = torch.tensor(0.0, device=device)

        return loss
def main():
    import torch
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    loss_fn = CrossEntropyLoss()
    batch_size, seq_len, vocab_size = 2, 5, 10

    # Create dummy data # Criar dados fictícios
    logits = torch.randn(batch_size, vocab_size, seq_len)
    input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, input_idx)
   # print(f"Computed loss on dummy data: {loss.item():.4f}")
    print(f"Perda calculada em dados fictícios: {loss.item():.4f}")

if __name__ == '__main__':
    main()




# """
# Module for training procedure class.
# """
# from custom_helpers import add_root_to_pythonpath
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import os
# import torch
# import numpy as np
# from tqdm import trange
# import matplotlib.pyplot as plt
# from typing import List, Tuple
#
# from LanguageModeling.task05_text_tools.task import TextTools as tt
# from LanguageModeling.task02_generation.task import Generator
#
#
# class TrainProcedure:
#     """
#     Class for training the model and computing losses.
#     """
#
#     @staticmethod
#     def _compute_loss(model, loss_fn, train_lines, batch_ids, device):
#         """Compute loss for a batch of training lines."""
#         batch_texts = [train_lines[i] for i in batch_ids]
#         input_matrix = tt.to_matrix(batch_texts).to(device)
#         logits = model(input_matrix)
#         loss = loss_fn(logits, input_matrix)
#         return loss
#
#     @staticmethod
#     def score_lines(model: torch.nn.Module, lines: List[str], loss_fn, batch_size: int, device: torch.device) -> float:
#         """Computes average loss over the entire dataset."""
#         all_losses = []
#         for i in range(0, len(lines), batch_size):
#             with torch.no_grad():
#                 batch_ids = range(i, min(i + batch_size, len(lines)))
#                 loss = TrainProcedure._compute_loss(model, loss_fn, lines, batch_ids, device)
#                 all_losses.append(loss.item())
#         return np.mean(all_losses)
#
#     @staticmethod
#     def produce_meta(train_history: List[Tuple[int, float]], dev_history: List[Tuple[int, float]], gen: Generator,
#                      gen_conf: dict):
#         """Produce meta information during training."""
#         plt.figure(figsize=[12, 4])
#         plt.scatter(*zip(*train_history), alpha=0.1, label='train_loss')
#         if len(dev_history):
#             plt.plot(*zip(*dev_history), color='red', label='dev_loss')
#         path_to_save = os.path.join(os.path.dirname(__file__), 'train_test_plot.png')
#         plt.legend()
#         plt.grid()
#         plt.savefig(path_to_save)
#         plt.close()
#
#         gen_conf_str = ', '.join([f'{k}={v}' for k, v in gen_conf.items()])
#         print(f"Generated examples for ({gen_conf_str}):")
#         for _ in range(3):
#             seq = gen.generate_sequence(**gen_conf)
#             print(''.join(seq))
#
#     @staticmethod
#     def train(model: torch.nn.Module, opt, loss_fn, train_lines: List[str], dev_lines: List[str],
#               device: torch.device, gen_conf: dict, batch_size: int = 64, draw_every: int = 50,
#               score_dev_every: int = 250, n_epochs: int = 5000) -> Tuple[
#         List[Tuple[int, float]], List[Tuple[int, float]], torch.nn.Module]:
#         """
#         Train the model using the training lines and evaluate on development lines.
#         """
#         train_history, dev_history = [], []
#         model = model.to(device)
#         gen = Generator(model, token_level='char')
#
#         for i in trange(n_epochs):
#             batch_ids = np.random.choice(len(train_lines), batch_size, replace=False)
#             loss_i = TrainProcedure._compute_loss(model, loss_fn, train_lines, batch_ids, device)
#
#             loss_i.backward()
#             opt.step()
#             opt.zero_grad()
#
#             train_history.append((i, loss_i.item()))
#
#             if (i + 1) % draw_every == 0:
#                 TrainProcedure.produce_meta(train_history, dev_history, gen, gen_conf)
#
#             if (i + 1) % score_dev_every == 0:
#                 print("Scoring dev...")
#                 dev_loss = TrainProcedure.score_lines(model, dev_lines, loss_fn, batch_size, device)
#                 dev_history.append((i, dev_loss))
#                 print(f'#{i} Dev loss: {dev_loss:.3f}')
#
#         return train_history, dev_history, model
#
#
# def main():
#     # Create tiny dummy data for demonstration
#     from LanguageModeling.task06_rnn.task import RNNLanguageModel
#     from LanguageModeling.task07_loss.task import CrossEntropyLoss
#
#     sample_lines = ["Hello world", "This is a test"]
#     dev_lines = ["Hello test"]
#
#     model = RNNLanguageModel(tt.TOKENS, emb_size=8, hid_size=16)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     loss_fn = CrossEntropyLoss()
#     device = torch.device("cpu")
#
#     gen_conf = {"prefix": "H", "max_len": 10, "mode": "sample", "temperature": 0.8}
#     train_history, dev_history, trained_model = TrainProcedure.train(
#         model, optimizer, loss_fn, sample_lines, dev_lines, device, gen_conf,
#         batch_size=2, draw_every=2, score_dev_every=2, n_epochs=4
#     )
#
#     print("Completed tiny training demonstration")
#
#
# if __name__ == '__main__':
#     main()


# """
# Module for computing loss functions for language models.
# """
# from custom_helpers import add_root_to_pythonpath
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#
# class CrossEntropyLoss(nn.Module):
#     """
#     Module for computing cross-entropy loss function.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Compute cross-entropy loss over non-eos tokens.
#
#         :param logits: Float32 tensor of shape [batch, n_tokens, seq_len]
#         :param input_idx: Int32 tensor of shape [batch, seq_len] with answers
#         :returns: Scalar float32 loss
#         """
#         device = logits.device
#         batch_size, n_tokens, seq_len = logits.shape
#
#         # Get mask for valid tokens (until first EOS)
#         masks = tt.compute_mask(input_idx).to(device)  # [batch, seq_len]
#
#         # Reshape logits from [batch, n_tokens, seq_len] to [batch * seq_len, n_tokens]
#         logits_flat = logits.permute(0, 2, 1).reshape(-1, n_tokens)
#
#         # Reshape targets and mask to 1D
#         targets_flat = input_idx.reshape(-1)
#         masks_flat = masks.reshape(-1)
#
#         # Exclude the first token (BOS) from loss calculation
#         # Create a mask that is False for the first position of each sequence
#         first_token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
#         first_token_mask = (first_token_positions == 0).reshape(-1)
#
#         # Combine masks: exclude first token AND only include tokens before first EOS
#         final_mask = masks_flat & (~first_token_mask)
#
#         # Select valid indices
#         valid_indices = final_mask.bool()
#
#         if valid_indices.sum() > 0:
#             valid_logits = logits_flat[valid_indices]
#             valid_targets = targets_flat[valid_indices]
#             loss = F.cross_entropy(valid_logits, valid_targets)
#         else:
#             loss = torch.tensor(0.0, device=device)
#
#         return loss
#
#
# def main():
#     import torch
#     from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#     loss_fn = CrossEntropyLoss()
#     batch_size, seq_len, vocab_size = 2, 5, 10
#
#     # Create dummy data
#     logits = torch.randn(batch_size, vocab_size, seq_len)
#     input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))
#
#     loss = loss_fn(logits, input_idx)
#     print(f"Computed loss on dummy data: {loss.item():.4f}")
#
#
# if __name__ == '__main__':
#     main()

# """
# Module for computing loss functions for language models.
# """
# from custom_helpers import add_root_to_pythonpath
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#
# class CrossEntropyLoss(nn.Module):
#     """
#     Module for computing cross-entropy loss function.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Note: nn.CrossEntropyLoss input:
#         logits -- (N, C, d1, d2, ..., dK) where N is batch size and C is number of classes
#         target -- (N, d1, d2, ..., dK) where each value is 0 <= targets[i] <= C-1
#         Compute the cross-entropy loss over non-eos tokens.
#
#         Your task: implement loss function as per formula above
#         your loss should only be computed on actual tokens, excluding padding
#         predicting actual tokens and first EOS do count. Subsequent EOS-es don't
#         you may or may not want to use the compute_mask function from above.
#
#         :param logits: Float32 tensor of shape [batch, n_tokens, seq_len].
#         :param input_idx: Int32 tensor of shape [batch, seq_len] with answers.
#         :returns: Scalar float32 loss function (mean crossentropy over non-eos tokens).
#
#         Please, use tt.compute_mask function to get masks for input_idx.
#         :note: Avoid using loops in your code (it will significantly slow down training)
#
#         Steps:
#         1. Compute the mask for input_idx.
#             Note: Don't forget to exclude BOS token from the mask. (i.e. mask[:, 1:])
#         2. Flatten the logits and input_idx tensors.
#             Note: Exclude the last token from each sequence (possible EOS).
#         3. Get target values from input_idx tensor and flatten them. (Again, exclude BOS token)
#         4. Extract valid logits and references using the computed mask.
#             Note: "valid" are ones which we want to see in the loss (that is all tokens after BOS (exclusive) and before EOS (inclusive)).
#         5. Compute the cross-entropy loss using F.cross_entropy function.
#         6. Return the computed loss.
#         """
#         device = logits.device
#         batch_size, n_tokens, seq_len = logits.shape
#
#         # Step 1: Compute the mask for input_idx
#         masks = tt.compute_mask(input_idx).to(device)  # [batch, seq_len]
#
#         # Step 2: Rearrange logits to [batch, seq_len, n_tokens] for easier processing
#         logits_permuted = rearrange(logits, 'b c s -> b s c')  # [batch, seq_len, n_tokens]
#
#         # We want to predict each token based on previous tokens
#         # So input target is input_idx, and we predict at each position
#         # Shift: logits at position t predict token at position t
#         # Therefore, we don't need to exclude the last token
#
#         # Step 3: Flatten everything for masking
#         # Flatten logits: [batch * seq_len, n_tokens]
#         logits_flat = rearrange(logits_permuted, 'b s c -> (b s) c')
#         # Flatten targets: [batch * seq_len]
#         targets_flat = rearrange(input_idx, 'b s -> (b s)')
#         # Flatten masks: [batch * seq_len]
#         masks_flat = rearrange(masks, 'b s -> (b s)')
#
#         # Step 4: Apply mask to exclude BOS and padding
#         # We don't predict the first token (BOS) since there's no previous context
#         # So we create a mask that is False for the first token of each sequence
#         # Create position mask: first token of each sequence should be False
#         position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
#         first_token_mask = (
#                     position_indices == 0)  # False for first token, True for others? Actually we want to exclude first token
#
#         # Combined mask: masks (EOS/padding) AND not first token
#         combined_mask = masks & (~first_token_mask)
#         combined_mask_flat = rearrange(combined_mask, 'b s -> (b s)')
#
#         # Get valid indices where we should compute loss
#         valid_indices = combined_mask_flat.bool()
#
#         # Step 5: Extract valid logits and targets
#         if valid_indices.sum() > 0:
#             valid_logits = logits_flat[valid_indices]
#             valid_targets = targets_flat[valid_indices]
#
#             # Step 6: Compute cross-entropy loss
#             loss = F.cross_entropy(valid_logits, valid_targets)
#         else:
#             # If no valid tokens, return zero loss
#             loss = torch.tensor(0.0, device=device)
#
#         return loss
#
#
# def main():
#     import torch
#     from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#     loss_fn = CrossEntropyLoss()
#     batch_size, seq_len, vocab_size = 2, 5, 10
#
#     # Create dummy data
#     logits = torch.randn(batch_size, vocab_size, seq_len)  # [batch, vocab, seq_len]
#     input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch, seq_len]
#
#     loss = loss_fn(logits, input_idx)
#     print(f"Computed loss on dummy data: {loss.item():.4f}")
#
#
# if __name__ == '__main__':
#     main()



# """
# Module for computing loss functions for language models.
# """
# from custom_helpers import add_root_to_pythonpath
#
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#
# class CrossEntropyLoss(nn.Module):
#     """
#     Module for computing cross-entropy loss function.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Note: nn.CrossEntropyLoss input:
#         logits -- (N, C, d1, d2, ..., dK) where N is batch size and C is number of classes
#         target -- (N, d1, d2, ..., dK) where each value is 0 <= targets[i] <= C-1
#         Compute the cross-entropy loss over non-eos tokens.
#
#         Your task: implement loss function as per formula above
#         your loss should only be computed on actual tokens, excluding padding
#         predicting actual tokens and first EOS do count. Subsequent EOS-es don't
#         you may or may not want to use the compute_mask function from above.
#
#         :param logits: Float32 tensor of shape [batch, n_tokens, seq_len].
#         :param input_idx: Int32 tensor of shape [batch, seq_len] with answers.
#         :returns: Scalar float32 loss function (mean crossentropy over non-eos tokens).
#
#         Please, use tt.compute_mask function to get masks for input_idx.
#         :note: Avoid using loops in your code (it will significantly slow down training)
#
#         Steps:
#         1. Compute the mask for input_idx.
#             Note: Don't forget to exclude BOS token from the mask. (i.e. mask[:, 1:])
#         2. Flatten the logits and input_idx tensors.
#             Note: Exclude the last token from each sequence (possible EOS).
#         3. Get target values from input_idx tensor and flatten them. (Again, exclude BOS token)
#         4. Extract valid logits and references using the computed mask.
#             Note: "valid" are ones which we want to see in the loss (that is all tokens after BOS (exclusive) and before EOS (inclusive)).
#         5. Compute the cross-entropy loss using F.cross_entropy function.
#         6. Return the computed loss.
#         """
#         device = logits.device
#         batch_size, n_tokens, seq_len = logits.shape
#
#         # Step 1: Compute the mask for input_idx
#         masks = tt.compute_mask(input_idx).to(device)  # [batch, seq_len]
#         # Exclude BOS token (first position) from the mask
#         masks = masks[:, 1:]  # [batch, seq_len - 1]
#
#         # Step 2: Flatten the logits and exclude the last token
#         # Rearrange logits from [batch, n_tokens, seq_len] to [batch, seq_len, n_tokens]
#         logits_permuted = rearrange(logits, 'b c s -> b s c')  # [batch, seq_len, n_tokens]
#         # Exclude the last token from each sequence (potential EOS for padding)
#         logits_permuted = logits_permuted[:, :-1, :]  # [batch, seq_len - 1, n_tokens]
#         # Flatten to [batch * (seq_len - 1), n_tokens]
#         logits_flat = rearrange(logits_permuted, 'b s c -> (b s) c')
#
#         # Step 3: Get target values and exclude BOS and last token
#         # Exclude the first token (BOS) and last token
#         targets = input_idx[:, 1:-1]  # [batch, seq_len - 2]
#         # Flatten targets
#         targets_flat = rearrange(targets, 'b s -> (b s)')
#
#         # Step 4: Flatten the mask and apply to valid positions
#         # Mask for valid positions (non-padding)
#         masks_flat = rearrange(masks, 'b s -> (b s)')  # [batch * (seq_len - 1)]
#
#         # Get valid indices (where mask is True)
#         valid_indices = masks_flat.bool()
#
#         # Extract valid logits and targets
#         if valid_indices.sum() > 0:
#             valid_logits = logits_flat[valid_indices]
#             valid_targets = targets_flat[valid_indices]
#
#             # Step 5: Compute cross-entropy loss
#             loss = F.cross_entropy(valid_logits, valid_targets)
#         else:
#             # If no valid tokens, return zero loss
#             loss = torch.tensor(0.0, device=device)
#
#         # Step 6: Return the computed loss
#         return loss
#
#
# def main():
#     import torch
#     from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#     loss_fn = CrossEntropyLoss()
#     batch_size, seq_len, vocab_size = 2, 5, 10
#
#     # Create dummy data
#     logits = torch.randn(batch_size, vocab_size, seq_len)  # [batch, vocab, seq_len]
#     input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch, seq_len]
#
#     loss = loss_fn(logits, input_idx)
#     print(f"Computed loss on dummy data: {loss.item():.4f}")
#
#
# if __name__ == '__main__':
#     main()




# """
# Módulo para calcular funções de perda para modelos de linguagem.
# Module for computing loss functions for language models.
# """
# from custom_helpers import add_root_to_pythonpath
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#
# class CrossEntropyLoss(nn.Module):
#     """
#     Module for computing cross-entropy loss function.
#     """
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, logits: torch.Tensor, input_idx: torch.Tensor) -> torch.Tensor:
#         """
#         Note: nn.CrossEntropyLoss input:
#         logits -- (N, C, d1, d2, ..., dK) where N is batch size and C is number of classes
#         target -- (N, d1, d2, ..., dK) where each value is 0 <= targets[i] <= C-1
#         Compute the cross-entropy loss over non-eos tokens.
#
#         Your task: implement loss function as per formula above
#         your loss should only be computed on actual tokens, excluding padding
#         predicting actual tokens and first EOS do count. Subsequent EOS-es don't
#         you may or may not want to use the compute_mask function from above.
#
#         :param logits: Float32 tensor of shape [batch, n_tokens, seq_len].
#         :param input_idx: Int32 tensor of shape [batch, seq_len] with answers.
#         :returns: Scalar float32 loss function (mean crossentropy over non-eos tokens).
#
#         Please, use tt.compute_mask function to get masks for input_idx.
#         :note: Avoid using loops in your code (it will significantly slow down training)
#
#         Steps:
#         1. Compute the mask for input_idx.
#             Note: Don't forget to exclude BOS token from the mask. (i.e. mask[:, 1:])
#         2. Flatten the logits and input_idx tensors.
#             Note: Exclude the last token from each sequence (possible EOS).
#         3. Get target values from input_idx tensor and flatten them. (Again, exclude BOS token)
#         4. Extract valid logits and references using the computed mask.
#             Note: "valid" are ones which we want to see in the loss (that is all tokens after BOS (exclusive) and before EOS (inclusive)).
#         5. Compute the cross-entropy loss using F.cross_entropy function.
#         6. Return the computed loss.
#         """
#         device = logits.device
#         batch_size, n_tokens, seq_len = logits.shape
#
#         # Etapa 1: Calcule a máscara para input_idx
#         masks = tt.compute_mask(input_idx).to(device) # [lote, comprimento_seq]
#         # Excluir token BOS (primeira posição) da máscara
#         masks = masks[:, 1:] # [lote, seq_len - 1]
#
#         # Etapa 2: Achatar os logits e excluir o último token
#         # Reorganizar os logits de [batch, n_tokens, seq_len] para [batch, seq_len, n_tokens]
#         logits_permuted = rearrange(logits, 'b c s -> b s c')  # [batch, seq_len, n_tokens]
#         # Exclude the last token from each sequence (potential EOS for padding)
#         logits_permuted = logits_permuted[:, :-1, :]  # [batch, seq_len - 1, n_tokens]
#         # Flatten to [batch * (seq_len - 1), n_tokens]
#         logits_flat = rearrange(logits_permuted, 'b s c -> (b s) c')
#
#         # Step 3: Get target values and exclude BOS and last token
#         # Exclude the first token (BOS) and last token
#         targets = input_idx[:, 1:-1]  # [batch, seq_len - 2]
#         # Flatten targets
#         targets_flat = rearrange(targets, 'b s -> (b s)')
#
#         # Step 4: Flatten the mask and apply to valid positions
#         # Mask for valid positions (non-padding)
#         masks_flat = rearrange(masks, 'b s -> (b s)')  # [batch * (seq_len - 1)]
#
#         # Get valid indices (where mask is True)
#         valid_indices = masks_flat.bool()
#
#         # Extract valid logits and targets
#         if valid_indices.sum() > 0:
#             valid_logits = logits_flat[valid_indices]
#             valid_targets = targets_flat[valid_indices]
#
#             # Step 5: Compute cross-entropy loss
#             loss = F.cross_entropy(valid_logits, valid_targets)
#         else:
#             # If no valid tokens, return zero loss
#             loss = torch.tensor(0.0, device=device)
#
#         # Step 6: Return the computed loss
#         return loss
# def main():
#     import torch
#     from LanguageModeling.task05_text_tools.task import TextTools as tt
#
#     loss_fn = CrossEntropyLoss()
#     batch_size, seq_len, vocab_size = 2, 5, 10
#
#     # Create dummy data
#     logits = torch.randn(batch_size, vocab_size, seq_len)  # [batch, vocab, seq_len]
#     input_idx = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch, seq_len]
#
#     loss = loss_fn(logits, input_idx)
#     print(f"Computed loss on dummy data: {loss.item():.4f}")
#
#
# if __name__ == '__main__':
#     main()
