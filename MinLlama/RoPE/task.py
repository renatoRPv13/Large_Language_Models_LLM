from typing import Tuple
import torch
import math


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.
    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
    Returns:
        torch.Tensor: Reshaped frequency tensor.
    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.

    Argumentos:
        freqs_cis (torch.Tensor): Tensor de frequência a ser remodelado.
         x (torch.Tensor): Tensor alvo para compatibilidade de broadcasting.
     Retorno:
        torch.Tensor: Tensor de frequência remodelado.
    Exceções:
        AssertionError: Se o tensor de frequência não corresponder ao formato esperado.
        AssertionError: Se o tensor alvo 'x' não tiver o número esperado de dimensões.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    Aplica embeddings rotativos aos tensores de entrada usando o tensor de frequência fornecido.

    Esta função aplica embeddings rotativos aos tensores de consulta e chave fornecidos. A rotação de cada token
    embedding é uma função da posição desse token na sequência, head_dim e theta.
    Os tensores de entrada são remodelados como números complexos para simplificar sua implementação.

    Argumentos:
        query (torch.Tensor): Tensor de consulta ao qual aplicar embeddings rotativos.

    Formato: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Tensor de chave ao qual aplicar embeddings rotativos.

    Formato: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimensão de cada cabeça de atenção.
        max_seq_len (int): Comprimento máximo da sequência suportado pelo modelo.
    Retorna:
        Tupla[torch.Tensor, torch.Tensor]: Tupla contendo o tensor de consulta modificado e o tensor chave com incorporações rotativas.
    """

    batch_size, seqlen, n_heads_q, _ = query.shape
    _, _, n_heads_k, _ = key.shape
    device = query.device

    # reshape xq and xk to match the complex representation
    # Redimensione xq e xk para corresponder à representação complexa
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...
    # Isso separa cada vetor de consulta/chave em seus índices pares e ímpares (assumindo *indexação única*).
    # query_real contém q_1, q_3, q_5, ... e query_imag contém q_2, q_4, q_6, ...

    # TODO: First, compute the trigonometric values in the second and fourth columns in slide 22 (linked above)
    # Compute position indices (0 to seqlen-1)

    # TODO: Primeiro, calcule os valores trigonométricos na segunda e quarta colunas do slide 22 (link acima)
    # Calcule os índices de posição (0 a seqlen-1)
    position_indices = torch.arange(seqlen, device=device).unsqueeze(0)  # (1, seqlen)

    # Compute frequencies: theta_i = 10000^(-2i/d) for i = 0, 1, ..., d/2 - 1
    # where d = head_dim # onde d = dim_cabeça
    half_dim = head_dim // 2
    freqs = theta ** (-torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)

    # Compute angles = position * freqs
    # Shape: (seqlen, half_dim)
    # Calcular ângulos = posição * frequências
    # Formato: (comprimento_da_sequência, metade_da_dimensões)
    angles = position_indices.unsqueeze(-1) * freqs.unsqueeze(0)  # (1, seqlen, half_dim)

    # Compute sine and cosine of the angles
    # Calcule o seno e o cosseno dos ângulos
    cos = torch.cos(angles)  # (1, seqlen, half_dim)
    sin = torch.sin(angles)  # (1, seqlen, half_dim)

    # Reshape for broadcasting to query and key tensors
    # Target shape: (1, seqlen, 1, half_dim) where half_dim = head_dim // 2
    # Redimensionar para transmissão para tensores de consulta e chave
    # Formato alvo: (1, seqlen, 1, half_dim) onde half_dim = head_dim // 2
    cos = cos.reshape(1, seqlen, 1, half_dim)
    sin = sin.reshape(1, seqlen, 1, half_dim)

    # TODO: Then, combine these trigonometric values with the tensors query_real, query_imag, key_real, and key_imag
    # TODO: Em seguida, combine esses valores trigonométricos com os tensores query_real, query_imag, key_real e key_imag
    # Apply rotary embedding to query
    # q_rotated = q * cos + rotate_90(q) * sin
    # where rotate_90(q) = [-q_imag, q_real]
    # Aplicar incorporação rotativa à consulta
    # q_rotated = q * cos + rotate_90(q) * sin
    # onde rotate_90(q) = [-q_imag, q_real

    # For query
    query_out_real = query_real * cos - query_imag * sin
    query_out_imag = query_real * sin + query_imag * cos

    # For key # Para chave
    key_out_real = key_real * cos - key_imag * sin
    key_out_imag = key_real * sin + key_imag * cos

    # Combine real and imaginary parts back to original shape
    # Combine as partes reais e imaginárias de volta à forma original
    query_out = torch.stack([query_out_real, query_out_imag], dim=-1).flatten(-2)
    key_out = torch.stack([key_out_real, key_out_imag], dim=-1).flatten(-2)

    # Return the rotary position embeddings for the query and key tensors
    # Retorna os embeddings de posição rotativa para os tensores de consulta e chave
    return query_out, key_out





# from typing import Tuple
# import torch
#
# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     """
#     Helper function to reshape the frequency tensor to have the same shape as the target tensor 'x'
#     for the purpose of broadcasting the frequency tensor during element-wise operations.
#
#     Args:
#         freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
#         x (torch.Tensor): Target tensor for broadcasting compatibility.
#
#     Returns:
#         torch.Tensor: Reshaped frequency tensor.
#
#     Raises:
#         AssertionError: If the frequency tensor doesn't match the expected shape.
#         AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
#     """
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(shape)
#
# def apply_rotary_emb(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     head_dim: int,
#     max_seq_len: int,
#     theta: float = 10000.0,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Apply rotary embeddings to input tensors using the given frequency tensor.
#
#     This function applies rotary embeddings to the given query and key tensors. The rotation to each token
#     embedding is a function of that token's position in the sequence, head_dim, and theta.
#     The input tensors are reshaped as complex numbers to simplify your implementation.
#
#     Args:
#         query (torch.Tensor): Query tensor to apply rotary embeddings.
#                               Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
#         key (torch.Tensor): Key tensor to apply rotary embeddings.
#                               Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
#         head_dim (int): Dimension of each attention head.
#         max_seq_len (int): Maximum sequence length supported by model.
#     Returns:
#         Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
#     """
#
#     _, seqlen, _, _ = query.shape
#     device = query.device
#     # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
#     # and Section 3 in https://arxiv.org/abs/2104.09864.
#
#     # reshape xq and xk to match the complex representation
#     query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
#     key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
#     # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
#     # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...
#
#     # TODO: First, compute the trigonometric values in the second and fourth columns in slide 22 (linked above)
#
#     # TODO: Then, combine these trigonometric values with the tensors query_real, query_imag, key_real, and key_imag
#
#     # Return the rotary position embeddings for the query and key tensors
#     return query_out, key_out
