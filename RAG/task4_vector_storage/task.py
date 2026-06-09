"""Simple vector search with dot-product (cosine) similarity."""
"""Busca vetorial simples com similaridade de produto escalar (cosseno)."""
import json
import pickle
import os
from pathlib import Path
from typing import List, Tuple, Union, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from custom_helpers import get_config

CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))


class SimpleVectorStore:
    """Simple vector store for similarity search with cosine similarity."""
    """Armazenamento vetorial simples para busca de similaridade com similaridade de cosseno."""

    def __init__(self, embeddings: np.ndarray, model: SentenceTransformer):
        """Initialize vector store.
        Args:
            embeddings: Array of embedding vectors
            model: SentenceTransformer model for encoding queries
        Raises:
            ValueError: If embeddings is not a 2D array
        Inicializa o armazenamento de vetores.
        Argumentos:
            embeddings: Array de vetores de incorporação
            model: Modelo SentenceTransformer para codificação de consultas
        Exceções:
            ValueError: Se embeddings não for um array bidimensional
        """
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        self.embeddings = embeddings.astype(np.float32)
        self.model = model
        self.dim = self.embeddings.shape[1]

        self._normalize_embeddings()

    def _normalize_embeddings(self) -> None:
        """Normalize the embeddings in place to unit length."""
        """Normalize the embeddings in place to unit length."""
        # Normalize each row to unit length (L2 norm)
        # Normalizar cada linha para comprimento unitário (norma L2)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero # Evite a divisão por zero
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms

    def _encode_and_normalize(self, text: str) -> np.ndarray:
        """Encode text and normalize the resulting vector.
        Args:
            text: Input text to encode
        Returns:
            Normalized embedding vector
        Codifica o texto e normaliza o vetor resultante.
        Argumentos:
            texto: Texto de entrada a ser codificado
        Retorno:
            Vetor de incorporação normalizado
        """
        # Encode the text # Codifique o texto
        embedding = self.model.encode(text)

        # Normalize the embedding # Normalizar o embedding
        norm = np.linalg.norm(embedding)

        # Handle edge case: embedding equal to zero # Lidar com caso extremo: incorporação igual a zero
        if norm == 0:
            return embedding

        return embedding / norm

    def search(self, query: str, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        Args:
            query: Query text
            k: Number of results to return
        Returns:
            Tuple of (indices, scores) for the top-k results
        Busca por vetores semelhantes.
        Argumentos:
            query: Texto da consulta
            k: Número de resultados a serem retornados
        Retorno:
            Tupla de (índices, pontuações) para os k melhores resultados
        """
        k = min(k, self.embeddings.shape[0])
        q = self._encode_and_normalize(query)

        # Compute cosine similarity (dot product because vectors are unit length)
        # Calcular a similaridade de cosseno (produto escalar, pois os vetores têm comprimento unitário)
        scores = np.dot(self.embeddings, q)

        # Get indices of top-k scores in descending order
        # Obtenha os índices das k melhores pontuações em ordem decrescente
        idx = np.argsort(scores)[::-1][:k]

        return idx, scores[idx]
class Searcher:
    """High-level helper that manages a SimpleVectorStore."""
    "Função auxiliar de alto nível que gerencia um SimpleVectorStore."

    def __init__(self, data: List[str], model: SentenceTransformer):
        """Initialize searcher.
        Args:
            data: List of text items for search
            model: SentenceTransformer model for encoding
        Inicializar o mecanismo de busca.
        Argumentos:
            dados: Lista de itens de texto para busca
            modelo: Modelo SentenceTransformer para codificaçãog
        """
        #print(f"📦 Creating embeddings for {len(data)} items...")
        print(f"📦 Criando embeddings para {len(data)} itens...")
        # Create embeddings for all data items
        # Criar incorporações para todos os itens de dados
        emb = model.encode(data)

        self.data = data
        self.store = SimpleVectorStore(emb, model)
        #print(f"✅ Vector store created with {len(data)} items and dimension {emb.shape[1]}")
        print(f"✅ Armazenagem vetorial criada com {len(data)} itens e dimensão {emb.shape[1]}")

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar items.
        Args:
            query: Query text
            k: Number of results to return
        Returns:
            List of (text, score) tuples for the top-k results
        Pesquisar itens semelhantes.
        Argumentos:
            query: Texto da consulta
            k: Número de resultados a serem retornados
        Retorno:
            Lista de tuplas (texto, pontuação) com os k melhores resultados
        """
        idx, scores = self.store.search(query, k)

        # Return texts and scores as list of tuples
        # Retorna textos e pontuações como listas de tuplas
        return [(self.data[i], float(scores[i])) for i in range(len(idx))]

    def save(self, path: Union[str, Path, None] = None) -> None:
        """Save searcher to disk.
        Args:
            path: Path to save to, defaults to CONFIG["VECTOR_STORE_PATH"]
        Salvar pesquisa no disco.
        Argumentos:
            caminho: Caminho para salvar, o padrão é CONFIG["VECTOR_STORE_PATH"]Salvar pesquisa no disco.

            caminho: Caminho para salvar, o padrão é CONFIG["VECTOR_STORE_PATH"]
        """
        path = Path(path or CONFIG["VECTOR_STORE_PATH"])
        # Ensure parent directory exists # Garantir que o diretório pai exista
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        #print(f"✅ Searcher saved to {path}")
        print(f"✅ Pesquisador salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> Any:
        """Load searcher from disk.
        Args:
            path: Path to load from, defaults to CONFIG["VECTOR_STORE_PATH"]
        Returns:
            Loaded Searcher instance
        Carregar o mecanismo de busca do disco.
        Argumentos:
            caminho: Caminho de onde carregar, o padrão é CONFIG["VECTOR_STORE_PATH"]
        Retorno:
            Instância do mecanismo de busca carregada
        """
        path = Path(path or CONFIG["VECTOR_STORE_PATH"])
        with path.open("rb") as f:
            searcher = pickle.load(f)
        #print(f"✅ Loaded searcher from {path} with {len(searcher.data)} items")
        print(f"✅ Pesquisador carregado de {path} com {len(searcher.data)} itens")
        return searcher

def load_definitions(path: Union[str, Path]) -> List[str]:
    """Load definitions from a JSON file.
    Args:
        path: Path to JSON file
    Returns:
        List of definition strings
    Carrega definições de um arquivo JSON.
    Argumentos:
        caminho: Caminho para o arquivo JSON
    Retorno:
        Lista de strings de definição
    """
    with Path(path).open() as f:
        definitions = json.load(f)
    #print(f"✅ Loaded {len(definitions)} definitions from {path}")
    print(f"✅ Carregadas {len(definitions)} definições de {path}")
    return definitions


def load_model(path: Union[str, Path]) -> SentenceTransformer:
    """Load a SentenceTransformer model.
    Args:
        path: Path to model directory
    Returns:
        Loaded SentenceTransformer model
    Carregar um modelo SentenceTransformer.
    Argumentos:
        caminho: Caminho para o diretório do modelo
    Retorno:
        Modelo SentenceTransformer carregado
    """
    model = SentenceTransformer(str(path), trust_remote_code=True)
    print(f"✅ Modelo carregado de {path}")
    return model

def main():
    """Example usage of the vector search."""
    "Exemplo de uso da busca vetorial."
    # Configurações de exemplo
    data = [ # word
        "palavra: exterior\ndefinição: parte de fora de um objeto",
        "palavra: complexidade\ndefinição: qualidade do que é complexo",
        "palavra: feliz\ndefinição: estado de alegria e satisfação",
    ]
    # Carregar modelo
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Criar searcher
    searcher = Searcher(data, model)

    # Buscar similaridade
    query = "parte externa"
    results = searcher.search(query, k=2)

    print(f"\n🔍 Consulta: '{query}'")
    print("📋 Resultados:")
    for i, (text, score) in enumerate(results):
        print(f"  {i + 1}. Pontuação: {score:.4f} - {text[:80]}...") # Score


if __name__ == "__main__":
    main()
