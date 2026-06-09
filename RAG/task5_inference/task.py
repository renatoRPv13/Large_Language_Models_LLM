"""Inference utilities contrasting baseline vs RAG-augmented prompts."""
"Utilidades de inferência contrastando prompts de linha de base com prompts aumentados por RAG."
from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2)

import os
import sys
from typing import List, Literal, Optional
from huggingface_hub import InferenceClient as HFInferenceClient

from RAG.task4_vector_storage.task import Searcher

# Adicionar o diretório RAG ao path
rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

# Tentar carregar config
config_path = os.path.join(rag_path, "config.yaml")
if os.path.exists(config_path):
    CONFIG = get_config(config_path)
else:
    CONFIG = {
        "VECTOR_STORE_PATH": "data/cache/vector_store.pkl",
        "DEFINITIONS_PATH": "data/wordnet_definitions.json"
    }
    print("⚠️ Usando valores padrão (config.yaml não encontrado)")


class InferenceClient:
    """Thin wrapper around the HF text-generation endpoint."""
    """Envoltório fino em torno do ponto de extremidade de geração de texto HF."""

    def __init__(self, api_token: str, model_id: str = "mistralai/Mistral-Nemo-Instruct-2407"):
        """Initialize client using the default Hugging Face inference API."""
       # Inicialize o cliente usando a API de inferência padrão do Hugging Face.
        self.model_id = model_id
        self.hf_client = HFInferenceClient(token=api_token)
       # print(f"✅ Initialized inference client for model: {model_id}")
        print(f"✅ Cliente de inferência inicializado para o modelo: {model_id}")

    def generate(self, prompt: str, max_tokens: int = 60) -> str:
        """Generate text from prompt using the chat completions endpoint."""
        """Gere texto a partir do prompt usando o endpoint de conclusões de chat."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Usar o endpoint de chat, que é compatível com modelos "Instruct"
            completion = self.hf_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            # Fornecer mais detalhes sobre o erro
            print(f"❌ Erro durante a geração de texto: {e}")
            # Tentar um fallback para text_generation se o chat falhar
            try:
                print("⚠️ Tentando fallback para 'text-generation'...")
                response = self.hf_client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                )
                return response
            except Exception as fallback_e:
                print(f"❌ Fallback também falhou: {fallback_e}")
                return "Não foi possível gerar uma resposta."


# Default prompt header for dictionary assistant
# Cabeçalho de prompt padrão para o assistente de dicionário
# PROMPT_HEADER = "You are an advanced dictionary assistant. Given a word, provide a clear, one-sentence definition."
PROMPT_HEADER = "Você é um assistente de dicionário avançado. Dada uma palavra, forneça uma definição clara em uma frase."

def build_prompt(query: str, definitions: List[str], mode: Literal["baseline", "rag"]) -> str:
    """Build prompt for generation.""" "Criar prompt para geração."
    if mode == "baseline":
        return f"""{PROMPT_HEADER}

Word: {query}
Definition:"""
    else:  # RAG-enhanced # Aprimorado pelo RAG
        definitions_text = "\n".join([f"- {d}" for d in definitions])
        return f"""{PROMPT_HEADER}

#Here are some similar words and their definitions for reference:
Aqui estão algumas palavras semelhantes e suas definições para referência:
{definitions_text}

# Now provide a definition for the word:
Agora, forneça uma definição para a palavra: {query}
Definition:"""

def compare_generations(
        query: str,
        client: InferenceClient,
        searcher: Optional[Searcher] = None,
        max_tokens: int = 60,
        k: int = 3  # ← Adicione este parâmetro
) -> None:
    """Compare baseline and RAG-augmented generations."""
    """Comparar as gerações de referência e as gerações aumentadas por RAG."""
    retrieved = []
    if searcher:
        print(f"🔍 Recuperando definições para '{query}'...") # Retrieving definitions for
        retrieved = [d for d, _ in searcher.search(query, k=k)]
        print(f"✅ Retrieved {len(retrieved)} definitions")

    p_base = build_prompt(query, [], mode="baseline")
    p_rag = build_prompt(query, retrieved, mode="rag")

    print("\n" + "="*60)
    #print("📝 BASELINE (no retrieval)")
    print("📝 LINHA DE BASE (sem recuperação)")
    print("="*60)
    print("🤖 Resposta:")
    baseline_response = client.generate(p_base, max_tokens=max_tokens)
    print(baseline_response)

    if searcher and retrieved:
        print("\n" + "="*60)
        print("🚀 RAG-AUMENTADO (com recuperação) ") #RAG-AUGMENTED (with retrieval)
        print("="*60)
        print("🤖 Resposta:") # Response
        rag_response = client.generate(p_rag, max_tokens=max_tokens)
        print(rag_response)

    print("\n" + "="*60)
    print("📊 COMPARAÇÃO") # COMPARISON
    print("="*60)
    print(f"Comprimento da linha de base: {len(baseline_response)} chars") # Baseline length


def main():
    """Example usage."""
    # Configurar caminhos
    rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Inicializar searcher
    vector_store_path = os.path.join(rag_path, "data/cache/vector_store.pkl")
    searcher = None
    if os.path.exists(vector_store_path):
        try:
            searcher = Searcher.load(vector_store_path)
            print(f"✅ Searcher carregado de {vector_store_path}")
        except Exception as e:
            print(f"⚠️ Erro ao carregar searcher: {e}")
    else:
        print(f"⚠️ Vector store não encontrado em {vector_store_path}")

    print("\n✅ Sistema de inferência pronto!")
    print("Para usar, configure HUGGINGFACE_API_TOKEN e execute compare_generations")

if __name__ == "__main__":
    main()
