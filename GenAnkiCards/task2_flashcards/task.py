import sys
import os
import openai
import json
import re
from typing import List, Optional

# Add project root to Python path  # Adicione a raiz do projeto ao caminho do Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from GenAnkiCards.task2_flashcards.config import settings
from GenAnkiCards.task2_flashcards.flashcard import Flashcard

# Initialize Openrouter client only if an API key is available
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY
) if settings.OPENROUTER_API_KEY else None


def extract_json_from_response(content: str) -> dict:
    """Extract JSON from API response, handling various formats."""
    "Extrair JSON da resposta da API, lidando com vários formatos."
    if not content or content.strip() == '':
        raise ValueError("Empty response from API")

    # Try to parse as direct JSON first # Tente analisar primeiro como JSON direto
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from Markdown code blocks (both objects and arrays)
    # Tente extrair JSON de blocos de código Markdown (tanto objetos quanto arrays)
    json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', content, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to extract JSON embedded in plain text (both objects and arrays)
    # Tente extrair JSON incorporado em texto simples (tanto objetos quanto arrays)
    json_match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {content}")


def generate_flashcards(words: List[str], language: str, level: Optional[str] = None, batch_size: int = 100) -> List[Flashcard]:
    """
    Generate flashcards for a list of words in the specified language and optional CEFR level.
    Each flashcard includes: word, language, level, translation, definition,
    example_sentence, example_translation, mnemonic. Returns a list of Flashcard models.

    Args:
        words: List of English words to create flashcards for
        language: Target language for the flashcard
        level: Optional CEFR level
        batch_size: Number of words to process in each batch (default: 100)

    Gera flashcards para uma lista de palavras no idioma especificado e no nível CEFR opcional.
    Cada flashcard inclui: palavra, idioma, nível, tradução, definição,
    frase_de_exemplo, tradução_de_exemplo, mnemônico. Retorna uma lista de modelos de Flashcard.

    Argumentos:
            palavras: Lista de palavras em inglês para as quais criar flashcards
            idioma: Idioma de destino para o flashcard
            nível: Nível CEFR opcional
            tamanho_do_lote: Número de palavras a serem processadas em cada lote (padrão: 100)
    """
    flashcards: List[Flashcard] = []

    if client is None:
        print("⚠️ Chave de API não configurada. Gerando apenas flashcards de fallback...")
        # Crie flashcards de fallback para todas as palavras
        for word in words:
            fallback_data = {
            "word": word,
            "language": language,
            "level": level,
            "translation": f"[Translation for {word} in {language}]",
            "definition": f"[Definition for {word}]",
            "example_sentence": f"[Example sentence with translated {word}]",
            "example_translation": f"[Example translation for {word}]",
            "mnemonic": f"[Mnemonic for {word} in {language}]"
        }
            flashcard = Flashcard(**fallback_data)
            flashcards.append(flashcard)
        return flashcards

    # Process words in batches # Processar palavras em lotes
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        # print(
        #     f"Processing batch {i // batch_size + 1}/{(len(words) + batch_size - 1) // batch_size} ({len(batch)} words)")
        print(
            f"Processando lote {i // batch_size + 1}/{(len(words) + batch_size - 1) // batch_size} ({len(batch)} palavras)")
        try:
            # Verificar se o cliente está configurado
            if client is None:
                raise ValueError("OPENROUTER_API_KEY is not configured")

            words_list = ', '.join([f'"{word}"' for word in batch])

            prompt = f"""
                Create flashcards for the following English words: {words_list}

                Respond with ONLY a valid JSON array where each object contains these exact keys:
                - "word": the original English word
                - "language": "{language}"
                - "translation": the word translated to {language}
                - "definition": definition in English
                - "example_sentence": example sentence using the translated word in {language}
                - "example_translation": English translation of the example sentence
                - "mnemonic": memory aid to remember the {language} word

                Do not include any explanatory text, just the JSON array with one object per word.
                Example format: [{{ "word": "example", "language": "{language}", "translation": "...", ... }}, ...]
            """

            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Parse the JSON response with error handling # Analisar a resposta JSON com tratamento de erros
            data = extract_json_from_response(content)

            # Ensure we have a list # Certifique-se de que temos uma lista
            if not isinstance(data, list):
                raise ValueError("Expected JSON array response")

            batch_flashcards = []

            # Process each flashcard in the response # Processar cada flashcard na resposta
            for idx, flashcard_data in enumerate(data):
                try:
                    # Add optional fields # Adicionar campos opcionais
                    flashcard_data.setdefault("level", level)

                    # Validate required fields  # Validar campos obrigatórios
                    required_fields = ["word", "language", "translation", "definition",
                                       "example_sentence", "example_translation", "mnemonic"]
                    for field in required_fields:
                        if field not in flashcard_data:
                            flashcard_data[field] = f"[{field} not provided]"

                    # Ensure canonical fields match input arguments
                    # Garantir que os campos canônicos correspondam aos argumentos de entrada
                    if idx < len(batch):
                        flashcard_data["word"] = batch[idx]
                    flashcard_data["language"] = language
                    if level is not None:
                        flashcard_data["level"] = level

                    # Validate and create a Flashcard instance # Validar e criar uma instância de Flashcard
                    flashcard = Flashcard(**flashcard_data)
                    batch_flashcards.append(flashcard)

                except Exception as e:
                    #print(f"Error processing flashcard data at index {idx}: {e}")
                    print(f"Erro ao processar dados do flashcard no índice {idx}: {e}")
                    # Create fallback flashcard if we have a corresponding word
                    # Criar cartão de memória alternativo se tivermos uma palavra correspondente
                    if idx < len(batch):
                        word = batch[idx]
                        fallback_data = {
                            "word": word,
                            "language": language,
                            "level": level,
                            "translation": f"[Translation for {word} in {language}]",
                            "definition": f"[Definition for {word}]",
                            "example_sentence": f"[Example sentence with translated {word}]",
                            "example_translation": f"[Example translation for {word}]",
                            "mnemonic": f"[Mnemonic for {word} in {language}]"
                        }
                        flashcard = Flashcard(**fallback_data)
                        batch_flashcards.append(flashcard)

            # If we didn't get enough flashcards from the API response, create fallbacks for missing ones
            # Se não obtivermos flashcards suficientes da resposta da API, criemos alternativas para os que estiverem faltando.
            while len(batch_flashcards) < len(batch):
                missing_idx = len(batch_flashcards)
                word = batch[missing_idx]
               # print(f"Creating fallback flashcard for missing word '{word}'")
                print(f"Criando flashcard alternativo para a palavra ausente '{word}'")
                fallback_data = {
                    "word": word,
                    "language": language,
                    "level": level,
                    "translation": f"[Translation for {word} in {language}]",
                    "definition": f"[Definition for {word}]",
                    "example_sentence": f"[Example sentence with translated {word}]",
                    "example_translation": f"[Example translation for {word}]",
                    "mnemonic": f"[Mnemonic for {word} in {language}]"
                }
                flashcard = Flashcard(**fallback_data)
                batch_flashcards.append(flashcard)

        except Exception as e:
            #print(f"Error generating flashcards for batch: {e}")
            # Create fallback flashcards for the entire batch
            print(f"Erro ao gerar flashcards para o lote: {e}")
            # Criar flashcards alternativos para todo o lote
            batch_flashcards = []
            for word in batch:
                fallback_data = {
                    "word": word,
                    "language": language,
                    "level": level,
                    "translation": f"[Translation for {word} in {language}]",
                    "definition": f"[Definition for {word}]",
                    "example_sentence": f"[Example sentence with translated {word}]",
                    "example_translation": f"[Example translation for {word}]",
                    "mnemonic": f"[Mnemonic for {word} in {language}]"
                }
                flashcard = Flashcard(**fallback_data)
                batch_flashcards.append(flashcard)

        # Add batch flashcards to the main list # Adicionar flashcards em lote à lista principal
        flashcards.extend(batch_flashcards)
        #print(f"Completed batch {i // batch_size + 1}. Total flashcards generated: {len(flashcards)}")
        print(f"Lote concluído {i // batch_size + 1}. Total de flashcards gerados: {len(flashcards)}")

    return flashcards


def save_flashcards(flashcards: List[Flashcard], filename: str = "flashcards.json") -> None:
    """Save flashcards to a JSON file."""
    "Salvar flashcards em um arquivo JSON."
    flashcards_data = [flashcard.model_dump() for flashcard in flashcards]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(flashcards_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import os

    # Verificar se o arquivo existe
    json_file = 'A2_english.json'
    if not os.path.exists(json_file):
        # Criar arquivo de exemplo se não existir
        sample_data = {
            "words": ["happy", "sad", "big", "small", "beautiful"]
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"Arquivo de exemplo criado: {json_file}")

    with open(json_file, 'r', encoding='utf-8') as f:
        a2_data = json.load(f)

   # words = a2_data['words'][:5]  # comment to run on the entire vocab
    words = a2_data['words']

    # Verificar se o cliente foi inicializado
    if client is None:
        # print("⚠️ API key not configured. Please set OPENROUTER_API_KEY in your environment.")
        # print("Generating fallback flashcards only...")
        print("?? Chave de API não configurada. Por favor, defina OPENROUTER_API_KEY em seu ambiente.")
        print("Gerando apenas flashcards de fallback...")

    flashcards = generate_flashcards(words, "Greek", "A2")
    save_flashcards(flashcards, "flashcards_example.json")
    # print(f"✅ Generated {len(flashcards)} flashcards saved to flashcards_example.json")
    print(f"? {len(flashcards)} flashcards gerados e salvos em flashcards_example.json")
