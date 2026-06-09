"""Run vector search on definitions.
Executar busca vetorial nas definições.

comando para salvar
cd /home/selvarpv/PycharmProjects/Mastering\ Large\ Language\ Models/RAG/task4_vector_storage
# Criar o vector store (--save para salvar)
python run.py --save --reuse
cd /home/selvarpv/PycharmProjects/Mastering\ Large\ Language\ Models/RAG/task5_inference
python run.py --query "agony"
"""
import os
import sys
from pathlib import Path
import click

# Adicionar o diretório RAG ao sys.path
rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if rag_path not in sys.path:
    sys.path.insert(0, rag_path)

# Adicionar o diretório raiz do projeto
project_root = os.path.abspath(os.path.join(rag_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importar helpers e tasks
from custom_helpers import get_config
from RAG.task4_vector_storage.task import load_model, load_definitions, Searcher

# Carregar configuração
config_path = os.path.join(rag_path, "config.yaml")
CONFIG = get_config(config_path)

# Garantir que os caminhos estão corretos
MODEL_PATHS = [
    CONFIG.get("MODEL_OUT", ""),
    "../task3_training/models/bi-encoder",
    "./models/bi-encoder",
    "/home/selvarpv/PycharmProjects/Mastering Large Language Models/RAG/task3_training/models/bi-encoder"
]

DEFINITIONS_PATHS = [
    CONFIG.get("DEFINITIONS_PATH", ""),
    "./data/dataset.json",
    "../data/dataset.json"
]


@click.command()
@click.option(
    "--query",
    default="agony",
    help="Query word to search for"
)
@click.option(
    "--top-k",
    default=5,
    type=int,
    help="Number of results to return"
)
@click.option(
    "--save",
    is_flag=True,
    help="Save the vector store to disk"
)
@click.option(
    "--reuse",
    is_flag=True,
    help="Reuse cached vector store if present"
)
@click.option(
    "--model-path",
    default=None,
    help="Override path to the sentence transformer model"
)
@click.option(
    "--definitions-path",
    default=None,
    help="Override path to the definitions JSON file"
)
def main(query: str, top_k: int, save: bool, reuse: bool, model_path: str, definitions_path: str) -> None:
    """Run vector search on definitions.""" """Executar busca vetorial nas definições."""

    # Determinar o caminho do modelo
    if model_path:
        model_path = Path(model_path)
    else:
        for path in MODEL_PATHS:
            if path and Path(rag_path).joinpath(path).exists():
                model_path = Path(rag_path).joinpath(path)
                break
        if not model_path:
            click.echo(f"❌ Modelo não encontrado. Procurei em: {MODEL_PATHS}")
            return

    # Determinar o caminho das definições
    if definitions_path:
        def_path = Path(definitions_path)
    else:
        for path in DEFINITIONS_PATHS:
            if path and Path(rag_path).joinpath(path).exists():
                def_path = Path(rag_path).joinpath(path)
                break
        if not def_path:
            click.echo(f"❌ Arquivo de definições não encontrado. Procurei em: {DEFINITIONS_PATHS}")
            return

    vector_store_path = Path(rag_path) / "data/cache/vector_store.pkl"
    vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to reuse cached vector store if requested
    # Tente reutilizar o armazenamento vetorial em cache, se solicitado
    searcher = None
    if reuse and vector_store_path.exists():
        click.echo(f"📦  Carregando o armazenamento de vetores em cache de {vector_store_path}...") # Loading cached vector store from
        try:
            searcher = Searcher.load(str(vector_store_path))
        except Exception as e:
            click.echo(f"⚠️ Falha ao carregar o armazenamento vetorial em cache.: {e}") # Failed to load cached vector store
            reuse = False

    # Create new vector store if not reusing
    # Crie um novo armazenamento de vetores se não for reutilizar
    if not reuse or searcher is None:
        click.echo(f"📦 Carregando modelo de {model_path}...") # Loading model from
        model_instance = load_model(str(model_path))

        click.echo(f"📦 Carregando definições de {def_path}...") # Loading definitions from
        definitions_data = load_definitions(str(def_path))

        click.echo(f"📦 Criando uma loja de vetores com {len(definitions_data)} items...") # Creating vector store with
        searcher = Searcher(definitions_data, model_instance)

        # Save vector store if requested
        # Salvar no armazenamento de vetores, se solicitado
        if save:
            click.echo(f"💾 Salvando o armazenamento de vetores em {vector_store_path}...") # Saving vector store to
            searcher.save(str(vector_store_path))

    # Run search # Executar pesquisa
    results = searcher.search(query, k=top_k)

    # Display results Exibir resultados
    click.echo(f"\n🔍 Top-{top_k} resultados para \"{query}\":") # results for
    for rank, (text, score) in enumerate(results, 1):
        display_text = text[:150] + "..." if len(text) > 150 else text
        click.echo(f"{rank:>2}. {score:.3f}  {display_text}")

if __name__ == "__main__":
    main()


# """Run vector search on definitions."""
# import os
# import sys
# from pathlib import Path
# import click
#
# # Adicionar o diretório RAG ao sys.path para encontrar custom_helpers
# rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if rag_path not in sys.path:
#     sys.path.insert(0, rag_path)
#
# # Adicionar o diretório raiz do projeto para importações absolutas
# project_root = os.path.abspath(os.path.join(rag_path, '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
#
# # Importar helpers e tasks
# from custom_helpers import get_config
# from RAG.task4_vector_storage.task import load_model, load_definitions, Searcher
#
# # Carregar configuração
# config_path = os.path.join(rag_path, "config.yaml")
# CONFIG = get_config(config_path)
#
# # Importar task3_run apenas para definir a variável de ambiente
# try:
#     import RAG.task3_training.run as task3_run
# except ImportError as e:
#     click.echo(f"Aviso: Não foi possível importar task3_run: {e}")
#
#
# @click.command()
# @click.option(
#     "--query",
#     default="agony",
#     help="Query word to search for"
# )
# @click.option(
#     "--top-k",
#     default=5,
#     type=int,
#     help="Number of results to return"
# )
# @click.option(
#     "--save",
#     is_flag=True,
#     help="Save the vector store to disk"
# )
# @click.option(
#     "--reuse",
#     is_flag=True,
#     help="Reuse cached vector store if present"
# )
# def main(query: str, top_k: int, save: bool, reuse: bool) -> None:
#     """Run vector search on definitions."""
#     "Executar busca vetorial nas definições."
#
#     # Construir caminhos absolutos a partir da pasta RAG
#     model_path = Path(rag_path) / CONFIG["MODEL_OUT"].replace("${DATA_DIR}", "data")
#     definitions_path = Path(rag_path) / CONFIG["DEFINITIONS_PATH"].replace("${DATA_DIR}", "data")
#     vector_store_path = Path(rag_path) / CONFIG["VECTOR_STORE_PATH"].replace("${DATA_DIR}", "data")
#
#     # Try to reuse cached vector store if requested
#     if reuse and vector_store_path.exists():
#         click.echo(f"🔄 Loading cached vector store from {vector_store_path}...")
#         try:
#             searcher = Searcher.load(str(vector_store_path))
#         except Exception as e:
#             click.echo(f"⚠️ Failed to load cached vector store: {e}")
#             reuse = False
#
#     # Create new vector store if not reusing
#     if not reuse:
#         click.echo(f"🔄 Loading model from {model_path}...")
#         model_instance = load_model(str(model_path))
#
#         click.echo(f"📚 Loading definitions from {definitions_path}...")
#         definitions_data = load_definitions(str(definitions_path))
#
#         click.echo(f"🔍 Creating vector store...")
#         searcher = Searcher(definitions_data, model_instance)
#
#         # Save vector store if requested
#         if save:
#             click.echo(f"💾 Saving vector store to {vector_store_path}...")
#             searcher.save(str(vector_store_path))
#
#     # Run search
#     results = searcher.search(query, k=top_k)
#
#     # Display results
#     click.echo(f"\n🔎 Top‑{top_k} results for \"{query}\":")
#     for rank, (text, score) in enumerate(results, 1):
#         click.echo(f"{rank:>2}. {score:0.3f}  {text}")
#
#
# if __name__ == "__main__":
#     main()
