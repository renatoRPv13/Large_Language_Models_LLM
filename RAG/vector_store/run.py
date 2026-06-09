import os
import sys
import click
import numpy as np
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from RAG.task3_training.task import CONFIG

def create_vector_store(
    model_path: str, 
    data_path: str, 
    output_path: str
):
    """
    Creates a vector store from a dataset using a fine-tuned bi-encoder model.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Loading model from {model_path}...")
    model = SentenceTransformer(model_path)

    print(f"📚 Loading data from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Assuming the data is a list of strings or dicts with a 'text' key
    if isinstance(documents[0], dict):
        texts = [doc.get('text', '') for doc in documents]
    else:
        texts = documents

    print(f"🧠 Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print(f"💾 Saving vector store to {output_path}...")
    np.save(output_path, embeddings)
    
    # Save the documents as well for later lookup
    doc_output_path = Path(output_path).with_suffix('.json')
    with open(doc_output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"✅ Vector store created successfully at {output_path} and {doc_output_path}")


@click.command()
@click.option(
    "--model_path", 
    default=CONFIG["MODEL_OUT"],
    help="Path to the fine-tuned bi-encoder model."
)
@click.option(
    "--data_path", 
    default=CONFIG["DATASET_PATH"],
    help="Path to the JSON data file to be indexed."
)
@click.option(
    "--output_path", 
    default="./vector_store/vector_store.npy",
    help="Path to save the output vector store (as a .npy file)."
)
def main(model_path: str, data_path: str, output_path: str):
    """
    Main function to create the vector store.
    """
    create_vector_store(model_path, data_path, output_path)


if __name__ == "__main__":
    main()
