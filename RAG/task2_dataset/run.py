"""Generate triplet datasets and save to disk."""
"Gere conjuntos de dados de triplas e salve em disco."

from pathlib import Path
from typing import Dict, Any

import click
from custom_helpers import add_root_to_pythonpath, get_config

add_root_to_pythonpath(n_up=2)

from RAG.task2_dataset.task import TripletDatasetBuilder


DEFAULT_CONFIG: Dict[str, Any] = {
    "DATASET_PATH": str(Path(__file__).resolve().parents[1] / "data" / "dataset.json"),
    "RAG_DATASET_DIR": str(Path(__file__).resolve().parents[1] / "data" / "rag_dataset"),
    "TRAIN_FRAC": 0.8,
}


def load_config() -> Dict[str, Any]:
    """Load config from likely locations, falling back to safe defaults."""
    "Carregar a configuração de locais prováveis, recorrendo a valores padrão seguros."
    current_file = Path(__file__).resolve()

    candidate_paths = [
        current_file.parents[1] / "config.yaml",  # RAG/config.yaml
        current_file.parents[1] / "conf.yaml",    # RAG/conf.yaml
        current_file.parents[2] / "config.yaml",  # project root/config.yaml
        current_file.parents[2] / "conf.yaml",    # project root/conf.yaml
    ]

    for path in candidate_paths:
        if path.exists():
            loaded_config = dict(get_config(str(path)))
            return {
                **DEFAULT_CONFIG,
                **loaded_config,
            }

    return DEFAULT_CONFIG.copy()


CONFIG = load_config()


@click.command()
@click.option(
    "--data",
    default=CONFIG["DATASET_PATH"],
    show_default=True,
    help="Path to the raw dataset file",
)
@click.option(
    "--out-dir",
    default=CONFIG["RAG_DATASET_DIR"],
    show_default=True,
    help="Output directory for the processed dataset",
)
def main(data: str, out_dir: str) -> None:
    """Generate triplet datasets and save to disk."""
    "Gere conjuntos de dados de triplas e salve em disco."
    data_path = Path(data).expanduser().resolve()
    out_dir_path = Path(out_dir).expanduser().resolve()

    if not data_path.exists():
        raise click.ClickException(
            f"Dataset file not found: {data_path}\n"
            "Provide the correct file with --data, for example:\n"
            "python run.py --data /path/to/dataset.json"
        )

    click.echo(f"🔍 Loading raw data from {data_path}")
    raw = TripletDatasetBuilder.load_raw(data_path)
    click.echo(f"📊 Loaded {len(raw)} raw items")

    ds_random = TripletDatasetBuilder.build_triplets(raw, "random_definition")
    ds_sentence = TripletDatasetBuilder.build_triplets(raw, "sentence")
    ds_close = TripletDatasetBuilder.build_triplets(raw, "close_definition")

    click.echo("📚 Built triplet datasets with sizes:")
    click.echo(f"  - Random negatives:  {len(ds_random)} items")
    click.echo(f"  - Sentence negatives: {len(ds_sentence)} items")
    click.echo(f"  - Close negatives:   {len(ds_close)} items")

    rag_ds = TripletDatasetBuilder.concat_and_split(
        [ds_random, ds_sentence, ds_close],
        train_frac=CONFIG["TRAIN_FRAC"],
    )

    out_dir_path.mkdir(parents=True, exist_ok=True)

    rag_ds.save_to_disk(out_dir_path)
    click.echo(f"💾 Saved Hugging-Face dataset to {out_dir_path}")

    for split in ("train", "eval"):
        outfile = out_dir_path / f"{split}.jsonl"
        rag_ds[split].to_json(str(outfile), lines=True)
        click.echo(f"📄 Saved {split}.jsonl with {len(rag_ds[split])} examples")

    click.echo("\n📊 Final Dataset Summary:")
    click.echo(f"  - Training set: {len(rag_ds['train'])} examples")
    click.echo(f"  - Evaluation set: {len(rag_ds['eval'])} examples")


if __name__ == "__main__":
    main()



# """Generate triplet datasets and save to disk."""
# from custom_helpers import add_root_to_pythonpath, get_config
# add_root_to_pythonpath(n_up=2)
#
# import os
# import click
# from pathlib import Path
#
# from RAG.task2_dataset.task import TripletDatasetBuilder
#
# CONFIG = get_config(os.path.join(__file__, "..", "..", "config.yaml"))
#
#
# @click.command()
# @click.option(
#     "--data",
#     default=CONFIG["DATASET_PATH"],
#     help="Path to the raw dataset file"
# )
# @click.option(
#     "--out-dir",
#     default=CONFIG["RAG_DATASET_DIR"],
#     help="Output directory for the processed dataset"
# )
# def main(data: str, out_dir: str) -> None:
#     """Generate triplet datasets and save to disk."""
#     # Ensure data and output directories exist
#     data_path = Path(data)
#     out_dir_path = Path(out_dir)
#
#     click.echo(f"🔍 Loading raw data from {data_path}")
#     raw = TripletDatasetBuilder.load_raw(data_path)
#     click.echo(f"📊 Loaded {len(raw)} raw items")
#
#     # Build triplet datasets with different negative selection strategies
#     ds_random = TripletDatasetBuilder.build_triplets(raw, "random_definition")
#     ds_sentence = TripletDatasetBuilder.build_triplets(raw, "sentence")
#     ds_close = TripletDatasetBuilder.build_triplets(raw, "close_definition")
#
#     click.echo(f"📚 Built triplet datasets with sizes:")
#     click.echo(f"  - Random negatives:  {len(ds_random)} items")
#     click.echo(f"  - Sentence negatives: {len(ds_sentence)} items")
#     click.echo(f"  - Close negatives:   {len(ds_close)} items")
#
#     # Concatenate and split datasets
#     rag_ds = TripletDatasetBuilder.concat_and_split(
#         [ds_random, ds_sentence, ds_close],
#         train_frac=CONFIG["TRAIN_FRAC"]
#     )
#
#     # Ensure output directory exists
#     out_dir_path.mkdir(parents=True, exist_ok=True)
#
#     # Save datasets
#     rag_ds.save_to_disk(out_dir_path)
#     click.echo(f"💾 Saved Hugging‑Face dataset to {out_dir_path.resolve()}")
#
#     # Also export JSONL for human inspection
#     for split in ("train", "eval"):
#         outfile = out_dir_path / f"{split}.jsonl"
#         rag_ds[split].to_json(outfile, lines=True)
#         click.echo(f"📄 Saved {split}.jsonl with {len(rag_ds[split])} examples")
#
#     # Print dataset summary
#     click.echo("\n📊 Final Dataset Summary:")
#     click.echo(f"  - Training set: {len(rag_ds['train'])} examples")
#     click.echo(f"  - Evaluation set: {len(rag_ds['eval'])} examples")
#
#
# if __name__ == "__main__":
#     main()