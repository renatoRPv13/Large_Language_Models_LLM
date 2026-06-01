from custom_helpers import add_root_to_pythonpath, get_config
add_root_to_pythonpath(n_up=2, verbose=True)

from FineTuning.task3_dataset.task import DatasetHandler

def main() -> None:
    """Main function to process and save datasets."""
    conf = get_config("../conf.yaml")
    model_and_tokenizer_name = conf.model
    num_proc = 4  # TO-DO: your number here
    handler = DatasetHandler(tokenizer_name=model_and_tokenizer_name, num_proc=num_proc)
    
    print("Loading and converting dataset...")
    dataset = handler.convert_to_hf(conf.data.words)
    
    print("Splitting dataset...")
    train, test = handler.train_test_split(dataset)
    
    print("Processing datasets...")
    train = handler.process(train)
    test = handler.process(test)
    
    print("Saving datasets...")
    handler.save(train, test, save_dir=conf.data.train_test_dir)
    
    print("\nDataset processing complete!")
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

if __name__ == "__main__":
    main()
