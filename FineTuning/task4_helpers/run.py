from custom_helpers import add_root_to_pythonpath, get_config
import  torch
add_root_to_pythonpath(n_up=2, verbose=True)

import os
# Comente ou remova esta linha se não tiver GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Desabilita CUDA completamente se não tiver GPU
if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from FineTuning.task4_helpers.task import Helpers
from FineTuning.task3_dataset.task import DatasetHandler


def setup_model(conf: dict) -> tuple:
    """Inicializar e retornar o modelo com a configuração adequada"""
    return Helpers.load_model_and_tokenizer(
        conf["model"],
        quantize=False,
        is_prompt_tuning=False
    )


def load_datasets(conf: dict) -> tuple:
    """Carregar e retornar conjuntos de dados processados"""
    handler = DatasetHandler(conf.model)
    return handler.load(conf.data.train_test_dir)


def create_trainer(conf, model, tokenizer, train, test) -> Trainer:
    """Criar e configurar o treinador de avaliação"""
    training_args = TrainingArguments(
        output_dir=conf.exp.eval_base_model_dir,
        per_device_eval_batch_size=16,  # Not really important
        report_to="none",
        no_cuda=True,  # Força uso de CPU
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )


def main() -> None:
    """Fluxo de execução principal para o cálculo das perdas do modelo base"""
    # Setup components
    conf = get_config("../conf.yaml")
    model, tokenizer = setup_model(conf)
    train, test = load_datasets(conf)
    trainer = create_trainer(conf, model, tokenizer, train, test)

    # Calculate and display losses
    test_loss = trainer.evaluate()
    train_loss = trainer.evaluate(eval_dataset=train)
    # print("\nFinal Results:")
    # print(f"Test Loss: {test_loss['eval_loss']:.4f}")
    # print(f"Train Loss: {train_loss['eval_loss']:.4f}")
    print("\nResultados Finais:")
    print(f"Perda no Teste: {test_loss['eval_loss']:.4f}")
    print(f"Perda no Treinamento: {train_loss['eval_loss']:.4f}")


if __name__ == "__main__":
    main()

# from custom_helpers import add_root_to_pythonpath, get_config
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# import torch
# from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
#
# from FineTuning.task4_helpers.task import Helpers
# from FineTuning.task3_dataset.task import DatasetHandler
#
# def setup_model(conf: dict) -> tuple:
#     """Initialize and return model with proper configuration"""
#     return Helpers.load_model_and_tokenizer(
#         conf["model"],
#         quantize=False,
#         is_prompt_tuning=False
#     )
#
# def load_datasets(conf: dict) -> tuple:
#     """Load and return processed datasets"""
#     handler = DatasetHandler(conf.model)
#     return handler.load(conf.data.train_test_dir)
#
# def create_trainer(conf, model, tokenizer, train, test) -> Trainer:
#     """Create and configure evaluation trainer"""
#     training_args = TrainingArguments(
#         output_dir=conf.exp.eval_base_model_dir,
#         per_device_eval_batch_size=16,  # Not really important
#         report_to="none",
#     )
#
#     return Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train,
#         eval_dataset=test,
#         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
#     )
#
# def main() -> None:
#     """Main execution flow for calculating base model losses"""
#     # Setup components
#     conf = get_config("../conf.yaml")
#     model, tokenizer = setup_model(conf)
#     train, test = load_datasets(conf)
#     trainer = create_trainer(conf, model, tokenizer, train, test)
#
#     # Calculate and display losses
#     test_loss = trainer.evaluate()
#     train_loss = trainer.evaluate(eval_dataset=train)
#     print("\nFinal Results:")
#     print(f"Test Loss: {test_loss['eval_loss']:.4f}")
#     print(f"Train Loss: {train_loss['eval_loss']:.4f}")
#
# if __name__ == "__main__":
#     main()