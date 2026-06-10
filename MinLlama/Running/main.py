import sys
import os
import argparse

# Add project root to Python path # Adicione a raiz do projeto ao caminho do Python
# Add project root to Python path # Adicione a raiz do projeto ao caminho do Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from MinLlama.HelperCode.run_llama import train, generate_sentence, test_with_prompting, test, seed_everything

def setup_args_for_option(option):
    parser = argparse.ArgumentParser(description="Run LLAMA model options")
    parser.add_argument("--train", type=str, default="../Data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="../Data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="../Data/cfimdb-test.txt")
    parser.add_argument("--label-names", type=str, default="../Data/cfimdb-label-mapping.json", help="Label names mapping file")
    parser.add_argument("--pretrained-model-path", type=str, default="../SanityCheck/stories42M.pt")
    parser.add_argument("--tokenizer-path", type=str, default="../HelperCode/tokenizer.model", help="Path to the SentencePiece tokenizer model")
    parser.add_argument("--max_sentence_len", type=int, default=512, help="Max sentence length for tokenizer")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--option", type=str, required=True,
                        help='prompt: the Llama parameters are frozen; finetune: Llama parameters are updated',
                        choices=('generate', 'prompt', 'finetune'))
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for training")
    parser.add_argument("--generated_sentence_low_temp_out", type=str, default="generated-sentence-temp-0.txt")
    parser.add_argument("--generated_sentence_high_temp_out", type=str, default="generated-sentence-temp-1.txt")
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-prompting-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-prompting-output.txt")

    # Hyperparameters # Hiperparâmetros
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for text generation")

    args = parser.parse_args(["--option", option])  # Directly set the option here for testing # Defina a opção diretamente aqui para teste
    if option == "generate":
        args.train = None  # No train data needed for generation # Não são necessários dados de trem para a geração
    elif option in ["prompt", "finetune"]:
        # Ensure paths are set for prompt and finetune options
        # Certifique-se de que os caminhos estejam configurados para opções de prompt e finetune.
        args.train = "../Data/sst-train.txt" if args.train is None else args.train
        args.dev = "../Data/sst-dev.txt" if args.dev is None else args.dev
        args.test = "../Data/sst-test.txt" if args.test is None else args.test
        args.label_names = "../Data/sst-label-mapping.json" if args.label_names is None else args.label_names
        if option == "finetune":
            args.dev_out = "sst-dev-finetuning-output.txt"
            args.test_out = "sst-test-finetuning-output.txt"

    return args


def main(option):
    args = setup_args_for_option(option)
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}.pt'  # save path # caminho de salvamento
    seed_everything(args.seed)

    if args.option == "generate":
        # Call text generation function # Função de geração de texto de chamada
        #print("Running text generation...")
        print("Executando geração de texto...")
        # Step 1
        # Complete this sentence to test your implementation!
        # Etapa 1
        # Complete esta frase para testar sua implementação!
        prefix = "I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"
        generate_sentence(args, prefix, args.generated_sentence_low_temp_out, max_new_tokens=75, temperature=0.0)
        generate_sentence(args, prefix, args.generated_sentence_high_temp_out, max_new_tokens=75, temperature=1.0)
    elif args.option == "prompt":
        #print("Running zero-shot prompting...")
        print("Executando prompts zero-shot...")
        # Step 2
        # Solve this task with prompted language modeling
        # Etapa 2
        # Resolva esta tarefa com modelagem de linguagem guiada
        test_with_prompting(args)
    elif args.option == "finetune":
        #print("Running fine-tuning...")
        print("Executando ajustes finos...")
        # Step 3
        # Finetune a classification model
        train(args)
        # Step 4
        # Evaluate your model on the dev and test sets
        # Etapa 4
        # Avalie seu modelo nos conjuntos de desenvolvimento e teste
        test(args)
    else:
        raise ValueError(f"Invalid option: {args.option}")

if __name__ == "__main__": # Altere para "prompt" ou "generate" para testar as outras opções
    main("finetune") # Change this to "prompt" or "generate" to test the other options
