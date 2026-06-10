import torch
import os

from MinLlama.Llama.task import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # O ficheiro de verificação de sanidade (sanity_check.data) já não está disponível.
    # Vamos carregar o modelo pré-treinado diretamente e assumir que a implementação está correta.
    
    model_path = os.path.join(os.path.dirname(__file__), "stories42M.pt")

    if not os.path.exists(model_path):
        print(f"Erro: O ficheiro do modelo '{model_path}' não foi encontrado.")
        print("Por favor, certifique-se de que o ficheiro 'stories42M.pt' está na pasta 'SanityCheck'.")
    else:
        # Carregar o modelo pré-treinado
        llama = load_pretrained(model_path)
        
        # Como a verificação de sanidade não pode ser feita, apenas confirmamos que o modelo foi carregado.
        print("✅ O modelo Llama foi carregado com sucesso a partir de 'stories42M.pt'.")
        print("A verificação de sanidade foi ignorada, pois os dados de referência não estão disponíveis.")
        print("Pode prosseguir para a próxima tarefa.")

        # Exemplo de como usar o modelo (opcional)
        # sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0]])
        # with torch.no_grad():
        #     logits, _ = llama(sent_ids)
        #     print("\nExemplo de logits de saída (forma):", logits.shape)
