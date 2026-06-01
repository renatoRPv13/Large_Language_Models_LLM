import torch
import pynvml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import Dict, Any


def _process_line(line: str, prefix: str) -> str:
    """
     Extrai o conteúdo principal de uma linha formatada, removendo o prefixo fornecido,
    removendo um marcador de enumeração (por exemplo, "1.", "a"), etc.) e qualquer pontuação final.
    A função executa os seguintes passos:
    1. Remove o prefixo especificado da linha.
    2. Busca marcadores de enumeração conhecidos (como "1.", "a.", "1)", "a)") e extrai o texto após o primeiro marcador encontrado.
    A extração é interrompida no próximo marcador (por exemplo, "2.", "b.", "2)" ou "b)"), se presente.
    3. Remove quaisquer caracteres de pontuação finais (.,;:()[]{}).
    Examples:
        >>> process_line("PREFIX 1. Hello, world! 2. Goodbye", "PREFIX ")
        'Hello, world!'
        >>> process_line(">>> a) This is an example text. b) Next text", ">>> ")
        'This is an example text'
    Motivation:
        LLM's is prone to generate more than one definition / example. For example, for 'apple', it might easily generate something like:
        word: apple
        definition: 1. a fruit that grows on trees. 2. a company that makes phones.
    """
    #Remova o prefixo e os espaços em branco iniciais/finais.
    content = line[len(prefix):].strip()

    # Defina pares de tokens: o token inicial a ser extraído e o próximo token que indica o fim do segmento desejado.
    token_pairs = [("1.", "2."), ("a.", "b."), ("1)", "2)"), ("a)", "b)")]

    for start_token, end_token in token_pairs:
        if start_token in content:
            start_index = content.find(start_token) + len(start_token)
            end_index = content.find(end_token, start_index)
            if end_index == -1:
                end_index = len(content)
            content = content[start_index:end_index].strip()
            break

    # Remova os caracteres de pontuação finais.
    content = content.strip(".,;:()[]{}")

    return content


# Crie um alias para doctests (para que tanto _process_line quanto process_line funcionem).
process_line = _process_line

class Helpers:
    @staticmethod
    def load_model_and_tokenizer(model_name: str, quantize: bool = False, is_prompt_tuning: bool = False) -> tuple:
        """Carregue o modelo com a configuração adequada para quantização e ajuste rápido. O modelo será congelado."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # Não suportado na versão mais recente do bitsandbytes para MacOS.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quantize,
            bnb_4bit_compute_dtype=torch.float32,
        ) if quantize else None

        # Configurar device_map baseado na disponibilidade de GPU
        if torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = None  # Para CPU, não usar device_map

        # Load the model with proper configuration
        # Carregue o modelo com a configuração adequada
        if device_map is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                offload_state_dict=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
        else:
            # Para CPU, carregar normalmente
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            model = model.to("cpu")

        # Ative o checkpoint de gradiente para otimização de memória.
        model.gradient_checkpointing_enable()

        # Se houver necessidade de ajuste imediato, habilite os gradientes de entrada.
        if is_prompt_tuning:
            model.enable_input_require_grads()

        # Congelar parâmetros do modelo
        for param in model.parameters():
            param.requires_grad = False

        return model, tokenizer

    @staticmethod
    def get_output(model, tokenizer, prompt: str, params: Dict[str, Any], device: str) -> str:
        """Gere uma saída de texto usando o modelo com os parâmetros fornecidos."""
        # Get input from tokenizer and move it to the device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Gere com os parâmetros fornecidos.
        outputs = model.generate(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=params.get("max_new_tokens", 100),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            top_k=params.get("top_k", 50),
            do_sample=params.get("do_sample", True),
            num_return_sequences=params.get("num_return_sequences", 1),
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def get_example_and_definition(model, tokenizer, word: str, params: Dict[str, Any], device: str) -> Dict:
        """Obtenha a definição e o exemplo processados ​​usando heurísticas de notebook."""
        # Generate definition # Gerar definição
        def_prompt = f"Provide a concise, one-sentence definition for the following word. Do not include any additional comments or explanations.\n\nword: {word}\ndefinition: "
        def_output = Helpers.get_output(model, tokenizer, def_prompt, params, device)
        
        definition = ""
        def_lines = def_output.split("\n")
        for line in reversed(def_lines):
            if line.strip():
                definition = _process_line(line, "definition: ")
                break

        # Generate example
        ex_prompt = f"word: {word}\ndefinition: {definition}\nProvide a simple, clear example of how to use this word in a sentence. Do not include any additional comments or explanations.\nexample: "
        ex_output = Helpers.get_output(model, tokenizer, ex_prompt, params, device)
        
        example = ""
        ex_lines = ex_output.split("\n")
        for line in reversed(ex_lines):
            if line.strip():
                example = _process_line(line, "example: ")
                break

        return {
            "word": word,
            "definition": definition,
            "example": example
        }

    @staticmethod
    def get_cuda_device_with_most_free_memory(verbose: bool = False) -> str:
        """
        Retorna o identificador do dispositivo CUDA (por exemplo, "cuda:0") que possui a maior quantidade de memória livre disponível.
            Esta função utiliza a biblioteca NVML (via pynvml) para consultar a memória livre de cada GPU.
            Ela compara a memória livre (em bytes) entre todas as GPUs disponíveis e retorna o identificador do dispositivo
            correspondente à GPU com a maior quantidade de memória livre.
        Args:
            verbose: Se verdadeiro, imprime a memória livre de cada GPU.
        Raises:
           Erro de tempo de execução: Se nenhum dispositivo CUDA for encontrado.
        Returns:
            Uma string que representa o dispositivo CUDA (por exemplo, "cuda:0") com a maior quantidade de memória livre.
        """
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            raise RuntimeError("No CUDA devices found.")

        best_device = 0
        best_free_memory = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mem = mem_info.free  # Free memory in bytes
            if verbose:
                print(f"### Device {i}")
                print(f"Total memory: {mem_info.total / 1024 ** 2:.2f} MiB")
                print(f"Used memory: {mem_info.used / 1024 ** 2:.2f} MiB")
                print(f"Free memory: {mem_info.free / 1024 ** 2:.2f} MiB")

            if free_mem > best_free_memory:
                best_free_memory = free_mem
                best_device = i

        pynvml.nvmlShutdown()
        return f"cuda:{best_device}"

    @staticmethod
    def convert_to_serializable(obj):
        """
        Converter um objeto para um formato serializável (por exemplo, dicionário, lista, etc.) convertendo recursivamente seus componentes.
        :param obj: O objeto a ser convertido.
        :return: O objeto em um formato serializável.
        """
        if isinstance(obj, dict):
            return {k: Helpers.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [Helpers.convert_to_serializable(item) for item in obj]
        return obj
