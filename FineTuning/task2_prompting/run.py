from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import os
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from FineTuning.task2_prompting.task import Prompter

class InferenceClient:
    """
    Very small wrapper around a local Hugging Face causal-LM.
    """

    def __init__(self, model_id: str = "gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Download / load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:           # GPT-2 has no pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device).eval()

        print(f"🤖 Ready! Using '{model_id}' on {self.device}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 60,
        generation_params: Optional[Dict] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum new tokens to generate.
            generation_params: Any extra kwargs accepted by `model.generate`,
                               e.g. temperature, top_p, do_sample.:contentReference[oaicite:4]{index=4}
        """
        generation_params = generation_params or {
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        
        encoded = self.tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_params,
        )
        # strip the prompt portion
        generated_ids = output_ids[0, len(encoded["input_ids"][0]):]  # trim the input part
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)




def main():
    # Sample data for few-shot examples
    sample_data = [
        {"word": "computer", "definition": "um dispositivo eletrônico para armazenar e processar dados",
         "example": "Eu uso um computador para trabalhar todos os dias"},
        {"word": "book", "definition": "uma obra escrita ou impressa composta por páginas coladas",
         "example": "Ela leu um livro sobre história antiga"},
        {"word": "algorithm", "definition": "um conjunto de regras a serem seguidas em cálculos",
         "example": "O mecanismo de busca usa algoritmos complexos para encontrar resultados relevantes"}
    ]
    
    word = "quantum"
    template_types = Prompter.PROMPT_TEMPLATES.keys()
    api = InferenceClient()

    for template_type in template_types:
        prompter = Prompter(template_type=template_type)
        prompt = prompter.build_prompt(word, examples=sample_data)
        print(f"PROMPT ({template_type}):")
        print(prompt)
        print(f'TEXTO GERADO (palavra: {word}):')
        generated_text = api.generate(prompt)
        print(generated_text)
        print('-' * 50)

if __name__ == "__main__":
    main()