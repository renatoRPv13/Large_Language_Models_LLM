from custom_helpers import add_root_to_pythonpath

add_root_to_pythonpath(n_up=2, verbose=True)

import random


class Prompter:
    # Task: Write prompts & shot_examples
    # Tarefa: Escreva prompts e exemplos de cenas
    PROMPT_TEMPLATES = {
        # Basic prompt: simplesmente pede a definição
        "basic": "word: {word}\ndefinition: ",

        # Instruction prompt: dá uma instrução clara sobre o que fazer
        "instruction": """Você é um assistente de dicionário prestativo. Sua tarefa é fornecer uma definição clara e concisa para a palavra dada..

Word: {word}
Definition: """,

        # Few-shot prompt: inclui exemplos de palavras semelhantes
        "few_shot": """Aqui estão alguns exemplos de definições de palavras.:

{examples}

Agora, forneça uma definição para a seguinte palavra.:
Word: {word}
Definition: """,

        # Structured prompt: formato mais organizado
        "structured": "Aqui está uma definição e um exemplo para a palavra.\word: {word}\definition: "
    }

    def __init__(self, template_type="structured"):
        self.template_type = template_type
        self.template = self.PROMPT_TEMPLATES[template_type]

    def build_prompt(self, word, examples=None, n_shots=3):
        if self.template_type == "few_shot" and examples:
            shot_examples = "\n".join(
                f"word: {ex['word']}\ndefinition: {ex['definition']}\nexample: {ex['example']}"
                for ex in random.sample(examples, min(n_shots, len(examples)))
            )
            return self.template.format(word=word, examples=shot_examples)
        return self.template.format(word=word)


# Exemplo de uso
if __name__ == "__main__":
    # Exemplos de flashcards para few-shot
    examples = [
        {"word": "apple", "definition": "uma fruta redonda com casca vermelha ou verde", "example": "Eu comi uma maçã."},
        {"word": "car", "definition": "um veículo com quatro rodas", "example": "Ela dirige um carro vermelho."},
        {"word": "happy", "definition": "sentindo ou demonstrando prazer", "example": "As crianças estavam felizes."}
    ]

    # Testar diferentes tipos de prompt
    for template_type in ["basic", "instruction", "few_shot", "structured"]:
        prompter = Prompter(template_type=template_type)
        prompt = prompter.build_prompt("computer", examples if template_type == "few_shot" else None)
        print(f"\n{'=' * 50}")
        print(f"Template: {template_type.upper()}")
        print(f"{'=' * 50}")
        print(prompt)




# from custom_helpers import add_root_to_pythonpath
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import random
#
# class Prompter:
#     # Task: Write prompts & shot_examples
#     PROMPT_TEMPLATES = {
#         "basic": # TODO: basic prompt,
#         "instruction": # TODO: instruction prompt,
#         "few_shot": # TODO: few-shot prompt,
#         "structured": "Here is a definition and an example for the word.\nword: {word}\ndefinition: "
#     }
#
#     def __init__(self, template_type="structured"):
#         self.template_type = template_type
#         self.template = self.PROMPT_TEMPLATES[template_type]
#
#     def build_prompt(self, word, examples=None, n_shots=3):
#         if self.template_type == "few_shot" and examples:
#             shot_examples = "\n".join(
#                 f"word: {ex['word']}\ndefinition: {ex['definition']}\nexample: {ex['example']}"
#                 for ex in random.sample(examples, min(n_shots, len(examples)))
#             )
#             return self.template.format(word=word, examples=shot_examples)
#         return self.template.format(word=word)
