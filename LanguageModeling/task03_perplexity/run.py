from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

from LanguageModeling.task01_ngrams.task import NGramLanguageModel, EOS
from LanguageModeling.task01_ngrams.download_data import get_train_test
from LanguageModeling.task02_generation.task import Generator
from LanguageModeling.task03_perplexity.task import Evaluator

def run_one(generation_conf, model, test_lines, prefix):
    evaluator = Evaluator()
    perplexity = evaluator.perplexity(model, test_lines)
    print(f"Perplexidade para{model.n}-gram model: {perplexity}")
    # few examples
    generator = Generator(model)
    print(f"Conclusão de amostra para '{prefix}':")
    for _ in range(3):
        seq = generator.generate_sequence(prefix, **generation_conf)
        if seq[-1] == EOS:
            seq = seq[:-1]
        print(' '.join(seq))
    print('-' * 50)

def run(generation_conf):
    train_lines, test_lines = get_train_test()
    for n in range(1, 4):
        model = NGramLanguageModel(n=n, lines=train_lines)
        prefix = 'qualquer coisa que você quiser'  # more of your ideas here # Mais ideias suas aqui
        # prefix = 'nonsense nonsense' # shouldn't be able to continue such a text
        run_one(generation_conf, model, test_lines, prefix)

def main():
    # generation_conf=dict(  # you can change these parameters
    #     mode='sample',
    #     temperature=0.0,  # greedy (not recommended)
    # )
    # other option is to use nucleus sampling
    generation_conf = dict(
        max_len=30,
        mode='nucleus',  
        nucleus=0.9,
    )
    run(generation_conf)

if __name__ == '__main__':
    main()
