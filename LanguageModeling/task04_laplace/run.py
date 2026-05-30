from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)


from LanguageModeling.task04_laplace.task import LaplaceLanguageModel
from LanguageModeling.task01_ngrams.download_data import get_train_test
from LanguageModeling.task03_perplexity.run import run_one


def run(generation_conf):
    train_lines, test_lines = get_train_test()

    for n in range(1, 4):
        model = LaplaceLanguageModel(n=n, lines=train_lines, delta=1e-5)
        prefix = 'preenchendo a lacuna' # Mais ideias suas aqui
        run_one(generation_conf, model, test_lines, prefix)


def main():
    # generation_conf=dict(  # you can change these parameters
    #     mode='sample',
    #     temperature=0.6,
    #     max_len=30,
    # )
    # other option is to use nucleus sampling
    generation_conf = dict(
        max_len=30,
        mode='nucleus',  
        nucleus=0.8,
    )
    run(generation_conf)

if __name__ == '__main__':
    main()
