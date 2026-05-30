from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import torch

from LanguageModeling.task01_ngrams.download_data import get_train_test
from LanguageModeling.task05_text_tools.task import TextTools as tt
from LanguageModeling.task06_rnn.task import RNNLanguageModel
from LanguageModeling.task08_train.task import TrainProcedure


def run(generation_conf, model_conf):
    train_lines, test_lines = get_train_test()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    model = RNNLanguageModel(**model_conf).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())

    train_history, dev_history, model_trained = TrainProcedure.train(
        model=model,
        opt=opt,
        loss_fn=loss_fn,
        train_lines=train_lines,
        dev_lines=test_lines,
        device=device,
        gen_conf=generation_conf,
    )

def main():
    generation_conf = dict(
        mode='sample',
        temperature=0.5,
        max_len=100,
    )

    model_conf = dict(
        tokens=tt.TOKENS,
    )

    run(generation_conf, model_conf)


if __name__ == '__main__':
    main()



# from custom_helpers import add_root_to_pythonpath
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
#
# from LanguageModeling.task01_ngrams.download_data import get_train_test
# from LanguageModeling.task05_text_tools.task import TextTools as tt
# from LanguageModeling.task06_rnn.task import RNNLanguageModel
# from LanguageModeling.task07_loss.task import CrossEntropyLoss
# from LanguageModeling.task08_train.task import TrainProcedure
#
#
# def run(generation_conf, model_conf):
#     train_lines, test_lines = get_train_test()
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"device: {device}")
#
#     model = RNNLanguageModel(**model_conf).to(device)
#
#     loss_fn = CrossEntropyLoss()
#     opt = torch.optim.Adam(model.parameters())
#
#     train_history, dev_history, model_trained = TrainProcedure.train(
#         model=model,
#         opt=opt,
#         loss_fn=loss_fn,
#         train_lines=train_lines,
#         dev_lines=test_lines,
#         device=device,
#         gen_conf=generation_conf,
#     )
#
#
# def main():
#     generation_conf = dict(
#         mode='sample',
#         temperature=0.5,
#         max_len=100,
#     )
#
#     model_conf = dict(
#         tokens=tt.TOKENS,
#     )
#
#     run(generation_conf, model_conf)
#
#
# if __name__ == '__main__':
#     main()




# from custom_helpers import add_root_to_pythonpath
# add_root_to_pythonpath(n_up=2, verbose=True)
#
# import torch
#
# from LanguageModeling.task01_ngrams.download_data import get_train_test
# from LanguageModeling.task05_text_tools.task import TextTools as tt
# from LanguageModeling.task06_rnn.task import RNNLanguageModel
# from LanguageModeling.task07_loss.task import CrossEntropyLoss
# from LanguageModeling.task08_train.task import TrainProcedure
#
#
# def run(generation_conf, model_conf):
#     train_lines, test_lines = get_train_test()
#
#     model = RNNLanguageModel(**model_conf)
#     # TODO: Custom loss works much slower
#     loss_fn = CrossEntropyLoss()
#
#     # ignore_index = tt.TOKEN_TO_ID[EOS]
#     # loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
#     opt = torch.optim.Adam(model.parameters())
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print(f"device: {device}")
#
#     train_history, dev_history, model_trained = TrainProcedure.train(
#         model=model,
#         opt=opt,
#         loss_fn=loss_fn,
#         train_lines=train_lines,
#         dev_lines=test_lines,
#         device=device,
#         gen_conf=generation_conf,
#     )
#     # actually we have all the metrics outputed during training
#
#
# def main():
#     # generation_conf=dict(  # you can change these parameters
#     #     mode='nucleus',
#     #     nucleus=0.9,
#     #     max_len=100,
#     # )
#     generation_conf=dict(  # you can change these parameters
#         mode='sample',
#         temperature=0.5,
#         max_len=100,
#     )
#     model_conf = dict(
#         tokens=tt.TOKENS,
#         # you can add emb_size, hid_size params
#     )
#     run(generation_conf, model_conf)
#
# if __name__ == '__main__':
#     main()
