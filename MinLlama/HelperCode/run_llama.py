# from contextlib import nullcontext
# import json
# import time, random, numpy as np, argparse, sys, re, os
# from types import SimpleNamespace
# from tqdm import tqdm
# from typing import Optional
#
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
#
# # change it with respect to the original model
# from MinLlama.Classifier.task import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
# from MinLlama.Llama.task import Llama, load_pretrained
# from MinLlama.Optimizer.task import AdamW
# from .tokenizer import Tokenizer


from contextlib import nullcontext
import json
import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

# change it with respect to the original model
# from MinLlama.Classifier.task import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
# from MinLlama.Llama.task import Llama, load_pretrained
# from MinLlama.Optimizer.task import AdamW
# from MinLlama.HelperCode.tokenizer import Tokenizer

import sys
import os

# Adicionar o caminho raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Agora as importações devem funcionar
from MinLlama.Classifier.task import LlamaZeroShotClassifier, LlamaEmbeddingClassifier
from MinLlama.Llama.task import Llama, load_pretrained
from MinLlama.Optimizer.task import AdamW
from MinLlama.HelperCode.tokenizer import Tokenizer


TQDM_DISABLE=False
# fix the random seed # Corrigir a semente aleatória
def seed_everything(seed=11711):

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# create a custom Dataset Class to be used for the dataloader
# Crie uma classe de conjunto de dados personalizada para ser usada pelo carregador de dados
class LlamaDataset(Dataset):
	def __init__(self, dataset, args, eos=False):
		self.dataset = dataset
		self.p = args
		self.tokenizer = Tokenizer(max_len=args.max_sentence_len)
		self.eos = eos

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		ele = self.dataset[idx]
		return ele

	def pad_data(self, data):
		sents = [x[0] for x in data]
		labels = [x[1] for x in data]
		encoding = [self.tokenizer.encode(s, bos=True, eos=self.eos) for s in sents]
		max_length_in_batch = max([len(sentence) for sentence in encoding])
		encoding_padded = [sentence + [self.tokenizer.pad_id] * (max_length_in_batch - len(sentence)) for sentence in encoding]
		token_ids = torch.LongTensor(encoding_padded)
		labels = torch.LongTensor(labels)

		return token_ids, labels, sents

	def collate_fn(self, all_data):

		token_ids, labels, sents = self.pad_data(all_data)
		batched_data = {
				'token_ids': token_ids,
				'labels': labels,
				'sents': sents,
			}

		return batched_data


# create the data which is a list of (sentence, label, token for the labels)
# Crie os dados, que são uma lista de (frase, rótulo, token para os rótulos)
def create_data(filename, tokenizer: Tokenizer, flag: str ='train', lower: bool = False, eos: bool = True, prompt_suffix: Optional[str]=None):
	# specify the tokenizer # especifique o tokenizador
	num_labels = {}
	data = []

	with open(filename, 'r') as fp:
		for line in fp:
			label, org_sent = line.split(' ||| ')
			if lower:
				org_sent = org_sent.lower()
			sent = org_sent.strip()
			if prompt_suffix is not None:
				sent = f"{sent} {prompt_suffix}"
			tokens = tokenizer.encode(sent, bos=True, eos=eos)
			label = int(label.strip())
			if label not in num_labels:
				num_labels[label] = len(num_labels)
			data.append((sent, label, tokens))
	print(f"load {len(data)} data from {filename}")
	if flag == 'train':
		return data, len(num_labels)
	else:
		return data

# perform model evaluation in terms of the accuracy and f1 score.
# Realizar a avaliação do modelo em termos de precisão e pontuação F1.
def model_eval(dataloader, model, device):
	model.eval() # switch to eval model, will turn off randomness like dropout
	y_true = [] # Alternar para o modelo de avaliação desativará a aleatoriedade, como o dropout.
	y_pred = []
	sents = []
	for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
		b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

		b_ids = b_ids.to(device)

		logits = model(b_ids)
		logits = logits.detach().cpu().numpy()
		preds = np.argmax(logits, axis=1).flatten()

		b_labels = b_labels.flatten()
		y_true.extend(b_labels)
		y_pred.extend(preds)
		sents.extend(b_sents)

	f1 = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)

	return acc, f1, y_pred, y_true, sents

def save_model(model, optimizer, args, config, filepath):
	save_info = {
		'model': model.state_dict(),
		'optim': optimizer.state_dict(),
		'args': args,
		'model_config': config,
		'system_rng': random.getstate(),
		'numpy_rng': np.random.get_state(),
		'torch_rng': torch.random.get_rng_state(),
	}

	torch.save(save_info, filepath)
	#print(f"save the model to {filepath}")
	print(f"Salve o modelo em {filepath}")
def train(args):
	device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
	#### Load data
	# create the data and its corresponding datasets and dataloader
	#### Carregar dados
	# criar os dados e seus respectivos conjuntos de dados e carregador de dados
	tokenizer = Tokenizer(args.max_sentence_len)
	train_data, num_labels = create_data(args.train, tokenizer, 'train')
	dev_data = create_data(args.dev, tokenizer, 'valid')

	train_dataset = LlamaDataset(train_data, args)
	dev_dataset = LlamaDataset(dev_data, args)

	train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
								  collate_fn=train_dataset.collate_fn)
	dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
								collate_fn=dev_dataset.collate_fn)

	#### Init model #### Inicializar modelo
	config = {'hidden_dropout_prob': args.hidden_dropout_prob,
			  'pretrained_model_path': args.pretrained_model_path,
			  'num_labels': num_labels,
			  'data_dir': '.',
			  'option': args.option}

	config = SimpleNamespace(**config)

	# initialize the Senetence Classification Model # Inicializar o Modelo de Classificação de Sentenças
	model = LlamaEmbeddingClassifier(config)
	model = model.to(device)

	lr = args.lr
	## specify the optimizer  ## especificar o otimizador
	optimizer = AdamW(model.parameters(), lr=lr)
	best_dev_acc = 0

	## run for the specified number of epochs ## Executar pelo número especificado de épocas
	for epoch in tqdm(range(args.epochs)):
		model.train()
		train_loss = 0
		num_batches = 0
		for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
			b_ids, b_labels, b_sents = batch['token_ids'], batch['labels'], batch['sents']

			b_ids = b_ids.to(device)
			b_labels = b_labels.to(device)

			optimizer.zero_grad()
			logits = model(b_ids)
			loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			num_batches += 1

		train_loss = train_loss / (num_batches)

		train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
		dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			save_model(model, optimizer, args, config, args.filepath)

		#print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
		print(f"época {epoch}: perda de treino :: {train_loss :.3f}, acurácia de treino :: {train_acc :.3f}, acurácia de desenvolvimento :: {dev_acc:.3f}")

def generate_sentence(args, prefix, outfile, max_new_tokens = 75, temperature = 0.0):
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float32) if args.use_gpu else nullcontext()
		llama = load_pretrained(args.pretrained_model_path)
		llama = llama.to(device)
		#print(f"load model from {args.pretrained_model_path}")
		print(f"Carregar modelo de {args.pretrained_model_path}")
		enc = Tokenizer(args.max_sentence_len)

		start_ids = enc.encode(prefix, bos=True, eos=False)
		x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

		# run generation # executar geração
		with torch.no_grad():
			with ctx:
				y = llama.generate(x, max_new_tokens, temperature=temperature)
				sentence = enc.decode(y[0].tolist())
				print(f"Temperatura is {temperature}")
				print(sentence)
				print('---------------')
				writer = open(outfile, 'w')
				writer.write(sentence)
				#print(f"Wrote generated sentence to {outfile}.")
				print(f"Escrevi a frase gerada para{outfile}.")
				writer.close()

def write_predictions_to_file(split: str, outfile: str, acc: float, pred: list[str], sents: list[str]):
	with open(outfile, "w+") as f:
		print(f"{split} acc :: {acc :.3f}")
		for s, p in zip(sents, pred):
			f.write(f"{p} ||| {s}\n")

def test_with_prompting(args):
	assert args.dev_out.endswith("dev-prompting-output.txt"), 'For saving prompting results, please set the dev_out argument as "<dataset>-dev-prompting-output.txt"'
	assert args.test_out.endswith("test-prompting-output.txt"), 'For saving prompting results, please set the test_out argument as "<dataset>-test-prompting-output.txt"'

	with torch.no_grad():

		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		#### Load data
		# create the data and its corresponding datasets and dataloader
		#### Carregar dados
		# criar os dados e seus respectivos conjuntos de dados e carregador de dados
		tokenizer = Tokenizer(args.max_sentence_len)
		label_names = json.load(open(args.label_names, 'r'))
		_, num_labels = create_data(args.train, tokenizer, 'train')

		#### Init model
		config = {'pretrained_model_path': args.pretrained_model_path,
				'label_names': label_names,
				'num_labels': num_labels,
				'data_dir': '.',
				'option': args.option}

		config = SimpleNamespace(**config)

		if len(label_names) == 2:
			label_name_str = " or ".join(label_names)
		else:
			label_name_str = ", ".join(label_names[:-1]) + ", or " + label_names[-1]
		prompt_suffix=f"Is this movie {label_name_str}? This movie is "
		model = LlamaZeroShotClassifier(config, tokenizer, label_names)
		model = model.to(device)

		dev_data = create_data(args.dev, tokenizer, 'valid', eos=False, prompt_suffix=prompt_suffix)
		dev_dataset = LlamaDataset(dev_data, args, eos=False)
		dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

		test_data = create_data(args.test, tokenizer, 'test', eos=False, prompt_suffix=prompt_suffix)
		test_dataset = LlamaDataset(test_data, args, eos=False)
		test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

		dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
		test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

		write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
		write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)

def test(args):
	assert args.dev_out.endswith("dev-finetuning-output.txt"), 'For saving finetuning results, please set the dev_out argument as "<dataset>-dev-finetuning-output.txt"'
	assert args.test_out.endswith("test-finetuning-output.txt"), 'For saving finetuning results, please set the test_out argument as "<dataset>-test-finetuning-output.txt"'
	with torch.no_grad():
		device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
		saved = torch.load(args.filepath, weights_only=False)
		config = saved['model_config']
		model = LlamaEmbeddingClassifier(config)
		model.load_state_dict(saved['model'])
		model = model.to(device)
		#print(f"load model from {args.filepath}")
		print(f"modelo de carga de {args.filepath}")
		tokenizer = Tokenizer(args.max_sentence_len)
		dev_data = create_data(args.dev, tokenizer, 'valid')
		dev_dataset = LlamaDataset(dev_data, args)
		dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

		test_data = create_data(args.test, tokenizer, 'test')
		test_dataset = LlamaDataset(test_data, args)
		test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

		dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
		test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)
	
		write_predictions_to_file("dev", args.dev_out, dev_acc, dev_pred, dev_sents)
		write_predictions_to_file("test", args.test_out, test_acc, test_pred, test_sents)
