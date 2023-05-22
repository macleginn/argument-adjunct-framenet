from collections import defaultdict
import json
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

from train_classifier import *
from bert_wrapper import *

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

frame_to_class = defaultdict(int)
with open('all_frames_train.json', 'r', encoding='utf-8') as inp:
    data_dev = json.load(inp)

batch_size = 32
n_steps_dev = ceil(len(data_dev) / batch_size)

n_layers = 12
embedding_size = 768
hidden_states = torch.zeros((n_layers, len(data_dev), embedding_size))
ff_activations = torch.zeros((n_layers, len(data_dev), embedding_size))
for batch_n in tqdm(range(n_steps_dev)):
    lo = batch_size * batch_n
    hi = batch_size * (batch_n + 1)
    annotated_sentences = data_dev[lo: hi]
    FEE_labels, FEE_idx, batch_sentences, FEE_ranges, CE_ranges, NCE_ranges, ETE_ranges = prepare_batch(
        annotated_sentences, tokenizer, frame_to_class)
    # Correct the endpoint to match the actual slice length
    # because some of the sentences may need to be discarded.
    hi = lo + len(batch_sentences)
    with torch.no_grad():
        outputs, ff_outputs = apply_bert_model(
            bert_model, tokenizer, batch_sentences)
        for i in [9, 11]:
            hidden_states[i][lo: hi] = outputs[i][:, 0, :]
            ff_activations[i][lo: hi] = ff_outputs[i][:, 0, :]

for i in tqdm([9, 11], desc='Saving results'):
    np.savetxt(f'../csv/level_{i+1}_activations_CLS_vanilla.csv',
               hidden_states[i].numpy(), delimiter=',')
    np.savetxt(f'../csv/level_{i+1}_activations_CLS_ff.csv',
               ff_activations[i].numpy(), delimiter=',')
