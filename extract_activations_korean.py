import sys
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from train_classifier import *

mask_fee = len(sys.argv) > 1

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
bert_model = nn.DataParallel(bert_model)
bert_model.load_state_dict(torch.load(
    f'../checkpoints/mbert_masked-{mask_fee}_linear.pt'))
bert_model.cuda()

n_classes = 50
classification_head = ClassificationHeadSimple(n_classes)
classification_head = nn.DataParallel(classification_head)
classification_head.load_state_dict(
    torch.load(f'../checkpoints/head_masked-{mask_fee}_linear.pt'))
classification_head.cuda()

with open('top_50_frames.json', 'r', encoding='utf-8') as inp:
    major_frames = json.load(inp)
frame_to_class = {frame: i for i, frame in enumerate(major_frames)}
with open('top_50_frames_korean.json', 'r', encoding='utf-8') as inp:
    data_dev = json.load(inp)

batch_size = 256
n_steps_dev = ceil(len(data_dev) / batch_size)

level_09_activations = torch.zeros((len(data_dev), 768))
level_10_activations = torch.zeros((len(data_dev), 768))
level_11_activations = torch.zeros((len(data_dev), 768))
for batch_n in tqdm(range(n_steps_dev)):
    lo = batch_size * batch_n
    hi = min(batch_size * (batch_n + 1), len(data_dev))
    annotated_sentences = data_dev[lo: hi]
    FEE_labels, FEE_idx, batch_sentences, FEE_ranges = prepare_batch(
        annotated_sentences, tokenizer, frame_to_class)
    mbert_inputs = {
        k: v.cuda() for k, v in tokenizer(
            batch_sentences, padding=True, truncation=True, return_tensors='pt'
        ).items()
    }
    if mask_fee:
        for i in range(len(mbert_inputs['input_ids'])):
            for j in range(FEE_ranges[i][0], FEE_ranges[i][1]):
                mbert_inputs['input_ids'][i][j] = 103
    with torch.no_grad():
        mbert_outputs = bert_model(**mbert_inputs).hidden_states
        level_09_activations[lo: hi] = torch.vstack([mbert_outputs[-4][i, FEE_idx[i]]
                                                     for i in range(mbert_outputs[-4].size(0))])
        level_10_activations[lo: hi] = torch.vstack([mbert_outputs[-3][i, FEE_idx[i]]
                                                     for i in range(mbert_outputs[-3].size(0))])
        level_11_activations[lo: hi] = torch.vstack([mbert_outputs[-2][i, FEE_idx[i]]
                                                     for i in range(mbert_outputs[-2].size(0))])

if mask_fee:
    np.savetxt(f'../csv/level_09_activations_masked_korean.csv',
               level_09_activations.numpy(), delimiter=',')
    np.savetxt(f'../csv/level_10_activations_masked_korean.csv',
               level_10_activations.numpy(), delimiter=',')
    np.savetxt(f'../csv/level_11_activations_masked_korean.csv',
               level_11_activations.numpy(), delimiter=',')
else:
    np.savetxt(f'../csv/level_09_activations_korean.csv',
               level_09_activations.numpy(), delimiter=',')
    np.savetxt(f'../csv/level_10_activations_korean.csv',
               level_10_activations.numpy(), delimiter=',')
    np.savetxt(f'../csv/level_11_activations_korean.csv',
               level_11_activations.numpy(), delimiter=',')
