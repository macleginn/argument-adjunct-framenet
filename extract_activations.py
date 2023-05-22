from collections import defaultdict
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from train_classifier import *
from bert_wrapper import *

# mask_fee = len(sys.argv) > 1

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
bert_model = nn.DataParallel(bert_model)
bert_model.load_state_dict(torch.load(
    f'../checkpoints/mbert_masked-None_linear.pt'))
bert_model.cuda()

with open('top_50_frames.json', 'r', encoding='utf-8') as inp:
    major_frames = json.load(inp)
frame_to_class = defaultdict(int)
with open('top_50_frames_dev.json', 'r', encoding='utf-8') as inp:
# with open('top_50_frames_train_ete.json', 'r', encoding='utf-8') as inp:
# with open('all_frames_train.json', 'r', encoding='utf-8') as inp:
    data_dev = json.load(inp)

batch_size = 256
n_steps_dev = ceil(len(data_dev) / batch_size)

level_09_activations = torch.zeros(len(data_dev), 768)
level_11_activations = torch.zeros(len(data_dev), 768)
for batch_n in tqdm(range(n_steps_dev)):
    lo = batch_size * batch_n
    hi = batch_size * (batch_n + 1)
    annotated_sentences = data_dev[lo: hi]
    FEE_labels, FEE_idx, batch_sentences, FEE_ranges, CE_ranges, NCE_ranges, ETE_ranges = prepare_batch(
        annotated_sentences, tokenizer, frame_to_class)
    # Correct the endpoint to match the actual slice length
    hi = lo + len(batch_sentences)
    with torch.no_grad():
        mbert_inputs = {
            k: v.cuda()
            for k, v in tokenizer(batch_sentences, return_tensors='pt', padding=True, truncation=True).items()
        }
        mbert_outputs = bert_model(**mbert_inputs).hidden_states
        # level_06_activations[lo: hi] = torch.vstack([mbert_outputs[6][i, FEE_idx[i]]
        #                                              for i in range(mbert_outputs[6].size(0))])
        # level_07_activations[lo: hi] = torch.vstack([mbert_outputs[7][i, FEE_idx[i]]
        #                                              for i in range(mbert_outputs[7].size(0))])
        # level_08_activations[lo: hi] = torch.vstack([mbert_outputs[8][i, FEE_idx[i]]
        #                                              for i in range(mbert_outputs[8].size(0))])
        level_09_activations[lo: hi] = torch.vstack([mbert_outputs[9][i, FEE_idx[i]]
                                                     for i in range(mbert_outputs[9].size(0))])
        # level_10_activations[lo: hi] = torch.vstack([mbert_outputs[-3][i, FEE_idx[i]]
        #                                              for i in range(mbert_outputs[-3].size(0))])
        level_11_activations[lo: hi] = torch.vstack([mbert_outputs[-2][i, FEE_idx[i]]
                                                     for i in range(mbert_outputs[-2].size(0))])
        # level_12_activations[lo: hi] = torch.vstack([mbert_outputs[-1][i, FEE_idx[i]]
        #                                              for i in range(mbert_outputs[-2].size(0))])
        #level_12_activations[lo: hi] = torch.vstack([mbert_outputs[-1][i, FEE_idx[i]]
        #                                             for i in range(mbert_outputs[-1].size(0))])
        # CLS

# np.savetxt(f'../csv/level_06_activations_vanilla.csv',
#            level_06_activations.numpy(), delimiter=',')
# np.savetxt(f'../csv/level_07_activations_vanilla.csv',
#            level_07_activations.numpy(), delimiter=',')
# np.savetxt(f'../csv/level_08_activations_vanilla.csv',
#            level_08_activations.numpy(), delimiter=',')
np.savetxt(f'../csv/level_09_activations_FEE_vanilla_50.csv',
            level_09_activations.numpy(), delimiter=',')
# np.savetxt(f'../csv/level_10_activations_vanilla.csv',
#            level_10_activations.numpy(), delimiter=',')
# np.savetxt(f'../csv/level_11_activations_vanilla.csv',
#            level_11_activations.numpy(), delimiter=',')
#np.savetxt(f'../csv/level_12_activations_FEE_vanilla.csv',
#           level_12_activations.numpy(), delimiter=',')
np.savetxt(f'../csv/level_11_activations_FEE_vanilla_50.csv',
           level_11_activations.numpy(), delimiter=',')
# if mask_fee:
#     np.savetxt(f'../csv/level_06_activations_masked.csv',
#                level_06_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_07_activations_masked.csv',
#                level_07_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_08_activations_masked.csv',
#                level_08_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_09_activations_masked.csv',
#                level_09_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_10_activations_masked.csv',
#                level_10_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_11_activations_masked.csv',
#                level_11_activations.numpy(), delimiter=',')
# else:
#     np.savetxt(f'../csv/level_06_activations.csv',
#                level_06_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_07_activations.csv',
#                level_07_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_08_activations.csv',
#                level_08_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_09_activations.csv',
#                level_09_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_10_activations.csv',
#                level_10_activations.numpy(), delimiter=',')
#     np.savetxt(f'../csv/level_11_activations.csv',
#                level_11_activations.numpy(), delimiter=',')
