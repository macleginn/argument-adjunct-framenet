import json
import pandas as pd
from tqdm.auto import tqdm

with open('top_50_frames_dev.json', 'r', encoding='utf-8') as inp:
# with open('top_50_frames_korean.json', 'r', encoding='utf-8') as inp:
# with open('all_frames_train.json', 'r', encoding='utf-8') as inp:
    data_dev = json.load(inp)
all_columns = set()

for s in tqdm(data_dev, desc='Collecting FEs'):
    for el in s:
        if el[1] is not None:
            all_columns.add(el[1])

columns = sorted(all_columns)
fe_binary = pd.DataFrame(index=list(range(len(data_dev))),
                         columns=columns).fillna(0).astype(int)
for i, s in tqdm(list(enumerate(data_dev)), desc='Filling the matrix'):
    for el in s:
        if el[1] is not None:
            fe_binary.loc[i, el[1]] = 1
fe_binary.to_csv('fe_binary_matrix.csv', index=False)
