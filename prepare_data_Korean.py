import json
from glob import glob
from multiprocessing.sharedctypes import Value
from xml.etree import ElementTree as etree
import numpy as np
from tqdm.auto import tqdm
from FrameNetLib import *

with open('elements_by_frame.json', 'r', encoding='utf-8') as inp:
    elements_by_frame = json.load(inp)

with open('korean_framenet_data.json', 'r', encoding='utf-8') as inp:
    korean_raw_data = json.load(inp)

annotated_sentences = []
for tokens, target, frame, arguments in korean_raw_data:
    buffer = []
    for f in frame:
        if f != '_':
            current_frame = f
            break
    else:
        raise ValueError('No frame')
    fe_types = {}
    for fe in elements_by_frame[current_frame]['core']:
        fe_types[fe] = 'CORE:'
    for fe in elements_by_frame[current_frame]['non-core']:
        fe_types[fe] = 'NONCORE:'
    for t, f, arg in zip(tokens, frame, arguments):
        if f != '_':
            buffer.append([t, f'FRAME:{f}'])
        elif arg != 'O':
            arg = arg.split('-')[1]
            try:
                buffer.append([t, f'{fe_types[arg]}{arg}'])
            except KeyError:
                print(f'Arg {arg} of {current_frame} unrecognised')
                buffer.append([t, f'NONCORE:{arg}'])
        else:
            buffer.append([t, None])
    annotated_sentences.append(buffer)
with open('top_50_frames_korean.json', 'w', encoding='utf-8') as out:
    json.dump(annotated_sentences, out, indent=2, ensure_ascii=False)


# for path in tqdm(glob('../data/fndata-1.7/lu/*.xml'), desc='Lexical units', leave=False):
#     lu_tree = etree.parse(path).getroot()
#     if lu_tree.attrib['POS'] != 'V':
#         continue
#     frame = lu_tree.attrib['frame']
#     if frame not in major_frames:
#         continue
#     for child in lu_tree:
#         if child.tag.endswith('subCorpus'):
#             for sentence in child:
#                 try:
#                     tokens, labels = prepare_sentence(
#                         extract_FEs(sentence), frame, elements_by_frame[frame])
#                     annotated_sentences.append(list(zip(tokens, labels)))
#                 except NoManualAnnotation:
#                     continue
#                 except FeeFeCollision:
#                     continue
# n_dev = len(annotated_sentences) // 10
# n_train = len(annotated_sentences) - n_dev
# indicators = np.array([0] * n_train + [1] * n_dev)
# assert len(indicators) == len(annotated_sentences)
# np.random.shuffle(indicators)
# data_train = []
# data_dev = []
# for sentence, i in zip(annotated_sentences, indicators):
#     if i:
#         data_dev.append(sentence)
#     else:
#         data_train.append(sentence)
# with open('top_50_frames_train.json', 'w', encoding='utf-8') as out:
#     json.dump(data_train, out, indent=2)
# with open('top_50_frames_dev.json', 'w', encoding='utf-8') as out:
#     json.dump(data_dev, out, indent=2)
