import json
from glob import glob
from xml.etree import ElementTree as etree
import numpy as np
from tqdm.auto import tqdm
from FrameNetLib import *

# with open('top_50_frames.json', 'r', encoding='utf-8') as inp:
#     major_frames = json.load(inp)

elements_by_frame = {}
for path in tqdm(
    glob('../data/fndata-1.7/frame/*.xml'),
    desc='Frame metadata', leave=False
):
    frame_tree = etree.parse(path).getroot()
    frame_data = extract_FEs_for_frame(frame_tree)
    elements_by_frame[frame_data['frame']] = frame_data['frame_elements']

# with open('elements_by_frame_top_50.json', 'w', encoding='utf-8') as out:
#     json.dump(elements_by_frame, out, indent=2)

annotated_sentences = []
for path in tqdm(glob('../data/fndata-1.7/lu/*.xml'), desc='Lexical units', leave=False):
    lu_tree = etree.parse(path).getroot()
    if lu_tree.attrib['POS'] != 'V':
        continue
    frame = lu_tree.attrib['frame']
    # if frame not in major_frames:
    #     continue
    for child in lu_tree:
        if child.tag.endswith('subCorpus'):
            for sentence in child:
                try:
                    tokens, labels = prepare_sentence(
                        extract_FEs(sentence), frame, elements_by_frame[frame])
                    annotated_sentences.append(list(zip(tokens, labels)))
                except NoManualAnnotation:
                    continue
                except FeeFeCollision:
                    continue

n_dev = len(annotated_sentences) // 10
n_train = len(annotated_sentences) - n_dev
indicators = np.array([0] * n_train + [1] * n_dev)
assert len(indicators) == len(annotated_sentences)
np.random.shuffle(indicators)
data_train = []
data_dev = []
for sentence, i in zip(annotated_sentences, indicators):
    if i:
        data_dev.append(sentence)
    else:
        data_train.append(sentence)
with open('all_frames_train.json', 'w', encoding='utf-8') as out:
    json.dump(data_train, out, indent=2)
with open('all_frames_dev.json', 'w', encoding='utf-8') as out:
    json.dump(data_dev, out, indent=2)
