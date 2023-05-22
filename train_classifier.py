import json
from math import ceil
from random import shuffle

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_scheduler

from tqdm.auto import tqdm


class ClassificationHead(nn.Module):
    def __init__(self, n_classes, input_size=768):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 768)
        self.linear2 = nn.Linear(768, n_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return self.linear2(x)


class ClassificationHeadSimple(nn.Module):
    def __init__(self, n_classes, input_size=768):
        super().__init__()
        self.linear = nn.Linear(input_size, n_classes)

    def forward(self, x):
        return self.linear(x)


def get_tokens_with_ranges(gold_tokens, tokeniser):
    '''
    In the normal case, BERT adds "##" before token
    pieces that do not start a new token. However, it does
    not do this for kanji. Cf.:

    > tokeniser.tokenize('chronomoscopy')
    ['ch', '##rono', '##mos', '##co', '##py']

    > tokeniser.tokenize('米国')
    ['米', '国']

    Therefore we need to build the tokenised sequence
    one word at a time.
    '''

    tokens = ['<s>']
    ranges = [(0, 1)]
    for gold_token in gold_tokens:
        range_start = len(tokens)
        range_end = range_start
        for subtoken in tokeniser.tokenize(gold_token):
            range_end += 1
            tokens.append(subtoken)
        ranges.append((range_start, range_end))
    ranges.append((len(tokens), len(tokens) + 1))
    tokens.append('</s>')
    return tokens, ranges


def get_tokens_with_labels(gold_tokens, labels, tokeniser):
    assert len(gold_tokens) == len(labels)
    out_tokens = [tokeniser.cls_token_id]
    out_labels = [None]
    for t, l in zip(gold_tokens, labels):
        if len(out_tokens) == 511:
            break
        tok_ids = tokeniser.encode(t, add_special_tokens=False)
        out_labels.append(l)
        out_tokens.append(tok_ids[0])
        if len(out_tokens) == 511:
            break
        for tok_id in tok_ids[1:]:
            out_tokens.append(tok_id)
            out_labels.append(None)
            if len(out_tokens) == 511:
                break
    out_tokens.append(tokeniser.sep_token_id)
    out_labels.append(None)
    assert len(out_tokens) == len(out_labels)
    seq_length = len(out_tokens)
    padding_length = 512 - seq_length
    tokeniser_output = {
        'input_ids': torch.tensor(out_tokens + [0] * padding_length),
        'token_type_ids': torch.tensor([0] * seq_length +
        [tokeniser._pad_token_type_id] * padding_length),
        'attention_mask': torch.tensor([1] * seq_length + [0] * padding_length)
    }
    return tokeniser_output, out_labels


def prepare_batch(annotated_sentences, tokenizer, frame_to_class):
    FEE_labels=[]
    FEE_idx=[]
    batch_sentences=[]
    FEE_ranges=[]
    CE_ranges=[]
    NCE_ranges=[]
    ETE_ranges=[]
    for s in annotated_sentences:
        CE_range_buffer=[]
        NCE_range_buffer=[]
        ETE_range_buffer=[]
        gold_tokens=[p[0] for p in s]
        gold_labels=[p[1] for p in s]
        _, ranges=get_tokens_with_ranges(gold_tokens, tokenizer)
        ranges=ranges[1:-1]
        assert len(ranges) == len(gold_tokens)
        FEE_found=False
        for i, label in enumerate(gold_labels):
            if label is not None:
                if label.startswith('FRAME:'):
                    # First token from the FEE.
                    # Each sentence contributes only one range.
                    if FEE_found:
                        continue
                    FEE_found=True
                    FEE_labels.append(frame_to_class[label.split(':')[-1]])
                    FEE_idx.append(ranges[i][0])
                    FEE_ranges.append(ranges[i])
                # For masking CEs and NCEs. More than one range can come
                # from a single sentence.
                elif label.startswith('CORE:'):
                    CE_range_buffer.append(ranges[i])
                elif label.startswith('NONCORE:'):
                    NCE_range_buffer.append(ranges[i])
                elif label.startswith('EXTRATHEMATIC:'):
                    ETE_range_buffer.append(ranges[i])
        if not FEE_found:
            print(f'No FEE was found in: {s}.')
            continue
        CE_ranges.append(CE_range_buffer)
        NCE_ranges.append(NCE_range_buffer)
        ETE_ranges.append(ETE_range_buffer)
        batch_sentences.append(' '.join(gold_tokens))
    return FEE_labels, FEE_idx, batch_sentences, FEE_ranges, CE_ranges, NCE_ranges, ETE_ranges


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        mask=None
    else:
        mask=sys.argv[1]
        assert mask in {'fee', 'ce', 'nce'}, f'Wrong mask: {mask}.'

    model_name="bert-base-multilingual-cased"
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    bert_model=AutoModel.from_pretrained(model_name)
    bert_model.cuda()
    bert_model=nn.DataParallel(bert_model)

    n_classes=50
    classification_head=ClassificationHeadSimple(n_classes)
    classification_head.cuda()
    classification_head=nn.DataParallel(classification_head)

    optimizer=AdamW(list(bert_model.parameters()) +
                      list(classification_head.parameters()), lr=1e-5)

    with open('top_50_frames_train.json', 'r', encoding='utf-8') as inp:
        data_train=json.load(inp)
    shuffle(data_train)
    with open('top_50_frames_dev.json', 'r', encoding='utf-8') as inp:
        data_dev=json.load(inp)
    shuffle(data_dev)
    with open('top_50_frames.json', 'r', encoding='utf-8') as inp:
        major_frames=json.load(inp)
    frame_to_class={frame: i for i, frame in enumerate(major_frames)}

    n_epochs=20
    n_training_steps=n_epochs * len(data_train)
    lr_scheduler=get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=n_training_steps
    )
    loss_function=nn.CrossEntropyLoss()

    batch_size=384

    min_dev_loss=float('inf')
    for epoch in range(n_epochs):
        # Train
        bert_model.train()
        epoch_train_losses=[]
        n_steps_train=ceil(len(data_train) / batch_size)
        for batch_n in tqdm(range(n_steps_train), desc=f'Epoch {epoch+1}, train'):
            annotated_sentences=data_train[batch_size *
                                             batch_n: batch_size * (batch_n + 1)]
            FEE_labels, FEE_idx, batch_sentences, FEE_ranges, CE_ranges, NCE_ranges=prepare_batch(
                annotated_sentences, tokenizer, frame_to_class)
            mbert_inputs={
                k: v.cuda() for k, v in tokenizer(
                    batch_sentences, padding=True, truncation=True, return_tensors='pt'
                ).items()
            }
            if mask == 'FEE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for j in range(FEE_ranges[i][0], FEE_ranges[i][1]):
                        mbert_inputs['input_ids'][i][j]=103
            elif mask == 'CE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for CE_range in CE_ranges[i]:
                        for j in range(CE_range[0], CE_range[1]):
                            mbert_inputs['input_ids'][i][j]=103
            elif mask == 'NCE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for NCE_range in NCE_ranges[i]:
                        for j in range(NCE_range[0], NCE_range[1]):
                            mbert_inputs['input_ids'][i][j]=103
            mbert_outputs=bert_model(**mbert_inputs).last_hidden_state
            FEE_embeddings=[mbert_outputs[i, FEE_idx[i]]
                              for i in range(mbert_outputs.size(0))]
            FEE_embeddings=torch.vstack(FEE_embeddings)
            logits=classification_head(FEE_embeddings)
            loss=loss_function(logits, torch.tensor(FEE_labels).cuda())
            loss.backward()
            epoch_train_losses.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        print(f'Epoch train loss: {torch.tensor(epoch_train_losses).mean()}')

        # Validate
        bert_model.eval()
        epoch_dev_losses=[]
        n_steps_dev=ceil(len(data_dev) / batch_size)
        for batch_n in tqdm(range(n_steps_dev), desc=f'Epoch {epoch+1}, validate'):
            annotated_sentences=data_dev[batch_size *
                                           batch_n: batch_size * (batch_n + 1)]
            FEE_labels, FEE_idx, batch_sentences, FEE_ranges, CE_ranges, NCE_ranges=prepare_batch(
                annotated_sentences, tokenizer, frame_to_class)
            mbert_inputs={
                k: v.cuda() for k, v in tokenizer(
                    batch_sentences, padding=True, truncation=True, return_tensors='pt'
                ).items()
            }
            if mask == 'FEE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for j in range(FEE_ranges[i][0], FEE_ranges[i][1]):
                        mbert_inputs['input_ids'][i][j]=103
            elif mask == 'CE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for CE_range in CE_ranges[i]:
                        for j in range(CE_range[0], CE_range[1]):
                            mbert_inputs['input_ids'][i][j]=103
            elif mask == 'NCE':
                for i in range(len(mbert_inputs['input_ids'])):
                    for NCE_range in NCE_ranges[i]:
                        for j in range(NCE_range[0], NCE_range[1]):
                            mbert_inputs['input_ids'][i][j]=103
            with torch.no_grad():
                mbert_outputs=bert_model(**mbert_inputs).last_hidden_state
                FEE_embeddings=[mbert_outputs[i, FEE_idx[i]]
                                  for i in range(mbert_outputs.size(0))]
                FEE_embeddings=torch.vstack(FEE_embeddings)
                logits=classification_head(FEE_embeddings)
                loss=loss_function(logits, torch.tensor(FEE_labels).cuda())
            epoch_dev_losses.append(loss.item())
        mean_dev_loss=torch.tensor(epoch_dev_losses).mean()
        print(f'Epoch dev loss: {mean_dev_loss}')
        if mean_dev_loss < min_dev_loss:
            min_dev_loss=mean_dev_loss
            print('Saving the model.')
            torch.save(bert_model.state_dict(),
                       f"../checkpoints/mbert_masked-{mask}_linear.pt")
            torch.save(classification_head.state_dict(),
                       f"../checkpoints/head_masked-{mask}_linear.pt")
        print()
