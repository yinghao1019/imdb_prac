from torch.nn.utils.rnn import pad_sequence
from google.cloud.storage import Blob
from torch.utils.data import Dataset
from collections import namedtuple
import pandas as pd

import logging
import torch
import os

from imdb_prac.process.text import nlp_preprocess, nlp_potsprocess
from imdb_prac.settings import LABEL_MAPPING

logger = logging.getLogger(__name__)

Example = namedtuple("Example", ['text', 'ents', 'label'])


class ImdbDataset(Dataset):
    def __init__(self, data, tokenizer, sent_num, max_token):
        self.data = data
        self.tokenizer = tokenizer
        self.sent_num = sent_num
        self.max_token = max_token

    def __getitem__(self, idx):
        doc = []
        doc_ent = []
        #load data
        text, label = self.data.iloc[idx, :2].values
        label = LABEL_MAPPING[label]

        # split doc to sent & nlp preprocess for each sent
        for n, s in enumerate(text.strip().split("\n")):
            # avoid exceed max sent num
            if n > self.sent_num:
                break
            sent, ents = nlp_preprocess(s, self.tokenizer)
            sent, ents = nlp_potsprocess(
                sent[:self.max_token], ents[:self.max_token], self.tokenizer)

            # convert to tensor
            sent = torch.tensor(sent, dtype=torch.short)
            ents = torch.tensor(ents, dtype=torch.uint8)
            doc.append(sent)
            doc_ent.append(ents)

        # tensor shape=[sent_n,self.max_token]
        doc = pad_sequence(doc, batch_first=True)
        doc_ent = pad_sequence(doc_ent, batch_first=True)

        return Example(text=doc, ents=doc_ent, label=label)

    def __len__(self):
        return self.data.shape[0]


def collate_batch(examples):
    texts, ents, labels = zip(*examples)

    # pad sentnum for doc
    texts = pad_sequence(texts, batch_first=True)
    ents = pad_sequence(ents, batch_first=True)

    # detect & handle dataset which has label or not
    if labels[0]:
        labels = torch.tensor(labels)
    else:
        labels = None

    return {"text": texts, "ents": ents, "labels": labels}


