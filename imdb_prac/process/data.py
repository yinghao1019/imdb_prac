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
        # load data
        doc_ids=[]
        doc_ents=[]
        text,label = self.data.iloc[idx, :2].values
        label = LABEL_MAPPING[label]

        # split doc to sent & nlp preprocess for each sent
        for n, s in enumerate(text.strip().split("\n")):
            # avoid exceed max sent num
            if n > self.sent_num:
                break
            sent, ents = nlp_preprocess(s,self.tokenizer)

            if sent or ents:
                sent, ents = nlp_potsprocess(sent[:self.max_token],ents[:self.max_token], self.tokenizer)

            sent = list(sent)+[0]*(self.max_token-len(sent))
            ents = list(ents)+[0]*(self.max_token-len(ents))

            doc_ids.append(sent)
            doc_ents.append(ents)
        # convert to tensor
        doc_ids = torch.tensor(doc_ids, dtype=torch.long)
        doc_ents = torch.tensor(doc_ents, dtype=torch.long)

        return Example(text=doc_ids, ents=doc_ents, label=label)

    def __len__(self):
        return self.data.shape[0]


def collate_batch(examples):
    texts, ents, labels = zip(*examples)

    # pad sentnum for doc
    texts = pad_sequence(texts, batch_first=True)
    ents = pad_sequence(ents, batch_first=True)

    # detect & handle dataset which has label or not
    if labels[0] is not None:
        labels = torch.tensor(labels)
    else:
        labels = None

    return {"text": texts, "ents": ents, "labels": labels}







