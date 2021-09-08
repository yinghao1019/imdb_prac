import torch
from torch.utils.data import Dataset
import os
import logging
from collections import namedtuple

from imdb_prac.process.text import nlp_preprocess,nlp_potsprocess

logger = logging.getLogger(__name__)

Example=namedtuple("Example",['text','ents','label'])

class ImdbDataset(Dataset):
    def __init__(self,data,tokenizer,mode,sent_num,max_token):
        self.data=data
        self.tokenizer=tokenizer
        self.sent_num=sent_num
        self.max_token=max_token
        self.mode=mode
    def __getitem__(self,idx):
        label=None
        doc=[]
        doc_ent=[]

        if self.mode=='train':
            text,label=self.data.iloc[idx,:2].values
        else:
            text=self.data.iloc[idx,:1].values

        #split doc to sent & nlp preprocess for each sent
        for n,s in enumerate(text.strip().split("\n")):
            #avoid exceed max sent num
            if n>self.sent_num:
                break
            sent,ents=nlp_preprocess(s,self.tokenizer)
            sent,ents=nlp_potsprocess(sent[:self.max_token],ents[:self.max_token],self.tokenizer)

            doc.append(sent)
            doc_ent.append(ents)
        
        return Example(text=doc,ents=doc_ent,label=label)
        
    def __len__(self):
        return self.data.shape[0]




    