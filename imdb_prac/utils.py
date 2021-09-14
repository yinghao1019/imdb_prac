from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from google.cloud.storage import Blob
from tokenizers import Tokenizer
import numpy as np
import pandas as pd
import logging
import random
import torch

def load_tokenizer(client,token_path):
    src=Blob.from_string(token_path,client)
    tokenizer=Tokenizer.from_str(src.download_as_text())
    return tokenizer
def load_data(client,data_path):
    src=Blob.from_string(data_path,client)
    with src.open("r",encoding="utf-8") as f_r:
        df=pd.read_csv(f_r,encoding="utf-8")
    return df
def count_parameters(model):
    """It's will count total params num for model."""
    return sum(p.numel()for p in model.parameters() if p.requires_grad)

# set random seed
def set_rnd_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True

def get_classMetrics(outputs, targets):
    assert targets.size()==outputs.size(),"The target'shape don't equal to output's"
    
    # convert to class idx
    binary_c = True if len(outputs.size()) == 1 else False
    eval_type = 'binary' if binary_c else 'weighted'
    # get correct label
    if binary_c:
        preds = torch.where(torch.sigmoid(outputs) >= 0.5, 1, 0)
        preds = preds.cpu().detach().numpy()
    else:
        preds = torch.argmax(outputs).cpu().detach().numpy()

    labels = targets.cpu().detach().numpy()  # get correct label
    print(labels)
    print(preds)
    acc = accuracy_score(labels, preds)
    f_score = f1_score(labels, preds, average=eval_type)
    precision = precision_score(labels, preds, average=eval_type)
    recall = recall_score(labels, preds, average=eval_type)

    return {
        'acc': acc, 'precision': precision,
        'f1_score': f_score, 'recall': recall,
    }



# set logger logging message
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt=r'%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
