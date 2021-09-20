from google.cloud.storage import bucket
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Lowercase, StripAccents
from google.cloud import storage
from tqdm import tqdm
import pandas as pd
import logging
import argparse

import os

from imdb_prac.settings import *
from imdb_prac.process.text import rm_punct, strip_html, en_nlp,nlp_preprocess
from imdb_prac.utils import load_data,load_tokenizer

# get module name to log message
logger = logging.getLogger(__name__)


def generate_bpe(data_iter, min_freq, max_size, blob):

    # build init tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix="##",
                              end_of_word_suffix="##"))
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], min_frequency=min_freq,
                         vocab_size=max_size)

    # insert text clean process
    norm_pipe = Sequence([StripAccents(), Lowercase()])
    tokenizer.normalizer = norm_pipe
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(data_iter, trainer, len(data_iter))

    # save trained tokenizer to gcloud
    model_params = tokenizer.to_str(pretty=True)
    blob.upload_from_string(model_params)


def fetch_raw_data(folder_path, label, data_num):
    i = 0
    polar_path = path.join(folder_path, label)
    for f_name in tqdm(os.listdir(polar_path), desc="nums:"):
        data = []
        with open(os.path.join(polar_path, f_name), "r", encoding='utf-8') as f_r:
            doc = strip_html(f_r.read().strip())
            doc = en_nlp(rm_punct(doc))
            # seprate line and add symbol for split
            doc = "\n".join(list(str(s) for s in doc.sents))
            data.append(doc)
        # add label
        data.append(label)
        if i > data_num:
            break
        logger.info(i)
        i += 1
        yield data


def create_csv(src_path, dest_path, data_num):
    if not path.isfile(dest_path):
        logger.info(f"------Start to get data from folder:{src_path}-------")
        datas = list(d for l in ['pos', 'neg']
                     for d in fetch_raw_data(src_path, l, data_num))
        df = pd.DataFrame(datas, columns=['text', 'label'])
        df = df.sample(frac=1)
        df.to_csv(dest_path, index=False, encoding="utf-8")
    else:
        logger.info(f"Data already exists in {dest_path}")






def main(args):
    client = storage.Client(PROJECT_ID)
    bucket = client.get_bucket(BUCKET_NAME)

    if args.subparser_name == "data":
        load_path = os.path.join(args.src, args.type)
        create_csv(load_path, args.dst, args.num)
        cloud_path = "/".join(args.dst.split("\\")[1:])
        blob = bucket.blob(cloud_path)
        blob.upload_from_filename(args.dst)

    elif args.subparser_name == "token":
        # load from gcloud
        df=load_data(client,args.src)

        data_iter = list(sent for r_v in df["text"]
                         for sent in r_v.split("\n") if sent)

        # train & save to gcloud
        blob = bucket.blob(args.dst)
        generate_bpe(data_iter, args.minFreq, args.maxSize, blob)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommand", dest="subparser_name",
                                       description="data: used for produce data and tokenizer",
                                       help="addition help")
    parser_data = subparsers.add_parser("data", help="data help")
    parser_data.add_argument("src", type=str, help="The Data file path that you want to processed with.\
                             Ex:Data/path/to/file")
    parser_data.add_argument("dst", type=str, help="The path for processed data output.")
    parser_data.add_argument("-n", "--num", type=int,
                             help="The data num you want to extract.")
    parser_data.add_argument("--type", type=str, choices=["train", "test"],
                             help="convert spec data that used for train or testing.")

    # subcommand for training bpe tokenizer
    parser_token = subparsers.add_parser("token", help="token help")
    parser_token.add_argument("src", type=str, 
                              help="The path for training tokenizer's data on gcloud storage.")
    parser_token.add_argument("dst", type=str, 
                               help="THe path for save tokenizer on gcloud storage.")
    parser_token.add_argument('-mf', "--minFreq", type=int,
                              default=1, help="The word leaset freq in dataset.Default is 1.")
    parser_token.add_argument("-ms", "--maxSize", type=int,
                              default=25000, help="Max vocabulary size for tokenizer.Default is 25000")

    args = parser.parse_args()
    main(args)
