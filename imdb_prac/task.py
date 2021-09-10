from torch.optim.lr_scheduler import MultiplicativeLR, ExponentialLR
from torch.utils.data import DataLoader, SequentialSampler
from google.cloud.storage import Client, Blob
from torch.optim import Adam
import pandas as pd
import argparse
import torch

from imdb_prac.utils import set_rnd_seed, init_logger, get_classMetrics, load_tokenizer
from imdb_prac.process.data import ImdbDataset, collate_batch
from imdb_prac.models.hsrnn import HierAttentionRNN
from imdb_prac.trainer.pipeline import ModelPipeline
from imdb_prac.settings import *


def main(args):
    # coonect to cloud
    client = Client(PROJECT_ID)

    # load train & test data from gcloud
    tokenizer = load_tokenizer(client, args.token_path)
    train_data = pd.read_csv(CLOUD_TRAINDATA_PATH)
    test_data = pd.read_csv(CLOUD_TESTDATA_PATH)

    # build torch dataset
    train_data = ImdbDataset(train_data, tokenizer,
                             args.max_sentN, args.max_token_num)
    test_data = ImdbDataset(test_data, tokenizer,
                            args.max_sentN, args.max_token_num)

    # build iterator
    train_iter = DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_batch)
    test_iter = DataLoader(test_data, batch_size=args.batch_size,
                           sampler=SequentialSampler(test_data), collate_fn=collate_batch)

    # initialze model & releated train tool
    model = HierAttentionRNN(MAX_VOCAB_SIZE, len(ENTITY_MAPPINGS.keys()), args.embed_dim,
                             args.ent_embed, args.hid_dim, args.output_dim, args.n_layers,
                             args.drop_rate, args.pad_idx)
    if torch.cuda.is_available():
        model.to(torch.device("cuda:0"))
    
    optimizer = Adam(model.parameters(), args.lr)
    lmbda=lambda epoch:epoch/args.warm_ep
    warm_scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    model_outputs = Blob.from_string(AIP_MODEL_DIR+"/model.pt", client)

    # build pipeline
    train_pipe = ModelPipeline(train_iter, test_iter, model, optimizer,
                               warm_scheduler, scheduler)

    # training
    train_pipe.amp_training(args.ep, args.save_step, model_outputs,
                            get_classMetrics, warm_step=args.warm_ep)


if __name__ == "__main__":
    init_logger()
    set_rnd_seed(1234)

    parser=argparse.ArgumentParser()
    parser.add_argument("--lr",default=1e-3,type=float,help="Init learning rate for update model weight.")
    parser.add_argument("--warm_ep",default=6,type=int,help="The epochs num that ")
    parser.add_argument("--ep",default=12,type=int,help="Total epoch nums for train.")
    parser.add_argument("--batch_size",default=64,type=int,help="The batch data num that iter in one epoch.")
    parser.add_argument("--save_step",default=4,type=int,help="Save model to gcloud in every N epochs.")

    parser.add_argument("--embed_dim",default=256,type=int,help="")
    parser.add_argument("--ent_embed",default=30,type=int,help="The embed dim for entity.")
    parser.add_argument("--hid_dim",default=256,type=int,help="The hid dim for rnn_layer.")
    parser.add_argument("--drop_rate",default=0.2,type=float,help="The rnn_layer dropout rate")
    parser.add_argument("--output_dim",default=1,type=int,help="Model output class num.")
    parser.add_argument("--n_layers",default=2,type=int,help="The nums of FFN layer that in model.")
    parser.add_argument("--pad_idx",default=0,type=int,help="The token/ent index for zero-value vector in embed matrix")

    args=parser.parse_args()

    main(args)
