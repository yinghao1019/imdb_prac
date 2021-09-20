from torch import nn as NN
from torch.nn import functional as F
import torch
from torch.nn.modules.linear import Linear


class TextEncoder(NN.Module):
    def __init__(self, input_dim, hid_dim):
        super(TextEncoder, self).__init__()

        self.rnn1 = NN.GRU(input_dim, hid_dim, batch_first=True)
        self.rnn2 = NN.GRU(input_dim+hid_dim, hid_dim, batch_first=True)
        self.atn = NN.Linear(hid_dim, 1)
        self.tanh = NN.Tanh()

    def forward(self, inputs):
        # inputs={Bs,doc/sent len,embed}

        # outputs=[Bs,doc/sent len/hid_dim]
        hids, _ = self.rnn1(inputs)
        # concatenate input to agument Repr.
        hids, _ = self.rnn2(torch.cat((hids, inputs), dim=-1))

        # summarize Repr by
        # compute attention weight
        # weights={Bs,doc/sent len,1}
        weights = self.tanh(self.atn(hids))
        weights = F.softmax(weights, dim=-1)
        # attn_c=[Bs,hid_dim]
        attn_c = torch.matmul(torch.movedim(weights, 2, 1), hids).squeeze(1)

        # compute max pooling
        # max_c=[Bs,hid_dim]
        max_c = F.max_pool1d(torch.transpose(hids, 1, 2),
                             kernel_size=hids.shape[1]).squeeze(2)

        # outputs=[Bs,hid_dim*2]
        return torch.cat((attn_c, max_c), dim=1)


class HierAttentionRNN(NN.Module):
    def __init__(self, input_num, ent_num, input_embed, ent_embed,
                 hid_dim, output_dim, n_layers, pad_idx):

        super(HierAttentionRNN, self).__init__()
        assert hid_dim % 2 == 0, "Hid dim must be even"

        self.rnn_output_dim = hid_dim*2

        self.tok_embed = NN.Embedding(
            input_num, input_embed, padding_idx=pad_idx)
        self.ent_embd = NN.Embedding(ent_num, ent_embed, padding_idx=pad_idx)
        self.sent_encoder = TextEncoder(input_embed+ent_embed, hid_dim)
        self.doc_encoder = TextEncoder(self.rnn_output_dim, hid_dim)

        self.fc_layers = NN.ModuleList([NN.Linear(int(self.rnn_output_dim/pow(2, i)), int(self.rnn_output_dim/pow(2, i+1)))
                                       for i in range(n_layers)])
        self.output_layer = NN.Linear(
            int(self.rnn_output_dim/pow(2, n_layers)), 1)

        self.relu = NN.ReLU()
        self.loss_fn = NN.BCEWithLogitsLoss()

    def forward(self, text, ents, labels=None):

        Bs, sent_num = text.size()[:2]
        # texts,ents=[Bs,sent_num,sent_len]
        # labels=[Bs,]

        # convert idx to embed and combined
        # text/ent embed=[Bs,sent_num,sent_len,embd_dim]
        text_embed = self.tok_embed(text)
        ent_embed = self.ent_embd(ents)

        # Bs & sent num to
        # Embed=[Bs*sent_num,sent_len,embed_dim]
        embed = torch.cat((text_embed, ent_embed), dim=3)
        embed = embed.reshape((-1,)+embed.size()[2:])

        # encoding token that in sent to context by using self-attention & Max pooling
        # hiddens=[Bs*sent_num,hid_dim*2]
        hiddens = self.sent_encoder(embed)

        # encoding sent that in doc to context
        # hiddens=[Bs,hid_dim*2]
        assert hiddens.size()[0]/Bs == sent_num, ""
        hiddens = hiddens.reshape((Bs, -1, hiddens.size()[-1]))
        hiddens = self.doc_encoder(hiddens)

        # ffn to extract info.
        for l in self.fc_layers:
            hiddens = self.relu(l(hiddens))

        # compute output prob
        # outputs=[Bs,]
        outputs = self.output_layer(hiddens).squeeze(1)

        loss = None
        if labels is not None:
            loss = self.loss_fn(outputs, labels.float())

        return outputs, loss
