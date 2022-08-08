import math
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim    
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()] 
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)


class EnvMOQTransformer(nn.Module):
    def __init__(self, num_hid, cond_dim, max_len, vocab_size, num_actions, num_obj, dropout, num_layers,
                num_head, **kwargs):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.cond_embed = nn.Linear(cond_dim, num_hid)
        self.embedding = nn.Embedding(vocab_size, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        # self.output = nn.Linear(num_hid + num_hid, num_actions)
        self.output = MLP(num_hid + num_hid, num_actions * num_obj, [4 * num_hid, 4 * num_hid], dropout)
        self.num_hid = num_hid
        self.num_actions = num_actions
        self.num_obj = num_obj

    def H(self, Q, w, s_num, w_num):
        # mask for reordering the batch
        mask = torch.cat(
            [torch.arange(i, s_num * w_num + i, s_num)
             for i in range(s_num)]).long()
        reQ = Q.view(-1, self.num_actions * self.num_obj
                     )[mask].view(-1, self.num_obj)

        # extend Q batch and preference batch
        reQ_ext = reQ.repeat(w_num, 1)
        w_ext = w.unsqueeze(2).repeat(1, self.num_actions * w_num, 1)
        w_ext = w_ext.view(-1, self.num_obj)

        # produce the inner products
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze()

        # mask for take max over actions and weights
        prod = prod.view(-1, self.num_actions * w_num)
        inds = prod.max(1)[1]
        mask = ByteTensor(prod.size()).zero_()
        mask.scatter_(1, inds.data.unsqueeze(1), 1)
        mask = mask.view(-1, 1).repeat(1, self.num_obj)

        # get the HQ
        HQ = reQ_ext.masked_select(Variable(mask)).view(-1, self.num_obj)

        return HQ

    def forward(self, x, cond, prefs, w_num=1):
        s_num = int(prefs.size(0) / w_num)
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x)
        pooled_x = x[0, :]
        cond_var = self.cond_embed(cond) # batch x hidden_dim
        
        q = self.output(torch.cat((pooled_x, cond_var), axis=-1))

        q = q.view(q.size(0), self.num_actions, self.num_obj)

        hq = self.H(q.detach().view(-1, self.num_obj), prefs, s_num, w_num)

        return hq, q


# Taken from the PyTorch Transformer tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)