import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_dim, n_layers, nclusters=None, dropout=0.5):
        super(RNNModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.nclusters = nclusters

        # encoder layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout,
                             batch_first=True)#, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim, encoder_dim)

        # decoder layers
        self.fc2 = nn.Linear(encoder_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, input_dim, n_layers, dropout=dropout,
                             batch_first=True)


        self.cluster_layer = Parameter(torch.Tensor(self.nclusters, 50))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    def forward(self, x, hidden1=None, hidden2=None):
        batch_size = x.size(0)

        # decode
        lstm_dec, hidden1 = self.lstm1(x, hidden1)
        lstm_dec = lstm_dec.contiguous().view(-1, self.hidden_dim)
        dec = self.dropout(lstm_dec)
        dec = F.relu(self.fc1(dec))

        # encode
        enc = F.relu(self.fc2(dec))
        enc = self.dropout(enc)

        lstm_enc = enc.view(batch_size, -1, self.hidden_dim)
        lstm_enc, hidden2 = self.lstm2(lstm_enc, hidden2)

        q = 1.0 / (1.0 + torch.sum(torch.pow(dec.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return lstm_enc, dec.unsqueeze(1), q

    def init_hidden(self, batch_size, gpu=True):
        weight = next(self.parameters()).data

        if (gpu):
            hidden1 = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
            hidden2 = (weight.new(self.n_layers, batch_size, self.input_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.input_dim).zero_().cuda())
        else:
            hidden1 = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
            hidden2 = (weight.new(self.n_layers, batch_size, self.input_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.input_dim).zero_())

        return hidden1, hidden2

'''
import torch

x = torch.randn(28, 1, 286)
model = RNNModel(286, 250, 50, 3)
a, b, c = model(x, None, None)
print(b.shape)
'''
