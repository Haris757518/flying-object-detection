"""Shared recurrent heads; checkpoints must specify input basis and six-class order."""
import torch
from torch import nn

class RecurrentClassifier(nn.Module):
    def __init__(self, input_dim, bidirectional=False, attention=False, hidden_dim=64, num_classes=6):
        super().__init__()
        self.rnn=nn.LSTM(input_dim,hidden_dim,num_layers=2,batch_first=True,
                         bidirectional=bidirectional,dropout=.2)
        width=hidden_dim*(2 if bidirectional else 1)
        self.attention=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,1)) if attention else None
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,num_classes))

    def forward(self,x):
        sequence,_=self.rnn(x)
        pooled=(sequence*self.attention(sequence).softmax(1)).sum(1) if self.attention is not None else sequence[:,-1]
        return self.head(pooled)
