from torch import nn
import torch

class CNNHead(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(512,128,3,padding=1),nn.ReLU(),nn.Conv1d(128,64,3,padding=1),nn.ReLU());self.fc=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,6))
    def forward(self,x):return self.fc(self.conv(x.transpose(1,2)).mean(-1))

class LSTMHead(nn.Module):
    def __init__(self,bi=False,attention=False):
        super().__init__();self.rnn=nn.LSTM(512,64,2,batch_first=True,bidirectional=bi,dropout=.2)
        width=128 if bi else 64
        self.att=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,1)) if attention else None
        self.fc=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,6))
    def forward(self,x):
        z,_=self.rnn(x)
        z=(z*self.att(z).softmax(dim=1)).sum(1) if self.att else z[:,-1]
        return self.fc(z)

class TransformerHead(nn.Module):
    def __init__(self):
        super().__init__();self.embed=nn.Linear(512,128);self.pos=nn.Parameter(torch.zeros(1,16,128))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,dim_feedforward=256,dropout=.1,batch_first=True),num_layers=2,enable_nested_tensor=False)
        self.fc=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,6))
    def forward(self,x):return self.fc(self.encoder(self.embed(x)+self.pos).mean(1))

METHODS=['CNN','LSTM','Transformer','BiLSTM_Attention','cGAN_Transformer']
def make_head(name):
    if name=='CNN':return CNNHead()
    if name=='LSTM':return LSTMHead()
    if name=='BiLSTM_Attention':return LSTMHead(bi=True,attention=True)
    if name in ('Transformer','cGAN_Transformer'):return TransformerHead()
    raise ValueError(name)

