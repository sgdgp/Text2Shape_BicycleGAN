import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.autograd import Variable


class S2T_D(nn.Module):
    def __init__(self, embedding_dim = 300, hidden_dim=256):
        super(S2T_D, self).__init__()
        # the input is a sequence of prob values over vocab for each timestep (upto max timesteps T)
        # B x T x V
        use_bias = True
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,1)
    
        
    def forward(self, input, vocab_embedding):
        """Standard forward."""
        vocab_embedding = vocab_embedding.squeeze(0)
        # print("S2T_D")
        # print(input.size())
        if  len(input.size()) == 3 : # B x T x V
            x = torch.matmul(input, vocab_embedding)
        else:
            # print(input)
            # print(input.size())
            x = F.one_hot(input, num_classes=vocab_embedding.size()[0]).float()
            x = torch.matmul(x,vocab_embedding)
        
        # print("x = ", x)
        self.lstm.flatten_parameters()
        out,(hidden,cell) = self.lstm(x)
        hidden = hidden.squeeze(0).unsqueeze(1)
        # print("hidden = ", hidden)
        # score = F.relu(self.linear2(F.relu(self.linear1(hidden))))
        score = self.linear2(F.relu(self.linear1(hidden)))
        # score = torch.sigmoid(self.linear2(F.relu(self.linear1(hidden))))
        # print("score  ", score)
        # print(score.size())
        return score