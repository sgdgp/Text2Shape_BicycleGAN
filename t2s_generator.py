import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import functools


class T2S_G(nn.Module):
    def __init__(self, config, embedding_dim, hidden_dim, data, num_vocab):
        super(T2S_G, self).__init__()
        self.config = config
        self.embedding_dim = embedding_dim
        self.hidden_dim = 256
        
        self.embedding_layer = nn.Embedding(num_vocab, self.embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(data.emb_weight_matrix))
        self.embedding_layer.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        
        self.fc1 = nn.Linear(256, 128, bias=True)
        self.fc2 = nn.Linear(128, 32768, bias=True)

        self.conv_trans1 = nn.ConvTranspose3d(512, 512, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm3d(512)
        self.conv_trans2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv_trans3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv_trans4 = nn.ConvTranspose3d(128, 4, kernel_size=4, stride=2, padding=19)
        

    def get_embedding_weight_as_tensor(self):
        # print(type(self.embedding_layer.weight.data))
        # print(self.embedding_layer.weight.data.size())
        # import sys
        # sys.exit()
        # if self.config.data_parallel:
        #     return self.module.embedding_layer.weight.data
        # else:
        return self.embedding_layer.weight.data

    def forward(self, input):
        # print("T2S_G")
        # print(input.size())
        # print("input = ", input)
        embedding = self.embedding_layer(input)
        self.lstm.flatten_parameters()
        output, hidden_cell = self.lstm(embedding)

        hidden = hidden_cell[0]
        # print(hidden.size())
        hidden = hidden.squeeze(0)
        # print("hidden = ", hidden)
        
        # generate shape from this hidden which is the latent representation
        latent = self.fc2(self.fc1(hidden))
        # print("latent = ", latent)
        
        latent = latent.view(latent.size()[0], 512, 4, 4, 4)
        c1 = F.relu(self.bn1(self.conv_trans1(latent)))
        # print("c1 = ", c1)
        c2 = F.relu(self.bn2(self.conv_trans2(c1)))
        # print("c2 = ", c2)
        c3 = F.relu(self.bn3(self.conv_trans3(c2)))
        # print("c3 = ", c3)
        # c4 = F.relu(self.conv_trans4(c3))
        c4 = torch.sigmoid(self.conv_trans4(c3))
        # print("c4 = ", c4)
        
        # print(c4.size())
        

        return c4        

