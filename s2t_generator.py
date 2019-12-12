import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from torch.autograd import Variable

class S2T_G(nn.Module):
    def __init__(self, config, norm_layer=nn.BatchNorm2d):
        super(S2T_G, self).__init__()

        # define a rnn sequence for T time steps
        # init hidden state with latent vector learnt from shape
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.latent_net = [
            nn.Conv3d(4, 64, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=0, bias=True),
            # norm_layer(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(128,256, kernel_size=4, stride=2, padding=0, bias=True),
            nn.Conv3d(256,256, kernel_size=4, stride=2, padding=0, bias=True)]

        self.latent_net = nn.Sequential(*self.latent_net)

        self.lstm = nn.LSTM(300,256,num_layers=1, bidirectional=False, batch_first=True)
        self.linear1 = nn.Linear(256, config.num_vocab)
        self.config = config
        # self.max_time_steps = 10

    def sample_gumbel(self,shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self,logits, temperature=0.5):
        y = F.log_softmax(logits, dim=-1) + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature=0.5):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward(self,input,vocab_embedding, max_time_steps):
        # print(max_time_steps)
        # print(max_time_steps.get_device())
        # import sys
        # sys.exit()
        vocab_embedding = vocab_embedding.squeeze(0)
        max_time_steps = int(max_time_steps.squeeze(0).item())
        # print("S2T_G")
        batch_size = input.size()[0]
        # print("Input : ", input)

        latent = self.latent_net(input)
        # print("Latent : ", latent.size())
        latent = latent.view(1, latent.size()[0], latent.size()[1])

        
        decoder_hidden = latent
        decoder_cell = latent.clone()
        decoder_input = torch.from_numpy(self.config.bos_embedding).float().to(self.config.device)
        decoder_input = decoder_input.view(1,1,decoder_input.size()[-1])
        decoder_input = decoder_input.repeat(batch_size, 1 ,1)

        # print(decoder_input.size())
        # import sys
        # sys.exit()

        all_outs=[]
        self.lstm.flatten_parameters()

        for t in range(max_time_steps):
            # print("decoder input :", decoder_input.size())
            # print("decoder hidden : ", decoder_hidden.size())
            # print("decoder cell : ", decoder_cell.size())

            output , (decoder_hidden, decoder_cell) = self.lstm(decoder_input, (decoder_hidden,decoder_cell))

            # gumbel softmax    
            output = output.squeeze(1)
            logits = self.linear1(output)
            gumbled = self.gumbel_softmax(logits)
            # print("Gumbled : ", gumbled.size())
            # print("vocab : ", vocab_embedding.size())

            # print("gumbled :", gumbled.size())
            # print("vocab_embedding : ", vocab_embedding.size())
            # import sys
            # sys.exit()
            decoder_input = torch.matmul(gumbled.detach(), vocab_embedding.detach())
            decoder_input = decoder_input.detach()
            decoder_input = decoder_input.view(batch_size,1,decoder_input.size()[-1])
            all_outs.append(gumbled)

        all_outs = torch.stack(all_outs, dim=1)
        # print("All outs : ", all_outs.size())
        # import sys
        # sys.exit()
        
        return all_outs