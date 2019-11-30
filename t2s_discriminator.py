import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
class T2S_D(nn.Module):
    def __init__(self, input_nc=4, ndf=64, norm_layer=nn.BatchNorm3d):
        super(T2S_D, self).__init__()

        # Pixel Discriminator

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)
        self.linear1 = nn.Linear(32*32*32, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, input):
        """Standard forward."""
        # print("T2S_D")
        # print(input.shape)
        out = self.net(input)
        out = out.view(input.size(0), -1)
        out = F.tanh(self.linear1(out))
        out = self.linear2(out)
        # print(out)
        # print(out.shape)
        return out