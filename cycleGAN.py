import torch
import torch.nn as nn
import itertools
from t2s_generator import *
from s2t_generator import *
from t2s_discriminator import *
from s2t_discriminator import *
from losses import *
from torch.nn import DataParallel as DP

class T2S_CycleGAN(nn.Module):
    def __init__(self, config, data):
        super(T2S_CycleGAN, self).__init__()
        
        self.config = config
        self.device = config.device
        self.t2s_g = T2S_G(config, 300, 256, data, self.config.num_vocab).to(config.device)
        self.s2t_g = S2T_G(config=config).to(config.device)
        self.t2s_d = T2S_D().to(config.device)
        self.s2t_d = S2T_D().to(config.device)

        self.criterionGAN = GANLoss()
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()

        # self.optimizer_g = torch.optim.Adam(itertools.chain(self.t2s_g.parameters(), self.s2t_g.parameters()), lr=config.lr)
        # self.optimizer_d = torch.optim.Adam(itertools.chain(self.t2s_d.parameters(), self.s2t_d.parameters()), lr=config.lr)

        self.optimizer_g = torch.optim.RMSprop(itertools.chain(self.t2s_g.parameters(), self.s2t_g.parameters()), lr=config.lr)
        self.optimizer_d = torch.optim.RMSprop(itertools.chain(self.t2s_d.parameters(), self.s2t_d.parameters()), lr=config.lr)


    def make_data_parallel(self):
        self.t2s_g = DP(self.t2s_g)
        self.t2s_d = DP(self.t2s_d)
        self.s2t_g = DP(self.s2t_g)
        self.s2t_d = DP(self.s2t_d)

    def save_models(self, model_dict):
        if self.config.data_parallel:
            torch.save(self.t2s_g.module.state_dict(),model_dict["t2s_g"])
            torch.save(self.t2s_d.module.state_dict(),model_dict["t2s_d"])
            torch.save(self.s2t_g.module.state_dict(),model_dict["s2t_g"])
            torch.save(self.s2t_d.module.state_dict(),model_dict["s2t_d"])
        else:
            torch.save(self.t2s_g.state_dict(),model_dict["t2s_g"])
            torch.save(self.t2s_d.state_dict(),model_dict["t2s_d"])
            torch.save(self.s2t_g.state_dict(),model_dict["s2t_g"])
            torch.save(self.s2t_d.state_dict(),model_dict["s2t_d"])
    
    def load_models(self,model_dict):
        if self.config.data_parallel:
            self.t2s_g.module.load_state_dict(torch.load(model_dict["t2s_g"]))
            self.t2s_d.module.load_state_dict(torch.load(model_dict["t2s_d"]))
            self.s2t_g.module.load_state_dict(torch.load(model_dict["s2t_g"]))
            self.s2t_d.module.load_state_dict(torch.load(model_dict["s2t_d"]))
        else:
            self.t2s_g.load_state_dict(torch.load(model_dict["t2s_g"]))
            self.t2s_d.load_state_dict(torch.load(model_dict["t2s_d"]))
            self.s2t_g.load_state_dict(torch.load(model_dict["s2t_g"]))
            self.s2t_d.load_state_dict(torch.load(model_dict["s2t_d"]))
        
    def set_train(self):
        self.t2s_g.train()
        self.t2s_d.train()
        self.s2t_g.train()
        self.s2t_d.train()

    def set_eval(self):
        self.t2s_g.eval()
        self.t2s_d.eval()
        self.s2t_g.eval()
        self.s2t_d.eval()


    def forward(self, max_length, mode="train"):
        self.fake_s = self.t2s_g(self.real_t)  
        # print("fake_s shape : ", self.fake_s.size())
        vocab_embedding = None
        if self.config.data_parallel:
            vocab_embedding = self.t2s_g.module.get_embedding_weight_as_tensor().detach()
        else:
            vocab_embedding = self.t2s_g.get_embedding_weight_as_tensor().detach()

        # print("vocab_embedding :", vocab_embedding.size())
        vocab_embedding = vocab_embedding.unsqueeze(0)
        vocab_embedding = vocab_embedding.expand(self.config.num_gpus_active,-1,-1)

        # print(type(self.s2t_g))
        # print(self.fake_s.get_device())
        # print(self.fake_s.size())
        # print(vocab_embedding.get_device())
        # print(vocab_embedding.size())
        # print(max_length.get_device())
        # print(max_length.size())
        
        
        self.rec_t = self.s2t_g(self.fake_s, vocab_embedding, max_length) 

        # print(self.rec_t.size())
        
        
        # print("rec_t shape : ", self.rec_t.size())
        # self.fake_s = self.t2s_g(self.real_t)
    


    def t2s_pred(self):
        vocab_embedding = None
        # if self.config.data_parallel:
        #     vocab_embedding = self.t2s_g.module.get_embedding_weight_as_tensor().detach()
        # else:
        vocab_embedding = self.t2s_g.get_embedding_weight_as_tensor().detach()

        vocab_embedding = vocab_embedding.unsqueeze(0)
        vocab_embedding = vocab_embedding.expand(self.config.test_batch_size,-1,-1)

        self.fake_s = self.t2s_g(self.real_t)

        max_tensor = torch.from_numpy(np.array([6.0])).float().to(self.config.device)
        self.rec_t = self.s2t_g(self.fake_s, vocab_embedding, max_tensor)   
            

    def backward_D_basic(self, netD, real, fake, flag):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
            flag  = 1 for T2S and 0 for S2T
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        vocab_embedding = None
        if self.config.data_parallel:
            vocab_embedding = self.t2s_g.module.get_embedding_weight_as_tensor().detach()
        else:
            vocab_embedding = self.t2s_g.get_embedding_weight_as_tensor().detach()
        vocab_embedding = vocab_embedding.unsqueeze(0)
        vocab_embedding = vocab_embedding.expand(self.config.num_gpus_active,-1,-1)

        # Real
        pred_real = netD(real) if flag else netD(real,vocab_embedding)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())  if flag else netD(fake.detach(),vocab_embedding)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        print("loss_D_real =", loss_D_real)
        print("loss_D_fake =", loss_D_fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)
        loss_D.backward()
        return loss_D

    def backward_d_t2s(self):
        """Calculate GAN loss for discriminator D_A"""
        # fake_B = self.fake_B_pool.query(self.fake_B)
        print("Backward_d_T2S")
        self.loss_d_t2s = self.backward_D_basic(self.t2s_d, self.real_s, self.fake_s, 1)

        # grad_pen, grad = cal_gradient_penalty(self.t2s_d, self.real_s, self.fake_s, self.device, type='mixed', constant=1.0, lambda_gp=10.0)

        # self.loss_d_t2s += grad_pen 

    def backward_d_s2t(self):
        """Calculate GAN loss for discriminator D_B"""
        # detaching one hot conversion history as well
        print("Backward_d_S2T")
        self.loss_d_s2t = self.backward_D_basic(self.s2t_d, self.real_t, self.rec_t, 0)

    def backward_g(self):
        vocab_embedding = None
        if self.config.data_parallel:
            vocab_embedding = self.t2s_g.module.get_embedding_weight_as_tensor().detach()
        else:
            vocab_embedding = self.t2s_g.get_embedding_weight_as_tensor().detach()
        vocab_embedding = vocab_embedding.unsqueeze(0)
        vocab_embedding = vocab_embedding.expand(self.config.num_gpus_active,-1,-1)

        # GAN loss D_A(G_A(A))
        self.loss_g_t2s = self.criterionGAN(self.t2s_d(self.fake_s), True)
        self.loss_g_t2s_L1 = self.criterionL1(self.fake_s, self.real_s)
        self.loss_g_t2s_mse = self.criterionMSE(self.fake_s, self.real_s)

        # GAN loss D_B(G_B(B))
        self.loss_g_s2t = self.criterionGAN(self.s2t_d(self.rec_t,vocab_embedding), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        
        # target = F.one_hot(self.real_t, num_classes=77).float()
        # self.loss_cycle = -1 * torch.sum(target * self.rec_t) #the crossentropy formula is -1 * sum( log(output_dist) * target_dist)
        
        # combined loss and calculate gradients
        # self.loss_g = self.loss_g_t2s + self.loss_g_s2t + self.loss_g_t2s_L1 + self.loss_g_t2s_mse + self.loss_cycle
        # self.loss_g = self.loss_g_t2s + self.loss_g_s2t + self.loss_g_t2s_L1 + self.loss_cycle
        self.loss_g = self.loss_g_t2s + self.loss_g_s2t + self.loss_g_t2s_L1 + self.loss_g_t2s_mse
        self.loss_g.backward()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self, update_only_d, maxlen):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward(maxlen)      
        # print("Forward done")
        # self.loss_g_t2s = 0
        # self.loss_g_s2t = 0
        # self.loss_d_t2s = 0
        # self.loss_d_s2t = 0
        loss_list = []
        if not update_only_d:
            self.set_requires_grad([self.t2s_d, self.s2t_d], False)  # Ds require no gradients when optimizing Gs
            self.optimizer_g.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_g()             # calculate gradients for G_A and G_B
            self.optimizer_g.step()       # update G_A and G_B's weights
            # print("G backward done")
            loss_list.append(self.loss_g_t2s.item())
            loss_list.append(self.loss_g_s2t.item())
            print("loss_list = ", loss_list)
            print("update_only_d = ", update_only_d)

        # D_A and D_B
        self.set_requires_grad([self.t2s_d, self.s2t_d], True)
        self.optimizer_d.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_d_t2s()      # calculate graidents for D_B
        self.backward_d_s2t()      # calculate gradients for D_A
        
        self.optimizer_d.step()
        # print("D backward done")

        # Discriminator weight clamping
        for p in self.t2s_d.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        for p in self.s2t_d.parameters():
            p.data.clamp_(-0.01, 0.01)

        loss_list.append(self.loss_d_t2s.item())
        loss_list.append(self.loss_d_s2t.item())

        print("loss_list = ", loss_list)
        print("update_only_d = ", update_only_d)
        return loss_list

