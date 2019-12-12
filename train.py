import torch
import torch.nn as nn
import numpy as np
from utils import *
from dataloader import *
from tqdm import tqdm 
from torch.nn import DataParallel as DP

class Trainer():
    def __init__(self, model, config, data):
        self.model = model
        self.config = config
        self.data = data

        
        # print(config.data_parallel)
        if config.data_parallel:
            # print("here")
            # self.model = DP(self.model)
            # self.model.module.make_data_parallel()
            self.model.make_data_parallel()
            # import sys
            # sys.exit


        self.train_epochs(num_epochs=self.config.num_epochs)

    def save_model(self):
        # if self.config.data_parallel:
        #     # print(type(self.model))
        #     # torch.save(self.model.module.state_dict(),self.config.model_path)
        #     self.model.module.save_models(self.config.model_path_dict)
            
        # else:
        #     # torch.save(self.model.state_dict(),self.config.model_path)
        #     self.model.save_models(self.config.model_path_dict)


        torch.save(self.model.state_dict(),self.config.model_path)
        self.model.save_models(self.config.model_path_dict)

    def load_model(self):
        # if self.config.data_parallel:
        #     # print(type(self.model)
        #     # self.model.module.load_state_dict(torch.load(self.config.model_path))
        #     self.model.module.load_models()

        # else:
        #     # self.model.load_state_dict(torch.load(self.config.model_path))
            # self.model.load_models()


        self.model.load_state_dict(torch.load(self.config.model_path))
        self.model.load_models()


    def train_epochs(self,num_epochs):
        self.model.train()
        # if self.config.data_parallel:
        #     self.model.module.set_train()
        # else:
        #     self.model.set_train()
        self.model.set_train()


        for ep in tqdm(range(num_epochs)):
        # for ep in range(1):
            # do 1000 mini-batches in each epoch
            train_data_generator = self.data.generate_batches_train()
            for b in range(self.config.num_minibatches):
            # for b in range(1):
                # get data
                train_data = next(train_data_generator)
                train_data_desc = [i[0] for i in train_data]
                voxel_file = [i[1] for i in train_data]
                voxel = [get_voxel_from_file(i) for i in voxel_file]

                
                # pad text data
                padded_text_data = []
                seq = [len(s) for s in train_data_desc]
                maxlen = max(seq)
                for i in range(len(seq)):
                    temp = train_data_desc[i].copy()
                    for j in range(maxlen - len(train_data_desc[i])):
                        temp.append(76)
                    padded_text_data.append(temp)
                
                # convert to tensor
                # print(type(train_data_desc))
                # print(train_data_desc)
                
                # if self.config.data_parallel:
                #     self.model.module.real_t = torch.from_numpy(np.array(padded_text_data)).cuda()
                #     self.model.module.real_s = torch.from_numpy(np.array(voxel)).cuda()
                #     self.model.module.real_s = self.model.module.real_s.permute(0,4,1,2,3)
                # else:
                self.model.real_t = torch.from_numpy(np.array(padded_text_data)).cuda()
                self.model.real_s = torch.from_numpy(np.array(voxel)).cuda()
                self.model.real_s = self.model.real_s.permute(0,4,1,2,3)
                
                # print("Input shapes : ")
                # print("real text : ", self.model.real_t.size())
                # print("real shape : ", self.model.real_s.size())
                # text_tensor.cuda()
                # voxel_tensor.cuda()
                # import sys
                # sys.exit()
                # optimise params (forward called inside)
                update_only_d = True
                if b % 10 == 0 and b != 0:
                    update_only_d = False 

                l = None
                # if self.config.data_parallel:
                #     l = self.model.module.optimize_parameters(update_only_d, maxlen)
                # else:
                maxlen_tensor = torch.from_numpy(np.array([float(maxlen)])).float().to(self.config.device)
                maxlen_tensor = maxlen_tensor.unsqueeze(0).expand(self.config.num_gpus_active,-1)

                l = self.model.optimize_parameters(update_only_d, maxlen_tensor)
                
                if (b+1) % 100 == 0 :
                    print("Iter : ",b)
                    if len(l) == 4 : 
                        print("T2S G loss = ",l[0])
                        print("S2T G loss = ",l[1])
                    print("T2S D loss = ",l[-2])
                    print("S2T D loss = ",l[-1])
                    print("====================================")

                # import sys
                # sys.exit() 


            self.save_model()
