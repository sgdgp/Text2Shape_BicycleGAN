import torch
import torch.nn as nn
import numpy as np
from utils import *
from dataloader import *

class Trainer():
    def __init__(self, model, config, data):
        self.model = model.to(config.device)
        self.config = config
        self.data = data

        self.train_epochs(num_epochs=self.config.num_epochs)

    def save_model(self):
        torch.save(self.model.state_dict(),self.config.model_path)

    def train_epochs(self,num_epochs):
        for ep in range(num_epochs):
            # do 1000 mini-batches in each epoch
            train_data_generator = self.data.generate_batches_train()
            for b in range(self.config.num_minibatches):
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
                l = self.model.optimize_parameters(update_only_d)
                
                if b % 100 == 0 :
                    print("Iter : ",b)
                    if len(l) == 4 : 
                        print("T2S G loss = ",l[0])
                        print("S2T G loss = ",l[1])
                    print("T2S D loss = ",l[-2])
                    print("S2T D loss = ",l[-1])
                    print("====================================")


            self.save_model()
