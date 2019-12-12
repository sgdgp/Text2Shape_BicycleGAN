import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils import *
from dataloader import *


class Tester:
    def __init__(self, model, config, data):
        self.config = config
        self.model = model
        self.data = data
        
        self.model.eval()
        self.model.load_state_dict(torch.load(self.config.model_path), strict=False)

    def get_shape_prediction(self):
        self.model.eval()
        self.model.set_eval()


        test_data_generator = self.data.generate_batches_test()
        b = 0

        f = open(self.config.output_folder + "text_gt_outputs.txt", "w")

        while True:
            if b==500:
                break
            test_data = next(test_data_generator)
            test_data_desc = [i[0] for i in test_data]
            voxel_file = [i[1] for i in test_data]


            if "black" in voxel_file[0]:
                continue
            
            # print(test_data)
            # print(test_data_desc)
            # print(voxel_file)

            voxel = [get_voxel_from_file(i) for i in voxel_file]


            self.model.real_t = torch.from_numpy(np.array(test_data_desc)).cuda()
            self.model.real_s = torch.from_numpy(np.array(voxel)).cuda()
            self.model.real_s = self.model.real_s.permute(0,4,1,2,3)

            self.model.t2s_pred()


            real_s = self.model.real_s.detach().cpu().numpy()
            fake_s = self.model.fake_s.detach().cpu().numpy()
            real_t = self.model.real_t.detach().cpu().numpy()
            fake_t = self.model.rec_t.detach().cpu().numpy()

            print(fake_s.any()>0 and fake_s.any()<=255)
            print(real_s.any()>0 and real_s.any()<=255)
            # print(voxel)
            # print(voxel_file)
            # break
            
            # real_s /= 255.
            # fake_s /= 255.

            for id in range(real_s.shape[0]):

                # get pngs
                f1 = write_temp_voxel_png(real_s[id], self.config.output_folder, self.config.voxel_renderer_path)
                f2 = write_temp_voxel_png(fake_s[id], self.config.output_folder, self.config.voxel_renderer_path)
                # f3 = write_text_to_png(real_t[id], self.data.id2word, self.config.output_folder)

                # stick together pngs and write to output folder
                
                print(f1)
                print(f2)
                
                text_gt = get_text_from_ids(real_t[0], self.data.id2word, 1)
                text_fake = get_text_from_ids(fake_t[0], self.data.id2word, 0)

                f.write("output_" +str(int(b*self.config.test_batch_size)+id) +"\t" +text_gt + "\t" + text_fake + "\n")

                merge_and_write_to_output_folder([f1,f2],self.config.output_folder+"output_"+str(int(b*self.config.test_batch_size)+id)+".png")

                #clear temp files
                # remove_temp_files([f1,f2])

            # break
            
            b += 1

        f.close()