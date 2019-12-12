import torch
import os

class Config:
    def __init__(self):
        self.dataset_folder = "../text2shape/primitives/"

        self.processed_data_folder = "./data/"
        self.word2id_filename = self.processed_data_folder + "word2id.json"
        self.id2word_filename = self.processed_data_folder + "id2word.json"
        self.id2voxel_filename = self.processed_data_folder + "id2voxel.json"

        self.train_samples_filename = self.processed_data_folder + "train_samples.json"
        self.val_samples_filename = self.processed_data_folder + "val_samples.json"
        self.test_samples_filename = self.processed_data_folder + "test_samples.json"

        self.google_word2vec_filepath = "../VirtualHome/data/GoogleNews-vectors-negative300.bin"
        self.embedding_matrix_filename = self.processed_data_folder + "embedding_matrix.npy"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.lr = 5e-5 # for WGAN using RMSProp
        self.num_epochs = 10000
        self.num_minibatches = 100
        self.batch_size = 16 * 4
        self.data_parallel = True
        # self.num_gpus_active = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.num_gpus_active = 4
        # print(self.num_gpus_active)
        # import sys
        # sys.exit()
        
        self.test_batch_size = 1
        self.embedding_dim = 300

        self.normalised_3d_model = True

        self.output_folder = "./output_WGAN_gradclip_RMSProp_NormalisedOutput_2/"
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        
        self.model_path = self.processed_data_folder + "model_l1_mse_RMSProp_gradclip_NormalisedOutput_2.pth"
        self.model_path_dict = {
                                "t2s_g" : self.processed_data_folder + "t2s_g_l1_mse_RMSProp_gradclip_NormalisedOutput_2.pth",
                                "t2s_d" : self.processed_data_folder + "t2s_d_l1_mse_RMSProp_gradclip_NormalisedOutput_2.pth",
                                "s2t_g" : self.processed_data_folder + "s2t_g_l1_mse_RMSProp_gradclip_NormalisedOutput_2.pth",
                                "s2t_d" : self.processed_data_folder + "s2t_d_l1_mse_RMSProp_gradclip_NormalisedOutput_2.pth",
                                
                                }

        self.voxel_renderer_path = "../sstk/ssc/render-voxels.js"