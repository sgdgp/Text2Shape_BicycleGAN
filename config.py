import torch
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
        self.lr = 0.01
        self.num_epochs = 100
        self.num_minibatches = 1000
        self.batch_size = 4
        self.test_batch_size = 1
        self.embedding_dim = 300


        self.model_path = self.processed_data_folder + "model.pth"
        self.output_folder = "./output/"
        self.voxel_renderer_path = "../sstk/ssc/render-voxels.js"