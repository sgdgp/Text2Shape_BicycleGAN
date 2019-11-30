import numpy as np
import nrrd
import os
import sys
from glob import glob
import itertools
import random
import gensim
import json

class PrimitiveSample:
    def __init__(self, desc_file, voxel_file_list):
        self.desc_list = self.get_desc_list(desc_file)
        self.voxel_file_list = voxel_file_list
        self.gen_samples_and_split()
        
    def get_desc_list(self, desc_file):
        desc_list = []
        with open(desc_file,"r") as f:
            for line in f:
                desc_list.append(line.strip())
        return desc_list

    
    def gen_samples_and_split(self):
        
        # split is done randomly at this stage to have samples from each kind of primitive 
        # in train/val/test splits.
        # Splitting at the end may have test or val sets skewed on some kind of primitive

        # train/test/val - list of tuples (desc, voxel)
        # print(len(self.desc_list))
        # print(len(self.voxel_list))
        all_samples = list(itertools.product(self.desc_list, self.voxel_file_list))
        # print(len(all_samples))
        random.shuffle(all_samples)
        # print(type(all_samples))
        
        train_cnt = int(0.8 * len(all_samples))
        val_cnt = int(0.1 * len(all_samples))

        self.train_samples = all_samples[0:train_cnt]
        self.val_samples = all_samples[train_cnt:train_cnt+val_cnt]
        self.test_samples = all_samples[train_cnt+val_cnt : ]





class PrimitiveDataset:
    def __init__(self, train_samples, val_samples, test_samples, config, load_from_disk = True):
        if not load_from_disk:
            self.config = config    
            self.generate_processed_samples(train_samples, val_samples, test_samples)
            
            print("Begin writing processed data (tokenized description with path of voxel)")
            print(len(self.train_samples))
            print(len(self.val_samples))
            print(len(self.test_samples))
            self.write_processed_data()
            self.form_embedding_matrix()
            np.save(self.config.embedding_matrix_filename, self.emb_weight_matrix)

        else :
            self.config = config
            self.word2id = json.load(open(config.word2id_filename, "r"))
            self.word2id = {k:int(v) for k,v in self.word2id.items()}
            self.id2word = json.load(open(config.id2word_filename, "r"))
            self.id2word = {int(k):v for k,v in self.id2word.items()}
            self.train_samples = train_samples
            self.val_samples = val_samples
            self.test_samples = test_samples
            self.emb_weight_matrix = np.load(self.config.embedding_matrix_filename)            
        
    
    @classmethod
    def read_from_disk(cls, config):
        # self.id2voxel = json.load(open(config.id2voxel_filename, "r"))
        # self.id2voxel = {int(k):v for k,v in self.id2voxel.items()}
        train_samples = json.load(open(config.train_samples_filename, "r"))
        val_samples = json.load(open(config.val_samples_filename, "r"))
        test_samples = json.load(open(config.test_samples_filename, "r"))
        return cls(train_samples,val_samples,test_samples,config)

    def form_embedding_matrix(self):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.config.google_word2vec_filepath, binary=True)
       
        # create embedding matrix for words
    
        self.embedding_dim = self.config.embedding_dim
        self.emb_weight_matrix = np.zeros((len(self.word2id), self.embedding_dim))
        print("Embedding matrix init, ",self.emb_weight_matrix.shape)
        
        for key in self.id2word.keys():
            try:
                # print(key)
                # print(id2word[key]) 
                self.emb_weight_matrix[key] = word2vec[self.id2word[key]]
            except KeyError:
                self.emb_weight_matrix[key] = np.random.normal(scale=0.6, size=(self.embedding_dim, ))
    
        print("Embedding matrix created, ",self.emb_weight_matrix.shape)
    
    def write_processed_data(self):
        with open(self.config.word2id_filename, "w") as f:
            json.dump(self.word2id, f)
        with open(self.config.id2word_filename, "w") as f:
            json.dump(self.id2word, f)
        
        # with open(self.config.id2voxel_filename, "w") as f:
        #     json.dump(self.id2voxel, f)
        
        with open(self.config.train_samples_filename, "w") as f:
            json.dump(self.train_samples, f)
    
        with open(self.config.val_samples_filename, "w") as f:
            json.dump(self.val_samples, f)
    
        with open(self.config.test_samples_filename, "w") as f:
            json.dump(self.test_samples, f)
    
    def generate_processed_samples(self, train_samples, val_samples, test_samples):
        # UNK token 0
        self.word2id = {"<unk>" : 0}
        self.id2word = {0 : "<unk>"}
        for sample in train_samples:
            desc = sample[0]
            desc = desc.strip().split()
            for word in desc:
                if word not in self.word2id.keys():
                    self.word2id[word] = len(self.word2id)
                    self.id2word[len(self.id2word)] = word
        
        
        # write a id2voxel list, to reduce space of train/val/test json files
        self.id2voxel = {}
        
        # generate processed lists 
        self.train_samples = []
        for sample in train_samples:
            w_l = []
            desc = sample[0]
            desc = desc.strip().split()
            for word in desc:
                if word in self.word2id.keys():
                    w_l.append(self.word2id[word])
                else:
                    w_l.append(self.word2id["<unk>"])
            # self.id2voxel[len(self.id2voxel)] = sample[1]

            self.train_samples.append([w_l , sample[1]])

        self.val_samples = []
        for sample in val_samples:
            w_l = []
            desc = sample[0]
            desc = desc.strip().split()
            for word in desc:
                if word in self.word2id.keys():
                    w_l.append(self.word2id[word])
                else:
                    w_l.append(self.word2id["<unk>"])
            

            self.val_samples.append([w_l , sample[1]])
        
        self.test_samples = []
        for sample in test_samples:
            w_l = []
            desc = sample[0]
            desc = desc.strip().split()
            for word in desc:
                if word in self.word2id.keys():
                    w_l.append(self.word2id[word])
                else:
                    w_l.append(self.word2id["<unk>"])
            

            self.test_samples.append([w_l , sample[1]])

    def generate_batches_train(self):
        self.num_batches = int(len(self.train_samples) / self.config.batch_size)
        for i in range(self.num_batches):
            yield self.train_samples[i * self.config.batch_size : (i+1)* self.config.batch_size]

    def generate_batches_val(self):
        for i in range(len(self.val_samples)):
            yield self.val_samples[i]

    def generate_batches_test(self):
        for i in range(len(self.test_samples)):
            yield self.test_samples[i * self.config.test_batch_size : (i+1)* self.config.test_batch_size]
    
    



def read_dataset(config):
    # print("here")
    folder_list = [x[0] for x in os.walk(config.dataset_folder)]
    folder_list = folder_list[1:]
    average_len = 0
    # primitive_dataset = []
    train_samples = []
    val_samples = []
    test_samples = []
    for f in folder_list:
        desc_file = os.path.join(f,"descriptions.txt")
        # print(desc_file)
        voxel_file_list = glob(os.path.join(f,"*.nrrd"))
        average_len += len(voxel_file_list)
        temp = PrimitiveSample(desc_file,voxel_file_list)
        train_samples.extend(temp.train_samples)
        val_samples.extend(temp.val_samples)
        test_samples.extend(temp.test_samples)
        
    return PrimitiveDataset(train_samples, val_samples, test_samples, config, False)           
    # print(float(average_len)/ float(len(folder_list)) )
