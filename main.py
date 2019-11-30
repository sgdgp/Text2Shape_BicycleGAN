from config import Config
from dataloader import *
import os 
from cycleGAN import *
from train import *
from test import *
import torch
import random
import numpy as np

def main():
    # set seeds
    seed_val = 123
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    os.environ['PYTHONHASHSEED']=str(seed_val)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed_val)


    conf = Config()
    primitive_dataset = None
    if os.path.exists(conf.word2id_filename):
        primitive_dataset = PrimitiveDataset.read_from_disk(conf)
    else:
        primitive_dataset = read_dataset(conf)
    # add few attributes to config
    # print(type(primitive_dataset))
    conf.num_vocab = len(primitive_dataset.word2id)
    conf.bos_embedding = np.random.normal(scale=0.6, size=(conf.embedding_dim, ))

    model = T2S_CycleGAN(conf, primitive_dataset)
    # trainer = Trainer(model,conf,primitive_dataset)

    tester = Tester(model,conf,primitive_dataset)
    tester.get_shape_prediction()

    

main()