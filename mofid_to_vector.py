from tokenizer.mof_tokenizer import MOFTokenizer
from model.transformer import TransformerRegressor, Transformer, regressoionHead
from model.utils import *
from datetime import datetime, timedelta
from time import time
from torch.utils.data import dataset, DataLoader

import os
import csv
import yaml
import shutil
import argparse
import sys
import time
import warnings
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset
import joblib
from sklearn.decomposition import IncrementalPCA

data = joblib.load('mofids.pkl')
vectors = []

def pca_reduction(data):
    """
    Converts a 3D array to a 1D array using PCA.

    Args:
    data: The 3D input array.
    n_components: The number of principal components to keep.

    Returns:
    The 1D array after PCA transformation.
    """

    # Reshape the 3D array into a 2D matrix
    flattened_data = data.reshape(-1, data.shape[2])

    # Apply PCA
    pca = IncrementalPCA(n_components=512)
    pca.fit(flattened_data)
    reduced_data = pca.transform(flattened_data)

    # Flatten the reduced data into a 1D array
    return reduced_data.flatten()

def _load_pre_trained_weights(model, config):
    try:
        checkpoints_folder = config['fine_tune_from']
        load_state = torch.load(os.path.join(checkpoints_folder, 'model_transformer_14.pth'),  map_location=config['gpu']) 
        model_state = model.state_dict()
        for name, param in load_state.items():
            if name not in model_state:
                print('NOT loaded:', name)
                continue
            else:
                print('loaded:', name)
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            model_state[name].copy_(param)
        print("Loaded pre-trained model with success.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Training from scratch.")
    return model

def get_attention():
    #filtered_data = list(filter(lambda row: row[0] is not None, data))
    modified_data = [item.replace(" ", "&&") for item in data]

    config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)
    config['gpu'] = 'cuda'
    device = torch.device('cuda')
    transformer = Transformer(**config['Transformer']).to(device).to(torch.float16)
    model = _load_pre_trained_weights(transformer, config)
    tokenizer = MOFTokenizer('tokenizer/vocab_full.txt', model_max_length = 512, padding_side='right')

    for mofid in modified_data:

        token = np.array(tokenizer.encode(mofid, max_length=512, truncation=True,padding='max_length'))
        token = torch.from_numpy(token).to(device).long()  # Keep token as Long tensor
        vectors.append(model(token.reshape(512, 1)).detach().cpu().numpy())
        
        print(mofid)
    
    joblib.dump(vectors, 'mof_vectors.pkl', compress='zlib')

get_attention()