
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
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset_finetune_transformer import MOF_ID_Dataset
# from dataset.dataset_finetune import collate_pool, get_train_val_test_loader
#from model.cgcnn_finetune import CrystalGraphConvNet

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)

class InferenceModel(object):
    def __init__(self, config, log_dir):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.random_seed = self.config['dataloader']['randomSeed']

        # self.mofdata = np.load(self.config['dataset']['dataPath'], allow_pickle=True)
        with open(self.config['dataset']['dataPath']) as f:
            reader = csv.reader(f)
            self.mofdata = [row for row in reader]
        self.mofdata = np.array(self.mofdata)
        self.vocab_path = self.config['vocab_path']
        self.tokenizer = MOFTokenizer(self.vocab_path, model_max_length = 512, padding_side='right')

        self.test_dataset = MOF_ID_Dataset(data = self.mofdata, tokenizer = self.tokenizer)

        self.test_loader = DataLoader(
        self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], drop_last=False, 
        shuffle=False, pin_memory = False
        )

        self.criterion = nn.MSELoss()
        self.normalizer = joblib.load(os.path.join(self.writer.log_dir, 'normalizer.pkl'))


    def _get_device(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
            self.config['cuda'] = True
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)

        return device
 
    def _load_pre_trained_weights(self, model):
        try:
            # checkpoints_folder = os.path.join(self.config['fine_tune_from'], 'checkpoints')
            checkpoints_folder = self.config['fine_tune_from']
            load_state = torch.load(os.path.join(checkpoints_folder, 'model_transformer_3.pth'),  map_location=self.config['gpu']) 
 
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
    
    def inference(self):
        self.transformer = Transformer(**self.config['Transformer'])
        # Load state dict
        if self.config['cuda']:
            self.transformer = self.transformer.to(self.device)

        model_transformer = self._load_pre_trained_weights(self.transformer)

        model = TransformerRegressor(transformer=model_transformer, d_model=512).to(self.device)
        
        if self.config['cuda']:
            model = model.to(self.device)
            
        self.model = model

        # test steps
        print('Test on test set')
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        print(model_path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        test_preds = []

        with torch.no_grad():
            self.model.eval()
            for bn, (inputs, target) in enumerate(self.test_loader):

                input_var = inputs.to(self.device)

                # compute output
                output = self.model(input_var)
                
                test_pred = self.normalizer.denorm(output.data.cpu())
                test_pred = test_pred.view(-1).tolist()
                test_preds += test_pred
                print(len(test_preds))
            

        pd.DataFrame(test_preds).to_csv(os.path.join(self.writer.log_dir, 'inference_results.csv'), index=False, header=False)
        
        self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer finetuning')
    parser.add_argument('--seed', default=1, type=int,
                        metavar='Seed', help='random seed for splitting data (default: 1)')

    args = parser.parse_args(sys.argv[1:])

    config = yaml.load(open("config_inf_transformer.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    config['dataloader']['randomSeed'] = args.seed

    if 'hMOF' in config['dataset']['data_name']:
        # task_name = 'hMOF'
        task_name = config['dataset']['data_name']
        pressure = config['dataset']['data_name'].split('_')[-1]
    if 'QMOF' in config['dataset']['data_name']:
        task_name = 'QMOF'
    if 'regenerability' in config['dataset']['data_name']:
        task_name = 'regenerability'

    # ftf: finetuning from
    # ptw: pre-trained with
    if config['fine_tune_from'] == 'scratch':
        ftf = 'scratch'
        ptw = 'scratch'
    else:
        ftf = config['fine_tune_from'].split('/')[-1]
        ptw = config['trained_with']

    seed = config['dataloader']['randomSeed']

    log_dir = os.path.join(
        'training_results/finetuning/Transformer',
        'Trans_{}_{}_{}'.format(ptw,task_name,seed)
    )

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fine_tune = InferenceModel(config, log_dir)
    fine_tune.inference()

    #CO2 Model: Val - MSE 1.0135 (0.6730) MAE 1.037 (0.999) Test - MSE 0.2805 (0.4837) MAE 0.683 (0.878)

    #N2 Model:
    #Val: Epoch [%d] Validate: [282/282], Loss 0.4431 (0.5568), MAE 0.088 (0.084)MAE 0.084
    #Test: [281/282], Loss 0.5729 (0.4725), MAE 0.091 (0.080) MAE 0.080