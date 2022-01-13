
import argparse
from utils.wandb import Wandb
from utils.config import config
import os, sys, time, shortuuid, pathlib, json, logging, os.path as osp

import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
import torch.nn as nn
import torch.nn.functional as F
import os

sys.path.insert(1, '/home/lbrahiea/OHrate')

from ipynb.fs.full.Utils import set_seed
from ipynb.fs.full.Model import build_model, model_initialize
from ipynb.fs.full.Data_loader import make_data_loader, Scaler, map_to_scale
from ipynb.fs.full.Get_fingerprints import get_rdkit_fingerprint, get_cp_fingerprint
from ipynb.fs.full.Evaluate import split_data
from sklearn.utils import shuffle
from train_model_wandb2 import train_model
from sklearn.model_selection import KFold


from torch.utils.tensorboard import SummaryWriter

import wandb


hyperparameter_defaults = dict(
    n_ensamble = 0,
    lr = 0.001,
    seed=10,
    scheduler_param=0.5,
    n_epochs = 100,
    weights = [10,1,1,1],
    bweights = [1,1,1,1],
    scheduler_gamma = 0.1,
    batch_size =128,
    num_folds = 5,
    input_type = 'Morgan',
    model_type = 'Point',
    dropout = 0.2,
    scale_features = True,
    scale_targets = False,
    batch_norm = True,
    target = 'OH',
    teston = 'All',
    scale_k= False
    )

resume = sys.argv[-1] == "--resume"
wandb.init(config=hyperparameter_defaults, project="RRP-All", resume=resume)
config = wandb.config
 

def main():
    
    # setup param
    
    config.scheduler_step = config.scheduler_param * config.n_epochs
    
    hidden = np.zeros([config.n_layers],dtype=int)
    temp = [config.h1,config.h2,config.h3,config.h4,config.h5,config.h6]
    for i in range(config.n_layers):
        hidden[i] = temp[i]
    config.hidden = list(hidden)
    
    param = config
    
    # load in dataframe
    
    df = pd.read_csv('/home/lbrahiea/OHrate/Data/df.csv')
    df = df.join(get_rdkit_fingerprint(df))
    cp_input = pd.read_csv('/home/lbrahiea/OHrate/Data/VOCensamble{}.csv'.format(config.n_ensamble))
    cp_features = cp_input.iloc[:,cp_input.columns != 'smiles']
    df = df.join(cp_features,)
    
    # run ensemble
    vvss=0
    test=0
    seeds=5
    clist = np.ndarray([seeds,1])
    vlist = np.ndarray([seeds,1])
    for i in range(seeds):
        set_seed(param.seed)
        seed = param.seed+i*10
        unique = df['smiles'].unique()
        K = int(unique.shape[0]*.2)
        options_test = unique[np.random.choice(unique.shape[0],size=K,replace=False)]
        train_mols, test_mols = split_data(df,options_test,param)
        train_molecules = shuffle(train_mols['smiles'].unique())
        kf = KFold(n_splits=param.num_folds)
        vvs=0
        # Do cross validation
        for j,(train_index, test_index) in enumerate(kf.split(train_molecules)):
            set_seed(param.seed)
            # Split data to train/test folds
            logging.info(f'####### Evaluating fold {j+1} in a total of {param.num_folds} folds #######')
            train_fold_mols, test_fold_mols = split_data(train_mols,train_molecules[test_index],param)
            # Do scaling if needed
            Scobj=Scaler()
            train_fold_mols,test_fold_mols = Scobj.Scale_data(train_fold_mols,test_fold_mols,param)
            # Create and run model
            model, criterion, optimizer, scheduler = model_initialize(df,param)
            vv=train_model(model,optimizer,criterion,j,param.teston,n_epochs=param.n_epochs,weight=param.weights,
                               train_loader=make_data_loader(train_fold_mols,param),
                               scheduler=scheduler,valid_loader=make_data_loader(test_fold_mols,param),scaler=Scobj,
                               seed=seed)
            vvs+=vv
        clist[i]=vvs/param.num_folds
        set_seed(param.seed)
        logging.info("========= Finished CV =========")
        # Evaluate model
        # Do scaling if needed
        Scobje=Scaler()
        train_mols,test_mols = Scobje.Scale_data(train_mols,test_mols,param)
        # Create and run model
        model, criterion, optimizer, scheduler = model_initialize(df,param)
        loss = train_model(model,optimizer,criterion,10,param.teston,n_epochs=param.n_epochs,weight=param.weights,
                               train_loader=make_data_loader(train_mols,param),
                               scheduler=scheduler,valid_loader=make_data_loader(test_mols,param),scaler=Scobje,
                               seed=seed)
        vlist[i]= loss
    metrics = {'CV-loss': np.mean(clist), 'CV-std': np.std(clist), 'test': np.mean(vlist), 'test-std': np.std(vlist)}
    wandb.log(metrics)
        
if __name__ == '__main__':
    main()
