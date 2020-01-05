import sys
sys.path.append('..')
from dataset import ModisDataset, Sentinel5Dataset
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import Model, snapshot, restore
import ignite.metrics
import pandas as pd
from train import train_epoch, test_epoch, test_model, fine_tune
from visualizations import make_and_plot_predictions, predict_future
from dataset import transform_data
from copy import deepcopy

import sklearn.metrics
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

region = "germany"
include_time = True

model_dir="/data2/igarss2020/tune/"
log_dir = "/data2/igarss2020/tune/"
name_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}_lr={lr}_weightdecay={weight_decay}_e={epoch}"
log_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}_lr={lr}_weightdecay={weight_decay}_log.csv"
epochs = 10

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def criterion(y_pred, y_data, log_variances):
    norm = (y_pred-y_data)**2
    loss = (torch.exp(-log_variances) * norm).mean()
    regularization = log_variances.mean()
    return 0.5 * (loss + regularization)

#def criterion(y_pred, y_data, log_variances):
#    norm = (y_pred-y_data)**2
#    return norm.mean()

def setup(hidden_size,num_layers,lr, weight_decay):
    model = Model(input_size=2,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  output_size=1,
                  device=device)

    #model.load_state_dict(torch.load("/tmp/model_epoch_0.pth")["model"])
    model.train()

    enddate = '2010-01-01'

    dataset = ModisDataset(region=region,
                           fold="train",
                           znormalize=True,
                           augment=True,
                           overwrite=False,
                           include_time=include_time,
                           filter_date=(None,enddate))

    validdataset = ModisDataset(region=region,
                                fold="validate",
                                znormalize=True,
                                augment=False,
                                include_time=include_time)

    #dataset = Sentinel5Dataset(fold="train", seq_length=300)
    #validdataset = Sentinel5Dataset(fold="validate", seq_length=300)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=512,
                                             shuffle=True,
                                             #sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(10000))
                                             )
    validdataloader = torch.utils.data.DataLoader(validdataset,
                                             batch_size=512,
                                             shuffle=False,
                                             #sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(10000))
                                             )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    return model, dataset, validdataset, dataloader, validdataloader, optimizer

def train(hidden_size,num_layers,lr, weight_decay):
    region = "germany"
    
    log_name = log_pattern.format(region=region, num_layers=num_layers, hidden_size=hidden_size, lr=lr, weight_decay=weight_decay)
    log_path = os.path.join(log_dir, log_name)
    
    if os.path.exists(log_path):
        print(f"{log_path} exists. skipping...")
        return
    
    model, dataset, validdataset, dataloader, validdataloader, optimizer = setup(hidden_size,num_layers,lr, weight_decay)
    stats=list()
    for epoch in range(epochs):
        trainloss = train_epoch(model,dataloader,optimizer, criterion, device)
        testmetrics, testloss = test_epoch(model,validdataloader,device, criterion, n_predictions=1)
        metric_msg = ", ".join([f"{name}={metric.compute():.2f}" for name, metric in testmetrics.items()])
        msg = f"epoch {epoch}: train loss {trainloss:.2f}, test loss {testloss:.2f}, {metric_msg}"
        print(msg)

        #test_model(model, validdataset, device)

        model_name = name_pattern.format(region=region, num_layers=num_layers, hidden_size=hidden_size, lr=lr, weight_decay=weight_decay, epoch=epoch)
        pth = os.path.join(model_dir, model_name+".pth")
        print(f"saving model snapshot to {pth}")
        snapshot(model, optimizer, pth)
        stat = dict()
        stat["epoch"] = epoch
        for name, metric in testmetrics.items():
            stat[name]=metric.compute()

        stat["trainloss"] = trainloss.cpu().detach().numpy()
        stat["testloss"] = testloss.cpu().detach().numpy()
        stats.append(stat)
        
    df = pd.DataFrame(stats)
    
    df.to_csv(log_path)
    print(f"saving log to {log_path}")

def sample_hparams():
    hidden_size = np.random.choice([16,32,64,128,256,512])
    num_layers = np.random.choice([1,2,3,4,5,6])
    lr=np.random.choice([1e-2,1e-3])#np.power(10,np.random.uniform(low=-4, high=-1, size=1))
    weight_decay=np.random.choice([1e-4,1e-5,1e-6])
    return int(hidden_size), int(num_layers), float(lr), float(weight_decay)

while True:
    hidden_size, num_layers, lr, weight_decay = sample_hparams()
    train(hidden_size, num_layers, lr, weight_decay)
