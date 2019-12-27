from dataset import ModisDataset, Sentinel5Dataset
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import Model, snapshot
from scipy.spatial.distance import mahalanobis
import ignite.metrics
import pandas as pd

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    num_layers = 1
    hidden_size = 128
    region = "germany"
    epochs = 100

    model_dir="/data2/igarss2020/models/"
    log_dir = "/data2/igarss2020/models/"
    name_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}_e={epoch}"
    log_pattern = "LSTM_{region}_l={num_layers}_h={hidden_size}"

    model = Model(input_size=1,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  output_size=1,
                  device=device)

    #model.load_state_dict(torch.load("/tmp/model_epoch_0.pth")["model"])
    model.train()

    dataset = ModisDataset(region=region,fold="train", znormalize=False, augment=False)
    validdataset = ModisDataset(region=region, fold="validate", znormalize=False, augment=False)

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

    #criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    L2Norm = torch.nn.MSELoss(reduction='none')
    def criterion(y_pred, y_data, log_variances):
        norm = (y_pred-y_data)**2
        loss = (torch.exp(-log_variances) * norm).mean()
        regularization = log_variances.mean()
        return 0.5 * (loss + regularization)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)


    stats=list()
    for epoch in range(epochs):

        trainloss = train_epoch(model,dataloader,optimizer, criterion, device)
        testmetrics, testloss = test_epoch(model,validdataloader,device, criterion, n_predictions=1)
        metric_msg = ", ".join([f"{name}={metric.compute():.2f}" for name, metric in testmetrics.items()])
        msg = f"epoch {epoch}: train loss {trainloss:.2f}, test loss {testloss:.2f}, {metric_msg}"
        print(msg)

        test_model(model, validdataset, device)

        model_name = name_pattern.format(region=region, num_layers=num_layers, hidden_size=hidden_size, epoch=epoch)
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

def train_epoch(model,dataloader,optimizer, criterion, device):
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    losses = list()
    for idx, batch in iterator:
        x_data, y_true = batch

        x_data = x_data.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        # y is used for teacher forcing during training
        y_pred, log_variances = model(x_data, y=y_true)
        loss = criterion(y_pred, y_true, log_variances)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        iterator.set_description(f"loss {loss:.4f}, mean log_variances {log_variances.mean():.6f}")

    return torch.stack(losses).mean()

def test_epoch(model,dataloader, device, criterion, n_predictions):

    metrics = dict(
        mae=ignite.metrics.MeanAbsoluteError(),
        mse=ignite.metrics.MeanSquaredError(),
        rmse=ignite.metrics.RootMeanSquaredError()
    )

    losses = list()

    with torch.no_grad():
        iterator = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in iterator:
            x_data, y_true = batch

            x_data = x_data.to(device)
            y_true = y_true.to(device)

            # make single forward pass to get test loss
            y_pred,log_variances = model(x_data)
            loss = criterion(y_pred, y_true, log_variances)
            losses.append(loss.cpu())

            # make multiple MC dropout inferences for further metrics
            y_pred, epi_var, ale_var = model.predict(x_data.to(device), n_predictions)
            #mae.update((y_pred, y_true))
            #metrics(y_true, y_pred, epi_var+ale_var)
            for name, metric in metrics.items():
                metric.update((y_pred.view(-1), y_true.view(-1)))

    return metrics, torch.stack(losses).mean()

def test_model(model, dataset, device):
    from visualizations import make_and_plot_predictions

    if isinstance(dataset,ModisDataset):

        idx = 1

        x = dataset.data[idx].astype(float)
        date = dataset.date[idx].astype(np.datetime64)
        N_seen_points = 250
        N_predictions = 10

        make_and_plot_predictions(model, x, date, N_seen_points=N_seen_points, N_predictions=N_predictions, device=device)
        plt.show()

        idx = 20

        x = dataset.data[idx].astype(float)
        date = dataset.date[idx].astype(np.datetime64)
        make_and_plot_predictions(model, x, date, N_seen_points=N_seen_points, N_predictions=N_predictions,
                                  device=device)
        plt.show()

    elif isinstance(dataset,Sentinel5Dataset):
        d = dataset.data["Napoli"]
        x = d[:,1].astype(float)[:,None]

        x = x - dataset.mean
        x = x / dataset.std

        date = d[:,0].astype(np.datetime64)

        N_seen_points = 300
        N_predictions = 20

        make_and_plot_predictions(model, x, date, N_seen_points=N_seen_points, N_predictions=N_predictions,
                                  device=device)
        plt.show()

    else:
        return


    """
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    
    future = x.shape[0] - N_seen_points


    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(date[:N_seen_points], x[:N_seen_points,0], c="#000000", alpha=1, label="seen input sequence")
    ax.plot(date[N_seen_points:], x[N_seen_points:,0], c="#000000", alpha=.1, label="unseen future")

    for idx in range(N_predictions):
        model.train()
        x_data = torch.Tensor(x)[None, :N_seen_points].to(device)
        y_pred, _ = model(x_data, future=future)
        y_pred = np.append([np.nan], y_pred[0, :-1, 0].cpu().detach().numpy())
        label = "prediction" if idx == 0 else None
        ax.plot(date[:N_seen_points + future], y_pred, c="#0065bd", label=label, alpha=(1 / N_predictions) ** 0.7)

    ax.axvline(x=date[N_seen_points], ymin=0, ymax=1)
    ax.legend()
    plt.show()
    """

if __name__ == '__main__':
    main()
