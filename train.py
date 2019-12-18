from dataset import ModisDataset
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from model import Model, snapshot

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dir="/tmp"

    include_time = False

    # model = Model(hidden_size=128,num_layers=2,dropout=0.5)
    model = Model(input_size=2 if include_time else 1, hidden_size=256, output_size=2 if include_time else 1, device=device)

    #model.load_state_dict(torch.load("/tmp/model_epoch_0.pth")["model"])
    model.train()

    dataset = ModisDataset(region="germany",fold="train", include_time=include_time)
    validdataset = ModisDataset(region="germany", fold="validate", include_time=include_time)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=512,
                                             shuffle=True,
                                             #sampler=torch.utils.data.sampler.SubsetRandomSampler(np.arange(10000))
                                             )

    #criterion = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    L2Norm = torch.nn.MSELoss(reduction='none')
    def criterion(y_pred, y_data, log_variances):
        norm = (y_pred-y_data)**2
        loss = (torch.exp(-log_variances) * norm).mean()
        regularization = log_variances.mean()
        return 0.5 * (loss + regularization)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 30

    for epoch in range(epochs):

        train_epoch(model,dataloader,optimizer, criterion, device)

        test_model(model, validdataset, device)

        path = f"model_epoch_{epoch}.pth"
        print(f"saving model snapshot to {os.path.join(model_dir, path)}")
        snapshot(model, optimizer, os.path.join(model_dir, path))

def train_epoch(model,dataloader,optimizer, criterion, device):
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, batch in iterator:
        x_data, y_data = batch
        optimizer.zero_grad()
        y_pred, log_variances = model(x_data.to(device), y = y_data)
        loss = criterion(y_pred, y_data.to(device), log_variances)
        loss.backward()
        optimizer.step()
        iterator.set_description(f"loss {loss:.4f}, mean log_variances {log_variances.mean():.6f}")

def test_model(model, dataset, device):
    idx = 1

    x = dataset.data[idx].astype(float)
    date = dataset.date[idx].astype(np.datetime64)

    N_seen_points = 200
    N_predictions = 50
    future = x.shape[0] - N_seen_points

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

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


if __name__ == '__main__':
    main()
