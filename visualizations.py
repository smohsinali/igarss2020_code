import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import torch

tumblack = "#000000"
tumblue = "#0065bd"
tumorange = "#e37222"
tumbluelight = "#64a0c8"
tumgray = "#999999"
tumlightgray = "#dad7cb"

def make_and_plot_predictions(model, x, date, N_seen_points=250, N_predictions=50, ylim=None, device=torch.device('cpu')):
    def variance(y_hat, var_hat):
        """eq 9 in Kendall & Gal"""
        T = y_hat.shape[0]
        sum_squares = (1 / T) * (y_hat ** 2).sum(0)
        squared_sum = ((1 / T) * y_hat.sum(0)) ** 2
        epi_var = sum_squares - squared_sum
        ale_var = (1 / T) * (var_hat ** 2).sum(0)
        return epi_var + ale_var, epi_var, ale_var

    future = x.shape[0] - N_seen_points

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))
    axs = np.array(axs).reshape(-1)

    axs[0].set_title("epistemic uncertainty")
    axs[1].set_title("aleatoric uncertainty")
    axs[2].set_title("combined uncertainty")


    x_ = torch.Tensor(x)[None, :].to(device)
    if x_.shape[2] == 2:
        doy_seen = x_[:, :N_seen_points, 1]
        doy_future = x_[:, N_seen_points:, 1]
    else:
        doy_seen = None
        doy_future=None
    x_data = x_[:, :N_seen_points, 0].unsqueeze(2)

    mean, epi_var, ale_var = model.predict(x_data, N_predictions, future, date=doy_seen, date_future=doy_future)
    var = epi_var + ale_var

    mean = mean.cpu().squeeze()
    var = var.cpu().squeeze()
    epi_var = epi_var.cpu().squeeze()
    ale_var = ale_var.cpu().squeeze()

    n_sigma = 1
    axs[0].fill_between(date[1:], mean[1:] + n_sigma * np.sqrt(epi_var[1:]), mean[1:] - n_sigma * np.sqrt(epi_var[1:]),
                        alpha=.5, label=f"epistemic {n_sigma}" + r"$\sigma$")
    axs[1].fill_between(date[1:], mean[1:] + n_sigma * np.sqrt(ale_var[1:]), mean[1:] - n_sigma * np.sqrt(ale_var[1:]),
                        alpha=.5, label=f"aleatoric {n_sigma}" + r"$\sigma$")
    axs[2].fill_between(date[1:], mean[1:] + n_sigma * np.sqrt(var)[1:], mean[1:] - n_sigma * np.sqrt(var)[1:], alpha=.5,
                        label=r"combined $\sigma$")

    for ax in axs:
        ax.plot(date[:N_seen_points], x[:N_seen_points, 0], c="#000000", alpha=1, label="seen input sequence")
        ax.plot(date[N_seen_points:], x[N_seen_points:, 0], c="#000000", alpha=.1, label="unseen future")
        ax.axvline(x=date[N_seen_points], ymin=0, ymax=1)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(date[1:], mean[1:],c=tumorange)

    #for y_pred in y_hat:
    #    label = "prediction" if idx == 0 else None
    #    ax.plot(date[:N_seen_points + future], y_pred, c=tumorange, label=label, alpha=(1 / N_predictions) ** 0.7)

    [ax.legend(ncol=3) for ax in axs]

    return fig, axs