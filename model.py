import torch
import numpy as np
import torch.nn as nn

def variance(y_hat, var_hat):
    """eq 9 in Kendall & Gal"""
    T = y_hat.shape[0]
    sum_squares = (1/T) * (y_hat**2).sum(0)
    squared_sum = ((1/T) * y_hat.sum(0))**2
    epi_var = sum_squares - squared_sum
    ale_var = (1/T) * (var_hat**2).sum(0)
    return epi_var, ale_var

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, device=torch.device("cpu"), num_layers=1):
        super(Model, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.inlinear = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(self.hidden_size,
                                  self.hidden_size,
                                  num_layers=num_layers,
                                  dropout=dropout if num_layers > 1 else 0.0, # dropout only applied with more than one layer
                                  bidirectional=False,
                                  batch_first=True)
        #self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size) #nn.LSTMCell
        self.outlinear = nn.Linear(self.hidden_size, self.output_size + 1)
        self.to(device)

    def forward(self, input, future=0, y=None, date=None, date_future=None):
        # reset the state of LSTM
        # the state is kept till the end of the sequence

        #input = input - input.mean(1)[:, :, None]
        #input = input / input.std(1)[:, :, None]

        if date is not None:
            input = input[:,:,0]
            input = torch.stack([input,date],2)

        h_t = torch.zeros(self.num_layers, input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(self.num_layers, input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)

        outputs, log_variances = self.encode(input,h_t, c_t)

        if future > 0:
            future_outputs, future_logvariances = self.decode(outputs[:,-1], h_t, c_t, future,y=y, date=date_future)
            outputs = torch.cat([outputs,future_outputs],1)
            log_variances = torch.cat([log_variances,future_logvariances],1)

        return outputs, log_variances

    def predict(self, input, n_predictions, future=0, date=None, date_future=None, return_yhat=False):
        outputs = list()
        variances = list()

        for n in range(n_predictions):
            with torch.no_grad():
                output, log_variance = self.forward(input, future, date=date, date_future=date_future)
            outputs.append(output)
            variances.append(log_variance.exp())

        var_hat = torch.stack(variances)
        y_hat = torch.stack(outputs)

        epi_var, ale_var = variance(y_hat, var_hat)
        mean = y_hat.mean(0)

        if return_yhat:
            return mean, epi_var, ale_var, y_hat
        else:
            return mean, epi_var, ale_var



    def encode(self,input, h_t, c_t):

        input = self.inlinear(input)
        input = self.relu(input)
        input = self.dropout(input)

        output, (h_t, c_t) = self.lstm(input, (h_t, c_t))
        output = self.dropout(output)
        output = self.outlinear(output)

        outputs = output[:, :, 0, None]
        log_variances = output[:, :, -1, None]

        return outputs, log_variances

    def decode(self, future_input, h_t, c_t, future, y=None, date=None):
        future_outputs = list()
        future_logvariances = list()
        for i in range(future):
            if y is not None and np.random.random() > 0.5:
                future_input = y[:, [i]]  # teacher forcing
            if date is not None:
                # take next time instance [1,1]
                next_date = date[:,i].unsqueeze(0)

                # concatenate with future input and ensure [N, D] dimensions
                future_input = torch.stack([future_input, next_date],2).squeeze(1)

            future_input = self.inlinear(future_input)
            future_input = self.relu(future_input)
            future_input = self.dropout(future_input)

            future_input, (h_t, c_t) = self.lstm(future_input[:, None, :], (h_t, c_t))
            future_input = self.dropout(future_input)
            future_input_logvariance = self.outlinear(future_input)

            future_input = future_input_logvariance[:, :, 0]
            future_logvariance = future_input_logvariance[:, :, 1]

            future_outputs.append(future_input)
            future_logvariances.append(future_logvariance)

        future_outputs = torch.stack(future_outputs, 1)
        future_logvariances = torch.stack(future_logvariances, 1)
        return future_outputs, future_logvariances

def snapshot(model, optimizer, path):
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict()),
        path)

def restore(path, model, optimizer=None):
    snapshot = torch.load(path)
    model.load_state_dict(snapshot["model"])
    if optimizer is not None:
        optimizer.load_state_dict(snapshot["optimizer"])
