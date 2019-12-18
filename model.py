import torch
import numpy as np
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5, device=torch.device("cpu")):
        super(Model, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inlinear = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size) #nn.LSTMCell
        self.linear = nn.Linear(self.hidden_size, self.output_size + 1)
        self.to(device)

    def forward(self, input, future=0, y=None):
        outputs = []
        log_variances = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence

        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            input_t = self.relu(self.inlinear(input_t))
            input_t = self.dropout(input_t)

            h_t, c_t = self.lstm(input_t.squeeze(1), (h_t, c_t))
            h_t = self.dropout(h_t)
            output_logvariance = self.linear(h_t)
            #output = output + bias
            output = output_logvariance[:,0][:,None]
            outputs += [output]
            log_variance = self.relu(output_logvariance[:,-1])[:,None]
            log_variances += [log_variance]
        for i in range(future):
            if y is not None and np.random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing

            output = self.relu(self.inlinear(output).squeeze(1))
            output = self.dropout(output)

            h_t, c_t = self.lstm(output, (h_t, c_t))
            h_t = self.dropout(h_t)
            output_logvariance = self.linear(h_t)

            output = output_logvariance[:,0][:,None]
            outputs += [output]
            log_variance = self.relu(output_logvariance[:,-1])[:,None]
            log_variances += [log_variance]

        outputs = torch.stack(outputs, 1)
        log_variances = torch.stack(log_variances, 1)
        return outputs, log_variances

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
