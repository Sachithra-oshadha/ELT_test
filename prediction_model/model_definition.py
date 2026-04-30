import torch
import torch.nn as nn

class BiLSTMQuantile(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1, quantiles: int = 3):
        super(BiLSTMQuantile, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_size * quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = out.view(batch_size, self.output_size, self.quantiles)

        return out