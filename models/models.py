from . import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Example Models
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)  # output size: 32x32
        self.conv2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)  # output size: 16x16
        self.dense = nn.Linear(16 * 16 * 16, 512)  # Added dense layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.dense(x) # F.relu(self.dense(x))  # Apply dense layer
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(512, 16 * 16 * 16)  # Added dense layer
        self.t_conv1 = nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1)  # output size: 32x32
        self.t_conv2 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)  # output size: 64x64

    def forward(self, x):
        x = self.dense(x) # F.relu(self.dense(x))  # Apply dense layer
        x = x.view(x.size(0), 16, 16, 16)  # Reshape the input
        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)  # No activation here
        return x

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)  # we only want the last 3 time steps

        return out