import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, n_channels, ts_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.ts_length = ts_length

        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, n_channels * ts_length)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.n_channels, self.ts_length)


class Discriminator(nn.Module):
    def __init__(self, n_channels, ts_length):
        super().__init__()
        self.n_channels = n_channels
        self.ts_length = ts_length
        self.fc1 = nn.Linear(n_channels * ts_length, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
