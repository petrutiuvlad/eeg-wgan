import torch
import torch.nn as nn


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 512
        n_out = 1

        self.hidden0 = nn.Sequential(nn.Linear(n_features, 1024),
                                     nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2),
                                     nn.Dropout(0.3))
        self.hidden2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2),
                                     nn.Dropout(0.3))

        self.out = nn.Sequential(torch.nn.Linear(256, n_out),
                                 torch.nn.Sigmoid())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        output = self.out(x)
        return output


class Discriminator(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 512
        n_out = 1

        self.init_model = nn.Sequential(nn.Linear(n_features, 512),
                                        nn.LeakyReLU(0.2, inplace=True))

        self.conv_model = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 256, kernel_size=2, stride=2), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True), nn.Conv1d(256, 512,
                                                       kernel_size=2),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True))

        self.out = nn.Sequential(torch.nn.Linear(512 * 254, n_out),
                                 torch.nn.Sigmoid())

    def forward(self, x):
        x = self.init_model(x)
        x = x.view(-1, 1, 512)
        features = self.conv_model(x)
        features = features.view(-1, 254 * 512)
        output = self.out(features)
        return output


class WGANDiscriminator(nn.Module):
    def __init__(self, input_size, layers=[1024, 512, 256]):
        super(WGANDiscriminator, self).__init__()

        self.input_size = input_size

        self.model = nn.Sequential(nn.Linear(self.input_size, layers[0]),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(layers[0], layers[1]),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(layers[1], layers[2]),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(layers[2], 1))

    def forward(self, signal):
        validity = self.model(signal)
        return validity
