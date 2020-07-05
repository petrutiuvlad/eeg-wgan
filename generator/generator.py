import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Sequential(nn.Linear(400, 256), nn.LeakyReLU(0.2))

        self.up = nn.Upsample(size=512, mode='nearest')

        # Input is the latent vector Z.
        self.conv1 = nn.Conv1d(1,
                               256,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv1_same = nn.Conv1d(256,
                                    256,
                                    kernel_size=1,
                                    padding=0,
                                    bias=False)
        self.bn1_same = nn.BatchNorm1d(256)

        # Input Dimension: (ngf*8) x 4 x 4
        self.conv2 = nn.Conv1d(256, 128, 2, 2, 0, bias=False)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv2_same = nn.Conv1d(128,
                                    128,
                                    kernel_size=1,
                                    padding=0,
                                    bias=False)
        self.bn2_same = nn.BatchNorm1d(128)

        # Input Dimension: (ngf*4) x 8 x 8
        self.conv3 = nn.Conv1d(128, 64, 2, 2, 0, bias=False)
        self.bn3 = nn.BatchNorm1d(64)

        # Input Dimension: (ngf*2) x 16 x 16
        self.conv4 = nn.Conv1d(128, 1, 2, 2, 0, bias=False)

        self.avg = nn.AvgPool1d(2, stride=2)

        self.relu = nn.LeakyReLU(0.2)

        self.out = nn.Sequential(nn.Linear(32, 512))

    def forward(self, x):
        x = self.input(x)
        x = x.view(-1, 1, 256)
        x = self.up(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1_same(x)
        x = self.bn1_same(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_same(x)
        x = self.bn2_same(x)
        x = self.relu(x)
        x = self.avg(x)

        x = self.conv4(x)

        x = self.out(x)
        x = x.view(-1, 512)
        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 400
        n_out = 512

        self.hidden0 = nn.Sequential(nn.Linear(n_features, 256),
                                     nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))

        self.out = nn.Sequential(nn.Linear(1024, n_out), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class WGANGenerator(nn.Module):
    def __init__(self, output_size=512, layers=[256, 512, 1024]):
        super(WGANGenerator, self).__init__()

        self.input_size = 256
        self.latent_dim = 400
        self.output_size = output_size

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, self.input_size, normalize=False),
            *block(self.input_size, layers[0]), *block(layers[0], layers[1]),
            *block(layers[1], layers[2]), nn.Linear(layers[2],
                                                    self.output_size),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z):
        signal = self.model(z)
        return signal