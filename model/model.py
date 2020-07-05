import torch
import torch.nn as nn


class Linear(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(Linear, self).__init__()
        n_out = 1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.batchNorm1 = nn.BatchNorm1d(16)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.batchNorm2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=2)
        self.batchNorm3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=2, stride=2)
        self.batchNorm4 = nn.BatchNorm1d(32)
        self.out1 = nn.Linear(32 * 127, 256)
        self.out2 = nn.Linear(256, n_out)
        self.final = nn.Sigmoid()

        self.gradients = None

    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = x.view(-1, 1, 512)
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.leakyRelu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.leakyRelu(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = self.leakyRelu(x)
        x = self.drop(x)
        x = self.conv4(x)
        features = x
        x.register_hook(self.save_gradients)
        x = self.batchNorm4(x)
        x = self.leakyRelu(x)
        x = self.drop(x)
        x = x.view(-1, 32 * 127)
        x = self.out1(x)
        x = self.leakyRelu(x)
        x = self.drop(x)
        x = self.out2(x)
        output = self.final(x)
        return output, features, self.gradients


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