import numpy as np

import torch
import torch.nn as nn

import pickle
import torch.utils.data as data_utils
from torch.utils import data
from model.model import Linear, weights_init


def load_data(seen):
    # ------------LOAD DATA SIGNAL----------#
    if seen == False:
        with open('./data/signal_unseen.pkl', 'rb') as file:
            signals = pickle.load(file)
    else:
        with open('./data/signal_seen.pkl', 'rb') as file:
            signals = pickle.load(file)
    for key in signals.keys():
        signals[key] = torch.from_numpy(signals[key])
        signals[key] = signals[key].float()
    return signals


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def train():
    signals_seen = load_data(False)
    signals_unseen = load_data(True)

    # BALANCE THE DATASETS
    minimum = min(len(signals_seen), len(signals_unseen))
    number_seen = len(signals_seen) - minimum
    number_unseen = len(signals_unseen) - minimum

    for i in range(
            len(signals_seen) - 1,
            len(signals_seen) - number_seen - 1, -1):
        signals_seen.pop(i)

    for i in range(
            len(signals_unseen) - 1,
            len(signals_unseen) - number_unseen - 1, -1):
        signals_unseen.pop(i)

    #SPLIT IN TRAIN AND TEST
    size_train = round(len(signals_seen) * 0.8)
    train_unseen = []
    train_seen = []

    for i in range(0, size_train):
        train_unseen.append(signals_unseen[i])
        train_seen.append(signals_seen[i])

    test_unseen = []
    test_seen = []

    for i in range(size_train, len(signals_unseen)):
        test_unseen.append(signals_unseen[i])
        test_seen.append(signals_seen[i])

    #TRANSFORM TO TENSOR
    train_seen = torch.stack(train_seen)
    train_unseen = torch.stack(train_unseen)

    test_seen = torch.stack(test_seen)
    test_unseen = torch.stack(test_unseen)
    train = torch.cat((train_seen, train_unseen))
    test = torch.cat((test_seen, test_unseen))

    targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
    targets_seen = torch.from_numpy(np.ones(len(train_seen)))
    targets = torch.cat((targets_seen, targets_unseen))

    test_targets_unseen = torch.from_numpy(np.zeros(len(test_seen)))
    test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
    test_targets = torch.cat((test_targets_seen, test_targets_unseen))

    train = data_utils.TensorDataset(train, targets)
    test = data_utils.TensorDataset(test, test_targets)
    loader = data.DataLoader(train, batch_size=5, shuffle=True)
    loader_test = data.DataLoader(test, batch_size=5, shuffle=False)

    discriminator = Linear()
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)
    num_epochs = 100
    loss = nn.BCELoss()

    discriminator = discriminator.cuda()

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        discriminator.train()
        for _, (real_batch, targets) in enumerate(loader):
            optimizer_D.zero_grad()

            preds, _ = discriminator(real_batch.cuda())

            error = loss(preds.cuda(), targets.unsqueeze(1).float().cuda())
            acc = binary_acc(preds.cuda(), targets.unsqueeze(1).cuda())
            error.backward()

            optimizer_D.step()
            train_loss += error
            train_acc += acc

        train_losses.append(train_loss / len(loader))
        train_accuracies.append(train_acc / len(loader))

        val_loss = 0
        val_acc = 0

        discriminator.eval()
        for _, (real_batch, targets) in enumerate(loader_test):

            preds, _ = discriminator(real_batch.cuda())

            error = loss(preds.detach().cuda(),
                         targets.unsqueeze(1).float().cuda())

            acc = binary_acc(preds.cuda(), targets.unsqueeze(1).cuda())

            val_loss += error
            val_acc += acc

        test_losses.append(val_loss / len(loader_test))
        test_accuracies.append(val_acc / len(loader_test))
    return train_losses, train_accuracies, test_losses, test_accuracies
