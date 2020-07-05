import struct
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import os
from os import path
import pickle
from discriminator.discriminator import *
from torch.utils import data
from generator.generator import *
from loss.loss import *
import itertools

cuda = True if torch.cuda.is_available() else False

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
batch_size = 10


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def load_data(seen):
    # ------------LOAD DATA SIGNAL----------#
    if seen == False:
        with open(
                '/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/signal_unseen.pkl',
                'rb') as file:
            signals = pickle.load(file)
    else:
        with open(
                '/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/signal_seen.pkl',
                'rb') as file:
            signals = pickle.load(file)
    for key in signals.keys():
        signals[key] = torch.from_numpy(signals[key])
        signals[key] = signals[key].float()
    train_dataset = data.DataLoader(signals,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    shuffle=True)
    return train_dataset


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 400))
    return n


def train_discriminator(discriminator, loss, optimizer, real_data, fake_data):
    N = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = discriminator(real_data)

    error_real = loss(prediction_real, ones_target(N).cuda())
    error_real.backward()

    prediction_fake = discriminator(fake_data)

    error_fake = loss(prediction_fake, zeros_target(N).cuda())
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake


def train_generator(discriminator, loss, optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # Sample noise and generate fake data
    pred_fake = discriminator(fake_data)

    error = loss(pred_fake, ones_target(N).cuda())
    error.backward()

    optimizer.step()
    # Return error
    return error


def init():

    #-----------INIT GENERATOR-----------#
    generator = GeneratorNet()
    generator.apply(weights_init)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005)

    #----------INIT DISCRIMINATOR--------#
    discriminator = DiscriminatorNet()
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    return discriminator, generator, optimizer_D, optimizer_G


def train(seen, path_save):
    train_dataset = load_data(seen)
    # num_batches = len(train_dataset)
    num_epochs = 1000
    # batches_done = 0
    loss = nn.BCELoss()

    disc_loss = []
    gen_loss = []

    discriminator, generator, optimizer_D, optimizer_G = init()

    for epoch in range(num_epochs):
        for index, real_batch in enumerate(train_dataset):

            N = real_batch.size(0)

            real_data = Variable(real_batch)

            noise_gen = Variable(noise(N))
            fake_data = generator(noise_gen.cuda()).detach()

            d_error = train_discriminator(discriminator, loss, optimizer_D,
                                          real_data.cuda(), fake_data.cuda())

            noise_gen = Variable(noise(N))
            fake_data = generator(noise_gen.cuda())

            g_error = train_generator(discriminator, loss, optimizer_G,
                                      fake_data.cuda())

            disc_loss.append(d_error)
            gen_loss.append(g_error)

    if seen:
        torch.save(discriminator.state_dict(),
                   path_save + 'weights/seen/discriminator_seen.pth')
        torch.save(generator.state_dict(),
                   path_save + 'weights/seen/generator_seen.pth')
    else:
        torch.save(discriminator.state_dict(),
                   path_save + 'weights/unseen/discriminator_unseen.pth')
        torch.save(generator.state_dict(),
                   path_save + 'weights/unseen/generator_unseen.pth')


# train(False)
# train(True)
