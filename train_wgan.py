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
                                    shuffle=True,
                                    drop_last=True)
    return train_dataset


def train_wgan(seen, path_save):
    generator = WGANGenerator()
    discriminator = WGANDiscriminator(input_size=512)

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    train_dataset = load_data(seen)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.00001)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)

    # ----------
    #  Training
    # ----------

    batches_done = 0
    num_epochs = 3000
    for epoch in range(num_epochs):

        for i, real_batch in enumerate(train_dataset):

            # Configure input
            real_imgs = Variable(real_batch.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 100, (real_batch.shape[0], 400))))

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(
                discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train the generator every n_critic iterations
            if i % 5 == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

    if seen:
        torch.save(generator.state_dict(),
                   path_save + 'weights/seen/generator_wgan_seen.pth')
    else:
        torch.save(generator.state_dict(),
                   path_save + 'weights/unseen/generator_wgan_unseen.pth')
