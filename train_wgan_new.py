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


def compute_gamma(signals):
    signals_mag = torch.FloatTensor(size=(signals.shape[0], 227))
    signals_phase = torch.FloatTensor(size=(signals.shape[0], 227))
    k = 0
    for signal in signals:
        signals_mag[k] = torch.rfft(torch.tensor(signal),
                                    signal_ndim=1,
                                    onesided=True)[30:257, 0]
        signals_phase[k] = torch.rfft(torch.tensor(signal),
                                      signal_ndim=1,
                                      onesided=True)[30:257, 1]
        k += 1

    return signals_mag.cuda(), signals_phase.cuda()


def compute_rest(signals):
    signals_mag = torch.FloatTensor(size=(signals.shape[0], 30))
    signals_phase = torch.FloatTensor(size=(signals.shape[0], 30))
    k = 0
    for signal in signals:
        signals_mag[k] = torch.rfft(torch.tensor(signal),
                                    signal_ndim=1,
                                    onesided=True)[0:30, 0]
        signals_phase[k] = torch.rfft(torch.tensor(signal),
                                      signal_ndim=1,
                                      onesided=True)[0:30, 1]
        k += 1

    return signals_mag.cuda(), signals_phase.cuda()


def train_wgan(seen, path_save):
    generator_mag_gamma = WGANGenerator(output_size=227)
    generator_phase_gamma = WGANGenerator(output_size=227)
    discriminator_mag_gamma = WGANDiscriminator(input_size=227)
    discriminator_phase_gamma = WGANDiscriminator(input_size=227)
    generator_mag_rest = WGANGenerator(output_size=30)
    generator_phase_rest = WGANGenerator(output_size=30)
    discriminator_mag_rest = WGANDiscriminator(input_size=30)
    discriminator_phase_rest = WGANDiscriminator(input_size=30)

    if cuda:
        generator_mag_gamma = generator_mag_gamma.cuda()
        generator_phase_gamma = generator_phase_gamma.cuda()
        discriminator_mag_gamma = discriminator_mag_gamma.cuda()
        discriminator_phase_gamma = discriminator_phase_gamma.cuda()
        generator_mag_rest = generator_mag_rest.cuda()
        generator_phase_rest = generator_phase_rest.cuda()
        discriminator_mag_rest = discriminator_mag_rest.cuda()
        discriminator_phase_rest = discriminator_phase_rest.cuda()

    train_dataset = load_data(seen)

    optimizer_G_mag_gamma = torch.optim.Adam(generator_mag_gamma.parameters(),
                                             lr=0.0001)
    optimizer_G_phase_gamma = torch.optim.Adam(
        generator_phase_gamma.parameters(), lr=0.0001)
    optimizer_D_mag_gamma = torch.optim.Adam(
        discriminator_mag_gamma.parameters(), lr=0.0001)
    optimizer_D_phase_gamma = torch.optim.Adam(
        discriminator_phase_gamma.parameters(), lr=0.0001)
    optimizer_G_mag_rest = torch.optim.Adam(generator_mag_rest.parameters(),
                                            lr=0.0001)
    optimizer_G_phase_rest = torch.optim.Adam(
        generator_phase_rest.parameters(), lr=0.0001)
    optimizer_D_mag_rest = torch.optim.Adam(
        discriminator_mag_rest.parameters(), lr=0.0001)
    optimizer_D_phase_rest = torch.optim.Adam(
        discriminator_phase_rest.parameters(), lr=0.0001)
    # ----------
    #  Training
    # ----------

    batches_done = 0
    num_epochs = 12000
    all_losses_D_gamma = []
    all_losses_D_rest = []
    all_losses_G = []
    all_losses_R = []
    for epoch in range(num_epochs):

        int_losses_D_gamma = []
        int_losses_D_rest = []
        int_losses_G = []
        int_losses_R = []
        for i, real_batch in enumerate(train_dataset):

            # Configure input
            real_imgs = Variable(real_batch.type(Tensor))

            real_signal_mag_gamma, real_signal_phase_gamma = compute_gamma(
                real_imgs)
            real_signal_mag_rest, real_signal_phase_rest = compute_rest(
                real_imgs)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            #optimizer_D.zero_grad()
            optimizer_D_mag_gamma.zero_grad()
            optimizer_D_phase_gamma.zero_grad()
            optimizer_D_mag_rest.zero_grad()
            optimizer_D_phase_rest.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 100, (real_batch.shape[0], 400))))

            # Generate a batch of images
            fake_signal_mag_gamma = generator_mag_gamma(z).detach()
            fake_signal_phase_gamma = generator_phase_gamma(z).detach()
            fake_signal_mag_rest = generator_mag_rest(z).detach()
            fake_signal_phase_rest = generator_phase_rest(z).detach()

            # Adversarial loss
            loss_D_mag_gamma = -torch.mean(
                discriminator_mag_gamma(real_signal_mag_gamma)) + torch.mean(
                    discriminator_mag_gamma(fake_signal_mag_gamma))
            loss_D_phase_gamma = -torch.mean(
                discriminator_phase_gamma(real_signal_phase_gamma)
            ) + torch.mean(discriminator_phase_gamma(fake_signal_phase_gamma))

            loss_D_mag_rest = -torch.mean(
                discriminator_mag_rest(real_signal_mag_rest)) + torch.mean(
                    discriminator_mag_rest(fake_signal_mag_rest))
            loss_D_phase_rest = -torch.mean(
                discriminator_phase_rest(real_signal_phase_rest)) + torch.mean(
                    discriminator_phase_rest(fake_signal_phase_rest))

            loss_D_gamma = (loss_D_mag_gamma + loss_D_phase_gamma)
            loss_D_rest = (loss_D_mag_rest + loss_D_phase_rest)

            int_losses_D_gamma.append(loss_D_gamma.item())
            loss_D_gamma.backward()
            optimizer_D_mag_gamma.step()
            optimizer_D_phase_gamma.step()

            int_losses_D_rest.append(loss_D_rest.item())
            loss_D_rest.backward()
            optimizer_D_mag_rest.step()
            optimizer_D_phase_rest.step()

            # Clip weights of discriminator
            for p in discriminator_mag_gamma.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in discriminator_phase_gamma.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in discriminator_mag_rest.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in discriminator_phase_rest.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train the generator every n_critic iterations
            if i % 5 == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G_mag_gamma.zero_grad()
                optimizer_G_phase_gamma.zero_grad()
                optimizer_G_mag_rest.zero_grad()
                optimizer_G_phase_rest.zero_grad()

                gen_signals_mag_gamma = generator_mag_gamma(z)
                gen_signals_phase_gamma = generator_phase_gamma(z)
                gen_signals_mag_rest = generator_mag_rest(z)
                gen_signals_phase_rest = generator_phase_rest(z)

                loss_G_mag_gamma = -torch.mean(
                    discriminator_mag_gamma(gen_signals_mag_gamma))
                loss_G_phase_gamma = -torch.mean(
                    discriminator_phase_gamma(gen_signals_phase_gamma))
                loss_G_mag_rest = -torch.mean(
                    discriminator_mag_rest(gen_signals_mag_rest))
                loss_G_phase_rest = -torch.mean(
                    discriminator_phase_rest(gen_signals_phase_rest))
                loss_G_gamma = (loss_G_mag_gamma + loss_G_phase_gamma)
                loss_G_rest = (loss_G_mag_rest + loss_G_phase_rest)

                # print("Generator loss {}".format(loss_G))
                int_losses_G.append(loss_G_gamma.item())
                loss_G_gamma.backward()
                optimizer_G_mag_gamma.step()
                optimizer_G_phase_gamma.step()

                loss_G_rest.backward()
                int_losses_R.append(loss_G_rest.item())
                optimizer_G_mag_rest.step()
                optimizer_G_phase_rest.step()

        all_losses_D_gamma.append(
            torch.mean(torch.FloatTensor(int_losses_D_gamma)))
        all_losses_D_rest.append(
            torch.mean(torch.FloatTensor(int_losses_D_rest)))
        all_losses_G.append(torch.mean(torch.FloatTensor(int_losses_G)))
        all_losses_R.append(torch.mean(torch.FloatTensor(int_losses_R)))

    torch.save(
        all_losses_G,
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_G.pt"
    )
    torch.save(
        all_losses_D_gamma,
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_D_gamma.pt"
    )
    torch.save(
        all_losses_D_rest,
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_D_rest.pt"
    )
    torch.save(
        all_losses_R,
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_R.pt"
    )

    if seen:

        torch.save(generator_mag_gamma.state_dict(),
                   path_save + 'generator_mag_gamma_wgan_seen.pth')

        torch.save(generator_phase_gamma.state_dict(),
                   path_save + 'generator_phase_gamma_wgan_seen.pth')

        torch.save(generator_mag_rest.state_dict(),
                   path_save + 'generator_mag_rest_wgan_seen.pth')

        torch.save(generator_phase_rest.state_dict(),
                   path_save + 'generator_phase_rest_wgan_seen.pth')
    else:
        torch.save(generator_mag_gamma.state_dict(),
                   path_save + 'generator_mag_gamma_wgan_unseen.pth')

        torch.save(generator_phase_gamma.state_dict(),
                   path_save + 'generator_phase_gamma_wgan_unseen.pth')

        torch.save(generator_mag_rest.state_dict(),
                   path_save + 'generator_mag_rest_wgan_unseen.pth')

        torch.save(generator_phase_rest.state_dict(),
                   path_save + 'generator_phase_rest_wgan_unseen.pth')


path_channel = '/home/vlad/Desktop/test_freq/'
train_wgan(True, path_channel)
train_wgan(False, path_channel)
