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
batch_size = 16


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
        signals_mag[k] = torch.rfft(signal.clone(),
                                    signal_ndim=1,
                                    onesided=True)[30:257, 0]
        signals_phase[k] = torch.rfft(signal.clone(),
                                      signal_ndim=1,
                                      onesided=True)[30:257, 1]
        k += 1

    return signals_mag.cuda(), signals_phase.cuda()


def compute_tetha(signals):
    signals_mag = torch.FloatTensor(size=(signals.shape[0], 8))
    signals_phase = torch.FloatTensor(size=(signals.shape[0], 8))
    k = 0
    for signal in signals:
        signals_mag[k] = torch.rfft(signal.clone(),
                                    signal_ndim=1,
                                    onesided=True)[0:8, 0]
        signals_phase[k] = torch.rfft(signal.clone(),
                                      signal_ndim=1,
                                      onesided=True)[0:8, 1]
        k += 1

    return signals_mag.cuda(), signals_phase.cuda()


def compute_alpha_beta(signals):
    signals_mag = torch.FloatTensor(size=(signals.shape[0], 22))
    signals_phase = torch.FloatTensor(size=(signals.shape[0], 22))
    k = 0
    for signal in signals:
        signals_mag[k] = torch.rfft(signal.clone(),
                                    signal_ndim=1,
                                    onesided=True)[8:30, 0]
        signals_phase[k] = torch.rfft(signal.clone(),
                                      signal_ndim=1,
                                      onesided=True)[8:30, 1]
        k += 1

    return signals_mag.cuda(), signals_phase.cuda()


def train_wgan_freq(seen, path_save):
    generator_mag_gamma = WGANGenerator(output_size=227)
    generator_phase_gamma = WGANGenerator(output_size=227)
    discriminator_mag_gamma = WGANDiscriminator(input_size=227)
    discriminator_phase_gamma = WGANDiscriminator(input_size=227)

    generator_mag_ab = WGANGenerator(output_size=22)
    generator_phase_ab = WGANGenerator(output_size=22)
    discriminator_mag_ab = WGANDiscriminator(input_size=22)
    discriminator_phase_ab = WGANDiscriminator(input_size=22)

    generator_mag_rest = WGANGenerator(output_size=8)
    generator_phase_rest = WGANGenerator(output_size=8)
    discriminator_mag_rest = WGANDiscriminator(input_size=8)
    discriminator_phase_rest = WGANDiscriminator(input_size=8)

    if cuda:
        generator_mag_gamma = generator_mag_gamma.cuda()
        generator_phase_gamma = generator_phase_gamma.cuda()
        discriminator_mag_gamma = discriminator_mag_gamma.cuda()
        discriminator_phase_gamma = discriminator_phase_gamma.cuda()

        generator_mag_ab = generator_mag_ab.cuda()
        generator_phase_ab = generator_phase_ab.cuda()
        discriminator_mag_ab = discriminator_mag_ab.cuda()
        discriminator_phase_ab = discriminator_phase_ab.cuda()

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

    optimizer_G_mag_ab = torch.optim.Adam(generator_mag_ab.parameters(),
                                          lr=0.0001)
    optimizer_G_phase_ab = torch.optim.Adam(generator_phase_ab.parameters(),
                                            lr=0.0001)
    optimizer_D_mag_ab = torch.optim.Adam(discriminator_mag_ab.parameters(),
                                          lr=0.0001)
    optimizer_D_phase_ab = torch.optim.Adam(
        discriminator_phase_ab.parameters(), lr=0.0001)

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
    all_losses_D_ab = []
    all_losses_D_rest = []

    all_losses_G_gamma = []
    all_losses_G_ab = []
    all_losses_G_rest = []
    for epoch in range(num_epochs):

        int_losses_D_gamma = []
        int_losses_D_ab = []
        int_losses_D_rest = []

        int_losses_G_gamma = []
        int_losses_G_ab = []
        int_losses_G_rest = []
        for i, real_batch in enumerate(train_dataset):

            # Configure input
            real_imgs = Variable(real_batch.type(Tensor))

            real_signal_mag_gamma, real_signal_phase_gamma = compute_gamma(
                real_imgs)
            real_signal_mag_ab, real_signal_phase_ab = compute_alpha_beta(
                real_imgs)
            real_signal_mag_rest, real_signal_phase_rest = compute_tetha(
                real_imgs)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            #optimizer_D.zero_grad()
            optimizer_D_mag_gamma.zero_grad()
            optimizer_D_phase_gamma.zero_grad()
            optimizer_D_mag_ab.zero_grad()
            optimizer_D_phase_ab.zero_grad()
            optimizer_D_mag_rest.zero_grad()
            optimizer_D_phase_rest.zero_grad()

            # Sample noise as generator input
            z = Variable(
                Tensor(np.random.normal(0, 100, (real_batch.shape[0], 400))))

            # Generate a batch of images
            fake_signal_mag_gamma = generator_mag_gamma(z).detach()
            fake_signal_phase_gamma = generator_phase_gamma(z).detach()

            fake_signal_mag_ab = generator_mag_ab(z).detach()
            fake_signal_phase_ab = generator_phase_ab(z).detach()

            fake_signal_mag_rest = generator_mag_rest(z).detach()
            fake_signal_phase_rest = generator_phase_rest(z).detach()

            # Adversarial loss
            loss_D_mag_gamma = -torch.mean(
                discriminator_mag_gamma(real_signal_mag_gamma)) + torch.mean(
                    discriminator_mag_gamma(fake_signal_mag_gamma))
            loss_D_phase_gamma = -torch.mean(
                discriminator_phase_gamma(real_signal_phase_gamma)
            ) + torch.mean(discriminator_phase_gamma(fake_signal_phase_gamma))

            loss_D_mag_ab = -torch.mean(
                discriminator_mag_ab(real_signal_mag_ab)) + torch.mean(
                    discriminator_mag_ab(fake_signal_mag_ab))
            loss_D_phase_ab = -torch.mean(
                discriminator_phase_ab(real_signal_phase_ab)) + torch.mean(
                    discriminator_phase_ab(fake_signal_phase_ab))

            loss_D_mag_rest = -torch.mean(
                discriminator_mag_rest(real_signal_mag_rest)) + torch.mean(
                    discriminator_mag_rest(fake_signal_mag_rest))
            loss_D_phase_rest = -torch.mean(
                discriminator_phase_rest(real_signal_phase_rest)) + torch.mean(
                    discriminator_phase_rest(fake_signal_phase_rest))

            loss_D_gamma = loss_D_mag_gamma + loss_D_phase_gamma
            loss_D_ab = loss_D_mag_ab + loss_D_phase_ab
            loss_D_rest = loss_D_mag_rest + loss_D_phase_rest

            int_losses_D_gamma.append(loss_D_gamma.item())
            int_losses_D_ab.append(loss_D_ab.item())
            int_losses_D_rest.append(loss_D_rest.item())

            loss_D_gamma.backward()
            optimizer_D_mag_gamma.step()
            optimizer_D_phase_gamma.step()

            loss_D_ab.backward()
            optimizer_D_mag_ab.step()
            optimizer_D_phase_ab.step()

            loss_D_rest.backward()
            optimizer_D_mag_rest.step()
            optimizer_D_phase_rest.step()

            # Clip weights of discriminator
            for p in discriminator_mag_gamma.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in discriminator_phase_gamma.parameters():
                p.data.clamp_(-0.01, 0.01)

            for p in discriminator_mag_ab.parameters():
                p.data.clamp_(-0.01, 0.01)
            for p in discriminator_phase_ab.parameters():
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

                optimizer_G_mag_ab.zero_grad()
                optimizer_G_phase_ab.zero_grad()

                optimizer_G_mag_rest.zero_grad()
                optimizer_G_phase_rest.zero_grad()

                gen_signals_mag_gamma = generator_mag_gamma(z)
                gen_signals_phase_gamma = generator_phase_gamma(z)

                gen_signals_mag_ab = generator_mag_ab(z)
                gen_signals_phase_ab = generator_phase_ab(z)

                gen_signals_mag_rest = generator_mag_rest(z)
                gen_signals_phase_rest = generator_phase_rest(z)

                loss_G_mag_gamma = -torch.mean(
                    discriminator_mag_gamma(gen_signals_mag_gamma))
                loss_G_phase_gamma = -torch.mean(
                    discriminator_phase_gamma(gen_signals_phase_gamma))

                loss_G_mag_ab = -torch.mean(
                    discriminator_mag_ab(gen_signals_mag_ab))
                loss_G_phase_ab = -torch.mean(
                    discriminator_phase_ab(gen_signals_phase_ab))

                loss_G_mag_rest = -torch.mean(
                    discriminator_mag_rest(gen_signals_mag_rest))
                loss_G_phase_rest = -torch.mean(
                    discriminator_phase_rest(gen_signals_phase_rest))

                loss_G_gamma = loss_G_mag_gamma + loss_G_phase_gamma
                loss_G_ab = loss_G_mag_ab + loss_G_phase_ab
                loss_G_rest = loss_G_mag_rest + loss_G_phase_rest

                # print("Generator loss {}".format(loss_G))
                int_losses_G_gamma.append(loss_G_gamma.item())
                int_losses_G_ab.append(loss_G_ab.item())
                int_losses_G_rest.append(loss_G_rest.item())

                loss_G_gamma.backward()
                optimizer_G_mag_gamma.step()
                optimizer_G_phase_gamma.step()

                loss_G_ab.backward()
                optimizer_G_mag_ab.step()
                optimizer_G_phase_ab.step()

                loss_G_rest.backward()
                optimizer_G_mag_rest.step()
                optimizer_G_phase_rest.step()

        all_losses_D_gamma.append(
            torch.mean(torch.FloatTensor(int_losses_D_gamma)))
        all_losses_D_ab.append(torch.mean(torch.FloatTensor(int_losses_D_ab)))
        all_losses_D_rest.append(
            torch.mean(torch.FloatTensor(int_losses_D_rest)))
        all_losses_G_gamma.append(
            torch.mean(torch.FloatTensor(int_losses_G_gamma)))
        all_losses_G_ab.append(torch.mean(torch.FloatTensor(int_losses_G_ab)))
        all_losses_G_rest.append(
            torch.mean(torch.FloatTensor(int_losses_G_rest)))

    # torch.save(
    #     all_losses_D_gamma,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_D_gamma.pt"
    # )
    # torch.save(
    #     all_losses_D_ab,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_D_ab.pt"
    # )
    # torch.save(
    #     all_losses_D_rest,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_D_rest.pt"
    # )
    # torch.save(
    #     all_losses_G_gamma,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_G_gamma.pt"
    # )
    # torch.save(
    #     all_losses_G_ab,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_G_ab.pt"
    # )
    # torch.save(
    #     all_losses_G_rest,
    #     "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/losses_G_rest.pt"
    # )

    if seen:

        torch.save(
            generator_mag_gamma.state_dict(),
            path_save + 'weights/seen/generator_mag_gamma_wgan_seen.pth')

        torch.save(
            generator_phase_gamma.state_dict(),
            path_save + 'weights/seen/generator_phase_gamma_wgan_seen.pth')

        torch.save(generator_mag_ab.state_dict(),
                   path_save + 'weights/seen/generator_mag_ab_wgan_seen.pth')

        torch.save(generator_phase_ab.state_dict(),
                   path_save + 'weights/seen/generator_phase_ab_wgan_seen.pth')

        torch.save(generator_mag_rest.state_dict(),
                   path_save + 'weights/seen/generator_mag_rest_wgan_seen.pth')

        torch.save(
            generator_phase_rest.state_dict(),
            path_save + 'weights/seen/generator_phase_rest_wgan_seen.pth')
    else:
        torch.save(
            generator_mag_gamma.state_dict(),
            path_save + 'weights/unseen/generator_mag_gamma_wgan_unseen.pth')

        torch.save(
            generator_phase_gamma.state_dict(),
            path_save + 'weights/unseen/generator_phase_gamma_wgan_unseen.pth')

        torch.save(
            generator_mag_ab.state_dict(),
            path_save + 'weights/unseen/generator_mag_ab_wgan_unseen.pth')

        torch.save(
            generator_phase_ab.state_dict(),
            path_save + 'weights/unseen/generator_phase_ab_wgan_unseen.pth')

        torch.save(
            generator_mag_rest.state_dict(),
            path_save + 'weights/unseen/generator_mag_rest_wgan_unseen.pth')

        torch.save(
            generator_phase_rest.state_dict(),
            path_save + 'weights/unseen/generator_phase_rest_wgan_unseen.pth')


# path_channel = '/home/vlad/Desktop/test_freq_new/'
# train_wgan(True, path_channel)
# train_wgan(False, path_channel)
