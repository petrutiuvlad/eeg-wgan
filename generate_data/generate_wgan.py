from generator.generator import WGANGenerator
import torch
from torch.autograd import Variable
import numpy as np

Tensor = torch.FloatTensor


def generate_data_wgan(path_weights, path):
    generator_seen = WGANGenerator()
    generator_seen.load_state_dict(
        torch.load(path_weights + 'weights/seen/generator_wgan_seen.pth'))

    generator_unseen = WGANGenerator()
    generator_unseen.load_state_dict(
        torch.load(path_weights + 'weights/unseen/generator_wgan_unseen.pth'))

    gen_noise_seen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))
    gen_noise_unseen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))

    seen = generator_seen(gen_noise_seen)
    unseen = generator_unseen(gen_noise_unseen)
    seen = seen.detach()
    unseen = unseen.detach()
    torch.save(seen, path + 'generated/seen_tensor.pt')
    torch.save(unseen, path + 'generated/unseen_tensor.pt')


def generate_data_wgan_freq(channel, path_weights, path, mode):

    generator_phase_gamma = WGANGenerator(227)
    generator_mag_gamma = WGANGenerator(227)
    generator_phase_ab = WGANGenerator(22)
    generator_mag_ab = WGANGenerator(22)
    generator_phase_rest = WGANGenerator(8)
    generator_mag_rest = WGANGenerator(8)
    generator_phase_gamma.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_phase_gamma_wgan_unseen.pth'))
    generator_mag_gamma.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_mag_gamma_wgan_unseen.pth'))
    generator_phase_rest.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_phase_rest_wgan_unseen.pth'))
    generator_mag_rest.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_mag_rest_wgan_unseen.pth'))
    generator_phase_ab.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_phase_ab_wgan_unseen.pth'))
    generator_mag_ab.load_state_dict(
        torch.load(path_weights +
                   'weights/unseen/generator_mag_ab_wgan_unseen.pth'))

    gen_noise_unseen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))
    unseen_phase_gamma = generator_phase_gamma(gen_noise_unseen)
    unseen_mag_gamma = generator_mag_gamma(gen_noise_unseen)
    unseen_phase_rest = generator_phase_rest(gen_noise_unseen)
    unseen_mag_rest = generator_mag_rest(gen_noise_unseen)
    unseen_phase_ab = generator_phase_ab(gen_noise_unseen)
    unseen_mag_ab = generator_mag_ab(gen_noise_unseen)

    if mode == 'all':
        unseen_phase = torch.cat(
            (unseen_phase_rest, unseen_phase_ab, unseen_phase_gamma), dim=1)
        unseen_mag = torch.cat(
            (unseen_mag_rest, unseen_mag_ab, unseen_mag_gamma), dim=1)
    elif mode == 'gamma':
        unseen_phase = torch.cat(
            (torch.zeros(200, 8), torch.zeros(200, 22), unseen_phase_gamma),
            dim=1)
        unseen_mag = torch.cat(
            (torch.zeros(200, 8), torch.zeros(200, 22), unseen_mag_gamma),
            dim=1)
    elif mode == 'ab':
        unseen_phase = torch.cat(
            (torch.zeros(200, 8), unseen_phase_ab, torch.zeros(200, 227)),
            dim=1)
        unseen_mag = torch.cat(
            (torch.zeros(200, 8), unseen_mag_ab, torch.zeros(200, 227)), dim=1)
    elif mode == 'theta':
        unseen_phase = torch.cat(
            (unseen_phase_rest, torch.zeros(200, 22), torch.zeros(200, 227)),
            dim=1)
        unseen_mag = torch.cat(
            (unseen_mag_rest, torch.zeros(200, 22), torch.zeros(200, 227)),
            dim=1)
    else:
        print('Not Valid Mode')

    signal_full = torch.FloatTensor(size=(200, 257, 2))
    for i in range(0, 200):
        signal_full[i][:, 0] = unseen_mag[i]
        signal_full[i][:, 1] = unseen_phase[i]
    all_signals = []
    for i in range(0, 200):
        all_signals.append(
            torch.irfft(signal_full, 1,
                        onesided=True).detach().numpy()[i][0:512])
    new_all_signals = torch.FloatTensor(all_signals)
    torch.save(new_all_signals, path + 'generated/unseen_tensor.pt')
    torch.save(
        new_all_signals,
        '/home/vlad/Desktop/Results_GAN/channel_{}/generated/unseen_tensor.pt'.
        format(channel))

    generator_phase_gamma.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_phase_gamma_wgan_seen.pth'))
    generator_mag_gamma.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_mag_gamma_wgan_seen.pth'))
    generator_phase_rest.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_phase_rest_wgan_seen.pth'))
    generator_mag_rest.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_mag_rest_wgan_seen.pth'))
    generator_phase_ab.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_phase_ab_wgan_seen.pth'))
    generator_mag_ab.load_state_dict(
        torch.load(path_weights +
                   'weights/seen/generator_mag_ab_wgan_seen.pth'))

    gen_noise_seen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))
    seen_phase_gamma = generator_phase_gamma(gen_noise_seen)
    seen_mag_gamma = generator_mag_gamma(gen_noise_seen)
    seen_phase_rest = generator_phase_rest(gen_noise_seen)
    seen_mag_rest = generator_mag_rest(gen_noise_seen)
    seen_phase_ab = generator_phase_ab(gen_noise_seen)
    seen_mag_ab = generator_mag_ab(gen_noise_seen)

    if mode == 'all':
        seen_phase = torch.cat(
            (seen_phase_rest, seen_phase_ab, seen_phase_gamma), dim=1)
        seen_mag = torch.cat((seen_mag_rest, seen_mag_ab, seen_mag_gamma),
                             dim=1)
    elif mode == 'gamma':
        seen_phase = torch.cat(
            (torch.zeros(200, 8), torch.zeros(200, 22), seen_phase_gamma),
            dim=1)
        seen_mag = torch.cat(
            (torch.zeros(200, 8), torch.zeros(200, 22), seen_mag_gamma), dim=1)
    elif mode == 'ab':
        seen_phase = torch.cat(
            (torch.zeros(200, 8), seen_phase_ab, torch.zeros(200, 227)), dim=1)
        seen_mag = torch.cat(
            (torch.zeros(200, 8), seen_mag_ab, torch.zeros(200, 227)), dim=1)
    elif mode == 'theta':
        seen_phase = torch.cat(
            (seen_phase_rest, torch.zeros(200, 22), torch.zeros(200, 227)),
            dim=1)
        seen_mag = torch.cat(
            (seen_mag_rest, torch.zeros(200, 22), torch.zeros(200, 227)),
            dim=1)
    else:
        print('Not Valid Mode')
    signal_full = torch.FloatTensor(size=(200, 257, 2))
    for i in range(0, 200):
        signal_full[i][:, 0] = seen_mag[i]
        signal_full[i][:, 1] = seen_phase[i]
    all_signals = []
    for i in range(0, 200):
        all_signals.append(
            torch.irfft(signal_full, 1,
                        onesided=True).detach().numpy()[i][0:512])
    new_all_signals = torch.FloatTensor(all_signals)
    torch.save(new_all_signals, path + 'generated/seen_tensor.pt')
    torch.save(
        new_all_signals,
        '/home/vlad/Desktop/Results_GAN/channel_{}/generated/seen_tensor.pt'.
        format(channel))