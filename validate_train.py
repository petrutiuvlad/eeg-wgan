import generate_data as data
import os
from train_wgan_new_2 import train_wgan_freq
from generator.generator import WGANGenerator
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from classify_raw_gen import train_per_channel_gen
from classify_raw import train_per_channel
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

Tensor = torch.FloatTensor


def get_clusters(channel):

    subjects_seen = []
    subjects_unseen = []

    all_seen = []
    all_unseen = []
    with open(
            '/home/vlad/Desktop/Statistics_GAN/PCA/ch_{}/Seen/dist.txt'.format(
                channel), 'r') as file_seen:
        for i in range(0, 11):
            line = file_seen.readline()
            value = float(line.split(':')[1])
            if value > 500:
                all_seen.append(300)
            else:
                all_seen.append(value)
    file_seen.close()

    with open(
            '/home/vlad/Desktop/Statistics_GAN/PCA/ch_{}/Unseen/dist.txt'.
            format(channel), 'r') as file_unseen:
        for i in range(0, 11):
            line = file_unseen.readline()
            value = float(line.split(':')[1])
            if value > 500:
                all_unseen.append(300)
            else:
                all_unseen.append(value)
    file_unseen.close()

    threshold_seen = sum(all_seen) / len(all_seen)
    threshold_unseen = sum(all_unseen) / len(all_unseen)

    for i in range(0, 11):
        if (all_unseen[i] <= threshold_unseen):
            subjects_unseen.append(i + 1)
        if (all_seen[i] <= threshold_seen):
            subjects_seen.append(i + 1)
    return subjects_seen, subjects_unseen


def create_directories(channel, number_valid):
    path_to_channel = '/home/vlad/Desktop/validate_ch{}/{}'.format(
        channel, number_valid)
    os.makedirs(path_to_channel, exist_ok=True)
    os.makedirs(path_to_channel + '/weights', exist_ok=True)
    os.makedirs(path_to_channel + '/weights/seen', exist_ok=True)
    os.makedirs(path_to_channel + '/weights/unseen', exist_ok=True)
    os.makedirs(path_to_channel + '/generated', exist_ok=True)
    os.makedirs(path_to_channel + '/raw', exist_ok=True)
    os.makedirs(path_to_channel + '/raw/files', exist_ok=True)
    os.makedirs(path_to_channel + '/gan', exist_ok=True)
    os.makedirs(path_to_channel + '/gan/files', exist_ok=True)
    os.makedirs(path_to_channel + '/wgan', exist_ok=True)
    os.makedirs(path_to_channel + '/wgan/files', exist_ok=True)
    os.makedirs(path_to_channel + '/11Fold', exist_ok=True)


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 400))
    return n


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


def generate_data_wgan_freq(channel, path_weights, path):

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

    # unseen_phase = torch.cat(
    #     (unseen_phase_rest, unseen_phase_ab, unseen_phase_gamma), dim=1)
    # unseen_mag = torch.cat((unseen_mag_rest, unseen_mag_ab, unseen_mag_gamma),
    #                        dim=1)
    # unseen_phase = torch.cat(
    #     (torch.zeros(200, 8), torch.zeros(200, 22), unseen_phase_gamma), dim=1)
    # unseen_mag = torch.cat(
    #     (torch.zeros(200, 8), torch.zeros(200, 22), unseen_mag_gamma), dim=1)
    # unseen_phase = torch.cat(
    #     (torch.zeros(200, 8), unseen_phase_ab, torch.zeros(200, 227)), dim=1)
    # unseen_mag = torch.cat(
    #     (torch.zeros(200, 8), unseen_mag_ab, torch.zeros(200, 227)), dim=1)
    # unseen_phase = torch.cat(
    #     (unseen_phase_rest, torch.zeros(200, 22), torch.zeros(200, 227)),
    #     dim=1)
    # unseen_mag = torch.cat(
    #     (unseen_mag_rest, torch.zeros(200, 22), torch.zeros(200, 227)), dim=1)
    unseen_phase = torch.cat(
        (unseen_phase_rest, torch.zeros(200, 22), unseen_phase_gamma), dim=1)
    unseen_mag = torch.cat(
        (unseen_mag_rest, torch.zeros(200, 22), unseen_mag_gamma), dim=1)
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

    # seen_phase = torch.cat(
    #     (torch.zeros(200, 8), torch.zeros(200, 22), seen_phase_gamma), dim=1)
    # seen_mag = torch.cat(
    #     (torch.zeros(200, 8), torch.zeros(200, 22), seen_mag_gamma), dim=1)
    # seen_phase = torch.cat((seen_phase_rest, seen_phase_ab, seen_phase_gamma),
    #                        dim=1)
    # seen_mag = torch.cat((seen_mag_rest, seen_mag_ab, seen_mag_gamma), dim=1)
    # seen_phase = torch.cat(
    #     (torch.zeros(200, 8), seen_phase_ab, torch.zeros(200, 227)), dim=1)
    # seen_mag = torch.cat(
    #     (torch.zeros(200, 8), seen_mag_ab, torch.zeros(200, 227)), dim=1)
    # seen_phase = torch.cat(
    #     (seen_phase_rest, torch.zeros(200, 22), torch.zeros(200, 227)), dim=1)
    # seen_mag = torch.cat(
    #     (seen_mag_rest, torch.zeros(200, 22), torch.zeros(200, 227)), dim=1)
    seen_phase = torch.cat(
        (seen_phase_rest, torch.zeros(200, 22), seen_phase_gamma), dim=1)
    seen_mag = torch.cat((seen_mag_rest, torch.zeros(200, 22), seen_mag_gamma),
                         dim=1)
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


def visualize_folds(path, valid_number, subject):

    acc_path = '/home/vlad/Desktop/validate_ch23/{}/11Fold/Accuracies/gan'.format(
        valid_number)
    loss_path = '/home/vlad/Desktop/validate_ch23/{}/11Fold/Losses/gan'.format(
        valid_number)
    conf_path = '/home/vlad/Desktop/validate_ch23/{}/11Fold/Confusion/gan'.format(
        valid_number)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(conf_path, exist_ok=True)

    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    acc_file = path + '/files/accuracies.txt'
    loss_file = path + '/files/losses.txt'
    with open(acc_file, 'r') as file:
        line = file.readline()
        while line:
            if ('Train' in line):
                value = line.split(':')[1]
                acc_train.append(float(value))
            if ('Test' in line):
                value = line.split(':')[1]
                acc_test.append(float(value))
            line = file.readline()

        plt.rcParams["figure.figsize"] = (20, 7)
        plt.plot(acc_train, 'b', label='Train Acc')
        plt.plot(acc_test, 'r', label='Test Acc')
        plt.legend(loc="upper left")
        plt.savefig(acc_path + '/{}.png'.format(subject))
        plt.clf()
        plt.close()
    with open(loss_file, 'r') as file:
        line = file.readline()
        while line:
            if ('Train' in line):
                value = line.split(':')[1]
                loss_train.append(float(value))
            if ('Test' in line):
                value = line.split(':')[1]
                loss_test.append(float(value))
            line = file.readline()

        plt.rcParams["figure.figsize"] = (20, 7)
        plt.plot(loss_train, 'b', label='Train Loss')
        plt.plot(loss_test, 'r', label='Test Loss')
        plt.legend(loc="upper left")
        plt.savefig(loss_path + '/{}.png'.format(subject))
        plt.clf()
        plt.close()

    preds = torch.load(path + '/files/preds.pt')
    gts = torch.load(path + '/files/gts.pt')
    predictions = []
    for lists in preds:
        for item in lists:
            predictions.append(item.item())
    grounds = []
    for lists in gts:
        for item in lists:
            grounds.append(item.item())
    data = {'y_Actual': grounds, 'y_Predicted': predictions}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    df.rename(columns={"0.0": "Unseen", "1.0": "Seen"})
    confusion = pd.crosstab(df['y_Actual'],
                            df['y_Predicted'],
                            rownames=['Actual'],
                            colnames=['Predicted'])
    plt.figure(figsize=(15, 4))
    sn.heatmap(confusion, annot=True, fmt='g')
    plt.savefig(conf_path + '/{}.png'.format(subject))
    plt.clf()
    plt.close()


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


current_channels = [23, 24]
for current_channel in current_channels:
    subjects_seen, subjects_unseen = get_clusters(current_channel)
    print(subjects_seen, subjects_unseen)
    data.generate_signals_seen(current_channel, subjects_seen)
    data.generate_signals_unseen(current_channel, subjects_unseen)
    data.generate_signals_per_subject(current_channel)
    path_channel = '/home/vlad/Desktop/Results_GAN/channel_{}/'.format(
        current_channel)
    # train_wgan_freq(True, path_channel)
    # train_wgan_freq(False, path_channel)
    for number_valid in range(1, 13):
        print('-----------------------{}-----------------------'.format(
            number_valid))
        create_directories(current_channel, number_valid)
        path_valid = '/home/vlad/Desktop/validate_ch{}/{}/'.format(
            current_channel, number_valid)
        path_weights = '/home/vlad/Desktop/Results_GAN/channel_{}/'.format(
            current_channel)
        generate_data_wgan_freq(channel=current_channel,
                                path_weights=path_weights,
                                path=path_valid)
        for index_test in intersection(subjects_seen, subjects_unseen):
            os.makedirs(path_valid + '11Fold/{}/gan/files'.format(index_test),
                        exist_ok=True)
            train_per_channel_gen(
                channel=current_channel,
                path_channel=path_valid + '11Fold/{}/'.format(index_test),
                good_subjects=intersection(subjects_seen, subjects_unseen),
                fold=True,
                index_test=index_test)
            # os.makedirs(path_valid + '11Fold/{}/raw/files'.format(index_test),
            #             exist_ok=True)
            # train_per_channel(
            #     channel=current_channel,
            #     path_channel=path_valid + '11Fold/{}/'.format(index_test),
            #     good_subjects=intersection(subjects_seen, subjects_unseen),
            #     fold=True,
            #     index_test=index_test)
        for index_test in intersection(subjects_seen, subjects_unseen):
            visualize_folds(path=path_valid +
                            '11Fold/{}/gan'.format(index_test),
                            valid_number=number_valid,
                            subject=index_test)
            # visualize_folds(path=path_valid +
            #                 '11Fold/{}/raw'.format(index_test),
            #                 valid_number=number_valid,
            #                 subject=index_test)
