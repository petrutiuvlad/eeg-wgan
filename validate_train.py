import generate_data as data
import os
from train_wgan import train_wgan_freq
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
from generate_data.generate_wgan import generate_data_wgan_freq, generate_data_wgan
from utils.utils import get_clusters, create_directories, noise, intersection

Tensor = torch.FloatTensor


def visualize_folds(channel, path, valid_number, subject, mode, wgan):

    acc_path = '/home/vlad/Desktop/validate_ch{}/{}/11Fold/Accuracies/{}'.format(
        channel, valid_number, wgan)
    loss_path = '/home/vlad/Desktop/validate_ch{}/{}/11Fold/Losses/{}'.format(
        channel, valid_number, wgan)
    conf_path = '/home/vlad/Desktop/validate_ch{}/{}/11Fold/Confusion/{}'.format(
        channel, valid_number, wgan)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(conf_path, exist_ok=True)

    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    acc_file = path + '/files/accuracies_{}.txt'.format(mode)
    loss_file = path + '/files/losses_{}.txt'.format(mode)
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
        plt.savefig(acc_path + '/{}_{}.png'.format(subject, mode))
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
        plt.savefig(loss_path + '/{}_{}.png'.format(subject, mode))
        plt.clf()
        plt.close()

    preds = torch.load(path + '/files/preds_{}.pt'.format(mode))
    gts = torch.load(path + '/files/gts_{}.pt'.format(mode))
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
    plt.savefig(conf_path + '/{}_{}.png'.format(subject, mode))
    plt.clf()
    plt.close()


mode = 'gamma'
current_channels = [1, 49, 118]
for current_channel in current_channels:
    subjects_seen, subjects_unseen = get_clusters(current_channel)
    print(subjects_seen, subjects_unseen)
    data.generate_signals_seen(current_channel, subjects_seen)
    data.generate_signals_unseen(current_channel, subjects_unseen)
    data.generate_signals_per_subject(current_channel)
    path_channel = '/home/vlad/Desktop/Results_GAN/channel_{}/'.format(
        current_channel)
    train_wgan_freq(True, path_channel)
    train_wgan_freq(False, path_channel)
    # for number_valid in range(1, 13):
    #     print('-----------------------{}-----------------------'.format(
    #         number_valid))
    #     create_directories(current_channel, number_valid)
    #     path_valid = '/home/vlad/Desktop/validate_ch{}/{}/'.format(
    #         current_channel, number_valid)
    #     path_weights = '/home/vlad/Desktop/Results_GAN/channel_{}/'.format(
    #         current_channel)
    #     generate_data_wgan_freq(channel=current_channel,
    #                             path_weights=path_weights,
    #                             path=path_valid,
    #                             mode=mode)
    #     for index_test in intersection(subjects_seen, subjects_unseen):
    #         os.makedirs(path_valid + '11Fold/{}/wgan/files'.format(index_test),
    #                     exist_ok=True)
    #         train_per_channel_gen(
    #             channel=current_channel,
    #             path_channel=path_valid + '11Fold/{}/'.format(index_test),
    #             good_subjects=intersection(subjects_seen, subjects_unseen),
    #             fold=True,
    #             index_test=index_test)
    #         os.makedirs(path_valid + '11Fold/{}/raw/files'.format(index_test),
    #                     exist_ok=True)
    #         train_per_channel(
    #             channel=current_channel,
    #             path_channel=path_valid + '11Fold/{}/'.format(index_test),
    #             good_subjects=intersection(subjects_seen, subjects_unseen),
    #             fold=True,
    #             index_test=index_test,
    #             mode=mode)
    #     for index_test in intersection(subjects_seen, subjects_unseen):
    #         visualize_folds(channel=current_channel,
    #                         path=path_valid +
    #                         '11Fold/{}/wgan'.format(index_test),
    #                         valid_number=number_valid,
    #                         subject=index_test,
    #                         mode=mode,
    #                         wgan='wgan')
    #         visualize_folds(channel=current_channel,
    #                         path=path_valid +
    #                         '11Fold/{}/raw'.format(index_test),
    #                         valid_number=number_valid,
    #                         subject=index_test,
    #                         mode=mode,
    #                         wgan='raw')
