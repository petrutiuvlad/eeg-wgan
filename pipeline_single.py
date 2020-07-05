import generate_data as data
import os
from train_wgan import train_wgan
from generator.generator import WGANGenerator
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from classify_raw import train_per_channel
from classify_raw_gen import train_per_channel_gen
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

Tensor = torch.FloatTensor


def create_directories(number_valid):
    path_to_channel = '/home/vlad/Desktop/single_train/{}'.format(number_valid)
    os.makedirs(path_to_channel, exist_ok=True)
    os.makedirs(path_to_channel + '/weights', exist_ok=True)
    os.makedirs(path_to_channel + '/weights/seen', exist_ok=True)
    os.makedirs(path_to_channel + '/weights/unseen', exist_ok=True)
    os.makedirs(path_to_channel + '/generated', exist_ok=True)
    os.makedirs(path_to_channel + '/raw', exist_ok=True)
    os.makedirs(path_to_channel + '/raw/files', exist_ok=True)
    os.makedirs(path_to_channel + '/gan', exist_ok=True)
    os.makedirs(path_to_channel + '/gan/files', exist_ok=True)


def generate_data_wgan(path):
    generator_seen = WGANGenerator()
    generator_seen.load_state_dict(
        torch.load(path + 'weights/seen/generator_wgan_seen.pth'))

    generator_unseen = WGANGenerator()
    generator_unseen.load_state_dict(
        torch.load(path + 'weights/unseen/generator_wgan_unseen.pth'))

    gen_noise_seen = Variable(Tensor(np.random.normal(0, 100, (100, 400))))
    gen_noise_unseen = Variable(Tensor(np.random.normal(0, 100, (100, 400))))

    seen = generator_seen(gen_noise_seen)
    unseen = generator_unseen(gen_noise_unseen)
    seen = seen.detach()
    unseen = unseen.detach()
    torch.save(seen, path + 'generated/seen_tensor.pt')
    torch.save(unseen, path + 'generated/unseen_tensor.pt')


def visualize_folds(path, valid_number, subject):

    acc_path = '/home/vlad/Desktop/single_train/{}/Accuracies'.format(
        valid_number)
    loss_path = '/home/vlad/Desktop/single_train/{}/Losses'.format(
        valid_number)
    conf_path = '/home/vlad/Desktop/single_train/{}/Confusion'.format(
        valid_number)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)
    os.makedirs(conf_path, exist_ok=True)

    acc_train = []
    acc_train_gen = []
    acc_test = []
    acc_test_gen = []
    loss_train = []
    loss_train_gen = []
    loss_test = []
    loss_test_gen = []
    acc_file = path + 'raw/files/accuracies.txt'
    acc_file_gen = path + 'gan/files/accuracies.txt'
    loss_file = path + 'raw/files/losses.txt'
    loss_file_gen = path + 'gan/files/losses.txt'
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

    with open(acc_file_gen, 'r') as file:
        line = file.readline()
        while line:
            if ('Train' in line):
                value = line.split(':')[1]
                acc_train_gen.append(float(value))
            if ('Test' in line):
                value = line.split(':')[1]
                acc_test_gen.append(float(value))
            line = file.readline()

    plt.rcParams["figure.figsize"] = (20, 7)
    plt.plot(acc_train, 'b', label='Train Acc Raw')
    plt.plot(acc_test, 'r', label='Test Acc Raw')
    plt.plot(acc_train_gen, 'g', label='Train Acc Raw+Gen')
    plt.plot(acc_test_gen, 'y', label='Test Acc Raw+Gen')
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

    with open(loss_file_gen, 'r') as file:
        line = file.readline()
        while line:
            if ('Train' in line):
                value = line.split(':')[1]
                loss_train_gen.append(float(value))
            if ('Test' in line):
                value = line.split(':')[1]
                loss_test_gen.append(float(value))
            line = file.readline()

    plt.rcParams["figure.figsize"] = (20, 7)
    plt.plot(loss_train, 'b', label='Train Loss Raw')
    plt.plot(loss_test, 'r', label='Test Loss Raw')
    plt.plot(loss_train_gen, 'g', label='Train Loss Raw+Gen')
    plt.plot(loss_test_gen, 'y', label='Test Loss Raw+Gen')
    plt.legend(loc="upper left")
    plt.savefig(loss_path + '/{}.png'.format(subject))
    plt.clf()
    plt.close()

    preds = torch.load(path + 'raw/files/preds.pt')
    gts = torch.load(path + 'raw/files/gts.pt')
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
    plt.savefig(conf_path + '/{}_raw.png'.format(subject))
    plt.clf()
    plt.close()

    preds = torch.load(path + 'gan/files/preds.pt')
    gts = torch.load(path + 'gan/files/gts.pt')
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
    plt.savefig(conf_path + '/{}_gan.png'.format(subject))
    plt.clf()
    plt.close()


for i in range(1, 12):
    subjects_seen = [i]
    subjects_unseen = [i]
    current_channel = 41
    create_directories(i)
    # data.generate_signals_seen(current_channel, subjects_seen)
    # data.generate_signals_unseen(current_channel, subjects_unseen)
    path_channel = '/home/vlad/Desktop/single_train/{}/'.format(i)
    # train_wgan(True, path_channel)
    # train_wgan(False, path_channel)
    # generate_data_wgan(path_channel)
    # train_per_channel(channel=current_channel,
    #                   path_channel=path_channel,
    #                   good_subjects=[i])
    # train_per_channel_gen(channel=current_channel,
    #                       path_channel=path_channel,
    #                       good_subjects=[i])
    visualize_folds(path=path_channel, valid_number=i, subject=i)
