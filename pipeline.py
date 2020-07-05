import generate_data as data
import os
from train_gan import train
from train_wgan import train_wgan
from generator.generator import GeneratorNet, WGANGenerator
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


def create_directories(channel):
    path_to_channel = '/home/vlad/Desktop/Results_GAN/channel_{}'.format(
        channel)
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
    os.makedirs('/home/vlad/Desktop/Results_GAN/Accuracies', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Losses', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Confusion', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Accuracies/raw', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Losses/raw', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Confusion/raw', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Accuracies/gan', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Losses/gan', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Confusion/gan', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Accuracies/wgan',
                exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Losses/wgan', exist_ok=True)
    os.makedirs('/home/vlad/Desktop/Results_GAN/Confusion/wgan', exist_ok=True)


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 400))
    return n


def generate_data_gan(path):
    generator_seen = GeneratorNet()
    generator_seen.load_state_dict(
        torch.load(path_channel + 'weights/seen/generator_seen.pth'))

    generator_unseen = GeneratorNet()
    generator_unseen.load_state_dict(
        torch.load(path_channel + 'weights/unseen/generator_unseen.pth'))

    gen_noise_seen = noise(200)
    gen_noise_unseen = noise(200)

    seen = generator_seen(gen_noise_seen)
    unseen = generator_unseen(gen_noise_unseen)
    seen = seen.detach()
    unseen = unseen.detach()
    torch.save(seen, path + 'generated/seen_tensor.pt')
    torch.save(unseen, path + 'generated/unseen_tensor.pt')


def generate_data_wgan(path):
    generator_seen = WGANGenerator()
    generator_seen.load_state_dict(
        torch.load(path_channel + 'weights/seen/generator_wgan_seen.pth'))

    generator_unseen = WGANGenerator()
    generator_unseen.load_state_dict(
        torch.load(path_channel + 'weights/unseen/generator_wgan_unseen.pth'))

    gen_noise_seen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))
    gen_noise_unseen = Variable(Tensor(np.random.normal(0, 100, (200, 400))))

    seen = generator_seen(gen_noise_seen)
    unseen = generator_unseen(gen_noise_unseen)
    seen = seen.detach()
    unseen = unseen.detach()
    torch.save(seen, path + 'generated/seen_tensor.pt')
    torch.save(unseen, path + 'generated/unseen_tensor.pt')


def visualize_raw(channel, path, generated=False):
    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []
    if generated == False:
        string = 'raw'
    else:
        string = 'gan'

    acc_file = path + '{}/files/accuracies.txt'.format(string)
    loss_file = path + '{}/files/losses.txt'.format(string)
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
        plt.savefig(
            '/home/vlad/Desktop/Results_GAN/Accuracies/{}/{}.png'.format(
                string, channel))
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
        plt.savefig('/home/vlad/Desktop/Results_GAN/Losses/{}/{}.png'.format(
            string, channel))
        plt.clf()
        plt.close()

    preds = torch.load(path + '{}/files/preds.pt'.format(string))
    gts = torch.load(path + '{}/files/gts.pt'.format(string))
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
    plt.savefig('/home/vlad/Desktop/Results_GAN/Confusion/{}/{}.png'.format(
        string, channel))
    plt.clf()
    plt.close()


def visualize_folds(path, channel, subject, generated=False):

    if generated == True:
        acc_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Accuracies/gan'.format(
            channel)
        loss_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Losses/gan'.format(
            channel)
        conf_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Confusion/gan'.format(
            channel)
        os.makedirs(acc_path, exist_ok=True)
        os.makedirs(loss_path, exist_ok=True)
        os.makedirs(conf_path, exist_ok=True)
    else:
        acc_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Accuracies/raw'.format(
            channel)
        loss_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Losses/raw'.format(
            channel)
        conf_path = '/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/Confusion/raw'.format(
            channel)
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


for current_channel in [23]:
    print('------------------{}-------------------------'.format(
        current_channel))
    subjects_seen, subjects_unseen = get_clusters(current_channel)
    print(subjects_seen, subjects_unseen)
    data.generate_signals_seen(current_channel, subjects_seen)
    data.generate_signals_unseen(current_channel, subjects_unseen)
    data.generate_signals_per_subject(current_channel)
    create_directories(current_channel)
    path_channel = '/home/vlad/Desktop/Results_GAN/channel_{}/'.format(
        current_channel)
    # train(True, path_channel)
    # train(False, path_channel)
    # train_wgan(True, path_channel)
    # train_wgan(False, path_channel)
    # generate_data_gan(path_channel)
    # generate_data_wgan(path_channel)
    # for index_test in intersection(subjects_seen, subjects_unseen):
    #     os.makedirs(path_channel + '11Fold/{}/raw/files'.format(index_test),
    #                 exist_ok=True)
    #     train_per_channel(
    #         channel=current_channel,
    #         path_channel=path_channel + '11Fold/{}/'.format(index_test),
    #         good_subjects=intersection(subjects_seen, subjects_unseen),
    #         fold=True,
    #         index_test=index_test)
    # for index_test in intersection(subjects_seen, subjects_unseen):
    #     visualize_folds(path=path_channel + '11Fold/{}/raw'.format(index_test),
    #                     channel=current_channel,
    #                     subject=index_test,
    #                     generated=False)
    for index_test in intersection(subjects_seen, subjects_unseen):
        os.makedirs(path_channel + '11Fold/{}/gan/files'.format(index_test),
                    exist_ok=True)
        train_per_channel_gen(
            channel=current_channel,
            path_channel=path_channel + '11Fold/{}/'.format(index_test),
            good_subjects=intersection(subjects_seen, subjects_unseen),
            fold=True,
            index_test=index_test)
    for index_test in intersection(subjects_seen, subjects_unseen):
        visualize_folds(path=path_channel + '11Fold/{}/gan'.format(index_test),
                        channel=current_channel,
                        subject=index_test,
                        generated=True)
    # os.makedirs(path_channel + '11Fold/shuffle/raw/files', exist_ok=True)
    # os.makedirs(path_channel + '11Fold/shuffle/gan/files', exist_ok=True)
    # train_per_channel(current_channel,
    #                   path_channel=path_channel + '11Fold/shuffle/',
    #                   good_subjects=intersection(subjects_seen,
    #                                              subjects_unseen),
    #                   shuffle=True)
    # visualize_folds(path=path_channel + '11Fold/shuffle/raw',
    #                 channel=current_channel,
    #                 subject='shuffle',
    #                 generated=False)
    # train_per_channel_gen(current_channel,
    #                       path_channel=path_channel,
    #                       good_subjects=intersection(subjects_seen,
    #                                                  subjects_unseen),
    #                       shuffle=True)
    # visualize_folds(path=path_channel + '11Fold/shuffle/gan',
    #                 channel=current_channel,
    #                 subject='shuffle',
    #                 generated=True)
