import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import os
import struct
from os import path
import pickle
from grad_cam import *
from model.model import *
from generate_data.read import *


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def read_raw_signal(channel_param: int, good_subjects):

    subjects_unseen = {}
    subjects_seen = {}

    counter_seen = 0
    counter_unseen = 0
    for jndex in good_subjects:
        id_patient = '{0:0=3d}'.format(jndex)
        channel = '{0:0=3d}'.format(channel_param)
        events = read_event(
            '/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Dots_30_{}-Event-Codes.bin'
            .format(id_patient, id_patient))
        timestamps = read_timestamp(
            '/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Dots_30_{}-Event-Timestamps.bin'
            .format(id_patient, id_patient))

        floats_channel = read_floats(
            "/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Dots_30_{}-Ch{}.bin"
            .format(id_patient, id_patient, channel))
        array = np.array(floats_channel)

        trial_states = []
        if path.exists(
                '/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Dots_30_{}.eti'
                .format(id_patient, id_patient)):
            with open(
                    '/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Dots_30_{}.eti'
                    .format(id_patient, id_patient)) as file:
                all_lines = file.readlines()
                for line in all_lines[4:]:
                    line_split = line.strip().split(",")
                    trial = line_split[0]
                    correct_answer = line_split[4]
                    patient_answer = line_split[7]
                    if (patient_answer != 'Nothing') and (patient_answer !=
                                                          'Something'):
                        if (patient_answer == correct_answer):
                            trial_states.append('Seen')
                        else:
                            trial_states.append('Wrong')
                    else:
                        trial_states.append(patient_answer)
        else:
            with open(
                    '/home/vlad/Desktop/Licenta/Date_EEG_without_Laplacean/Dots_30_{}/Trials_Info-Dots_30_{}.eti'
                    .format(id_patient, id_patient)) as file:
                all_lines = file.readlines()
                for line in all_lines[4:]:
                    line_split = line.strip().split(",")
                    trial = line_split[0]
                    correct_answer = line_split[4]
                    patient_answer = line_split[7]
                    if (patient_answer != 'Nothing') and (patient_answer !=
                                                          'Something'):
                        if (patient_answer == correct_answer):
                            trial_states.append('Seen')
                        else:
                            trial_states.append('Wrong')
                    else:
                        trial_states.append(patient_answer)

        indices_show = [i for i, x in enumerate(events) if x == 150]
        indices_response = [i for i, x in enumerate(events) if x == 129]
        indices_end_response = [
            i for i, x in enumerate(events) if x == 3 or x == 2 or x == 1
        ]
        indices_end = [i for i, x in enumerate(events) if x == 131]
        for index in range(0, 210):
            if trial_states[index] == 'Nothing':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                ch1 = ch1[:512]

                subjects_unseen.update({counter_unseen: ch1})
                counter_unseen += 1

            if trial_states[index] == 'Seen':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                ch1 = ch1[:512]

                subjects_seen.update({counter_seen: ch1})
                counter_seen += 1
    return subjects_unseen, subjects_seen


def prepare_11_fold(index_test, channel, subjects):
    test_seen = {}
    test_unseen = {}
    train_seen = {}
    train_unseen = {}
    count_seen = 0
    count_unseen = 0
    count_seen_test = 0
    count_unseen_test = 0

    for j in subjects:
        with open(
                "/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/all_subjects/signal_seen_{}.pkl"
                .format(channel, j), 'rb') as handle:
            seen = pickle.load(handle)
        handle.close()

        with open(
                "/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/all_subjects/signal_unseen_{}.pkl"
                .format(channel, j), 'rb') as handle:
            unseen = pickle.load(handle)
        handle.close()

        if index_test != j:
            for item in range(0, len(seen)):
                train_seen.update({count_seen: seen[item]})
                count_seen += 1
            for item in range(0, len(unseen)):
                train_unseen.update({count_unseen: unseen[item]})
                count_unseen += 1
        else:
            for item in range(0, len(seen)):
                test_seen.update({count_seen_test: seen[item]})
                count_seen_test += 1
            for item in range(0, len(unseen)):
                test_unseen.update({count_unseen_test: unseen[item]})
                count_unseen_test += 1

    minimum = min(len(train_seen), len(train_unseen))
    number_seen = len(train_seen) - minimum
    number_unseen = len(train_unseen) - minimum
    for i in range(len(train_seen) - 1, len(train_seen) - number_seen - 1, -1):
        train_seen.pop(i)
    for i in range(
            len(train_unseen) - 1,
            len(train_unseen) - number_unseen - 1, -1):
        train_unseen.pop(i)

    train_unseen_torch = []
    train_seen_torch = []
    for i in range(0, len(train_seen)):
        train_unseen_torch.append(torch.from_numpy(train_unseen[i]))
        train_seen_torch.append(torch.from_numpy(train_seen[i]))

    test_unseen_torch = []
    test_seen_torch = []
    for i in range(0, len(test_seen)):
        test_seen_torch.append(torch.from_numpy(train_seen[i]))
    for i in range(0, len(test_unseen)):
        test_unseen_torch.append(torch.from_numpy(test_unseen[i]))

    for i in range(0, len(train_seen)):
        train_unseen_torch[i] = train_unseen_torch[i].float()
        train_seen_torch[i] = train_seen_torch[i].float()
    for i in range(0, len(test_seen)):
        test_seen_torch[i] = test_seen_torch[i].float()
    for i in range(0, len(test_unseen)):
        test_unseen_torch[i] = test_unseen_torch[i].float()

    train_seen = torch.stack(train_seen_torch)
    train_unseen = torch.stack(train_unseen_torch)

    test_seen = torch.stack(test_seen_torch)
    test_unseen = torch.stack(test_unseen_torch)

    seen_gen = torch.load(
        '/home/vlad/Desktop/Results_GAN/channel_{}/generated/seen_tensor.pt'.
        format(channel))
    unseen_gen = torch.load(
        '/home/vlad/Desktop/Results_GAN/channel_{}/generated/unseen_tensor.pt'.
        format(channel))

    for i in range(0, len(seen_gen)):
        seen_gen[i] = seen_gen[i].float()
        unseen_gen[i] = unseen_gen[i].float()

    train = torch.cat((train_seen, train_unseen, seen_gen, unseen_gen))
    test = torch.cat((test_seen, test_unseen))

    targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
    targets_seen = torch.from_numpy(np.ones(len(train_seen)))
    targets_unseen_gen = torch.from_numpy(np.zeros(len(seen_gen)))
    targets_seen_gen = torch.from_numpy(np.ones(len(unseen_gen)))
    targets = torch.cat(
        (targets_seen, targets_unseen, targets_seen_gen, targets_unseen_gen))

    test_targets_unseen = torch.from_numpy(np.zeros(len(test_unseen)))
    test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
    test_targets = torch.cat((test_targets_seen, test_targets_unseen))

    return train, test, targets, test_targets


def prepare_data(signals_unseen, signals_seen, path_channel):
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

    size_train = round(len(signals_seen) * 0.8)

    train_unseen = []
    train_seen = []
    for i in range(0, size_train):
        train_unseen.append(torch.from_numpy(signals_unseen[i]))
        train_seen.append(torch.from_numpy(signals_seen[i]))

    test_unseen = []
    test_seen = []
    for i in range(size_train, len(signals_unseen)):
        test_unseen.append(torch.from_numpy(signals_unseen[i]))
        test_seen.append(torch.from_numpy(signals_seen[i]))

    for i in range(0, len(train_seen)):
        train_unseen[i] = train_unseen[i].float()
        train_seen[i] = train_seen[i].float()
    for i in range(0, len(test_seen)):
        test_unseen[i] = test_unseen[i].float()
        test_seen[i] = test_seen[i].float()

    train_seen = torch.stack(train_seen)
    train_unseen = torch.stack(train_unseen)

    test_seen = torch.stack(test_seen)
    test_unseen = torch.stack(test_unseen)

    seen_gen = torch.load(path_channel + 'generated/seen_tensor.pt')
    unseen_gen = torch.load(path_channel + 'generated/unseen_tensor.pt')

    for i in range(0, len(seen_gen)):
        seen_gen[i] = seen_gen[i].float()
        unseen_gen[i] = unseen_gen[i].float()

    train = torch.cat((train_seen, train_unseen, seen_gen, unseen_gen))
    # train = torch.cat((seen_gen, unseen_gen))

    test = torch.cat((test_seen, test_unseen))

    targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
    targets_seen = torch.from_numpy(np.ones(len(train_seen)))

    targets_unseen_gen = torch.from_numpy(np.zeros(len(seen_gen)))
    targets_seen_gen = torch.from_numpy(np.ones(len(unseen_gen)))

    targets = torch.cat(
        (targets_seen, targets_unseen, targets_seen_gen, targets_unseen_gen))
    # targets = torch.cat((targets_seen_gen, targets_unseen_gen))

    test_targets_unseen = torch.from_numpy(np.zeros(len(test_seen)))
    test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
    test_targets = torch.cat((test_targets_seen, test_targets_unseen))

    return train, test, targets, test_targets


def prepare_data_shuffle(path_channel):
    with open(path_channel + "11Fold/shuffle/train.pkl", 'rb') as file:
        train = pickle.load(file)
    with open(path_channel + "11Fold/shuffle/test.pkl", 'rb') as file:
        test = pickle.load(file)
    with open(path_channel + "11Fold/shuffle/targets.pkl", 'rb') as file:
        targets = pickle.load(file)
    with open(path_channel + "11Fold/shuffle/test_targets.pkl", 'rb') as file:
        test_targets = pickle.load(file)
    seen_gen = torch.load(path_channel + 'generated/seen_tensor.pt')
    unseen_gen = torch.load(path_channel + 'generated/unseen_tensor.pt')
    for i in range(0, len(seen_gen)):
        seen_gen[i] = seen_gen[i].float()
        unseen_gen[i] = unseen_gen[i].float()

    train = torch.cat((train, seen_gen, unseen_gen))

    targets_unseen_gen = torch.from_numpy(np.zeros(len(seen_gen)))
    targets_seen_gen = torch.from_numpy(np.ones(len(unseen_gen)))

    targets = torch.cat((targets, targets_seen_gen, targets_unseen_gen))

    return train, test, targets, test_targets


def train_per_channel_gen(channel: int,
                          path_channel,
                          good_subjects,
                          fold=False,
                          index_test=1,
                          shuffle=False):
    create_gradcam_dirs(path_channel)
    if (fold == True):
        train, test, targets, test_targets = prepare_11_fold(
            index_test, channel, good_subjects)
    elif (shuffle == True):
        train, test, targets, test_targets = prepare_data_shuffle(path_channel)
        path_channel += '11Fold/shuffle/'
    else:
        signals_unseen, signals_seen = read_raw_signal(channel, good_subjects)
        train, test, targets, test_targets = prepare_data(
            signals_unseen, signals_seen, path_channel)

    train = data_utils.TensorDataset(train, targets)
    test = data_utils.TensorDataset(test, test_targets)
    loader = data_utils.DataLoader(train, batch_size=16, shuffle=True)
    loader_test = data_utils.DataLoader(test, batch_size=16, shuffle=False)

    discriminator = Linear()
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)
    num_epochs = 200

    loss = nn.BCELoss()

    discriminator = discriminator.cuda()

    with open(path_channel + 'gan/files/accuracies.txt',
              'w+') as acc_file, open(path_channel + 'gan/files/losses.txt',
                                      'w+') as loss_file:
        for epoch in range(num_epochs):
            train_loss = 0
            train_acc = 0
            discriminator.train()
            for index, (real_batch, targets) in enumerate(loader):
                optimizer_D.zero_grad()

                preds, _, _ = discriminator(real_batch.cuda())

                error = loss(preds, targets.unsqueeze(1).float().cuda())
                acc = binary_acc(preds, targets.unsqueeze(1).float().cuda())
                error.backward()

                optimizer_D.step()
                train_loss += error
                train_acc += acc

            losssss = (train_loss / len(loader))
            acccccc = (train_acc / len(loader))
            loss_file.write('Train:' + str(losssss.item()) + '\n')
            acc_file.write('Train:' + str(acccccc.item()) + '\n')

            val_loss = 0
            val_acc = 0
            discriminator.eval()
            for index, (real_batch, targets) in enumerate(loader_test):
                preds, _, _ = discriminator(real_batch.cuda())

                error = loss(preds, targets.unsqueeze(1).float().cuda())

                acc = binary_acc(preds, targets.unsqueeze(1).float().cuda())

                val_loss += error
                val_acc += acc
            losssss = (val_loss / len(loader_test))
            acccccc = (val_acc / len(loader_test))
            loss_file.write('Test:' + str(losssss.item()) + '\n')
            acc_file.write('Test:' + str(acccccc.item()) + '\n')
            torch.cuda.empty_cache()

        loss_file.close()
        acc_file.close()

    discriminator.eval()
    predictions = []
    ground_truths = []
    for index, (real_batch, targets) in enumerate(loader_test):
        preds, _, _ = discriminator(real_batch.cuda())

        preds_rounded = torch.round(preds.detach())
        predictions.append(preds_rounded.cpu())
        ground_truths.append(targets.detach().cpu())
    torch.save(predictions, path_channel + 'gan/files/preds.pt')
    torch.save(ground_truths, path_channel + 'gan/files/gts.pt')
    plot_gradcam(path_channel,
                 discriminator,
                 test,
                 train_dataset=False,
                 generated=True)
    plot_gradcam(path_channel,
                 discriminator,
                 train,
                 train_dataset=True,
                 generated=True)


# for index in range(1, 129):
#     print('--------------------{}----------------------'.format(index))
#     train_per_channel(index)
# train_per_channel(70)
# for index_test in range(1, 12):
#     train_per_channel_gen(
#         channel=71,
#         path_channel='/home/vlad/Desktop/Statistics_GAN/11_Fold/ch71/{}/'.
#         format(index_test),
#         fold=True,
#         index_test=index_test)

# train_per_channel_gen(
#     channel=71,
#     path_channel='/home/vlad/Desktop/Statistics_GAN/11_Fold/ch71/shuffle/',
#     fold=False,
#     shuffle=True)
