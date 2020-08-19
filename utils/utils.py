from generate_data.read import *
from os import path
import os
import numpy as np
import pickle
import random
import torch
from torch.autograd import Variable
from torch.utils import data


def read_raw_signal(channel_param: int, good_subjects, window=2):

    subjects_unseen = {}
    subjects_seen = {}

    counter_seen = 0
    counter_unseen = 0
    for subject_number in good_subjects:
        id_patient = '{0:0=3d}'.format(subject_number)
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
                # TODO: Add for all windows
                if window == 2:
                    ch1 = ch1[:512]
                elif window == 3:
                    ch1 = ch1[-512:]

                subjects_unseen.update({counter_unseen: ch1})
                counter_unseen += 1

            if trial_states[index] == 'Seen':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                # TODO: Add for all windows
                if window == 2:
                    ch1 = ch1[:512]
                elif window == 3:
                    ch1 = ch1[-512:]

                subjects_seen.update({counter_seen: ch1})
                counter_seen += 1
    return subjects_unseen, subjects_seen


def prepare_11_fold(index_test, channel, subjects, generated=False):
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

    train = torch.cat((train_seen, train_unseen))
    test = torch.cat((test_seen, test_unseen))

    targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
    targets_seen = torch.from_numpy(np.ones(len(train_seen)))
    targets = torch.cat((targets_seen, targets_unseen))

    test_targets_unseen = torch.from_numpy(np.zeros(len(test_unseen)))
    test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
    test_targets = torch.cat((test_targets_seen, test_targets_unseen))

    if generated == True:
        seen_gen = torch.load(
            '/home/vlad/Desktop/Results_GAN/channel_{}/generated/seen_tensor.pt'
            .format(channel))
        unseen_gen = torch.load(
            '/home/vlad/Desktop/Results_GAN/channel_{}/generated/unseen_tensor.pt'
            .format(channel))

        for i in range(0, len(seen_gen)):
            seen_gen[i] = seen_gen[i].float()
            unseen_gen[i] = unseen_gen[i].float()

        train = torch.cat((train, seen_gen, unseen_gen))

        targets_unseen_gen = torch.from_numpy(np.zeros(len(seen_gen)))
        targets_seen_gen = torch.from_numpy(np.ones(len(unseen_gen)))
        targets = torch.cat((targets, targets_seen_gen, targets_unseen_gen))

    return train, test, targets, test_targets


def prepare_data(signals_unseen, signals_seen, path_channel, generated=False):
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

    train = torch.cat((train_seen, train_unseen))
    test = torch.cat((test_seen, test_unseen))

    targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
    targets_seen = torch.from_numpy(np.ones(len(train_seen)))
    targets = torch.cat((targets_seen, targets_unseen))

    test_targets_unseen = torch.from_numpy(np.zeros(len(test_seen)))
    test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
    test_targets = torch.cat((test_targets_seen, test_targets_unseen))

    if generated == True:
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


#TODO: This needs to be tested. After I have made the code beautiful I did not have time to test it.
#This method is not used btw
def prepare_data_shuffle(signals_unseen,
                         signals_seen,
                         path_channel,
                         generated=False):

    if generated:

        with open(path_channel + "11Fold/shuffle/train.pkl", 'rb') as file:
            train = pickle.load(file)
        with open(path_channel + "11Fold/shuffle/test.pkl", 'rb') as file:
            test = pickle.load(file)
        with open(path_channel + "11Fold/shuffle/targets.pkl", 'rb') as file:
            targets = pickle.load(file)
        with open(path_channel + "11Fold/shuffle/test_targets.pkl",
                  'rb') as file:
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

    else:

        list_unseen = list(signals_unseen.items())
        random.shuffle(list_unseen)

        list_seen = list(signals_seen.items())
        random.shuffle(list_seen)

        minimum = min(len(list_seen), len(list_unseen))
        number_seen = len(list_seen) - minimum
        number_unseen = len(list_unseen) - minimum
        for i in range(
                len(list_seen) - 1,
                len(list_seen) - number_seen - 1, -1):
            list_seen.pop(i)
        for i in range(
                len(list_unseen) - 1,
                len(list_unseen) - number_unseen - 1, -1):
            list_unseen.pop(i)

        size_train = round(len(list_seen) * 0.8)

        train_unseen = []
        train_seen = []
        for i in range(0, size_train):
            train_unseen.append(torch.from_numpy(list_unseen[i][1]))
            train_seen.append(torch.from_numpy(list_seen[i][1]))

        test_unseen = []
        test_seen = []
        for i in range(size_train, len(list_seen)):
            test_unseen.append(torch.from_numpy(list_unseen[i][1]))
            test_seen.append(torch.from_numpy(list_seen[i][1]))

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

        train = torch.cat((train_seen, train_unseen))
        test = torch.cat((test_seen, test_unseen))

        targets_unseen = torch.from_numpy(np.zeros(len(train_unseen)))
        targets_seen = torch.from_numpy(np.ones(len(train_seen)))
        targets = torch.cat((targets_seen, targets_unseen))

        test_targets_unseen = torch.from_numpy(np.zeros(len(test_seen)))
        test_targets_seen = torch.from_numpy(np.ones(len(test_seen)))
        test_targets = torch.cat((test_targets_seen, test_targets_unseen))

        f = open(path_channel + "train.pkl", "wb")
        pickle.dump(train, f)
        f.close()
        f = open(path_channel + "test.pkl", "wb")
        pickle.dump(test, f)
        f.close()
        f = open(path_channel + "targets.pkl", "wb")
        pickle.dump(targets, f)
        f.close()
        f = open(path_channel + "test_targets.pkl", "wb")
        pickle.dump(test_targets, f)
        f.close()

    return train, test, targets, test_targets


def load_data(seen):
    # ------------LOAD DATA SIGNAL----------#
    if seen == False:
        with open('./data/signal_unseen.pkl', 'rb') as file:
            signals = pickle.load(file)
    else:
        with open('./data/signal_seen.pkl', 'rb') as file:
            signals = pickle.load(file)
    for key in signals.keys():
        signals[key] = torch.from_numpy(signals[key])
        signals[key] = signals[key].float()

    return signals


def load_data_wgan(seen, batch_size):
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


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def transform(signals, mode):
    signals_mag = torch.FloatTensor(size=(signals.shape[0], 257))
    signals_phase = torch.FloatTensor(size=(signals.shape[0], 257))

    k = 0
    for signal in signals:
        signals_mag[k] = torch.rfft(signal.clone(),
                                    signal_ndim=1,
                                    onesided=True)[:, 0]
        signals_phase[k] = torch.rfft(signal.clone(),
                                      signal_ndim=1,
                                      onesided=True)[:, 1]
        k += 1

    for index in range(0, signals.shape[0]):
        if (mode == 'gamma'):
            signals_mag[index][0:30] = 0
            signals_phase[index][0:30] = 0
        elif (mode == 'ab'):
            signals_mag[index][0:8] = 0
            signals_phase[index][0:8] = 0
            signals_mag[index][30:257] = 0
            signals_phase[index][30:257] = 0
        elif (mode == 'theta'):
            signals_mag[index][8:257] = 0
            signals_phase[index][8:257] = 0

    signal_full = torch.FloatTensor(size=(signals.shape[0], 257, 2))
    for i in range(0, signals.shape[0]):
        signal_full[i][:, 0] = signals_mag[i]
        signal_full[i][:, 1] = signals_phase[i]

    all_signals = []
    for i in range(0, signals.shape[0]):
        all_signals.append(
            torch.irfft(signal_full, 1,
                        onesided=True).detach().numpy()[i][0:512])

    signals_torch = []
    for i in range(0, signals.shape[0]):
        signals_torch.append(torch.from_numpy(all_signals[i]))

    all_signals = torch.stack(signals_torch)

    return all_signals
