import numpy as np
from .read import *

import os
from os import path
import pickle


def generate_signals_seen(channel_param: int, subjects_correct_seen):

    subjects_seen = {}

    counter_seen = 0

    for jndex in subjects_correct_seen:
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
            if trial_states[index] == 'Seen':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                ch1 = ch1[:512]
                subjects_seen.update({counter_seen: ch1})
                counter_seen += 1

    f = open(
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/signal_seen.pkl",
        "wb")
    pickle.dump(subjects_seen, f)
    f.close()


def generate_signals_unseen(channel_param: int, subjects_correct_unseen):

    subjects_unseen = {}
    counter_unseen = 0
    for jndex in subjects_correct_unseen:
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

    f = open(
        "/home/vlad/Desktop/Licenta/functional_networks_dots_30/gan/data/signal_unseen.pkl",
        "wb")
    pickle.dump(subjects_unseen, f)
    f.close()


def generate_signals_per_subject(channel_param: int):

    os.makedirs(
        "/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/all_subjects".format(
            channel_param),
        exist_ok=True)

    for jndex in range(1, 12):
        subjects_seen = {}
        subjects_unseen = {}
        counter_seen = 0
        counter_unseen = 0
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
            if trial_states[index] == 'Seen':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                ch1 = ch1[:512]
                subjects_seen.update({counter_seen: ch1})
                counter_seen += 1
        for index in range(0, 210):
            if trial_states[index] == 'Nothing':
                ch1 = array[timestamps[indices_response[index]]:timestamps[
                    indices_end_response[index]]]
                ch1 = ch1[:512]
                subjects_unseen.update({counter_unseen: ch1})
                counter_unseen += 1

        f = open(
            "/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/all_subjects/signal_seen_{}.pkl"
            .format(channel_param, jndex), "wb")
        pickle.dump(subjects_seen, f)
        f.close()
        f = open(
            "/home/vlad/Desktop/Results_GAN/channel_{}/11Fold/all_subjects/signal_unseen_{}.pkl"
            .format(channel_param, jndex), "wb")
        pickle.dump(subjects_unseen, f)
        f.close()