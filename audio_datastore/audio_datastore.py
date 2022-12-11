import os
import os.path
from pathlib import Path
import copy

import numpy as np


class AudioDatastore:
    def __init__(self, folders=None, files=None, labels=None):
        self.folders = folders
        self.files: [] = files
        self.labels: [] = labels

    def set(self, folders=None, files=None, labels=None):
        if folders:
            self.folders = folders
        if files:
            self.files = files
        if labels:
            self.labels = labels

    def populate(self, folders, include_sub_folders=False, label_source=None):
        self.folders = folders

        files = []
        labels = []

        # if include_sub_folders:
        for folder_name in os.listdir(folders):
            folder_path = Path(folders) / folder_name
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".wav"):
                        file_path = Path(folder_path) / file_name
                        files.append(file_path)
                        if label_source:
                            labels.append(folder_name)

        self.files = files
        self.labels = labels


# by label
def subset(audio_datastore: AudioDatastore, label):
    labels = audio_datastore.labels
    files = audio_datastore.files
    folders = audio_datastore.folders
    new_labels = []
    new_files = []

    for i in range(len(labels)):
        if labels[i] in label:
            new_labels.append(labels[i])
            new_files.append(files[i])

    new_ads = AudioDatastore(folders, files=new_files, labels=new_labels)

    return new_ads


def filter(audio_datastore: AudioDatastore, files_to_avoid):
    labels = audio_datastore.labels
    files = audio_datastore.files
    folders = audio_datastore.folders
    new_labels = []
    new_files = []

    for i in range(len(files)):
        if files[i] not in files_to_avoid:
            new_labels.append(labels[i])
            new_files.append(files[i])

    new_ads = AudioDatastore(folders, files=new_files, labels=new_labels)

    return new_ads


# return two ads objects, one with the amount the other with what's left
def split(audio_datastore: AudioDatastore, amount):

    labels = audio_datastore.labels
    files = audio_datastore.files
    folders = audio_datastore.folders

    new_labels = []
    new_files = []
    new_labels_two = []
    new_files_two = []

    label_set = np.unique(audio_datastore.labels)

    for label in label_set:
        label_count = 0
        for i in range(len(labels)):
            current_label = labels[i]
            if current_label == label and label_count < amount:
                new_labels.append(labels[i])
                new_files.append(files[i])
                label_count += 1
            elif current_label == label and label_count >= amount:
                new_labels_two.append(labels[i])
                new_files_two.append(files[i])
                label_count += 1

    new_ads = AudioDatastore(folders, files=new_files, labels=new_labels)
    new_ads_two = AudioDatastore(folders, files=new_files_two, labels=new_labels_two)
    return new_ads, new_ads_two


def get_deep_copy(audio_datastore: AudioDatastore):
    labels = copy.copy(audio_datastore.labels)
    files = copy.copy(audio_datastore.files)
    folders = copy.copy(audio_datastore.folders)
    return AudioDatastore(folders=folders, files=files, labels=labels)


def clean_zeros(audio_datastore: AudioDatastore, features):
    ads_copy = get_deep_copy(audio_datastore)
    features_copy = copy.copy(features)

    zero_indices = []
    removed_files = []
    for i in range(len(features_copy)):
        if len(features_copy[i]) == 0 or len(features_copy[i]) < 15:
            zero_indices.append(i)
            removed_files.append(ads_copy.labels[i])

    for index in sorted(zero_indices, reverse=True):
        del features_copy[index]
        del ads_copy.labels[index]
        del ads_copy.files[index]

    for file in ads_copy.files:
        if file in removed_files:
            print('oh deary me')

    return ads_copy, features_copy
