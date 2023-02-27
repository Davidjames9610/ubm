import os
import os.path
from pathlib import Path
import copy
import my_torch.tuts2.torch_transforms as torch_t
import my_torch.tuts2.combo_torch_transforms as torch_c
import numpy as np
from collections import Counter
import random

import utils


class AudioDatastore:
    def __init__(self, folders=None, files=None, labels=None, data=None, ads_type='files', average_power=None):
        self.folders = folders
        self.files: [] = files
        self.labels: [] = labels
        self.data: [] = data
        self.ads_type = ads_type
        self.average_power = average_power

    def __getitem__(self, index):
        return getattr(self, self.ads_type)[index]

    def populate(self, folders, include_sub_folders=False, label_source=None):
        self.folders = folders

        files = []
        labels = []
        data = []

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
        self.data = data

    def set(self, folders=None, files=None, labels=None, data=None):
        if folders:
            self.folders = folders
        if files:
            self.files = files
        if labels:
            self.labels = labels
        if data:
            self.data = data

    def info(self, name=None):
        print(name, 'database info: ')
        print(Counter(self.labels).keys())
        print(Counter(self.labels).values())
        print('\n')

    def get_average_power(self, amount=50, process_method=torch_c.file_to_numpy()):
        data = getattr(self, self.ads_type)
        random_samples = random.sample(data, amount)
        average_power = []
        for data in random_samples:
            if self.ads_type == 'files':
                data = process_method(data)
            average_power.append(utils.get_average_power(data))
        self.average_power = round(np.mean(average_power), 5)


def subset(audio_datastore: AudioDatastore, label):
    labels = audio_datastore.labels
    files = audio_datastore.files
    data = audio_datastore.data
    new_labels = []
    new_files = []
    new_data = []

    for i in range(len(labels)):
        if labels[i] in label:
            new_labels.append(labels[i])
            new_files.append(files[i])
            if data and len(data) > 0:
                new_data.append(data[i])

    return AudioDatastore(
        folders=audio_datastore.folders,
        files=new_files,
        labels=new_labels,
        data=new_data,
        ads_type=audio_datastore.ads_type,
        average_power=audio_datastore.average_power
    )


# by label
def filter_out(audio_datastore: AudioDatastore, files_to_avoid):
    labels = audio_datastore.labels
    files = audio_datastore.files
    data = audio_datastore.data
    new_labels = []
    new_files = []
    new_data = []

    for i in range(len(files)):
        if files[i] not in files_to_avoid:
            new_labels.append(labels[i])
            new_files.append(files[i])
            if data and len(data) > 0:
                new_data.append(data[i])

    new_ads = AudioDatastore(
        folders=audio_datastore.folders,
        files=new_files,
        labels=new_labels,
        data=new_data,
        ads_type=audio_datastore.ads_type,
        average_power=audio_datastore.average_power
    )

    return new_ads


# return two ads objects, one with the amount the other with what's left
def split(audio_datastore: AudioDatastore, amount):
    labels = audio_datastore.labels
    files = audio_datastore.files
    data = audio_datastore.data

    new_labels = []
    new_files = []
    new_data = []

    new_labels_two = []
    new_files_two = []
    new_data_two = []

    label_set = np.unique(audio_datastore.labels)

    for label in label_set:
        label_count = 0
        for i in range(len(labels)):
            current_label = labels[i]
            if current_label == label and label_count < amount:
                new_labels.append(labels[i])
                new_files.append(files[i])
                if data and len(data) > 0:
                    new_data.append(data[i])
                label_count += 1
            elif current_label == label and label_count >= amount:
                new_labels_two.append(labels[i])
                new_files_two.append(files[i])
                if data and len(data) > 0:
                    new_data_two.append(data[i])
                label_count += 1

    new_ads = AudioDatastore(
        folders=audio_datastore.folders,
        files=new_files,
        labels=new_labels,
        data=new_data,
        ads_type=audio_datastore.ads_type,
        average_power=audio_datastore.average_power
    )
    new_ads_two = AudioDatastore(
        folders=audio_datastore.folders,
        files=new_files_two,
        labels=new_labels_two,
        data=new_data_two,
        ads_type=audio_datastore.ads_type,
        average_power=audio_datastore.average_power
    )
    return new_ads, new_ads_two


def get_deep_copy(audio_datastore: AudioDatastore):
    labels = copy.copy(audio_datastore.labels)
    files = copy.copy(audio_datastore.files)
    folders = copy.copy(audio_datastore.folders)
    data = copy.copy(audio_datastore.data)
    return AudioDatastore(
        folders=folders,
        files=files,
        labels=labels,
        data=data,
        ads_type=audio_datastore.ads_type,
        average_power=audio_datastore.average_power
    )
