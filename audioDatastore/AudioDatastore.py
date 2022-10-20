import os
import os.path
from pathlib import Path


class AudioDatastore:
    def __init__(self, folders=None, files=None, labels=None):
        self.folders = folders
        self.files = files
        self.labels = labels

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

def filter(audio_datastore: AudioDatastore, label):
    labels = audio_datastore.labels
    files = audio_datastore.files
    folders = audio_datastore.folders
    new_labels = []
    new_files = []

    for i in range(len(labels)):
        if labels[i] not in label:
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

    for i in range(len(labels)):
        if i < amount:
            new_labels.append(labels[i])
            new_files.append(files[i])
        else:
            new_labels_two.append(labels[i])
            new_files_two.append(files[i])

    new_ads = AudioDatastore(folders, files=new_files, labels=new_labels)
    new_ads_two = AudioDatastore(folders, files=new_files_two, labels=new_labels_two)
    return new_ads, new_ads_two

