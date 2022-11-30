import random
from typing import List
from scipy.io import wavfile
from annotations import Annotations
import decimal
import math


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def get_labeled_data(file, labels):
    all_samples = []
    for a in labels:
        sample = file[a[0]:a[1]]
        all_samples.append(sample)
    return all_samples


def seconds_to_samples(sec, fs):
    sam = math.floor(sec * fs)
    return sam


class GetDataBase:

    def __init__(self,
                 location_of_wav_file,
                 location_of_annotations,
                 annotations: Annotations,
                 window_length,
                 frame_step,
                 samples=True
                 ):
        self.location_of_wav_file = location_of_wav_file
        self.location_of_annotations = location_of_annotations

        fs, audio = wavfile.read(location_of_wav_file)
        self.fs = fs
        self.audio = audio

        annotations.set_annotations(location_of_annotations, fs)
        self.annotations = annotations

        self.window_length = None
        self.frame_step = None
        self.set_frequency_annotations(window_length, frame_step, samples)

        self.features = None
        self.spliced_features = None
        self.feature_type = None

    def __str__(self):
        return f"getData"

    def get_audio(self):
        return self.audio

    def set_features(self, features, feature_type='not given'):
        self.features = features
        self.feature_type = feature_type

    def set_frequency_annotations(self, window_length, frame_step, samples=True):
        if not samples:
            window_length = seconds_to_samples(window_length, self.fs)
            frame_step = seconds_to_samples(frame_step, self.fs)
        self.window_length = window_length
        self.frame_step = frame_step
        self.annotations.set_frequency_annotations(self.window_length, self.frame_step)

    def splice_wav_in_samples(self):
        sample_annotations = self.annotations.get_only_sample()
        return get_labeled_data(self.audio, sample_annotations)

    def splice_features_in_frequency(self):
        frequency_annotations = self.annotations.get_only_frequency()
        self.spliced_features = get_labeled_data(self.features, frequency_annotations)
        return self.spliced_features

    def get_window_in_seconds(self):
        win_len = math.floor(self.window_length * 1000 / self.fs) / 1000
        win_step = math.floor(self.frame_step * 1000 / self.fs) / 1000
        return win_len, win_step

    def get_random_train_test(self, train_index, percentage=True):
        total_length = len(self.spliced_features)
        if percentage:
            train_index = int(total_length * train_index / 100)
        randomised_features = self.spliced_features[:]
        random.shuffle(randomised_features)
        return randomised_features[:train_index], randomised_features[train_index:]
