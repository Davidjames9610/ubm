import math
from typing import List
import pandas as pd
from scipy.io import wavfile
import numpy as np
import decimal


def time_to_samples(string):
    h, m, s = string.split(':')
    s, ms = s.split('.')
    ms = int(ms)
    ms += int(s) * 1000
    ms += int(m) * 60 * 1000
    ms += int(h) * 60 * 60 * 1000
    return ms


class Annotation:
    def __init__(self, start_samples, end_samples, start_ms, end_ms):
        self.start = start_samples
        self.end = end_samples
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.start_fs = 0
        self.end_fs = 0

    def __str__(self):
        return f"annotation start/end"


class Annotations:
    def __init__(self):
        self.annotations = []

    def __str__(self):
        return f"annotations"

    def set_annotations(self, labels, fs):
        pass

    def set_frequency_annotations(self, frame_len, frame_step):
        for annot in self.annotations:
            annot.start_fs = max(2 + int(math.ceil((1.0 * annot.start - frame_len) / frame_step)), 0)
            annot.end_fs = max(1 + int(math.ceil((1.0 * annot.end - frame_len) / frame_step)), 0)
        return self.annotations

    def get_only_time(self):
        annotations = map(lambda x: [x.start_ms, x.end_ms], self.annotations)
        return annotations

    def get_only_frequency(self):
        annotations = map(lambda x: [x.start_fs, x.end_fs], self.annotations)
        return annotations

    def get_only_sample(self):
        annotations = map(lambda x: [x.start, x.end], self.annotations)
        return annotations


class AnnotationsJacques(Annotations):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"AnnotationsJacques"

    def set_annotations(self, labels, fs):
        df = pd.read_csv(labels, encoding='latin', sep='\t', names=['times'])
        annotation_list = df.times.to_numpy()
        annotations = []
        for annot in annotation_list:
            split = annot.split(' - ')
            start_ms = time_to_samples(split[0])
            end_ms = time_to_samples(split[1])
            start_samples = int((start_ms / 1000) * fs)
            end_samples = int((end_ms / 1000) * fs)
            annotations.append(Annotation(start_samples, end_samples, start_ms, end_ms))
        self.annotations = annotations


class AnnotationsAudacity(Annotations):
    def __init__(self):
        super().__init__()

    def set_annotations(self, labels, fs):
        df = pd.read_csv(labels, encoding='UTF-8', sep='\t', names=['start', 'end', ''])

        start = df.start.to_numpy()
        end = df.end.to_numpy()
        annotation_list = np.column_stack((start, end))

        annotations = []
        for annot in annotation_list:
            start_ms = annot[0] * 1000
            end_ms = annot[1] * 1000
            start_samples = int((start_ms / 1000) * fs)
            end_samples = int((end_ms / 1000) * fs)
            annotations.append(Annotation(start_samples, end_samples, start_ms, end_ms))
        self.annotations = annotations
