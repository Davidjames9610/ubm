import numpy as np

from audio_datastore.audio_datastore import *
from feature_extraction.fe_configs import NormFactor
from processing.process_method_base import ProcessMethodBase
from processing.processing import *


class FeatureExtractorBase:

    def __init__(self, normalize=False):
        self.norm: NormFactor | None = None
        self.normalize = normalize

    def __str__(self):
        return f"FeatureExtractorBase"

    def set_normalisation(self, ads: AudioDatastore, process_method: ProcessMethodBase):
        all_features = []
        for file in ads.files:
            signal = process_method.pre_process(file)
            feature = self.extract_feature(signal)
            all_features.append(feature)

        means = []
        std = []
        for feature in all_features:
            means.append(np.mean(feature, axis=0))
            std.append(np.std(feature, axis=0))

        means = np.array(means)
        means = np.mean(means, axis=0)

        std = np.array(std)
        std = np.mean(std, axis=0)

        self.norm = NormFactor(means, std)

    def extract_feature(self, signal):
        return np.array([])

    def normalize_feature(self, feature):
        norm_feature = (feature - self.norm.means) / self.norm.std
        norm_feature = norm_feature - np.mean(norm_feature)
        return norm_feature

    def extract_and_normalize_feature(self, signal):
        feature = self.extract_feature(signal)
        if self.normalize:
            feature = (feature - self.norm.means) / self.norm.std
            feature = feature - np.mean(feature)
        return feature
