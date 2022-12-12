import numpy as np

from audio_datastore.audio_datastore import *
from feature_extraction.fe_configs import NormFactor


class FeatureExtractorBase:

    def __init__(self):
        self.norm: NormFactor | None = None

    def __str__(self):
        return f"FeatureExtractorBase"

    def set_normalisation(self, ads: AudioDatastore):
        all_features = []
        for file in ads.files:
            # this will need to be updated with pre-processing too
            feature = self.extract_feature(file)
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

    def extract_and_normalize_feature(self, file):
        feature = self.extract_feature(file)
        norm_feature = (feature - self.norm.means) / self.norm.std
        norm_feature = norm_feature - np.mean(norm_feature)
        return norm_feature
