import numpy as np

from audio_datastore.audio_datastore import *
from feature_extraction.fe_configs import NormFactor
from processing.process_method_base import ProcessMethodBase
from processing.processing import *
import my_torch.tuts2.torch_transforms as torch_t
import utils

class FeatureExtractorBase:

    def __init__(self):
        self.norm: NormFactor | None = None

    def __str__(self):
        return f"FeatureExtractorBase"

    def extract_feature(self, signal):
        return np.array([])

    # def normalize_feature(self, feature):
    #     norm_feature = (feature - self.norm.means) / self.norm.std
    #     norm_feature = norm_feature - np.mean(norm_feature)
    #     return norm_feature

    def __call__(self, signal):
        feature = self.extract_feature(signal)
        if self.norm:
            feature = (feature - self.norm.means) / self.norm.std
            # feature = feature - np.mean(feature)
        return feature

    def set_normalisation(self, ads: AudioDatastore, process_method: torch_t.ComposeTransform | None = None,
                          verbose=False):
        # calculate normalisation
        all_features = []
        count = 0
        for i in range(len(ads.labels)):
            file = ads[i]
            if verbose and count % 10 == 0:
                print('processed files: ', count)
            if process_method is not None:
                file = process_method(file)
            feature = self.extract_feature(file)
            all_features.append(feature)
            count = count + 1

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
