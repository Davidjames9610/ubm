from feature_extraction.fe_base import FeatureExtractorBase
import numpy as np
import librosa
import scipy.io.wavfile as wav
import config as conf
# from spafe.features import gfcc, mfcc
from utils import get_class
from spafe.features.mfcc import *

MFCC = 'spafe.features.mfcc.mfcc'
GFCC = 'spafe.features.gfcc.gfcc'


class FeatureExtractorLogMel(FeatureExtractorBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'log mel spafe'

    def extract_feature(self, sig):
        features, _ = mel_spectrogram(sig=sig, fs=conf.SAMPLING_RATE, nfft=conf.N_FFT)
        features_no_zero = zero_handling(features)
        log_features = np.log(features_no_zero)
        return log_features

class FeatureExtractorMfcc(FeatureExtractorBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'mfcc spafe'

    def extract_feature(self, sig):
        features = mfcc(sig=sig, fs=conf.SAMPLING_RATE, num_ceps=conf.N_MFCC, nfft=conf.N_FFT)
        features = zero_handling(features)
        return features

class FeatureExtractorMfccDelta(FeatureExtractorMfcc):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'mfcc delta spafe'

    def extract_feature(self, sig):
        mfcc_feature = super().extract_feature(sig)
        delta_feature = librosa.feature.delta(mfcc_feature.T, width=3, order=1).T
        feature = np.concatenate([mfcc_feature, delta_feature], axis=1)
        return feature

class FeatureExtractorMfccDeltaDeltas(FeatureExtractorMfcc):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'mfcc delta delta spafe'

    def extract_feature(self, file):
        mfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(mfcc_feature.T, order=1).T
        delta_delta_feature = librosa.feature.delta(mfcc_feature.T, order=2).T
        feature = np.concatenate([mfcc_feature, delta_feature, delta_delta_feature], axis=1)
        return feature

#
# class FeatureExtractorSpafe(FeatureExtractorBase):
#
#     def __init__(self, fe_type):
#         super().__init__()
#         self.fe_type = fe_type
#         self.fe_func = get_class(self.fe_type)
#
#     def __str__(self):
#         return self.fe_type
#
#     def extract_feature(self, signal):
#         feature = self.fe_func(sig=signal, fs=conf.SAMPLING_RATE, num_ceps=conf.N_MFCC, nfft=conf.N_FFT)
#         return feature
#
#
# class FeatureExtractorSpafeDelta(FeatureExtractorSpafe):
#
#     def __str__(self):
#         return self.fe_type + 'Delta'
#
#     def extract_feature(self, signal):
#         mfcc_feature = super().extract_feature(signal)
#         delta_feature = librosa.feature.delta(mfcc_feature.T, width=3, order=1).T
#         feature = np.concatenate([mfcc_feature, delta_feature], axis=1)
#         return feature
#
#
# class FeatureExtractorSpafeDeltaDeltas(FeatureExtractorSpafe):
#
#     def __str__(self):
#         return self.fe_type + 'DeltaDeltas'
#
#     def extract_feature(self, file):
#         mfcc_feature = super().extract_feature(file)
#         delta_feature = librosa.feature.delta(mfcc_feature.T, order=1).T
#         delta_delta_feature = librosa.feature.delta(mfcc_feature.T, order=2).T
#         feature = np.concatenate([mfcc_feature, delta_feature, delta_delta_feature], axis=1)
#         return feature
