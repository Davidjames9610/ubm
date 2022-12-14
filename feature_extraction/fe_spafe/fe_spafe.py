from feature_extraction.fe_base import FeatureExtractorBase
import numpy as np
import librosa
import scipy.io.wavfile as wav
import config as conf
from spafe.features import gfcc, mfcc
from utils import *

MFCC = 'spafe.features.mfcc.mfcc'
GFCC = 'spafe.features.gfcc.gfcc'


class FeatureExtractorSpafe(FeatureExtractorBase):

    def __init__(self, fe_type):
        super().__init__()
        self.fe_type = fe_type
        self.fe_func = get_class(self.fe_type)

    def __str__(self):
        return self.fe_type

    def extract_feature(self, signal):
        feature = self.fe_func(sig=signal, fs=conf.SAMPLING_RATE, num_ceps=conf.N_MFCC, nfft=conf.N_FFT)
        return feature


class FeatureExtractorSpafeDelta(FeatureExtractorSpafe):

    def __str__(self):
        return self.fe_type + 'Delta'

    def extract_feature(self, signal):
        mfcc_feature = super().extract_feature(signal)
        delta_feature = librosa.feature.delta(mfcc_feature.T, width=3, order=1).T
        feature = np.concatenate([mfcc_feature, delta_feature], axis=1)
        return feature


class FeatureExtractorSpafeDeltaDeltas(FeatureExtractorSpafe):

    def __str__(self):
        return self.fe_type + 'DeltaDeltas'

    def extract_feature(self, file):
        mfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(mfcc_feature.T, order=1).T
        delta_delta_feature = librosa.feature.delta(mfcc_feature.T, order=2).T
        feature = np.concatenate([mfcc_feature, delta_feature, delta_delta_feature], axis=1)
        return feature
