from feature_extraction.fe_base import FeatureExtractorBase
import numpy as np
import librosa
import scipy.io.wavfile as wav
from feature_extraction import fe_configs as conf
from spafe.features import gfcc, mfcc, lfcc

mfcc_types = ['FeatureExtractorMFCC', 'FeatureExtractorMFCCDelta', 'FeatureExtractorMFCCDeltaDeltas',
              'FeatureExtractorGFCC', 'FeatureExtractorGFCCDelta', 'FeatureExtractorGFCCDeltaDeltas']


class FeatureExtractorMFCC(FeatureExtractorBase):

    def __str__(self):
        return f"FeatureExtractorMFCC"

    def extract_feature(self, file):
        signal, sr = librosa.load(file, sr=conf.SAMPLING_RATE)
        signal = librosa.util.normalize(signal)
        feature = mfcc.mfcc(sig=signal, fs=conf.SAMPLING_RATE, num_ceps=conf.N_MFCC, nfft=conf.N_FFT)
        return feature


class FeatureExtractorMFCCDelta(FeatureExtractorMFCC):

    def __str__(self):
        return f"FeatureExtractorMFCCDelta"

    def extract_feature(self, file):
        mfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(mfcc_feature.T, width=3, order=1).T
        feature = np.concatenate([mfcc_feature, delta_feature], axis=1)
        return feature


class FeatureExtractorMFCCDeltaDeltas(FeatureExtractorMFCC):

    def __str__(self):
        return f"FeatureExtractorMFCCDeltaDeltas"

    def extract_feature(self, file):
        mfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(mfcc_feature.T, order=1).T
        delta_delta_feature = librosa.feature.delta(mfcc_feature.T, order=2).T
        feature = np.concatenate([mfcc_feature, delta_feature, delta_delta_feature], axis=1)
        return feature


class FeatureExtractorGFCC(FeatureExtractorBase):

    def __str__(self):
        return f"FeatureExtractorGFCC"

    def extract_feature(self, file):
        signal, sr = librosa.load(file, sr=conf.SAMPLING_RATE)
        signal = librosa.util.normalize(signal)
        feature = gfcc.gfcc(sig=signal, fs=conf.SAMPLING_RATE, num_ceps=conf.N_MFCC, nfft=conf.N_FFT)
        return feature


class FeatureExtractorGFCCDelta(FeatureExtractorGFCC):
    def __str__(self):
        return f"FeatureExtractorGFCCDelta"

    def extract_feature(self, file):
        gfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(gfcc_feature.T, order=1).T
        feature = np.concatenate([gfcc_feature, delta_feature], axis=1)
        return feature


class FeatureExtractorGFCCDeltaDeltas(FeatureExtractorGFCC):

    def __str__(self):
        return f"FeatureExtractorGFCCDeltaDeltas"

    def extract_feature(self, file):
        gfcc_feature = super().extract_feature(file)
        delta_feature = librosa.feature.delta(gfcc_feature.T, order=1).T
        delta_delta_feature = librosa.feature.delta(gfcc_feature.T, order=2).T
        feature = np.concatenate([gfcc_feature, delta_feature, delta_delta_feature], axis=1)
        return feature
