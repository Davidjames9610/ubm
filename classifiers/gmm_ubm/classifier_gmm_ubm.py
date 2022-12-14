from sklearn.mixture import GaussianMixture, _gaussian_mixture

from audio_datastore.audio_datastore import AudioDatastore, subset, filter
from classifiers.classifier_base import ClassifierBase
from feature_extraction.fe_base import FeatureExtractorBase
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Optional, Tuple
import torch
import torchaudio
from processing.process_method_base import ProcessMethodBase

eps = np.finfo(np.float64).eps


class ClassifierGMMUBM(ClassifierBase):

    def __str__(self):
        return f"ClassifierGMMUBM"

    def __init__(self, fe_method: FeatureExtractorBase, process_method: ProcessMethodBase):
        super().__init__(fe_method, process_method)
        self.ubm: GaussianMixture | None = None
        self.enrolled_gmms: {GaussianMixture} = {}
        self.num_features = None
        self.num_components = 32
        self.relevance_factor = 16
        self.speakers = None
        self.test_results = {}

    def train(self, ads_train: AudioDatastore):
        print('training for: ', self.fe_method.__str__())
        train_features = []
        for i in range(len(ads_train.files)):
            signal = self.process_method.pre_process(ads_train.files[i])
            train_feature = self.fe_method.extract_and_normalize_feature(signal)
            train_features.append(train_feature)
        ubm = GaussianMixture(n_components=self.num_components, covariance_type='diag')
        train_features_flattened = np.array([item for sublist in train_features for item in sublist])
        ubm.fit(train_features_flattened)
        self.ubm = ubm
        self.num_features = train_features[0].shape[1]

    def enroll(self, ads_enroll: AudioDatastore):
        print('enrolling for ', self.fe_method.__str__())
        self.speakers = np.unique(ads_enroll.labels)

        speakers = np.unique(ads_enroll.labels)
        self.enrolled_gmms = {}

        for i in range(len(speakers)):
            ads_train_subset = subset(ads_enroll, speakers[i])
            N = np.zeros((1, self.num_components))
            F = np.zeros((self.num_features, self.num_components))
            S = np.zeros((self.num_features, self.num_components))

            for file in ads_train_subset.files:
                signal = self.process_method.pre_process(file)
                speaker_feature = self.fe_method.extract_and_normalize_feature(signal)
                if len(speaker_feature) > 0:
                    n, f, s, l = self.__helper_expectation(speaker_feature, self.ubm)
                    N = N + n
                    F = F + f
                    S = S + s
                else:
                    print('skipping train file because len = 0')
            N = np.maximum(N, eps)

            gmm = self.__helper_maximization(N, F, S, self.num_components)

            alpha = N / (N + self.relevance_factor)

            mu = (alpha.T * gmm.means_) + ((1 - alpha).T * self.ubm.means_)
            gmm.means_ = mu

            sigma = alpha * (S / N) + (1 - alpha) * (
                        self.ubm.covariances_.T + np.square(self.ubm.means_).T) - np.square(
                gmm.means_).T

            sigma = np.maximum(sigma, eps).T
            gmm.covariances_ = sigma

            gmm.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(gmm.covariances_, 'diag')

            weights = alpha * (N / np.sum(N)) + (1 - alpha) * self.ubm.weights_.T
            weights = np.squeeze(weights / np.sum(weights))
            gmm.weights_ = weights

            self.enrolled_gmms[speakers[i]] = gmm

    def set_effects(self, effects: List[List[str]]):
        self.effects = effects

    def test(self, ads_test: AudioDatastore):
        print('testing for ', self.fe_method.__str__())

        # confusion matrix

        scores = []
        labels = ads_test.labels
        for i in range(len(ads_test.files)):
            signal = self.process_method.pre_process(ads_test.files[i])
            signal = self.process_method.post_process(signal)
            speaker_feature = self.fe_method.extract_and_normalize_feature(signal)
            speakers_scores = []
            for s in range(len(self.speakers)):
                speaker_gmm = self.enrolled_gmms[self.speakers[s]]

                likelihood_speaker = logsumexp(speaker_gmm._estimate_weighted_log_prob(speaker_feature), axis=1)
                likelihood_ubm = logsumexp(self.ubm._estimate_weighted_log_prob(speaker_feature), axis=1)

                speakers_scores.append(np.mean(self.running_mean(likelihood_speaker - likelihood_ubm, 3)))

            scores.append(self.speakers[np.argmax(speakers_scores)])

        cm = confusion_matrix(np.array(scores), labels, labels=self.speakers, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.speakers)
        disp.plot(cmap=plt.cm.Blues, values_format='g')
        plt.show()

    @staticmethod
    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    @staticmethod
    def __helper_expectation(feature, ubm: GaussianMixture):
        post = ubm._estimate_weighted_log_prob(feature)
        l = logsumexp(post, axis=1)
        gamma = np.exp((post.T - l)).T
        n = np.sum(gamma, axis=0)
        f = np.dot(feature.T, gamma)
        s = np.dot(np.multiply(feature.T, feature.T), gamma)
        l = np.sum(l)
        return n, f, s, l

    @staticmethod
    # should be moved somewhere else
    def __create_gmm(w, m, c, num_components):
        gmm = GaussianMixture(n_components=num_components, covariance_type='diag')
        # quick check
        if w.shape[0] != num_components or m.shape[0] != num_components or c.shape[0] != num_components:
            print('somethings wrong')
        gmm.weights_ = w
        gmm.means_ = m
        gmm.covariances_ = c
        gmm.precisions_cholesky_ = _gaussian_mixture._compute_precision_cholesky(c, 'diag')
        return gmm

    def __helper_maximization(self, n, f, s, nc):
        n = np.maximum(n, eps)
        weights = np.maximum(n / np.sum(n), eps)
        weights = np.squeeze(weights / np.sum(weights))
        means = (f / n)
        covars = np.maximum((s / n) - np.square(means), eps)
        gmm = self.__create_gmm(w=weights.T, m=means.T, c=covars.T, num_components=self.num_components)
        return gmm
