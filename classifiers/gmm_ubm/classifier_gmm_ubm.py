from sklearn.mixture import GaussianMixture, _gaussian_mixture

from audio_datastore.audio_datastore import AudioDatastore, subset, filter_out
from classifiers.classifier_base import ClassifierBase
from feature_extraction.fe_base import FeatureExtractorBase
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from my_torch.tuts2.torch_transforms import ComposeTransform
from typing import List, Optional, Tuple
import torch
import torchaudio

from my_torch.tuts2.torch_transforms import ComposeTransform
from processing.process_method_base import ProcessMethodBase

eps = np.finfo(np.float64).eps


class ClassifierGMMUBM(ClassifierBase):

    def __str__(self):
        return f"ClassifierGMMUBM"

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform, info=None):
        super().__init__(train_process, test_process, info)
        self.ubm: GaussianMixture | None = None
        self.enrolled_gmms: {GaussianMixture} = {}
        self.num_features = None
        self.num_components = 32
        self.relevance_factor = 16
        self.speakers = None
        self.test_results = {}

    def train(self, ads: AudioDatastore):
        features = []
        for i in range(len(ads.labels)):
            feature = self.train_process(ads[i])
            features.append(feature)
        ubm = GaussianMixture(n_components=self.num_components, covariance_type='diag')
        features_flattened = np.array([item for sublist in features for item in sublist])
        ubm.fit(features_flattened)
        self.ubm = ubm
        self.num_features = features[0].shape[1]

    def enroll(self, ads_enroll: AudioDatastore):
        self.speakers = np.unique(ads_enroll.labels)

        speakers = np.unique(ads_enroll.labels)
        self.enrolled_gmms = {}

        for y in range(len(speakers)):
            ads_enroll_subset = subset(ads_enroll, speakers[y])
            N = np.zeros((1, self.num_components))
            F = np.zeros((self.num_features, self.num_components))
            S = np.zeros((self.num_features, self.num_components))

            for i in range(len(ads_enroll_subset.labels)):
                enroll_feature = self.train_process(ads_enroll_subset[i])
                if len(enroll_feature) > 0:
                    n, f, s, l = self.__helper_expectation(enroll_feature, self.ubm)
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

            self.enrolled_gmms[speakers[y]] = gmm

    def test_all(self, ads_test: AudioDatastore, thresholds=None):
        self.test_confusion_matrix(ads_test)
        self.test_frr(ads_test, thresholds)
        self.test_far(ads_test, thresholds)
        self.test_det(thresholds)

    def test_confusion_matrix(self, ads_test: AudioDatastore):
        scores = []
        labels = ads_test.labels
        for i in range(len(ads_test.labels)):
            speaker_feature = self.test_process(ads_test[i])
            speakers_scores = []
            for s in range(len(self.speakers)):
                speaker_gmm = self.enrolled_gmms[self.speakers[s]]

                likelihood_speaker = logsumexp(speaker_gmm._estimate_weighted_log_prob(speaker_feature), axis=1)
                likelihood_ubm = logsumexp(self.ubm._estimate_weighted_log_prob(speaker_feature), axis=1)

                speakers_scores.append(np.mean(self.running_mean(likelihood_speaker - likelihood_ubm, 3)))

            scores.append(self.speakers[np.argmax(speakers_scores)])

        self.test_results['cm'] = scores

        cm = confusion_matrix(np.array(scores), labels, labels=self.speakers, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.speakers)
        disp.plot(cmap=plt.cm.Blues, values_format='g')
        plt.show()

    def test_frr(self, ads_test, thresholds=None):
        llr = []
        for s in range(len(self.speakers)):
            cur_speaker = self.speakers[s]
            local_gmm = self.enrolled_gmms[cur_speaker]
            ads_test_subset = subset(ads_test, cur_speaker)
            llr_per_speaker = np.zeros(len(ads_test_subset.labels))

            for i in range(len(ads_test_subset.labels)):
                speaker_feature = self.test_process(ads_test_subset[i])

                log_likelihood = local_gmm._estimate_weighted_log_prob(speaker_feature)
                likelihood_speaker = logsumexp(log_likelihood, axis=1)

                log_likelihood = self.ubm._estimate_weighted_log_prob(speaker_feature)
                likelihood_ubm = logsumexp(log_likelihood, axis=1)

                llr_per_speaker[i] = np.mean(self.running_mean(likelihood_speaker - likelihood_ubm, 3))

            llr.append(llr_per_speaker)
        llr_cat = np.concatenate(llr, axis=0)



        if thresholds is None:
            thresholds = np.arange(-0.5, 2.5, 0.01)
        thresholds = np.expand_dims(thresholds, axis=1)
        ones = np.ones((1, len(llr_cat)))
        thresholds = thresholds * ones
        frr = np.mean((llr_cat < thresholds), axis=1)
        plt.plot(thresholds, frr * 100)
        plt.title('false rejection rate vs threshold')
        plt.show()
        self.test_results['frr'] = frr

    def test_far(self, ads_test, thresholds=None):

        llr = []

        for s in range(len(self.speakers)):
            cur_speaker = self.speakers[s]
            local_gmm = self.enrolled_gmms[cur_speaker]
            filtered = self.speakers[s] != self.speakers
            filtered_speakers = self.speakers[filtered]
            ads_test_subset = subset(ads_test, filtered_speakers)
            llr_per_speaker = np.zeros(len(ads_test_subset.labels))

            for i in range(len(ads_test_subset.labels)):
                speaker_feature = self.test_process(ads_test_subset[i])

                log_likelihood = local_gmm._estimate_weighted_log_prob(speaker_feature)
                likelihood_speaker = logsumexp(log_likelihood, axis=1)

                log_likelihood = self.ubm._estimate_weighted_log_prob(speaker_feature)
                likelihood_ubm = logsumexp(log_likelihood, axis=1)

                llr_per_speaker[i] = np.mean(self.running_mean(likelihood_speaker - likelihood_ubm, 3))

            llr.append(llr_per_speaker)

        if thresholds is None:
            thresholds = np.arange(-0.5, 2.5, 0.01)
        llr_cat = np.concatenate(llr, axis=0)
        thresholds = np.expand_dims(thresholds, axis=1)
        ones = np.ones((1, len(llr_cat)))
        thresholds = thresholds * ones
        far = np.mean((llr_cat > thresholds), axis=1)
        plt.plot(thresholds, far * 100)
        plt.title('false acceptance rate vs threshold')
        plt.show()
        self.test_results['far'] = far

    def test_det(self, thresholds=None):

        if thresholds is None:
            thresholds = np.arange(-0.5, 2.5, 0.01)

        far = self.test_results['far']
        frr = self.test_results['frr']
        # plot de
        x1 = far * 100
        y1 = frr * 100
        plt.plot(x1, y1)
        plt.title('DET tradeoff')
        plt.xlabel('far')
        plt.ylabel('frr')
        plt.show()

        diff = (np.abs(far - frr))
        idx = np.argmin(diff)
        EERThreshold = thresholds[idx]
        EER = np.mean([far[idx], frr[idx]])

        print(EERThreshold, EER*100)

        plt.plot(thresholds, far*100)
        plt.plot(thresholds, frr*100)
        plt.plot(EERThreshold, EER*100, marker="o", markersize=8, markeredgecolor="red", markerfacecolor="red")
        plt.title('EER')
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
