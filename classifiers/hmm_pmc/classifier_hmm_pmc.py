import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM

import utils
from audio_datastore.audio_datastore import AudioDatastore, subset
from classifiers.hmm.classifier_hmm import ClassifierHMM
from feature_extraction.fe_base import FeatureExtractorBase
from my_torch.tuts2.torch_transforms import ComposeTransform
from processing.process_method_base import ProcessMethodBase
from classifiers.fhmm.helper_functions import *

class ClassifierHMMPMC(ClassifierHMM):

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform, fe_method: ComposeTransform,
                 snr, signal_average_power, info=None, n_mix=2, n_components=4
                 ):
        super().__init__(train_process, test_process, fe_method, n_mix, n_components, info)
        self.n_components = 4
        self.noise_hmm: GaussianHMM | None = None
        self.hmms: {GaussianHMM} = {}
        self.snr = snr
        self.signal_average_power = signal_average_power
        self.snr_noise = 1

    def __str__(self):
        return f"ClassifierFHMM"

    def train_noise_hmm(self):
        # calculate noise power from snr
        # 10log10(s/n) = snr_db
        # # ln x = y, e^y = x
        # # log10 (s/n) = snr/10
        # signal_average_power / np.power((snr_db / 10), 10)
        noise_average_power = self.signal_average_power / np.power(10, self.snr / 10)
        noise_signal = np.random.normal(0, np.sqrt(noise_average_power), 10000)
        noise_hmm = GaussianHMM(n_components=1)
        noise_feature = self.fe_method(noise_signal)
        noise_hmm.fit(noise_feature)
        self.noise_hmm = noise_hmm

    def adapt_speaker_models(self):
        for key in self.hmms:
            self.hmms[key] = self.adapt_speaker_model(self.hmms[key])

    def adapt_speaker_model_log_max(self, speaker_hmm: hmm.GaussianHMM):
        # 01 deconstruct hmm
        n_states = speaker_hmm.transmat_.shape[0]

        signal_cept = StatParams(speaker_hmm.means_, speaker_hmm.covars_)
        noise_cept = StatParams(self.noise_hmm.means_, self.noise_hmm.covars_)

        pm_signal = ParamMapper(signal_cept.mu)
        pm_noisy = ParamMapper(noise_cept.mu)

        signal_log = pm_signal.map_cepstral_to_log(signal_cept)
        noise_log = pm_noisy.map_cepstral_to_log(noise_cept)

        # 03 combine using snr
        combined_log = StatParams(
            np.maximum(signal_log.mu, noise_log.mu),
            np.maximum(signal_log.cov, noise_log.cov)
        )

        combined_cept = pm_signal.map_log_to_cepstral(combined_log)

        # 05 combined hmm
        hmm_combined = hmm.GaussianHMM(n_states, covariance_type='diag')
        hmm_combined.n_features = speaker_hmm.covars_.shape[1]

        # hmm_combined.covars_ = np.array([np.diag(i) for i in combined_cept['cov']])
        hmm_combined.covars_ = np.array([np.diag(i) for i in combined_cept.cov])
        hmm_combined.means_ = combined_cept.mu
        hmm_combined.startprob_ = speaker_hmm.startprob_
        hmm_combined.transmat_ = speaker_hmm.transmat_

        return hmm_combined

    def adapt_speaker_model(self, speaker_hmm: hmm.GaussianHMM):
        # 01 deconstruct hmm
        n_states = speaker_hmm.transmat_.shape[0]

        signal_cept = StatParams(speaker_hmm.means_, speaker_hmm.covars_)
        noise_cept = StatParams(self.noise_hmm.means_, self.noise_hmm.covars_)

        pm_signal = ParamMapper(signal_cept.mu)
        pm_noisy = ParamMapper(noise_cept.mu)

        signal_lin = pm_signal.map_cepstral_to_linear(signal_cept)
        noise_lin = pm_noisy.map_cepstral_to_linear(noise_cept)

        # 03 combine using snr
        combined_lin = StatParams(
            signal_lin.mu + self.snr_noise * noise_lin.mu,
            signal_lin.cov + ((np.square(self.snr_noise)) * noise_lin.cov)
        )

        # 04 combined params in cept
        combined_cept = pm_signal.map_linear_to_cepstral(combined_lin)
        # combined_cept.mu[:, 0] = noise_cept.mu[:, 0]  # for some reason first gets confused
        # combined_cept['mu'][:, 1] = signal_cept['mu'][:, 1]

        # 05 combined hmm
        hmm_combined = hmm.GaussianHMM(n_states, covariance_type='diag')
        hmm_combined.n_features = speaker_hmm.covars_.shape[1]

        # hmm_combined.covars_ = np.array([np.diag(i) for i in combined_cept['cov']])
        hmm_combined.covars_ = np.array([np.diag(i) for i in combined_cept.cov])
        hmm_combined.means_ = combined_cept.mu
        hmm_combined.startprob_ = speaker_hmm.startprob_
        hmm_combined.transmat_ = speaker_hmm.transmat_

        return hmm_combined

    def train(self, ads_train: AudioDatastore):
        super().train(ads_train)

    def enroll(self, ads_enroll: AudioDatastore):
        pass

    def test(self, ads_test: AudioDatastore):
        super().test(ads_test)
