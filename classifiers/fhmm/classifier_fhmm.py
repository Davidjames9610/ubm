from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM

from audio_datastore.audio_datastore import AudioDatastore, subset
from classifiers.classifier_base import ClassifierBase
from classifiers.hmm.classifier_hmm import ClassifierHMM
from feature_extraction.fe_base import FeatureExtractorBase
from processing.process_method_base import ProcessMethodBase
from helper_functions import *


class ClassifierFHMM(ClassifierHMM):

    def __init__(self, fe_method: FeatureExtractorBase, process_method: ProcessMethodBase):
        super().__init__(fe_method, process_method)
        self.n_components = 4
        self.noise_hmm: GaussianHMM | None = None
        self.hmms: {GaussianHMM} = {}
        self.speakers: None
        self.snr_noise = None

    def __str__(self):
        return f"ClassifierFHMM"

    def train_noise_hmm(self, noise_example, snr_noise):
        hmm = GaussianHMM(n_components=self.n_components)
        hmm.fit(noise_example)
        self.noise_hmm = hmm
        self.snr_noise = snr_noise

    def adapt_speaker_models(self):
        for key in self.hmms:
            self.adapt_speaker_model(self.hmms[key])

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
            (signal_lin['mu']) + (self.snr_noise * noise_lin['mu']),
            (signal_lin['cov']) + ((np.square(self.snr_noise)) * noise_lin['cov'])
        )

        # 04 combined params in cept
        combined_cept = pm_signal.map_linear_to_cepstral(combined_lin)
        combined_cept.mu[:, 0] = signal_cept.mu[:, 0]  # for some reason first gets confused
        # combined_cept['mu'][:, 1] = signal_cept['mu'][:, 1]

        # 05 combined hmm
        hmm_combined = hmm.GaussianHMM(n_states, covariance_type='diag')
        hmm_combined.n_features = speaker_hmm.covars_.shape[1]

        # hmm_combined.covars_ = np.array([np.diag(i) for i in combined_cept['cov']])
        hmm_combined.covars_ = np.array([np.diag(i) for i in speaker_hmm.covars_])
        hmm_combined.means_ = combined_cept['mu']
        hmm_combined.startprob_ = speaker_hmm.startprob_
        hmm_combined.transmat_ = speaker_hmm.transmat_

        return hmm_combined, combined_cept

    def train(self, ads_train: AudioDatastore):
        super().train(ads_train)

    def enroll(self, ads_enroll: AudioDatastore):
        pass

    def test(self, ads_test: AudioDatastore):
        super().test(ads_test)

    # not sure if should normalize if mapping ?
    def set_normalisation(self, ads: AudioDatastore):
        super().set_normalisation(ads)
