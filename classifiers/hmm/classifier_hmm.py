from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture

from audio_datastore.audio_datastore import AudioDatastore
from classifiers.classifier_base import ClassifierBase
from feature_extraction.fe_base import FeatureExtractorBase
from processing.process_method_base import ProcessMethodBase
from hmmlearn.hmm import GMMHMM, GaussianHMM
import numpy as np
from audio_datastore.audio_datastore import AudioDatastore, subset, filter


class ClassifierHMM(ClassifierBase):

    def __str__(self):
        return f"ClassifierHMM"

    def __init__(self, fe_method: FeatureExtractorBase, process_method: ProcessMethodBase):
        super().__init__(fe_method, process_method)
        self.n_mix = 3
        self.hmms: {GaussianHMM} = {}
        self.num_features = None
        self.n_components = 4
        self.speakers = None
        self.test_results = {}

    def train(self, ads_train: AudioDatastore):

        speakers = np.unique(ads_train.labels)
        self.speakers = speakers
        self.hmms = {}

        for i in range(len(speakers)):
            ads_train_subset = subset(ads_train, speakers[i])
            speaker_features = []
            for file in ads_train_subset.files:
                signal = self.process_method.pre_process(file)
                speaker_features.append(self.fe_method.extract_and_normalize_feature(signal))
            speaker_features_flattened = np.array([item for sublist in speaker_features for item in sublist])
            hmm = GaussianHMM(n_components=self.n_components)
            hmm.fit(speaker_features_flattened)
            self.hmms[speakers[i]] = hmm

    def enroll(self, ads_enroll: AudioDatastore):
        pass

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
                speaker_hmm: GaussianHMM = self.hmms[self.speakers[s]]
                likelihood_hmm = speaker_hmm.score(speaker_feature)
                speakers_scores.append(likelihood_hmm)

            scores.append(self.speakers[np.argmax(speakers_scores)])

        cm = confusion_matrix(np.array(scores), labels, labels=self.speakers, normalize=None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.speakers)
        disp.plot(cmap=plt.cm.Blues, values_format='g')
        plt.show()
