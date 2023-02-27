from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture

from audio_datastore.audio_datastore import AudioDatastore
from classifiers.classifier_base import ClassifierBase
from feature_extraction.fe_base import FeatureExtractorBase
from my_torch.tuts2.torch_transforms import ComposeTransform
from processing.process_method_base import ProcessMethodBase
from hmmlearn.hmm import GMMHMM, GaussianHMM
import numpy as np
from audio_datastore.audio_datastore import AudioDatastore, subset


class ClassifierHMM(ClassifierBase):

    def __str__(self):
        return f"ClassifierHMM"

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform, info=None,
                 n_mix=2, n_components=4
                 ):
        super().__init__(train_process, test_process, info)
        self.hmms: {GaussianHMM} = {}
        self.n_mix = n_mix
        self.n_components = n_components
        self.speakers = None
        self.test_results = {}

    def train(self, ads_train: AudioDatastore):
        speakers = np.unique(ads_train.labels)
        self.speakers = speakers
        self.hmms = {}

        for s in range(len(speakers)):
            ads_train_subset = subset(ads_train, speakers[s])
            features = []
            for i in range(len(ads_train_subset.labels)):
                feature = self.train_process(ads_train_subset[i])
                features.append(feature)
            features_flattened = np.array([item for sublist in features for item in sublist])
            hmm = GaussianHMM(n_components=self.n_components)
            hmm.fit(features_flattened)
            self.hmms[speakers[s]] = hmm

    def enroll(self, ads_enroll: AudioDatastore):
        pass

    # todo add in other tests as well ! also need to continue the development of the
    # transforms and use them

    def test_all(self, ads_test: AudioDatastore, thresholds=None):
        self.test_confusion_matrix(ads_test)

    def test_confusion_matrix(self, ads_test: AudioDatastore):
        # confusion matrix
        scores = []
        labels = ads_test.labels
        for i in range(len(ads_test.labels)):
            feature = self.test_process(ads_test[i])
            speakers_scores = []
            for s in range(len(self.speakers)):
                speaker_hmm: GaussianHMM = self.hmms[self.speakers[s]]
                likelihood_hmm = speaker_hmm.score(feature)
                speakers_scores.append(likelihood_hmm)

            scores.append(self.speakers[np.argmax(speakers_scores)])

        cm = confusion_matrix(np.array(scores), labels, labels=self.speakers, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.speakers)
        disp.plot(cmap=plt.cm.Blues, values_format='g')
        plt.show()

