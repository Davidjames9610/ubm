from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from audio_datastore.audio_datastore import AudioDatastore, subset
from classifiers.classifier_base import ClassifierBase
from classifiers.hmm.classifier_hmm import ClassifierHMM
from my_torch.tuts2.torch_transforms import ComposeTransform
import numpy as np
from classifiers.hmm_nn.dnnhmm.development.dnnhmm import DNNHMM


class ClassifierHMMDNN(ClassifierBase):

    def __str__(self):
        return f"ClassifierHMMDNN"

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform, fe_method: ComposeTransform, info=None,
                 n_mix=2, n_components=4
                 ):
        super().__init__(train_process, test_process, fe_method, info)
        self.speakers = None
        self.n_mix = n_mix
        self.n_components = n_components
        self.models = {DNNHMM}

    def train(self, ads_train: AudioDatastore):
        speakers = np.unique(ads_train.labels)
        self.speakers = speakers
        self.models = {}

        for s in range(len(speakers)):
            ads_train_subset = subset(ads_train, speakers[s])
            features = []
            for i in range(len(ads_train_subset.labels)):
                feature = self.fe_method(self.train_process(ads_train_subset[i]))
                features.append(feature)
            # features_flattened = np.array([item for sublist in features for item in sublist])
            model = DNNHMM(n_mix=self.n_mix, n_components=self.n_components)
            model.fit(features)
            self.models[speakers[s]] = model

    def test_all(self, ads_test: AudioDatastore, thresholds=None):
        self.test_confusion_matrix(ads_test)

    def test_confusion_matrix(self, ads_test: AudioDatastore):
        # confusion matrix
        scores = []
        labels = ads_test.labels
        for i in range(len(ads_test.labels)):
            feature = self.fe_method(self.test_process(ads_test[i]))
            speakers_scores = []
            for s in range(len(self.speakers)):
                model: DNNHMM = self.models[self.speakers[s]]
                likelihood_model = model.score(feature)
                speakers_scores.append(likelihood_model)

            scores.append(self.speakers[np.argmax(speakers_scores)])

        cm = confusion_matrix(np.array(scores), labels, labels=self.speakers, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.speakers)
        disp.plot(cmap=plt.cm.Blues, values_format='g')
        plt.show()




