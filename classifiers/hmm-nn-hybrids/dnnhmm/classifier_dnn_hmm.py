from audio_datastore.audio_datastore import AudioDatastore, subset
from classifiers.classifier_base import ClassifierBase
from classifiers.hmm.classifier_hmm import ClassifierHMM
from my_torch.tuts2.torch_transforms import ComposeTransform
import numpy as np


class ClassifierHMMDNN(ClassifierBase):

    def __str__(self):
        return f"ClassifierHMMDNN"

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform, fe_method: ComposeTransform, info=None,
                 n_mix=2, n_components=4
                 ):
        super().__init__(train_process, test_process, fe_method, info)
        self.n_mix = n_mix
        self.n_components = n_components
        self.hmm_dnns = {}
        self.sequences = None  # forced alignment
        self.features = None # shouldn't need this ?

    def train(self, ads_train: AudioDatastore):
        speakers = np.unique(ads_train.labels)
        self.speakers = speakers
        self.hmms = {}

        for s in range(len(speakers)):
            ads_train_subset = subset(ads_train, speakers[s])
            features = []
            for i in range(len(ads_train_subset.labels)):
                feature = self.fe_method(self.train_process(ads_train_subset[i]))
                features.append(feature)
            features_flattened = np.array([item for sublist in features for item in sublist])
            hmm = GaussianHMM(n_components=self.n_components)
            hmm.fit(features_flattened)
            self.hmms[speakers[s]] = hmm



