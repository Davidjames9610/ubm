from audio_datastore.audio_datastore import AudioDatastore
from feature_extraction.fe_base import FeatureExtractorBase


class ClassifierBase:

    def __init__(self, fe_method: FeatureExtractorBase):
        self.fe_method: FeatureExtractorBase = fe_method

    def __str__(self):
        return f"Base Classifier"

    def train(self, ads_train: AudioDatastore):
        pass

    def enroll(self, ads_enroll: AudioDatastore):
        pass

    def test(self, ads_test: AudioDatastore):
        pass

    def set_normalisation(self, ads: AudioDatastore):
        self.fe_method.set_normalisation(ads)
