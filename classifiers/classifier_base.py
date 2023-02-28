from audio_datastore.audio_datastore import AudioDatastore
from feature_extraction.fe_base import FeatureExtractorBase
from processing.process_method_base import ProcessMethodBase
from my_torch.tuts2.torch_transforms import ComposeTransform


class ClassifierBase:

    def __init__(self, train_process: ComposeTransform, test_process: ComposeTransform,
                 fe_method: ComposeTransform, info=None,
                 ):
        self.train_process = train_process
        self.test_process = test_process
        self.fe_method = fe_method
        self.info = info

    def __str__(self):
        return f"Base Classifier"

    def train(self, ads_train: AudioDatastore):
        pass

    def enroll(self, ads_enroll: AudioDatastore):
        pass

    def test(self, ads_test: AudioDatastore):
        pass
