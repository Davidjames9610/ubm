from sklearn.mixture import GaussianMixture

from classifiers.classifier_base import ClassifierBase
from feature_extraction.fe_base import FeatureExtractorBase
from processing.process_method_base import ProcessMethodBase


class ClassifierHMM(ClassifierBase):

    def __str__(self):
        return f"ClassifierGMMUBM"

    def __init__(self, fe_method: FeatureExtractorBase, process_method: ProcessMethodBase):
        super().__init__(fe_method)
        self.ubm: GaussianMixture | None = None
        self.enrolled_gmms: {GaussianMixture} = {}
        self.num_features = None
        self.num_components = 32
        self.relevance_factor = 16
        self.speakers = None
        self.test_results = {}
        self.process_method = process_method