from audio_datastore.audio_datastore import AudioDatastoreProcessed

# these classifiers will only work AudioDatastoreProcessed
class Classifier:

    def __init__(self):
        pass

    def __str__(self):
        return f"Base Classifier"

    def train(self, ads_train: AudioDatastoreProcessed):
        pass

    def enroll(self, ads_enroll: AudioDatastoreProcessed):
        pass

    def test(self, ads_test: AudioDatastoreProcessed):
        pass
