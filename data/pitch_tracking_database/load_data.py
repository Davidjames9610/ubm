
import pickle

normalised_30 = r'/Users/david/Documents/mastersCode/ubm/data/pitch_tracking_database/pitch_tracking_db_normalised_audio_30.pickle'
def pitch_tracking_db_normalised_audio_30():
    return pickle.load(open(r'/library/pitch_tracking_db_normalised_audio_30.pickle', 'rb'))