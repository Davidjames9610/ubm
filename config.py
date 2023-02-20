import numpy as np

eps = np.finfo(np.float64).eps
SAMPLING_RATE = 16000
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 1024

# databases and mfccs

ads_pitch_tracking_db_mfcc = r'/Users/david/Documents/mastersCode/ubm/data/pitch_tracking_database' \
                             r'/ads_pitch_tracking_db_mfcc.pickle'
noise_db = r'/Users/david/Documents/data/MUSAN/musan/noise/free-sound'
