import os

# Directories for real and fake audio samples
REAL_AUDIO_DIR = os.path.join('data', 'real')
FAKE_AUDIO_DIR = os.path.join('data', 'fake')

# Model and scaler save paths
MODEL_FILENAME = 'svm_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

# Feature extraction parameters
MFCC_PARAMS = {
    'n_mfcc': 13,
    'n_fft': 2048,
    'hop_length': 512
}

# Training configuration
TRAIN_CONFIG = {
    'test_size': 0.3,
    'random_state': 42
}