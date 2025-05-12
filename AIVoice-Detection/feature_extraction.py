import os
import logging
import numpy as np
import librosa
import glob
from typing import Optional, Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_mfcc_features(
    audio_path: str, 
    n_mfcc: int = 13, 
    n_fft: int = 2048, 
    hop_length: int = 512
) -> Optional[np.ndarray]:
    """
    Extract MFCC features from an audio file.
    
    Args:
        audio_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        n_fft (int): FFT window size
        hop_length (int): Number of samples between successive frames
    
    Returns:
        Optional[np.ndarray]: Mean of MFCC features or None if extraction fails
    """
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sr, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        logger.error(f"Error extracting features from {audio_path}: {e}")
        return None

def create_dataset(
    directory: str, 
    label: int
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create a dataset from audio files in a directory.
    
    Args:
        directory (str): Directory containing audio files
        label (int): Label for the dataset (0 for real, 1 for fake)
    
    Returns:
        Tuple of feature lists and corresponding labels
    """
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    
    logger.info(f"Processing {len(audio_files)} files in {directory}")
    
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            logger.warning(f"Skipping audio file {audio_path}")
    
    logger.info(f"Number of samples in {directory}: {len(X)}")
    return X, y