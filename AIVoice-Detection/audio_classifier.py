import os
import logging
import joblib
import config
from feature_extraction import extract_mfcc_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_audio(input_audio_path):
    """
    Analyze an audio file for deepfake detection.
    
    Args:
        input_audio_path (str): Path to the input audio file
    
    Returns:
        str: Classification result
    """
    # Validate input
    if not os.path.exists(input_audio_path):
        logger.error("Error: The specified file does not exist.")
        return "File not found"
    
    if not input_audio_path.lower().endswith(".wav"):
        logger.error("Error: Only .wav files are supported.")
        return "Unsupported file type"

    # Load models
    try:
        model_path = os.path.join('models', config.MODEL_FILENAME)
        scaler_path = os.path.join('models', config.SCALER_FILENAME)
        
        svm_classifier = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        return "Model not trained"

    # Extract features
    mfcc_features = extract_mfcc_features(
        input_audio_path, 
        **config.MFCC_PARAMS
    )

    if mfcc_features is None:
        logger.error("Unable to process input audio")
        return "Feature extraction failed"

    # Scale and predict
    mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
    prediction = svm_classifier.predict(mfcc_features_scaled)

    result = "genuine" if prediction[0] == 0 else "deepfake"
    logger.info(f"Audio classified as: {result}")
    
    return result

def main():
    # user_input_file = "data/test/Abhishek_Fake.wav"
    user_input_file = "data/test/Abhishek_real_voice.wav"
    result = analyze_audio(user_input_file)
    print(f"Classification result: {result}")

if __name__ == "__main__":
    main()