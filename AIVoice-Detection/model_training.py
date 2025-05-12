import os
import logging
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import config
from feature_extraction import create_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def train_model(X, y):
    """
    Train an SVM model on the provided dataset.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Label vector
    """
    unique_classes = np.unique(y)
    logger.info(f"Unique classes: {unique_classes}")

    if len(unique_classes) < 2:
        raise ValueError("At least 2 classes are required to train")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TRAIN_CONFIG['test_size'], 
        random_state=config.TRAIN_CONFIG['random_state'], 
        stratify=y
    )

    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm_classifier = SVC(kernel='linear', random_state=config.TRAIN_CONFIG['random_state'])
    svm_classifier.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = svm_classifier.predict(X_test_scaled)
    
    logger.info("Model Evaluation:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm_classifier, os.path.join('models', config.MODEL_FILENAME))
    joblib.dump(scaler, os.path.join('models', config.SCALER_FILENAME))
    logger.info("Model and scaler saved successfully")

def main():
    """
    Main function to create dataset and train the model
    """
    X_real, y_real = create_dataset(config.REAL_AUDIO_DIR, label=0)
    X_fake, y_fake = create_dataset(config.FAKE_AUDIO_DIR, label=1)

    if len(X_real) < 2 or len(X_fake) < 2:
        logger.error("Each class should have at least two samples")
        return

    X = np.vstack((X_real, X_fake))
    y = np.hstack((y_real, y_fake))

    train_model(X, y)

if __name__ == "__main__":
    main()