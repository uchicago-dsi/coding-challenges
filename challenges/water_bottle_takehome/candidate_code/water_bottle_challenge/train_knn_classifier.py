import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import MODELS_DIR, RAW_DATA_DIR
from preprocessing import read_preprocessed_audio_file, get_single_dimension

def train_knn_classifier() -> KNeighborsClassifier:
    """Trains a KNN Classifier for audio file based on two examples"""
    top = read_preprocessed_audio_file(RAW_DATA_DIR / 'train/top.csv')
    bottom = read_preprocessed_audio_file(RAW_DATA_DIR / 'train/bottom.csv')

    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(
        np.array([get_single_dimension(top),get_single_dimension(bottom)]),
        np.array([0,1])
    )

    return knn

if __name__ == '__main__':
    model = train_knn_classifier()
    joblib.dump(model, MODELS_DIR / 'knn_classifier.pkl')
