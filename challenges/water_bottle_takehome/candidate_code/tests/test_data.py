import os

import joblib
import pytest

from water_bottle_challenge.config import MODELS_DIR, RAW_DATA_DIR
from water_bottle_challenge.preprocessing import read_preprocessed_audio_file, get_single_dimension
from water_bottle_challenge.water_bottle_challenge import classify_preprocessed_audio

model = joblib.load(MODELS_DIR / 'knn_classifier.pkl')

def test_classifier():
    top = read_preprocessed_audio_file(RAW_DATA_DIR / 'train/top.csv')
    bottom = read_preprocessed_audio_file(RAW_DATA_DIR / 'train/bottom.csv')


    assert model.predict([get_single_dimension(top)])[0] == 0
    assert model.predict([get_single_dimension(bottom)])[0] == 1

def test_classifier_function():
    res = classify_preprocessed_audio(RAW_DATA_DIR / 'train/unlabeled_00.csv')

    assert res == 0 or res == 1


