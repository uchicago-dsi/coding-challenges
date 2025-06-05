import joblib

from candidate_code.water_bottle_challenge.config import MODELS_DIR
from candidate_code.water_bottle_challenge.preprocessing import read_preprocessed_audio_file, get_single_dimension

model = joblib.load(MODELS_DIR / 'knn_classifier.pkl')

def classify_preprocessed_audio(fpath : str) -> int:
    """
    Reads in a preprocessed csv of an audio recording of a knife striking a
    water bottle and returns a 0 if the knife struck the top and a 1 if the
    knife struck the bottom
    """
    audio_df = read_preprocessed_audio_file(fpath)
    audio_reduced = get_single_dimension(audio_df)

    return int(model.predict([audio_reduced])[0])
