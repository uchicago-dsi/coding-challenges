"""Script to generate preprocessed data from raw mp3 files.

Note: This script is a record of how the preprocessed data, but it is not designed to be run again, as
it will generate different random filenames. After the files were generated, some were renamed to form
the labeled training data and the witheld validation data.
"""

import json
from pathlib import Path

import librosa
import librosa.display
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "preprocessed2"

RAW_MAP_FPATH = DATA_DIR / "raw_map.json"
PROCESSED_MAP_FPATH = DATA_DIR / "preprocessed2_map.json"

def preprocess_mp3(mp3_fpath: Path, output_fpath: Path):
    y, sr = librosa.load(mp3_fpath)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Compute pitches and magnitudes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get the time values in milliseconds
    times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sr) * 1000  # Convert to ms

    # Get the frequency bins in Hz
    freqs = librosa.fft_frequencies(sr=sr)

    # Create an empty DataFrame
    data = pd.DataFrame(data=magnitudes, index=freqs, columns=times)
    data.index.name = 'frequency'
    data.columns.name = 'time'
    
    # Save
    data.to_csv(output_fpath)


if __name__ == "__main__":
    raw_map = json.loads(RAW_MAP_FPATH.read_text())
    
    # Generate random names for the preprocessed files
    random_ints = np.random.randint(0, 100, len(raw_map))
    processed_keys = [f"unlabeled_{str(i).rjust(2, '0')}" for i in random_ints]
    
    # Preprocess the data and save to random names
    processed_map = {}
    for raw_key, processed_key in zip(raw_map, processed_keys):
        processed_map[processed_key] = raw_map[raw_key]
        preprocess_mp3(
            RAW_DATA_DIR / f"{raw_key}.mp3",
            PROCESSED_DATA_DIR / f"{processed_key}.csv",
        )
    
    # Save mapping to true classes
    json.dump(processed_map, PROCESSED_MAP_FPATH.open("w"), indent=2)