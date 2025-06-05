from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

RANDOM_STATE = 5262025
svd = TruncatedSVD(n_components=1, random_state=RANDOM_STATE)


def read_preprocessed_audio_file(filepath : Union[str, Path]) -> pd.DataFrame:
    """give a filepath reads in a csv version of a preprocessed audio file"""
    return pd.read_csv(filepath, index_col=0)

def get_single_dimension(audio_df : pd.DataFrame) -> np.ndarray:
    """reduces a dataframe to a single dimension array using Truncated SVD"""
    return svd.fit_transform(audio_df.to_numpy()).flatten()
