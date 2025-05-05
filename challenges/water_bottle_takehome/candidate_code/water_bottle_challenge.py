from pathlib import Path

import numpy as np
import pandas as pd

BOTTOM_FPATH = Path(__file__).parent / "train/bottom.csv"
TOP_FPATH = Path(__file__).parent / "train/top.csv"
SIGMA = 20

def import_data(fpath):
    ret = pd.read_csv(fpath, header = 0, index_col = 0)
    ret.columns = ret.columns.astype(float)
    return ret

BOTTOM_DF = import_data(BOTTOM_FPATH)
TOP_DF = import_data(TOP_FPATH)

def norm_max(df):
    overall_max = df.max().max()
    return df.max(axis=1)/overall_max

def gaussian(x,sigma):
    return np.exp(-0.5*(x/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def gaussian_filter(signal,sigma):
    signal = signal.sort_index()
    ret = pd.Series(index = signal.index)
    for x in signal.index:
        conv = gaussian(np.array(x - signal.index), sigma)
        ret[x] = (signal.values*conv).sum()
    return ret

def dist(a,b,sigma):
    n_a = norm_max(a)
    n_a = n_a/np.sqrt((n_a**2).sum())
    n_b = norm_max(b)
    n_b = n_b/np.sqrt((n_b**2).sum())
    filtered_diff = gaussian_filter(n_a-n_b, sigma)
    return (filtered_diff ** 2).sum()

def classify_preprocessed_audio(fpath: str) -> int:
    df = import_data(fpath)
    dist_top = dist(df, TOP_DF, SIGMA)
    dist_bottom = dist(df, BOTTOM_DF, SIGMA)
    return 0 if dist_top <= dist_bottom else 1

if __name__ == "__main__":
    fpath = BOTTOM_FPATH
    result = classify_preprocessed_audio(fpath)
    print(result)