import pandas as pd

# Hard Coded for now. Refactor later.
THRESHOLD = 2018.73779296875
MARGIN = 381.1376953125

def classify_preprocessed_audio(fpath: str) -> int:
    """
    Returns:
      0: Top strike
      1: Bottom strike
      None: Uncertain
    """
    try:
        df = pd.read_csv(fpath, index_col=0).astype(float)
    except Exception as e:
        print(f"Error loading '{fpath}': {e}")
        return None

    # Energy at different Hz 
    mean_freq = df.mean(axis=1)
    peak_hz   = mean_freq.idxmax()
    print(peak_hz)

    if abs(peak_hz - THRESHOLD) < MARGIN:
        return None
    
    return 0 if peak_hz > THRESHOLD else 1
