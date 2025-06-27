import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt





def classify_preprocessed_audio(fpath: str) -> int:
    """
    Classify the given preprocessed audio CSV as top (0) or bottom (1).
    Returns None if confidence is too low.
    """
    try:
        # Extract features for the test file
        test_features = extract_features(fpath)

        # Compute Euclidean distances
        dist_to_top = np.linalg.norm(test_features - TOP_FEATURES)
        dist_to_bottom = np.linalg.norm(test_features - BOTTOM_FEATURES)

        # Margin threshold for uncertainty
        if abs(dist_to_top - dist_to_bottom) < 0.05 * max(dist_to_top, dist_to_bottom):
            return None  # too similar

        return 0 if dist_to_top < dist_to_bottom else 1

    except Exception as e:
        print(f"Error classifying {fpath}: {e}")
        return None



def extract_features(fpath: str) -> np.ndarray:
    """
    Extracts a small set of summary features from a preprocessed frequency-time CSV.

    Returns:
        features (np.ndarray): shape (N,), where N = number of features
    """
    # Load CSV
    df = pd.read_csv(fpath)
    freqs = df.iloc[:, 0].values
    times = df.columns[1:].astype(float)
    matrix = df.iloc[:, 1:].values  # drop freq col

    # 1) Overall mean magnitude
    overall_mean = matrix.mean()

    # 2) Mean in low freq band (<500 Hz)
    low_band_mean = matrix[freqs < 500].mean() if any(freqs < 500) else 0

    # 3) Mean in mid freq band (500-2000 Hz)
    mid_band_mean = matrix[(freqs >= 500) & (freqs < 2000)].mean() if any((freqs >= 500) & (freqs < 2000)) else 0

    # 4) Mean in high freq band (>2000 Hz)
    high_band_mean = matrix[freqs >= 2000].mean() if any(freqs >= 2000) else 0

    # 5) Dominant frequency (peak freq averaged over time)
    mean_spectrum = matrix.mean(axis=1)
    peak_freq = freqs[np.argmax(mean_spectrum)]

    # 6) Energy decay slope (log-energy vs time)
    total_energy = matrix.sum(axis=0)  # sum over freqs for each time step
    # To avoid log(0), add tiny offset
    log_energy = np.log(total_energy + 1e-6)
    slope, _, _, _, _ = linregress(times, log_energy)  # decay slope should be negative

    # 7) Spectral centroid (weighted avg freq)
    spec_centroid = np.sum(freqs * mean_spectrum) / np.sum(mean_spectrum)

    # Combine
    features = np.array([
        overall_mean,
        low_band_mean,
        mid_band_mean,
        high_band_mean,
        peak_freq,
        slope,
        spec_centroid
    ])

    return features


def plot_feature_comparison(unlabeled_fpath: str):
    """
    Plot each feature:
    - bar for unlabeled file
    - horizontal lines for top and bottom reference values
    """
    # Extract features
    unlabeled = extract_features(unlabeled_fpath)
    top = TOP_FEATURES
    bottom = BOTTOM_FEATURES

    feature_names = [
        "Overall Mean",
        "Low Band Mean",
        "Mid Band Mean",
        "High Band Mean",
        "Peak Frequency",
        "Decay Slope",
        "Spectral Centroid"
    ]

    # Normalize for comparison
    stacked = np.vstack([top, bottom, unlabeled])
    max_per_feature = np.max(stacked, axis=0)
    max_per_feature[max_per_feature == 0] = 1

    unlabeled_norm = unlabeled / max_per_feature
    top_norm = top / max_per_feature
    bottom_norm = bottom / max_per_feature

    x = np.arange(len(feature_names))

    plt.figure(figsize=(12, 6))

    # Plot unlabeled bars
    plt.bar(x, unlabeled_norm, width=0.4, label='Unlabeled', color='skyblue')

    # Add horizontal lines for top and bottom per feature
    for i in range(len(feature_names)):
        plt.hlines(top_norm[i], i - 0.2, i + 0.2, colors='green', linestyles='--', label='Top' if i == 0 else "")
        plt.hlines(bottom_norm[i], i - 0.2, i + 0.2, colors='red', linestyles=':', label='Bottom' if i == 0 else "")

    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.ylabel('Normalized Feature Value')
    plt.title(f'Unlabeled vs Top/Bottom References: {unlabeled_fpath}')
    plt.legend()
    plt.tight_layout()
    plt.show()


    

TOP_PATH = 'data/preprocessed/top.csv'
BOTTOM_PATH = 'data/preprocessed/bottom.csv'

# Precompute and store:
TOP_FEATURES = extract_features(TOP_PATH)
BOTTOM_FEATURES = extract_features(BOTTOM_PATH)