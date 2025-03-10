import os
import numpy as np
import pandas as pd
import librosa
from typing import TypeVar
from sklearn.metrics.pairwise import cosine_similarity
# Function to read data
def read_spectrogram(file_name: str) -> pd.DataFrame:
    """
    Read spectrograms from CSV file where:
    - Row indices are frequency bands in Hz
    - Column names are time slices in milliseconds
    """
    # Load spectrogram
    spectrogram = pd.read_csv(file_name, index_col=0)
    
    # Convert column names to float (time values)
    spectrogram.columns = spectrogram.columns.astype(float)
    
    # Return spectrogram
    return spectrogram

def extract_spectral_features(spec_data: pd.DataFrame, bands: list) -> dict:
    """
    Extract numerical features from the spectrogram that could be useful for classification
    
    Args:
        spec_data: DataFrame containing the spectrogram data
        
    Returns:
        features: Dictionary of extracted features
    """
    # Get frequency and time values
    freq_values = spec_data.index.astype(float).values
    
    # Calculate frequency distribution
    freq_power = spec_data.sum(axis=1)
    time_power = spec_data.sum(axis=0)
    
    # Find peak frequency (frequency with maximum power)
    peak_freq_idx = freq_power.argmax()
    peak_frequency = freq_values[peak_freq_idx]
    
    # Calculate spectral centroid (weighted average of frequencies)
    spectral_centroid = np.sum(freq_values * freq_power) / np.sum(freq_power)
    
    # Calculate spectral bandwidth (weighted standard deviation of frequencies)
    spectral_bandwidth = np.sqrt(np.sum(((freq_values - spectral_centroid) ** 2) * freq_power) / np.sum(freq_power))
    
    # Calculate power in the bands
    band_power = []
    for band in bands:
        low_freq, high_freq = band
        band_mask = (freq_values >= low_freq) & (freq_values < high_freq)
        band_power.append(np.sum(freq_power[band_mask]))    

    total_band_power = np.sum(band_power)
    for i in range(len(band_power)):
        band_power[i] /= total_band_power  
 
    # Calculate temporal features
    max_power = np.max(time_power)
    max_power_idx = time_power.argmax()
    last_silence_idx = time_power[:max_power_idx][time_power < (0.01 * max_power)].index[-1]
    release_10pct_idx = time_power[time_power > (0.1 * max_power)].index[-1]
        
    attack_time = max_power_idx - last_silence_idx
    release_time = release_10pct_idx - max_power_idx 

    peak_time_idx = time_power.argmax()
    peak_time = time_power.index[peak_time_idx]

    ## Attack/release for each band    
    attack_times = []
    release_times = []
    for band in bands:
        low_freq, high_freq = band
        band_mask = (freq_values >= low_freq) & (freq_values < high_freq)
        band_time_power = spec_data[band_mask].sum(axis=0)

        max_power = np.max(band_time_power)
        max_power_idx = band_time_power.argmax()
        last_silence_idx = band_time_power[:max_power_idx][band_time_power < (0.01 * max_power)].index[-1]
        release_10pct_idx = band_time_power[band_time_power > (0.1 * max_power)].index[-1]
         
        attack_times.append(max_power_idx - last_silence_idx)
        release_times.append(release_10pct_idx - max_power_idx)
     
    
    # Return features as a dictionary
    features = {
        'peak_frequency': peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,  
        'peak_time': peak_time,
        'total_power': np.sum(spec_data.values),
        'max_power': np.max(spec_data.values), 
        'attack_time': attack_time,
        'release_time': release_time,
        'band_power': total_band_power,
        ## Add the band powers
        **{f'band_{i}_power': band_power[i] for i in range(len(band_power))},
        ## Add the band attack/release times
        **{f'band_{i}_attack_time': attack_times[i] for i in range(len(attack_times))}, 
        **{f'band_{i}_release_time': release_times[i] for i in range(len(release_times))}, 
    }
    
    return features 

def classify_preprocessed_audio(fpath: str) -> int:
    """
    Classify preprocessed audio as top (0) or bottom (1)
    
    Args:
        fpath: Path to CSV file containing preprocessed audio spectrogram
        
    Returns:
        0 if top, 1 if bottom, None if unable to classify
    """  
    # Read spectrogram
    spec_data = read_spectrogram(fpath) 
    
    # Extract features
    features = extract_spectral_features(spec_data, bands=[(750, 1400), (1600, 2000), (2600, 3200), (3500, 4000)])
    # Define important features based on analysis
    important_features = [
        'band_0_power', 'band_2_power',
        'band_0_attack_time', 'band_1_attack_time', 'band_2_attack_time',
        'band_0_release_time', 'band_1_release_time', 'band_2_release_time',
        'peak_frequency'
    ] 
   
    # Normalize using typical values (approximating StandardScaler)
    feature_means = np.array([
        4.76993993e-01,  # band_0_power
        3.65619867e-01,  # band_2_power
        1.18609221e+01,    # band_0_attack_time
        1.31475813e+01,    # band_1_attack_time
        1.28884165e+01,    # band_2_attack_time
        1.69287075e+03,   # band_0_release_time
        1.55832851e+03,   # band_1_release_time
        1.48647449e+03,   # band_2_release_time
        1.85768738e+03  # peak_frequency 
    ])
     
    feature_stds = np.array([
        1.87617691e-01,  # band_0_power
        1.75739236e-01,  # band_2_power
        7.18313494e+00,     # band_0_attack_time
        6.95145742e+00,     # band_1_attack_time
        6.84232406e+00,     # band_2_attack_time
        2.78582688e+02,     # band_0_release_time
        2.60876576e+02,     # band_1_release_time
        2.69027287e+02,     # band_2_release_time
        9.36869530e+02   # peak_frequency 
        ])
    # Prototype feature values for "top" class (0)
    top_prototype = {
        'band_0_power': -1.32916124,
        'band_2_power': 1.56748033 ,
        'band_0_attack_time': 1.21105459,
        'band_1_attack_time':  1.10360659,
        'band_2_attack_time': 0.79103378,
        'band_0_release_time': 1.43434984,
        'band_1_release_time': 0.97934363 ,
        'band_2_release_time': 1.05157336,
        'peak_frequency': 1.18895387 
    }
       
    # Prototype feature values for "bottom" class (1)
    bottom_prototype = {
        'band_0_power': 1.4675927,
        'band_2_power': -1.3384323,
        'band_0_attack_time': -0.45952519,
        'band_1_attack_time':-1.05421429,
        'band_2_attack_time': -1.10890547,
        'band_0_release_time': -0.43963221 ,
        'band_1_release_time': -0.56529173,
        'band_2_release_time': -0.45369842,
        'peak_frequency': -0.84514844 
    }
     
    # Feature weights (signal-to-noise ratio)
    feature_weights = {
        'band_0_power': 2.79675393,
        'band_2_power': 2.90591263,
        'band_0_attack_time': 1.67057978,
        'band_1_attack_time': 2.15782088,
        'band_2_attack_time': 1.89993924,
        'band_0_release_time': 1.87398205,
        'band_1_release_time': 1.54463536,
        'band_2_release_time': 1.50527178,
        'peak_frequency': 2.03410231, 
    }
    
    # Create feature vectors
    feature_vector = np.array([features.get(f) for f in important_features])
    
    
    # Apply normalization
    feature_vector_norm = (feature_vector - feature_means) / feature_stds
    
    # Get the prototype vectors for top and bottom classes
    top_vector_norm = np.array([top_prototype.get(f) for f in important_features])
    bottom_vector_norm = np.array([bottom_prototype.get(f) for f in important_features])
    weights = np.array([feature_weights.get(f) for f in important_features])
    
    # Apply weights
    feature_vector_weighted = feature_vector_norm * weights
    top_vector_weighted = top_vector_norm * weights
    bottom_vector_weighted = bottom_vector_norm * weights
    
    # Calculate cosine similarity
    similarity_to_top = cosine_similarity(
        feature_vector_weighted.reshape(1, -1), 
        top_vector_weighted.reshape(1, -1)
    )[0][0]
    
    similarity_to_bottom = cosine_similarity(
        feature_vector_weighted.reshape(1, -1), 
        bottom_vector_weighted.reshape(1, -1)
    )[0][0]
    
    print(f"Similarity to Top: {similarity_to_top:.4f}")
    print(f"Similarity to Bottom: {similarity_to_bottom:.4f}")
    print(f"Predicted Class: {'Top' if similarity_to_top > similarity_to_bottom else 'Bottom'}")

    # Return classification (0 for top, 1 for bottom)
    return 0 if similarity_to_top > similarity_to_bottom else 1
