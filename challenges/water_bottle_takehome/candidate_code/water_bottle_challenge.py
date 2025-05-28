# Nora Payne (paynenor@gmail.com)

import pandas as pd
import numpy as np

def classify_preprocessed_audio(fpath):
    '''
    This function uses PCA + LDA to classify preprocessed audio file of a knife hitting a waterbottle.
    
    Input: 
        fpath: filepath of preprocessed audio file (csv).
    
    Output:
        Integer-valued prediction of whether bottle was hit at top (0) or bottom (1).
    '''
    
    # Transform data matrix to vector of summed magnitudes at each frequency
    data_matrix = pd.read_csv(fpath)
    data_long = pd.melt(data_matrix, id_vars='frequency', var_name="time", value_name="magnitude")
    data_long['time'] = data_long.time.astype(float)
    v = data_long.groupby('frequency').apply(lambda x: sum(x['magnitude'])).to_numpy()
    
    # (Approximately) project vector onto PC space
    pc1_ix = np.array([14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
                        26,  28,  30,  42,  53,  54,  55,  62,  70,  71,  80,  97,
                        98,  99, 100, 160, 170, 175, 178, 179, 181, 182, 249, 275, 
                        276, 281, 282, 283, 284, 350, 351])
    pc2_ix = np.array([14,  15,  16,  18,  21,  22,  23,  24,  25,  26,  29,  30,  31,
                       36,  42,  54,  70,  80,  98,  99, 100, 160, 181, 184, 249, 250, 
                       275, 276, 279, 280, 281, 282, 283, 344, 350, 351])
    
    pc1_coeff = np.array([-0.006, -0.008, -0.013, -0.012, -0.017, -0.020,  0.016, -0.008, -0.022, 
                          -0.042, 0.031, 0.010, 0.013, 0.005, -0.014, 0.008, 0.033,  0.011, 0.006, -0.006,
                         -0.006,  -0.425, -0.154, -0.005, -0.005, -0.867, -0.006, -0.087, -0.009, -0.014,
                         -0.010, -0.005, -0.073,  0.018, -0.014,  0.099,  0.017,  0.089,  0.024,  0.009,
                         0.008, -0.005,  0.063])
    pc2_coeff = np.array([0.023, -0.007,  0.007,  0.007,  0.017,  0.032,  0.038,  0.016, -0.009,  0.011,
                    0.007,  0.006, -0.008, -0.006, -0.006, -0.013, -0.007,  0.110,  0.013, -0.015,
                    0.007, -0.084,  0.060, -0.009, -0.027, -0.025,  0.324, -0.921,  0.007,  0.006,
                    -0.122, -0.010, -0.010, -0.010, -0.031, -0.032])
               
    pv = np.array([np.dot(v[pc1_ix], pc1_coeff), np.dot(v[pc2_ix], pc2_coeff)])
    
    # LDA on PC1 and PC2
    b0 = 9.281324
    b1 = np.array([0.0133170681133271, -0.00653101905300485])
    score = np.dot(pv, b1) + b0
    pred = 0 if score > 0 else 1 # 0 = top; 1 = bottom
    
    return(pred)