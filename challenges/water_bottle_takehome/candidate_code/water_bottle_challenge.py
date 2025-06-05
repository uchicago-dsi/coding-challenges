LOW_BAND_MIN = 750
LOW_BAND_MAX = 1100
HIGH_BAND_MIN = 2900
HIGH_BAND_MAX = 3100

TOP_RATIO = 0.3558837397431573
BOTTOM_RATIO = 5.901926118095266

NOISE_THRESHOLD = 0.34


def classify_preprocessed_audio(fpath: str, threshold: float=NOISE_THRESHOLD) -> int:
    """
    Return 1 for top, 0 for bottom, or None for Unclear
    
    Based on the groundwork in `explore.py`, a top hit has
    a more powerful high band and a bottom hit has a more
    powerful low band. We will use the ratios in the reference
    events to make our labels here.

    If the ratio is within NOISE_THRESHOLD of these reference ratios, we
    will consider it a clean enough signal for a label. This hyperparameter
    should probably be tuned.
    """
    with open(fpath, 'r') as f:
        data = f.readlines()
    freq_sums = {}
    for d in data[1:]:          # Leave off the header line
        d_split = d.split(',')
        freq = float(d_split[0]) # Casting to float makes the graphing easier
        d_list = [float(x) for x in d_split[1:]]
        freq_sums[freq] = sum(d_list)

    low_band = {f: p for f, p in freq_sums.items() \
                if LOW_BAND_MIN < f and f < LOW_BAND_MAX}
    high_band = {f: p for f, p in freq_sums.items() \
                 if HIGH_BAND_MIN < f and f < HIGH_BAND_MAX}
    low_band_power = sum(low_band.values())
    high_band_power = sum(high_band.values())
    try:
        ratio = low_band_power / high_band_power
    except ZeroDivisionError:  # If there was no data in the high power band
        return None
    # So, if the result is `threshold` closer to top ratio than 1, label 1
    if ratio < TOP_RATIO + ((1 - TOP_RATIO) * (1 - NOISE_THRESHOLD)):
        return 0
    # And if it is `threshold` closer to bottom ratio than 1, label 0
    if 1 + BOTTOM_RATIO * NOISE_THRESHOLD - NOISE_THRESHOLD < ratio:
        return 1
    # If the ratios are equal, it is unknown
    return None
