import numpy as np
import matplotlib.pyplot as plt


#############
# Functions #
#############

def get_freq_sums(fname: str) -> dict:
    """
    Return a dict of {frequency: sum}
    
    As a simplifying assumption, we can take the sum of the frequencies
    over the time frame and then assess which frequencies in the clear
    signals are creating the most resonance.
    """
    with open(fname, 'r') as f:
        data = f.readlines()
    freq_sums = {}
    for d in data[1:]:          # Leave off the header line
        d_split = d.split(',')
        freq = float(d_split[0]) # Casting to float makes the graphing easier
        d_list = [float(x) for x in d_split[1:]]
        freq_sums[freq] = sum(d_list)
    return freq_sums


def norm_freq_sums(freq_sums: dict) -> dict:
    """
    Return a normalized version of the freq_sums dict.

    We'll want a normalized version in case some hits to the
    bottle are harder or softer. This prevents the model from
    attaching to absolute numbers, and instead looking at
    relative numbers.
    """
    total = sum(freq_sums.values())
    for f, v in freq_sums.items():
        freq_sums[f] = v / total
    return freq_sums


def graph_dists(d1: dict, d2: dict,
                d1_title: str="d1", d2_title: str="d2") -> None:
    """
    Create and show a bar graph of d1 and d2

    Allows for visual comparison between two sample
    distributions.
    """
    ind = np.arange(2)
    width = 10   # We're overlapping bars, but we want to make sure
                 # we aren't missing anything due to weird renders
                 # Plus, we want approximate frequencies, since those
                 # could change based on the exact hit
    xs = np.array(list(d1.keys()))
    plt.bar(xs, d1.values(), width, label=d1_title)
    plt.bar(xs + width, d2.values(), width, label=d2_title)
    plt.xlabel("Frequency")
    plt.ylabel("dB")
    plt.title(f"{d1_title} and {d2_title}")
    plt.legend()
    plt.show()


def get_time_series_sums(fname: str, low_freq: float=0, high_freq: float=None) -> dict:
    with open(fname, 'r') as f:
        data = f.readlines()
    sums = [0 for _ in data[0][1:]] # Initialize time slices sums
    times = data[0].split(',')[1:]  # Get time slice labels
    if not high_freq:               # Set to  max
        high_freq = max(data[1].split(',')[1:])
    for d in data[1:]:
        d_freq = float(d.split(',')[0])
        if low_freq <= d_freq and d_freq <= high_freq:
            for i, v in enumerate(d.split(',')[1:]):
                sums[i] += float(v)

    return dict(zip(times, sums))
    

def get_time_series_visual(fname: str, low_freq: float, high_freq: float) -> None:
    """
    Plot and show a bar graph of sound power between summed
    low_freq and high_freq.

    We can use this to check that the sound is a hit
    by making sure it has a peak and decay structure over time
    rather than noise throughout.
    """
    time_sums = get_time_series_sums(fname, low_freq=low_freq, high_freq=high_freq)
    # We can slice out just he parts we want to see
    i = min(time_sums.keys())
    for t in time_sums.keys():
        if time_sums[t] != 0:
            i = t
            break

    j = max(time_sums.keys())
    rev_keys = list(time_sums.keys())
    rev_keys.reverse()
    for t in rev_keys:
        if time_sums[t] != 0:
            j = t
            break

    time_sums = {t: s for t, s in time_sums.items() if i <= t and t <= j}
    onset = max(time_sums, key=time_sums.get)

    # Finally, plot it
    fig, ax = plt.subplots()
    ax.bar(time_sums.keys(), time_sums.values())
    title_name = fname.split('/')[1].split('.')[0]
    ax.set_title(f"{title_name} from {low_freq}Hz to {high_freq}Hz")
    ax.set_xlabel("Time")
    ax.set_ylabel("dB")
    xlims = ax.get_xlim()
    ax.set_xticks(xlims)
    plt.show()


def graph_onsets(fname: str) -> None:
    """
    For each frequency, graph the time of the max power.

    To check for onset alignment.
    """
    with open(fname, 'r') as f:
        data = f.readlines()

    times = data[0].split(',')[1:]        # Time slice labels
    max_power = {}
    for d in data[1:]:                    # For each frequency
        d_split = d.split(',')
        d_split = [float(x) for x in d_split]
        freq = d_split[0]
        max_p = max(d_split[1:])          # Get the max power
        # Remove zero-power frequencies
        if max_p > 10:
            max_i = d_split[1:].index(max_p)  # Find the index of the max power
            max_power[freq] = times[max_i]    # Set the associated time index

    # Plot the max (i.e., onsets)
    fig, ax = plt.subplots()
    ax.bar(max_power.keys(), max_power.values(), width=10)
    title_name = fname.split('/')[1].split('.')[0]
    ax.set_title(f"{title_name} max power")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Time")
    ylims = ax.get_ylim()
    ax.set_yticks(ylims)
    xlims = ax.get_xlim()
    ax.set_xticks(xlims)
    plt.show()


def comparative_power(fname: str,
                      low_band_min: float, low_band_max: float,
                      high_band_min: float, high_band_max: float):
    """
    Return the comparative power of the lower band with the higher band.

    Return a ratio of low band power over high band power
    """
    freq_sums = get_freq_sums(fname)
    low_band = {f: p for f, p in freq_sums.items() \
                if low_band_min < f and f < low_band_max}
    high_band = {f: p for f, p in freq_sums.items() \
                 if high_band_min < f and f < high_band_max}
    low_band_power = sum(low_band.values())
    high_band_power = sum(high_band.values())
    return low_band_power / high_band_power


def classify_preprocessed_audio(fpath: str,
                                low_band_min: float=700,
                                low_band_max: float=1100,
                                high_band_min: float=2900,
                                high_band_max: float=3100,
                                threshold: float=0.5) -> int:
    """
    Return 1 for top, 0 for bottom, or None for Unclear
    
    Based on the groundwork in `explore.py`, a top hit has
    a more powerful high band and a bottom hit has a more
    powerful low band. We will use the ratios in the reference
    events to make our labels here.

    If the low band / high band ratio is more similar to the
    top reference signal, we will label as 1
    If the low band / high band ratio is more similar to the
    bottom reference signal, label as 0
    """
    top_ratio = 0.3558837397431573
    bottom_ratio = 5.901926118095266

    freq_sums = get_freq_sums(fpath)
    low_band = {f: p for f, p in freq_sums.items() \
                if low_band_min < f and f < low_band_max}
    high_band = {f: p for f, p in freq_sums.items() \
                 if high_band_min < f and f < high_band_max}
    low_band_power = sum(low_band.values())
    high_band_power = sum(high_band.values())
    try:
        ratio = low_band_power / high_band_power
    except ZeroDivisionError:  # If there was no data in the high power band
        return None
    # So, if the result is closer to top ratio than 1, label 1
    if ratio < top_ratio + ((1 - top_ratio) * (1 - threshold)):
        return 0
    # And if it is closer to bottom ratio than 1, label 0
    if 1 + bottom_ratio * threshold - threshold < ratio:
        return 1
    # If the ratios are equal, it is unknown
    return None


def train_threshold() -> None:
    import os
    import re

    dir = os.listdir("./train")
    
    for t in range(10, 100, 2):
        tops = 0
        bottoms = 0
        unknown = 0
        for fname in dir:
            if re.match(r"unlabeled", fname):
                c = classify_preprocessed_audio(f"./train/{fname}", threshold=t/100)
                if c == 1:
                    tops += 1
                elif c == 0:
                    bottoms += 1
                else:
                    unknown += 1
        print(f"{t} threshold: {tops} tops, {bottoms} bottoms, {unknown} unknowns")
        
    
###############
# Begin Scipt #
###############
    
top_freq_sums = get_freq_sums("train/top.csv")
bottom_freq_sums = get_freq_sums("train/bottom.csv")

# Let's see if there are a bunch of zeroes we can get rid of
non_zeroes = []
for f, v in top_freq_sums.items():
    if v != 0 and bottom_freq_sums[f] != 0:
        non_zeroes.append(f)
print(f"Lowest detected frequency: {min(non_zeroes)}")
print(f"Highest detected frequency: {max(non_zeroes)}")

# So we can see that frequencies below 150Hz
# And everything above 3823Hz are zero on both, so
# We may as well get rid of those, since there
# is no resonance at those
top_freq_sums = {f: s for f, s in top_freq_sums.items() if 150 < f and f < 3823}
bottom_freq_sums = {f: s for f, s in bottom_freq_sums.items() if 150 < f and f < 3823}

# Then, we can check our visuals more easily, too
graph_dists(top_freq_sums, bottom_freq_sums,
            d1_title="Top Sums", d2_title="Bottom Sums")

# So it looks like the top frequencies tend to be around
# 2900-3100Hz, and the bottom frequencies tent to be around
# 700-1100Hz.

# Let's look at these if they are normalized over the total sound
# power in the file. This ensures that we aren't looking at some
# artifact of how hard the bottles are being hit in each file
top_norm = norm_freq_sums(top_freq_sums)
bottom_norm = norm_freq_sums(bottom_freq_sums)

graph_dists(top_norm, bottom_norm,
            d1_title="Top Norm", d2_title="Bottom Norm")

# Norms agree with the absolutes

# Let's double check that these are hits by seeing if they decay
# over time in the known samples
# We'll also check that onset (i.e. max volume) is at the same time

get_time_series_visual("train/top.csv", 750, 1100)
# Not super clean, but pretty good

get_time_series_visual("train/top.csv", 2900, 3100)
# That definitely looks like a hit and decay

get_time_series_visual("train/bottom.csv", 750, 1100)
# Not super clean, but not too bad

get_time_series_visual("train/bottom.csv", 2900, 3100)
# That definitely looks like a hit and decay

# Let's check all onsets to make sure everything is synced up
graph_onsets("train/top.csv")
graph_onsets("train/bottom.csv")

# These don't look so good. I'd want to dig into this more
# if this were being used beyond this exercise. It's likely
# just an error in my logic, but the maximum power should
# line up in time for each frequency, up to some limit
# of the speed of sound and recording, which aren't likely
# to effect things at this time scale.

top_ratio = comparative_power("train/top.csv", 750, 1100, 2900, 3100)
bottom_ratio = comparative_power("train/bottom.csv", 750, 1100, 2900, 3100)

print(f"For a top hit, low frequencies are {top_ratio} times more powerful")
print(f"For a bottom hit, low frequencies are {bottom_ratio} times more powerful")

# This makes sense. I used the bottle on my desk as an example, and I can
# definitely hear a lower pitch when hitting the bottom

# So let's define the model as follows:
# We will labels a dataset as 1 if the high frequencies are at least
# some number times more powerful, 0 if they are at most some portion
# powerful and None otherwise.
# I'll set a hyperparameter to define the bands

# train_threshold()

# The above calls give shows that a threshold of 0.34 gives an even split
# between labeled and unlabeled data. This is not particularly well
# movtivated, but gives a good-enough cutoff. This also works will with
# creating three numerically similar ratios for labels.
