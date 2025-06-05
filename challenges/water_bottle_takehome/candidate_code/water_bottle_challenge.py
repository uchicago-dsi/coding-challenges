import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv

'''
top.csv (1)
Freq with the most energy: 2971.58203125
Total energy 0-2000 hz: 1306.6336747139326
Total energy 2000-4000 hz: 1222.340218622739
Total energy 4000+ hz: 0.0

bottom.csv (0)
Freq with the most energy: 1065.8935546875
Total energy 0-2000 hz: 3241.127459887787
Total energy 2000-4000 hz: 415.74926504680906
Total energy 4000+ hz: 0.0
'''

FREQ_DIVIDER = 2000 # If the freq w the highest energy is > 2000 it is most likely top, if < 2000 then bottom (i decided this from looking at the graph w all the files' freqs)

# The code for the graphs doesn't really matter but I figured I would leave it in
# Graphed out top and bottom by frequency and time so I could see what their energies for each looked like
# Frequency had a clear difference in each
def graph_energy_by_freq(indeces: list, values: list, file_name: str):
    plt.plot(indeces, values)
    plt.title(f"Total energy by freq for {file_name}")
    plt.xlabel(f"freq (hz)")
    plt.ylabel(f"total energy")
    plt.xticks(np.arange(min(indeces), max(indeces)+1, 500), rotation=90)
    plt.show()

# The graph for time was pretty much the same for both top and bottom so looking at time was kind of useless
def graph_energy_by_time(indeces: list, values: list, file_name:str):
    plt.plot(indeces, values)
    plt.title(f"Total energy by time point for {file_name}")
    plt.xlabel('time points (ms)')
    plt.ylabel('total energy')
    plt.show()

# Returns the frequency that has the most energy in each csv file
def get_freq_with_most_energy(fpath: str) -> float:
    df = pd.read_csv(fpath, index_col=0)
    total_energy_by_freq = df.sum(axis=1) 
    # graph_energy_by_freq(total_energy_by_freq.index, total_energy_by_freq.values, fpath) # The graph for top and bottom energy totals by frequency
    return total_energy_by_freq.idxmax()
    
# Assuming these are all valid files being inputted and formatted the same
def classify_preprocessed_audio(fpath: str) -> int:
    freq_with_most_energy = get_freq_with_most_energy(fpath)
    if freq_with_most_energy < FREQ_DIVIDER:
        return 0
    elif freq_with_most_energy > FREQ_DIVIDER: 
        return 1 
    return None # if the max energy freq is exactly 2000

def write_output(output:list):
    with open("output.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(output)

# This graph shows a super clear divide between files that have the most energy at a frequency below or above a thousand so that is the only weight I used to classify
def graph_max_energy_freqs_by_file(dir: Path):
    fpaths = [str(f) for f in dir.iterdir() if f.is_file()]
    filenames = []
    frequencies = []
    for fpath in fpaths:
        filenames.append(fpath)
        frequencies.append(get_freq_with_most_energy(fpath))
    plt.scatter(range(len(frequencies)), frequencies)
    plt.xticks(range(len(filenames)), filenames, rotation=90)
    plt.ylabel("Frequency with the most energy")
    plt.xlabel("File names")
    plt.title("Frequency with the max energy for each file")
    plt.show()

def main():
    input_str = input("Folder or file path: ").strip() # added a folder because sending a single file in every time was annoying
    input_path = Path(input_str)

    if input_path.is_file():
        classification = classify_preprocessed_audio(input_path)
        write_output([(input_path, classification)])      
    elif input_path.is_dir():
        # graph_max_energy_freqs_by_file(input_path)
        fpath_classifications = []
        fpaths = [str(f) for f in input_path.iterdir() if f.is_file()]
        for fpath in fpaths:
            fpath_classifications.append((fpath, classify_preprocessed_audio(fpath)))
        write_output(fpath_classifications)
    else:
        write_output([(input_path, "bad file :(")]) 

if __name__ == "__main__":
    main()