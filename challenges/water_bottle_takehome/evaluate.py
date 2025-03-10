import json
from pathlib import Path

import pandas as pd

from candidate_code import water_bottle_challenge

DATA_DIR = Path("data/")
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
ANSWER_KEY_PATH = DATA_DIR / "class_map.json"


if __name__ == "__main__":
    # Load answer key
    answer_key = json.loads(ANSWER_KEY_PATH.read_text())

    # Run candidate's classifier function
    predictions = {}
    for fpath in PREPROCESSED_DATA_DIR.glob("*.csv"):
        predictions[fpath.stem] = water_bottle_challenge.classify_preprocessed_audio(fpath)

    # Join all info about predictions into a single DataFrame
    df = pd.DataFrame({
        "predicted": predictions,
        "actual": answer_key,
    })
    df["predicted"] = df["predicted"].map({0: "top", 1: "bottom"})
    df["is_correct"] = df["predicted"] == df["actual"]
    df["type"] = df.index.str.split("_").str[0]
    df["type"] = df["type"].map({"top": "labeled", "bottom": "labeled"}).fillna(df["type"])

    # Get counts of correct predictions
    correct_counts = df.groupby(["type", "is_correct"]).size().unstack("is_correct").fillna(0).astype(int)

    # Get counts of all combinations of actual and predicted
    all_combo_counts = df.groupby(["type", "actual", "predicted"]).size().reset_index()

    # Write to file in a readable format
    with open("candidate_code/evaluation_results.txt", "w") as f:
        # Write total correct and incorrect counts
        f.write("Top-line results:\n")
        for idx, row in correct_counts.iterrows():
            f.write(f"{idx}: {row[True]} correct, {row[False]} incorrect\n")
        # Write all combination counts, fixed width for readability
        f.write("\nAll combination counts:\n")
        for col in all_combo_counts.columns:
            all_combo_counts[col] = all_combo_counts[col].astype(str)
            all_combo_counts[col] = all_combo_counts[col].str.pad(all_combo_counts[col].str.len().max()+1, side='right')
        for idx, row in all_combo_counts.iterrows():
            f.write(f"{row['type']}\t{row['actual']}\t{row['predicted']}\t{row[0]}\n")