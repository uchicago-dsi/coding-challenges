"""Classify all pre-processed files in a directory.

This is a skeleton of a script for running the candidate's classifier function
on all pre-processed files in a directory.
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from candidate_code.water_bottle_challenge import classify_preprocessed_audio

def setup_logger(verbose=False):
    """Set up and configure root logger with appropriate level based on verbose flag."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.root.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create file handler and set level
    log_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)


def make_predictions(directory_path) -> dict:
    """Makes a prediction on all pre-processed files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing pre-processed files.
        
    Returns:
        dict: A dictionary where each key is a file name and each value is the prediction.
    """
    logging.info("Analysis completed successfully")
    predictions = {}
    for fpath in Path(directory_path).glob('*.csv'):
        predictions[fpath.stem] = classify_preprocessed_audio(fpath)

    return predictions


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run analysis on specified directory')
    parser.add_argument('directory', type=str, help='Path to the directory for analysis')
    args = parser.parse_args()
    
    # Set up logger
    setup_logger()

    # Run analysis
    predictions = make_predictions(args.directory)
    print(predictions)