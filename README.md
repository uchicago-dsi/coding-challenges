# DSI Coding Challenges

This repo contains code and data used to build, conduct, and evaluate take-home and in-person coding challenges for UChicago DSI's technical interviews.

## Challenges

### Water Bottle Take-Home

This challenge is intended for data science candidates and tests the the candidates' ability to take a common-sense approach to a simple problem when there is not sufficient data to build a standard supervised model.

Google Drive folder with description, data, and candidate results: https://drive.google.com/drive/folders/1g6Lytlix7sJjFzGY49P7p7UQXVKIkReN

**To evaluate:**
1. Copy the candidate's requirements.txt and and water_bottle_challenge.py into challenges/water_bottle_takehome/candidates/{candidate name}/
2. Create a virtual environment and install dependencies:
    ```
    cd challenges/water_bottle_takehome
    python -m venv venv
    source venv/bin/activate
    pip install -r candidate_code/requirements.txt
    ```
3. Run evaluation script, which will create a file called evaluation_results.txt in candidate_code, containing the counts of correct and incorrect predictions:
    ```
    python evaluate.py
    ```

**In-person live code editing challenge:**
Start with the code in challenges/water_bottle_takehome/make_predictions.py. Ask the candidate to make the following changes in succession (when they finish the first, request the second, etc.):
1. Make the script run your classifier on all files in a directory and output results as a CSV or JSON.
2. Update your code to log some debugging info if and only if the `-v` arg is required.
3. Make the script NOT error if it encounters an incorrectly formatted input file. Lines that errored should be marked as "error" in the CSV and errors should be logged even if `-v` was not used.
4. Make the script load the answer key using `json.load("data/preprocessed_map.json")` and save performance statistics for your classifier to a file.