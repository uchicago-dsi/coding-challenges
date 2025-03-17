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
Start with the code in challenges/water_bottle_takehome/make_predictions.py. Walk through the following steps with the candidate to make the script more robust:
0. Clone the repo, read the code, set up a virtual env, and run make_predictions.py.
1. How should we handle a file that is broken / doesn't load?
2. How should the results be saved?
3. Load up an answer key ("data/preprocessed_map.json") using `json.load(Path(answer_key_path).open()`. How should we generate and save performance stats?
4. We want to print out some debugging info if and only if `-v` is provided. How should we do that? Provide this code snippet: `parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')`
