# water_bottle_challenge


A simple classifier for whether a preprocessed audio file is of a knife hitting the top or bottom oof a water bottle


## Project Setup

Requires Python 3.13

Create a virtual environment in top director

```python3 -m venv {ENV_NAME}```

Activate environment

```source {ENV_NAME/}bin/activate```

Install requirements

```pip install -r requirements.txt```

The parameters for the knn classifer are already saved so no need to retrain the model however
for both running tests and training the classifier the folder of csvs of preprocessed audio files
should be placed in `/data/raw/`. Once that is done you can retrain the model using

```python water_bottle_challenge/train_knn_classifier.py```

With the train folder you can also run tests

```pytest```

The function `classify_preprocessed_audio` is in `/water_bottle_challenge/water_bottle_challenge.py`
and can be imported from there into a script (see `/notebooks/test_classifier` for example)


<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         water_bottle_challenge and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── water_bottle_challenge   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes water_bottle_challenge a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

