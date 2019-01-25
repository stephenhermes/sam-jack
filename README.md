sam-jack
==============================

This is a sample data science project done from start to (some degree of) finish, using the `cookiecutter` project structure. The goal is to perform <a target="_blank" href="https://en.wikipedia.org/wiki/Literate_programming">literate</a>, <a target="_blank" href="http://ropensci.github.io/reproducibility-guide/sections/introduction/">reproducible</a> work.

**The Project.** Here we use natural language processing to predict a film's genre based off of a plain text plot summary. Included are notebooks and scripts illustrating different techniques of data science, including:
- web scraping to get film plots and genre data,
- data visualization to explore the data,
- natural language processing to engineer features from plain text, and
- machine learning training, tuning and evaluation.

<!-- **Why Sam Jack?** This project started off with a different goal involving Samuel L. Jackson. -->

**Highlights.** Some specific highlights of the project are:
- *src/* - Constains scripts used.
    - *data/fetch_summaries.py* - A script to fetch film data from the OMDb web api.
    - *data/wikipedia_movies.py* - A webscraper to crawl Wikipedia for titles of films. The data pulled by this script is used as input to *fetch_summaries.py*.
    - *models/train.py* - Trains a multi-label random forest classifier using word embeddings to predict the genre of a film, based on it's raw test plot summary.
- *notebooks/* - Contains Jupyter notebooks for data analysis, exploration, model evaluation and prototyping.
    - *5.0-sh-genre-modeling* - The construction, development, and analysis of the machine learning model used.
    - *4.1-sh-raw-data-eda* - Exploratory data analysis of the film data pulled. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
