Chess Positions Recognition
==============================

The aim of this project is to determine chess positions form schematic chess boards, using computer vision. 
Specifically using the Kornia framework for transforming the data and some type of convolutional neural network. 
The focus of the project is on the machine learning operations associated with the development.

# Dataset

100000 images of a randomly generated chess positions of 5-15 pieces (2 kings and 3-13 pawns/pieces)
Images were generated using 28 styles of chess boards and 32 styles of chess pieces totaling 896 board/piece style combinations.

Images were generated using this custom-build tool

All images are 400 by 400 pixels.

    Training set: 80000 images
    Test set: 20000 images

Pieces were generated with the following probability distribution:

    30% for Pawn
    20% for Bishop
    20% for Knight
    20% for Rook
    10% for Queen

2 Kings are guaranteed to be on the board.

Labels are in a filename in Forsyth–Edwards Notation format, but with dashes instead of slashes.

# Data processing

After processing the dataset, from each original image, we get 64 images each depicting a single square on the chessboard. The images are further converted to greyscale to reduce model complexity.

# Model

We sill just use some vanilla classifier based on Kornia modules.




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
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
