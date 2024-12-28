This folder contains the scripts required to preprocess the ['Who is Alyx?'](https://github.com/cschell/who-is-alyx) dataset for training and evaluation of the classification models (CNN & GRU).

# Preparation

1. clone the dataset from https://github.com/cschell/who-is-alyx into `data/input`
2. install python requirements with `pip install -r requirements.txt`

# Usage

1. `python 01_aggregate.py` produces an intermediate file used by the following scripts
2. `python 02_generate_classifier_dataset.py` produces the HDF5 file for the classification-based model

The HDF5 files in `data/output` are used for training the machine learning models. Make sure to edit the path in the hydra config files.