This file describes the worklow

(1) store_cks_fileroots:
- takes cks spectra from cks-spectra/ directory and stores paths to each target in cks_physical_merged.csv so that both labels and spectra can be easily referenced.

(2) load_training_data:
- loads the CKS flux + labels and filters them to assemble a training set. This script also stores original fluxes to train a model for comparison and demonstrate improvement from wavelet-based filtering.
- saves training data ./data/cannon_training_data

(3) train_cannon_model:
- code to train a cannon model and store model, metrics, and diagnostic plots.
- pulls functions from cannon_model_diagnostics.py
- models and their diagnostics are saved to ./data/cannon_models
- this code is customizable to allow training models using different orders/wavelet filtered or unfiltered data, and has the option to save training data to .csv files.
- there are also some lines of code at the end to train a number of models that I use in my analysis.

(4) spectrum.py
- creates Spectrum object that fits a model to a given spectrum and reports fit metrics
- this code is customizable to allow fitting to different orders/wavelet filtered or unfiltered data
