This file describes the worklow

(1) rsync_cks_spectra:
- copies spectra from CKS and CKS-cool from cadence to local machine.
- saves CKS spectra to ./data/cks-spectra_{chip}, where chip=r or i

(2) shift_and_register_cks_spectra:
- uses SpecMatch-emp to shift spectra into the rest frame and register them onto a uniform HIRES wavelength scale.
- saved shifted spectra to ./data/cks-spectra_shifted_resampled_{chip}
- saves orders to individual files so that we can treat them individually in our analysis

(3) load_training_data:
- loads the CKS flux + labels and filters them to assemble a training set. This script also stores original fluxes to train a model for comparison and demonstrate improvement from wavelet-based filtering.
- saves training data ./data/cannon_training_data

(4) train_cannon_model:
- code to train a cannon model and store model, metrics, and diagnostic plots.
- pulls functions from cannon_model_diagnostics.py
- models and their diagnostics are saved to ./data/cannon_models
- this code is customizable to allow training models using different orders/wavelet filtered or unfiltered data, and has the option to save training data to .csv files.
- there are also some lines of code at the end to train a number of models that I use in my analysis.

(5) load_known_binaries:
- generates dataframes with flux, sigma arrays for known binaries
- user inputs HIRES orders to load flux data for

(6) spectrum.py
- creates Spectrum object that fits a model to a given spectrum and reports fit metrics
- this code is customizable to allow fitting to different orders/wavelet filtered or unfiltered data

question: do I need store_cks_fileroots anymore? in fact maybe it's deleted?
