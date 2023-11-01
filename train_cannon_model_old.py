from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import thecannon as tc
import cannon_model_diagnostics

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'hires_i_chip_segment'

# load training data for diagnostic one-to-one plot
training_flux_df = pd.read_csv('./training_flux.csv')
training_sigma_df = pd.read_csv('./training_sigma.csv')

# define training set labels
training_labels = ['cks_steff', 'cks_slogg', 'cks_smet','cks_svsini']

# Load the table containing the training set labels
training_set_table = Table.read('./training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

normalized_flux = fits.open('./training_flux.fits')[0].data
normalized_sigma = fits.open('./training_sigma.fits')[0].data
normalized_ivar = 1/normalized_sigma**2


# take segment of the code to train cannon model on
w = fits.open('./data/w_to_resample_to_i_chip.fits')[0].data
idx = (w>6670) & (w<6785)
normalized_flux = normalized_flux[:, idx]
normalized_ivar = normalized_ivar[:, idx]
training_flux_df = training_flux_df.iloc[idx]
training_sigma_df = training_sigma_df.iloc[idx]

# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer)

# train model
model_path = './data/cannon_models/'
model_filename = model_path + model_fileroot + '.model'
model.train()
print('finished training cannon model')
model.write(model_filename, include_training_set_spectra=True)
print('model written to {}'.format(model_filename))
# model = tc.CannonModel.read(model_filename)

# save model diagnostic plots
print('saving diagnostic plots...')

cannon_model_diagnostics.plot_one_to_one(
    training_set_table.to_pandas(), 
    training_flux_df, 
    training_sigma_df, 
    model, 
    model_path + model_fileroot + '_one_to_one.png',
    path_to_save_labels='training_set_cannon_labels')
print('one to one plot saved to {}'.format(model_path + model_fileroot + '_one_to_one.png'))


