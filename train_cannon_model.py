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
model_fileroot = 'rchip_order4_wavedec_sym5_level1-8.model'

# define training set labels
training_labels = ['cks_steff', 'cks_slogg', 'cks_smet','cks_svsini']

# Load the table containing the training set labels
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

normalized_flux = fits.open('./data/cannon_training_data/training_flux.fits')[0].data
normalized_sigma = fits.open('./data/cannon_training_data/training_sigma.fits')[0].data
normalized_ivar = 1/normalized_sigma**2

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

# next I want to write some code to generate one-to-one plots
# similar to the ones I made for the Cannon
print('generating one-to-one diagnostic plot using leave-one-out cross-validation...')
flux_df = pd.read_csv('./data/hires_spectra_dataframes/training_flux.csv')
sigma_df = pd.read_csv('./data/hires_spectra_dataframes/training_sigma.csv')
cannon_model_diagnostics.plot_one_to_one_leave1out(
	model, 
	training_set.to_pandas(), 
	flux_df, 
	sigma_df, 
	model_path + model_fileroot + '_one_to_one.png')

