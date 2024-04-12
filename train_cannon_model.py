from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import thecannon as tc
from cannon_model_diagnostics import *

# initialize file with order stats
order_data_path = './data/cannon_models/rchip_order_stats.csv'
# create an empty DataFrame with columns
empty_order_df = pd.DataFrame({'order': [],'label':[],'bias': [],'rms': []})
# write the DataFrame to a CSV file
empty_order_df.to_csv(order_data_path, index=False)

def train_single_order_cannon_model(order_n):
	"""
	Trains a Cannon model of a specific order
	order_n (int): order number to train on, 1-16 for HIRES r chip
	"""

	# path to save model files to, 
	# should be descriptive of current model to be trained
	model_fileroot = 'rchip_order{}.model'.format(order_n)

	# define training set labels
	training_labels = ['cks_steff', 'cks_slogg', 'cks_smet','cks_svsini']

	# Load the table containing the training set labels
	training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
	training_set = training_set_table[training_labels]

	training_data_path = './data/cannon_training_data/'
	normalized_flux_filename = training_data_path + 'training_flux_order{}.fits'.format(order_n)
	normalized_sigma_filename = training_data_path + 'training_sigma_order{}.fits'.format(order_n)
	normalized_flux = fits.open(normalized_flux_filename)[0].data
	normalized_sigma = fits.open(normalized_sigma_filename)[0].data
	normalized_ivar = 1/normalized_sigma**2

	# Create a vectorizer that defines our model form.
	vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

	# Create the model that will run in parallel using all available cores.
	model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
	                       vectorizer=vectorizer)
	# train model
	model_path = './data/cannon_models/'
	model_filename = model_path + model_fileroot# + '.model'
	model.train()
	print('finished training cannon model')
	model.write(model_filename, include_training_set_spectra=True, overwrite=True)
	print('model written to {}'.format(model_filename))

	# generate one-to-one plots
	print('generating one-to-one diagnostic plot using leave-one-out cross-validation...')
	training_df_path = './data/cks-spectra_dataframes/'
	flux_df_filename = training_df_path + 'training_flux_order{}.csv'.format(order_n)
	sigma_df_filename = training_df_path + 'training_sigma_order{}.csv'.format(order_n)
	flux_df = pd.read_csv(flux_df_filename)
	sigma_df = pd.read_csv(sigma_df_filename)
	plot_one_to_one_leave1out(
		order_n, 
		training_set.to_pandas(), 
		flux_df, 
		sigma_df, 
		model_path + model_fileroot + '_one_to_one.png',
		save_order_to = order_data_path)

# train cannon models + save stats for all 16 orders
for order_n in range(1, 17):
    train_single_order_cannon_model(order_n)

# insert code to train on all orders
# i think maybe what needs to happen is:
# the function can take in an array of order numbers instead of a single one
# this will allow me to rerun the script
# things that need to change: 
# part that loads the training data
# filename it gets saved to
# I think that's it! this should be reasonable.
