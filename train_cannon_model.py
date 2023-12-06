from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import thecannon as tc
import cannon_model_diagnostics

# initialize file with order stats
order_data_path = './data/cannon_models/cannon_order_model_stats.csv'
# create an empty DataFrame with columns
empty_order_df = pd.DataFrame({'order': [],'label':[],'bias': [],'rms': []})
# write the DataFrame to a CSV file
empty_order_df.to_csv(order_data_path, index=False)


def train_single_order_cannon_model(order_idx):

	# order numbers are not zero-indexed
	order_n = order_idx + 1

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

	# clip end of each order by 200 pixels (total of 10% clipped)
	normalized_flux = normalized_flux[:,200:-200]
	normalized_ivar = normalized_ivar[:,200:-200]

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
	training_df_path = './data/cks-spectra_dataframes/'
	flux_df_filename = training_df_path + 'training_flux_order{}.csv'.format(order_n)
	sigma_df_filename = training_df_path + 'training_sigma_order{}.csv'.format(order_n)
	flux_df = pd.read_csv(flux_df_filename)
	sigma_df = pd.read_csv(sigma_df_filename)
	cannon_model_diagnostics.plot_one_to_one_leave1out(
		model, 
		training_set.to_pandas(), 
		flux_df, 
		sigma_df, 
		model_path + model_fileroot + '_one_to_one.png',
		save_order_to = order_data_path,
		order_number = order_n)


# train cannon models + save stats for all 16 orders
for order_idx in range(0, 16):
    train_single_order_cannon_model(order_idx)



