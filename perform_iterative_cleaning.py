"""
Loads the original cannon model trained in train_cannon_model.py
and performs an iterative cleaning procedure (see El-Badry2018b)
to remove binaries from the training set and re-train the model.
"""
from astropy.table import Table
from shutil import copyfile
from cannon_model_diagnostics import *
import thecannon as tc
import pandas as pd
import os

# ========= load original model training data ==================

# label + spectrum order information to train model
training_labels = ['cks_teff', 'cks_logg', 'cks_feh','cks_vsini']
vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)
adopted_order_numbers = [i for i in range(1,17) if i not in [2,3,12]]
delta_BIC_threshold = 250 # conservative binary detection threshold
n_binaries = np.inf # intialize non-zero number for iterative cleaning

def clean(model_path_iter_n, iter_n_plus_1):
	"""
    function to clean (i.e. remove binaries) from training set and re-train
    binaries are identified as objects with delta_chisq>100, 
    used for iterative cleaning

	Args:
		model_path_iter_n (str): path to cannon model directory. This directory
		should contain a trained cannon model, and dataframes with associated
		training set flux, sigma, labels + metrics.
		iter_n_plus_1 (int): integer number corresponding to next iteration
		(so for example, original model will be 1, first iteration will be 2, etc.)

    """
	# load training data + binary detection metrics
	# computed during leave-20%-out validaion
	normalized_flux_iter_n = pd.read_csv(model_path_iter_n + 'training_flux.csv')
	normalized_sigma_iter_n= pd.read_csv(model_path_iter_n + 'training_sigma.csv')
	training_set_table_iter_n = pd.read_csv(model_path_iter_n + 'training_labels.csv')
	training_metrics_iter_n = pd.read_csv(model_path_iter_n + 'cannon_labels.csv')

	# create directory for next model iteration
	model_suffix_iter_n_plus_1 = 'adopted_orders_dwt_iter_{}'.format(iter_n)
	model_path_iter_n_plus_1 = model_path.replace('adopted_orders_dwt', model_suffix_iter_n_plus_1)
	os.mkdir(model_path_iter_n_plus_1)

	# query stars to keep (i.e., stars that are not recovered by
	# conservative binary threshold)
	stars_to_keep = training_metrics_iter_n.query('delta_BIC<@delta_BIC_threshold').id_starname.to_numpy()
	n_binaries = len(training_metrics_iter_n) - len(stars_to_keep)

	# index flux, sigma of of stars to keep for next iteration 
	normalized_flux_iter_n_plus_1 = normalized_flux_iter_n[stars_to_keep].to_numpy().T
	normalized_sigma_iter_n_plus_1 = normalized_sigma_iter_n[stars_to_keep].to_numpy().T
	normalized_ivar_iter_n_plus_1 = 1/normalized_sigma_iter_n_plus_1**2
	print('iteration {}: {} binaries removed, {} stars in final training set'.format(
		iter_n_plus_1,
		n_binaries,
		len(normalized_flux_iter_n_plus_1)))

	# index training labels of stars to keep for next iteration
	training_set_iter_n_plus_1 = training_set_table_iter_n[training_set_table_iter_n.id_starname.isin(stars_to_keep)]
	training_set_iter_n_plus_1 = Table.from_pandas(training_set_iter_n_plus_1[training_labels])

	# Create the model that will run in parallel using all available cores.
	model_iter_n_plus_1 = tc.CannonModel(
		training_set_iter_n_plus_1, 
		normalized_flux_iter_n_plus_1, 
		normalized_ivar_iter_n_plus_1,
		vectorizer=vectorizer)

	# train model
	model_iter_n_plus_1.train()

	# compute binary detection metrics for training set stars
	plot_one_to_one_leave1out(
	    adopted_order_numbers, 
	    training_set_iter_n_plus_1.to_pandas(), 
	    model_path_iter_n_plus_1 + 'one_to_one.png',
	    model_suffix_iter_n_plus_1,
	    save_binary_metrics=save_True)

	# store model + training data for next iteration
	model.write(model_path_iter_n_plus_1 + 'cannon_model.model', 
		include_training_set_spectra=True, 
		overwrite=True)
	training_flux[stars_to_keep].to_csv(model_path_iter_n_plus_1 + 'training_flux.csv')
	training_sigma[stars_to_keep].to_csv(model_path_iter_n_plus_1 + 'training_sigma.csv')
	training_set_iter_n_plus_1.to_csv(model_path_iter_n_plus_1 + 'training_labels.csv')


# iteratively clean models
# update n_binaries with each successive model
# and terminate when no binaries are found in the training set
n_plus_1 = 1
while n_binaries>0:

	# first iteration: path to original model
	if n_plus_1 == 1:
		model_path_iter_n = './data/cannon_models/rchip/adopted_orders_dwt/'
	# successive iterations: path to model iteration
	else: 
		model_path_iter_n = './data/cannon_models/rchip/adopted_orders_dwt_iter{}/'.format(n_plus_1)

	# clean model
	clean(model_path_iter_n)

	# update iteration number
	n_plus_1 += 1


# save cleaned model + associated data files
copyfile(
	model_path_iter_n+'training_flux.csv', 
	'./data/cannon_training_data/training_flux_adopted_orders_dwt_cleaned.csv')
copyfile(
	model_path_iter_n+'training_sigma.csv', 
	'./data/cannon_training_data/training_sigma_adopted_orders_dwt_cleaned.csv')
copyfile(
	model_path_iter_n+'training_labels.csv', 
	'./data/label_dataframes/training_labels_cleaned.csv')
os.path.rename(model_path_iter_n, './data/cannon_models/rchip/adopted_orders_dwt_cleaned')




