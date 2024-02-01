from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import thecannon as tc
import numpy as np
import pandas as pd

def plot_one_to_one_leave1out(model_to_validate, label_df, flux_df, sigma_df, figure_path, \
	save_order_to=None, order_number=None):
	"""
	Plot a one-to-one comparison of the training set labels from CKS and the Cannon
    labels inferred from the training set spectra. 
    note: For cross-validation, I train 5 models with 20% of the training set held out,
	and then use that model to compute the labels for the held out 20%.

    Args:
    	label_df (pd.Dataframe) : training labels of sample to plot (n_objects x n_labels)
    	flux_df (pd.Dataframe) : flux of sample to plot (n_pixels x n_objects)
    	sigma_df (pd.Dataframe) : sigma of sample to plot (n_pixels x n_objects)
    	model (tc.CannonModel) : cannon model object to test
    	figure_path (str) : full path to save plot to 
    	path_to_save_labels (str) : full path to save injected + recovered labels, if given
	"""
	pc = 'k';markersize=1;alpha_value=0.5
	labels_to_plot = ['cks_steff', 'cks_slogg', 'cks_smet', 'cks_svsini']

	def compute_cannon_labels(label_df, flux_df, sigma_df):
		# define training set labels
		cks_keys = labels_to_plot
		cannon_keys = [i.replace('cks', 'cannon') for i in cks_keys]
		vectorizer = tc.vectorizer.PolynomialVectorizer(cks_keys, 2)

		# bin training data into 5 test sets
		n_training = len(model_to_validate.training_set_labels)
		test_size = n_training // 5
		test_bins = np.arange(0,n_training,test_size)
		test_bins[-1]= n_training - 1 # include remainder in last chunk

		# perform leave-20%-out cross validation for each bin
		cannon_label_data = []
		for i in range(len(test_bins)-1):
		    
		    # define index bounds of left out sample
		    start_idx, stop_idx = test_bins[i], test_bins[i+1]
		    s = slice(start_idx, stop_idx)
		    print(start_idx, '-', stop_idx)
		    
		    # remove left out targets from original training data
		    training_set_labels_leave1out = np.delete(model_to_validate.training_set_labels, s, 0)
		    training_set_leave1out = Table(training_set_labels_leave1out, names=cks_keys)
		    normalized_flux_leave1out = np.delete(model_to_validate.training_set_flux, s, 0)
		    normalized_ivar_leave1out = np.delete(model_to_validate.training_set_ivar, s, 0)

		    # train model for cross validation
		    model_leave1out = tc.CannonModel(
		        training_set_leave1out, 
		        normalized_flux_leave1out, 
		        normalized_ivar_leave1out,
		        vectorizer=vectorizer, 
		        regularization=None)
		    model_leave1out.train()
		    
		    # store labels, flux + sigma for left out targets
		    for spectrum_idx in range(start_idx, stop_idx):
		        cks_labels = model_to_validate.training_set_labels[spectrum_idx]
		        flux = model_to_validate.training_set_flux[spectrum_idx]
		        ivar = model_to_validate.training_set_ivar[spectrum_idx]
		        
		        # fit cross validation model to data
		        cannon_labels = model_leave1out.test(flux, ivar)[0][0]

		        # store data for plot
		        keys = cks_keys + cannon_keys + ['test_number']
		        values = cks_labels.tolist() + cannon_labels.tolist() + [i]
		        cannon_label_data.append(dict(zip(keys, values)))

		# convert label data to dataframe
		cannon_label_df = pd.DataFrame(cannon_label_data)
		return cannon_label_df

	def plot_label_one_to_one(label_df, label):
		x = label_df['cks_{}'.format(label)]
		y = label_df['cannon_{}'.format(label)]
		diff = y - x
		bias = np.round(np.mean(diff), 3)
		rms = np.round(np.sqrt(np.sum(diff**2)/len(diff)), 3)
		subplot_label = 'bias, rms = {}, {}'.format(bias, rms)
		plt.plot(x, y, '.', color=pc, ms=markersize, alpha=alpha_value)
		plt.plot([], [], '.', color='w', label=subplot_label)
		plt.xlabel('CKS {}'.format(label));plt.ylabel('Cannon {}'.format(label))
		plt.plot([x.min(), x.max()], [x.min(), x.max()], lw=0.7, color='#AA8ED9')
		plt.legend(loc='upper left', frameon=False, labelcolor='firebrick')
		# save order stats to file
		if save_order_to is not None:
			stats_dict = {'order': order_number, 'label':label, 'bias':bias, 'rms': rms}
			existing_order_data = pd.read_csv(save_order_to)
			updated_order_data  = pd.concat(
				[existing_order_data, pd.DataFrame([stats_dict])])
			updated_order_data.to_csv(save_order_to, index=False)

	def plot_label_difference(label_df, label):
	    x = label_df['cks_{}'.format(label)]
	    y = label_df['cannon_{}'.format(label)]
	    diff = y - x
	    plt.hist(diff, histtype='step', color=pc)
	    plt.xlabel(r'$\Delta {}$'.format(label))

	cannon_label_df = compute_cannon_labels(
		label_df, 
		flux_df, 
		sigma_df)

	# if path_to_save_labels is not None:
	# 	cannon_label_filename = './data/label_dataframes/'+path_to_save_labels+'.csv'
	# 	cannon_label_df.to_csv(cannon_label_filename)
	# 	print('cannon label dataframe saved to {}'.format(cannon_label_filename))

	gs = gridspec.GridSpec(5, 2, width_ratios=[2,1])
	plt.figure(figsize=(10,17))
	for i in range(len(labels_to_plot)):
		plt.subplot(gs[2*i])
		plot_label_one_to_one(cannon_label_df, labels_to_plot[i][4:])
		plt.subplot(gs[2*i+1])
		plot_label_difference(cannon_label_df, labels_to_plot[i][4:])
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')


def plot_one_to_one(label_df, flux_df, sigma_df, model, 
	figure_path, path_to_save_labels=None):
	"""
	Plot a one-to-one comparison of the training set labels from GALAH and the Cannon
	labels inferred from the training set spectra.
	"""
	pc = 'k';markersize=1;alpha_value=0.5
	labels_to_plot = ['cks_steff', 'cks_slogg', 'cks_smet','cks_svsini']

	def compute_cannon_labels(label_df, flux_df, sigma_df, model):
		cks_keys = labels_to_plot
		cannon_keys = [key.replace('cks','cannon') for key in labels_to_plot]

		cannon_label_data = []
		# iterate over each object
		for id_starname in label_df.id_starname.to_numpy():
			# store galah labels
			row = label_df.loc[label_df.id_starname==id_starname]
			cks_labels = row[cks_keys].values.flatten().tolist()
			# fit cannon model
			flux = flux_df[id_starname]
			sigma = sigma_df[id_starname]
			ivar = 1/sigma**2
			result = model.test(flux, ivar)
			teff_fit, logg_fit, met_fit, vsini_fit = result[0][0]
			# store cannon labels
			cannon_labels = [teff_fit, logg_fit, met_fit, vsini_fit]
			# convert to dictionary
			keys = ['id_starname'] + cks_keys + cannon_keys
			values = [id_starname] + cks_labels + cannon_labels
			cannon_label_data.append(dict(zip(keys, values)))
		cannon_label_df = pd.DataFrame(cannon_label_data)
		return cannon_label_df

	def plot_label_one_to_one(cannon_label_df, training_label_df, label):
		x = cannon_label_df['cks_{}'.format(label)]
		y = cannon_label_df['cannon_{}'.format(label)]
		# model performance metrics
		diff = y - x
		bias = np.round(np.mean(diff), 3)
		rms = np.round(np.sqrt(np.sum(diff**2)/len(diff)), 3)

		# CKS label uncertainty (expected performance)
		error_arr1 = training_label_df['cks_'+label+'_err1'].to_numpy()
		error_arr2 = training_label_df['cks_'+label+'_err2'].to_numpy()
		error = np.abs(np.vstack((error_arr1, error_arr2))).flatten()
		error = np.round(np.median(error), 3)

		subplot_label = 'bias, rms = {}, {}\navg. CKS error={}'.format(bias, rms, error)
		plt.plot(x, y, '.', color=pc, ms=markersize, alpha=alpha_value)
		plt.plot([], [], '.', color='w', label=subplot_label)
		plt.xlabel('CKS {}'.format(label));plt.ylabel('Cannon {}'.format(label))
		plt.plot([x.min(), x.max()], [x.min(), x.max()], lw=0.7, color='#AA8ED9')
		plt.legend(loc='upper left', frameon=False, labelcolor='firebrick')

	def plot_label_difference(cannon_label_df, label):
		x = cannon_label_df['cks_{}'.format(label)]
		y = cannon_label_df['cannon_{}'.format(label)]
		diff = y - x
		plt.hist(diff, histtype='step', color=pc)
		plt.xlabel(r'$\Delta {}$'.format(label))

	cannon_label_df = compute_cannon_labels(
		label_df, 
		flux_df, 
		sigma_df, 
		model)

	if path_to_save_labels is not None:
		cannon_label_filename = './'+path_to_save_labels+'.csv'
		cannon_label_df.to_csv(cannon_label_filename)
		print('cannon label dataframe saved to {}'.format(cannon_label_filename))

	gs = gridspec.GridSpec(5, 2, width_ratios=[2,1])
	plt.figure(figsize=(10,17))
	for i in range(len(labels_to_plot)):
		plt.subplot(gs[2*i])
		plot_label_one_to_one(cannon_label_df, label_df, labels_to_plot[i][4:])
		plt.subplot(gs[2*i+1])
		plot_label_difference(cannon_label_df, labels_to_plot[i][4:])
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')