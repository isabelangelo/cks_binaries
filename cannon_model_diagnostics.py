from astropy.io import fits
from astropy.table import Table
from spectrum import Spectrum
from spectrum import tc 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os

# file with order stats
order_data_path = './data/cannon_models/rchip_order_stats.csv'

# dataframe with trainig labels + object names
training_label_df = pd.read_csv('./data/label_dataframes/training_labels.csv')

# create file if it doesn't already exist
if os.path.exists(order_data_path)==False:
	empty_order_df = pd.DataFrame({'model': [],'label':[],'bias': [],'rms': []})
	# write the DataFrame to a CSV file
	empty_order_df.to_csv(order_data_path, index=False)

def plot_one_to_one_leave1out(order_numbers, label_df, figure_path, model_suffix):
	"""
	Plot a one-to-one comparison of the training set labels from CKS and the Cannon
    labels inferred from the training set spectra. 
    note: For cross-validation, I train 5 models with 20% of the training set held out,
	and then use that model to compute the labels for the held out 20%.

    Args:
    	order_numbers (list): numbers for HIRES r chip orders in model 
                      		e.g., [1,2,6,15,16]
    	order number (int) : number of HIRES r chip spectrum order (1-17)
    	label_df (pd.Dataframe) : training labels of sample to plot (n_objects x n_labels)
    	figure_path (str) : full path to save plot to 
    	model_suffix (str) : string associated with model to go in filenames
	"""
	pc = 'k';markersize=1;alpha_value=0.5
	labels_to_plot = ['cks_steff', 'cks_slogg', 'cks_smet', 'cks_svsini']

	# compute model to validate based on order number
	model_dir = './data/cannon_models/rchip_{}'.format(model_suffix)
	model_path = model_dir+'/rchip_{}.model'.format(model_suffix)
	model_to_validate = tc.CannonModel.read(model_path)

	def compute_cannon_labels():
		# define training set labels
		cks_keys = labels_to_plot
		cannon_keys = [i.replace('cks', 'cannon') for i in cks_keys]
		metric_keys = ['fit_chisq', 'binary_fit_chisq','training_density', 'delta_chisq']
		vectorizer = tc.vectorizer.PolynomialVectorizer(cks_keys, 2)

		# bin training data into 5 test sets
		n_training = len(model_to_validate.training_set_labels)
		test_size = n_training // 5
		test_bins = np.arange(0,n_training,test_size)
		test_bins[-1]= n_training # include remainder in last chunk

		# perform leave-20%-out cross validation for each bin
		cannon_label_data = []
		for i in range(len(test_bins)-1):

			# define index bounds of left out sample
			start_idx, stop_idx = test_bins[i], test_bins[i+1]
			s = slice(start_idx, stop_idx)
			print('training model with objects {}-{} held out'.format(
				start_idx, stop_idx))

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
				# load object name from training label dataframe
				id_starname = training_label_df.iloc[spectrum_idx].id_starname
				# load object labels, flux, ivar from saved model data
				cks_labels = model_to_validate.training_set_labels[spectrum_idx]
				flux = model_to_validate.training_set_flux[spectrum_idx]
				ivar = model_to_validate.training_set_ivar[spectrum_idx]
				sigma = 1/np.sqrt(ivar)

				# fit cross validation mdoel to data
				spec = Spectrum(
					flux, 
					sigma, 
					order_numbers, 
					model_leave1out)
				spec.fit_single_star()
				spec.fit_binary()
				cannon_labels = spec.fit_cannon_labels

				# store data for plot
				keys = ['id_starname', 'test_number'] + cks_keys + cannon_keys + metric_keys
				values = [id_starname, i] + cks_labels.tolist() + cannon_labels.tolist() \
						+ [spec.fit_chisq, spec.binary_fit_chisq, spec.training_density, spec.delta_chisq]
				cannon_label_data.append(dict(zip(keys, values)))
				print(id_starname)

		# convert label data to dataframe
		cannon_label_df = pd.DataFrame(cannon_label_data)
		cannon_label_path = model_dir+'/cannon_labels.csv'
		print('saving training set cannon output labels to {}'.format(cannon_label_path))
		cannon_label_df.to_csv(cannon_label_path)
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
		stats_dict = {'model': model_suffix, 'label':label, 'bias':bias, 'rms': rms}
		existing_order_data = pd.read_csv(order_data_path)
		updated_order_data  = pd.concat(
			[existing_order_data, pd.DataFrame([stats_dict])])
		updated_order_data.to_csv(order_data_path, index=False)

	def plot_label_difference(label_df, label):
	    x = label_df['cks_{}'.format(label)]
	    y = label_df['cannon_{}'.format(label)]
	    diff = y - x
	    plt.hist(diff, histtype='step', color=pc)
	    plt.xlabel(r'$\Delta {}$'.format(label))

	cannon_label_df = compute_cannon_labels()

	gs = gridspec.GridSpec(5, 2, width_ratios=[2,1])
	plt.figure(figsize=(10,17))
	for i in range(len(labels_to_plot)):
		plt.subplot(gs[2*i])
		plot_label_one_to_one(cannon_label_df, labels_to_plot[i][4:])
		plt.subplot(gs[2*i+1])
		plot_label_difference(cannon_label_df, labels_to_plot[i][4:])
	plt.savefig(figure_path, dpi=300, bbox_inches='tight')






