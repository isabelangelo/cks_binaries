"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_fits
from astropy.io import fits
from spectrum import Spectrum
from spectrum import tc 
import pandas as pd
import numpy as np
import dwt

# define paths to spectrum files + labels
df_path = './data/spectrum_dataframes/'
shifted_resampled_path = './data/cks-spectra_shifted_resampled_r/'

# path to names + labels of Kraus 2016 + Kolbl 2015 binaries
kraus2016_binaries = pd.read_csv('./data/label_dataframes/kraus2016_binary_labels.csv')
kolbl2015_binaries = pd.read_csv('./data/label_dataframes/kolbl2015_binary_labels.csv')
known_binaries = pd.concat((
	kraus2016_binaries[['id_starname', 'obs_id']], 
	kolbl2015_binaries[['id_starname', 'obs_id']]))

# filter fluxes with wavelet decomposition
filter_wavelets=True
# store orders in relevant Cannon model
order_numbers = [i for i in np.arange(1,17,1).tolist() if i not in [2, 12]]
model_path = './data/cannon_models/rchip/orders_2.12_omitted_dwt/'
cannon_model = tc.CannonModel.read(model_path+'cannon_model.model')

# store flux, sigma for all orders
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
print('storing flux, sigma of binaries to dataframes')
for order_n in order_numbers:    
	# lists to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []

	# get order data for all stars in training set
	for i in range(len(known_binaries)):
		# load file data
		row = known_binaries.iloc[i]
		filename = shifted_resampled_path + 'order{}/{}.fits'.format(
            order_n, row.id_starname)
		id_starname_list.append(row.id_starname) # save star name for column

		# load spectrum from file
		# and process for Cannon training
		flux_norm, sigma_norm = dwt.load_spectrum(
			filename, 
			filter_wavelets)

		# save to lists
		flux_list.append(flux_norm)
		sigma_list.append(sigma_norm)

	# store flux, sigma data
	flux_df_n = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
	sigma_df_n = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

	# store order number
	flux_df_n.insert(0, 'order_number', order_n)
	sigma_df_n.insert(0, 'order_number', order_n)

	# save to final dataframe
	flux_df = pd.concat([flux_df, flux_df_n])
	sigma_df = pd.concat([sigma_df, sigma_df_n])

# write flux, sigma to .csv files
flux_path = '{}known_binary_flux_dwt.csv'.format(df_path)
sigma_path = '{}known_binary_sigma_dwt.csv'.format(df_path)

flux_df.to_csv(flux_path, index=False)
sigma_df.to_csv(sigma_path, index=False)
print('wavelet-filtered binary spectra saved to:')
print(flux_path)
print(sigma_path)

# # compute metrics for binary sample and save to dataframe
# print('computing metrics for binary sample:')
# metric_keys = ['fit_chisq', 'binary_fit_chisq','training_density', 'delta_chisq']
# metric_data = []
# for star in flux_df.columns[1:]:
# 	print(star)
# 	# load flux, sigma
# 	flux = flux_df[star]
# 	sigma = sigma_df[star]
# 	# create spectrum object
# 	spec = Spectrum(
# 		flux, 
# 		sigma, 
# 		order_numbers, 
# 		cannon_model)
# 	# calculate metrics
# 	spec.fit_single_star()
# 	spec.fit_binary()
# 	# store metrics in dataframe
# 	keys = ['id_starname'] + metric_keys
# 	values = [star] + [spec.fit_chisq, spec.binary_fit_chisq, \
# 	spec.training_density, spec.delta_chisq]
# 	metric_data.append(dict(zip(keys, values)))
# # convert metric data to dataframe
# metric_df = pd.DataFrame(metric_data)

# metric_path = 'data/metric_dataframes/known_binary_metrics.csv'
# metric_df.to_csv(metric_path)
# print('known binary metrics saved to {}'.format(metric_path))


