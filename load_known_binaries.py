"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from astropy.io import fits
import pandas as pd
import numpy as np
import dwt

# define paths to spectrum files + labels
df_path = './data/spectrum_dataframes/'
shifted_resampled_path = './data/cks-spectra_shifted_resampled_r/'

# path to names + labels of Kraus 2016 + Kolbl 2015 binaries
known_binaries = pd.read_csv('./data/label_dataframes/known_binary_labels.csv',
                                dtype={'spectrum_fileroot': str})

# filter fluxes with wavelet decomposition
filter_wavelets=True
# store orders in relevant Cannon model
order_numbers = [i for i in np.arange(1,17,1).tolist() if i not in [11,12]]

# store flux, sigma for all orders
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
for order_n in order_numbers:    
	# lists to store data
	id_starname_list = []
	flux_arr = np.array([])
	sigma_arr = np.array([])

	# get order data for all stars in training set
	for i in range(len(known_binaries)):
		# load file data
		row = known_binaries.iloc[i]
		row_id_starname = row.id_starname.replace('K','k')
		row_spectrum_fileroot = str(row.spectrum_fileroot)
		filename = shifted_resampled_path + 'order{}/cks-{}_rj{}.fits'.format(
			order_n,row_id_starname, row_spectrum_fileroot)
		id_starname_list.append(row.id_starname) # save star name for column
		# load spectrum from file
		# and process for Cannon training
		flux_norm, sigma_norm = dwt.load_spectrum(
			filename, 
			filter_wavelets)
		# save to arrays
		if flux_arr.size==0:
			flux_arr = flux_norm
			sigma_arr = sigma_norm
		else:
			flux_arr = np.vstack((flux_arr, flux_norm))
			sigma_arr = np.vstack((sigma_arr, sigma_norm))

	# store flux, sigma data
	flux_df_n = pd.DataFrame(dict(zip(id_starname_list, flux_arr)))
	sigma_df_n = pd.DataFrame(dict(zip(id_starname_list, sigma_arr)))

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


