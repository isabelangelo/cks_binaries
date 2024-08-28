"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.spectrum import read_fits
from spectrum import Spectrum
import thecannon as tc
import pandas as pd
import numpy as np
import dwt

# define paths to spectrum files + labels
df_path = './data/spectrum_dataframes'
shifted_path = './data/cks-spectra_shifted'

# path to names + labels of Kraus 2016 + Kolbl 2015 binaries
kraus2016_binaries = pd.read_csv('./data/label_dataframes/kraus2016_binary_labels.csv')
kolbl2015_binaries = pd.read_csv('./data/label_dataframes/kolbl2015_binary_labels.csv')
known_binaries = pd.concat((
	kraus2016_binaries[['id_starname', 'obs_id']], 
	kolbl2015_binaries[['id_starname', 'obs_id']]))

# filter fluxes with wavelet decomposition
filter_wavelets=True
# store orders in relevant Cannon model
order_numbers = [i for i in np.arange(1,17,1).tolist() if i not in [2,12]]
model_path = './data/cannon_models/rchip/adopted_orders_dwt'
cannon_model = tc.CannonModel.read(model_path+'/cannon_model.model')

# load original wavelength data for rescaling
# this is from the KOI-1 original r chip file
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') 
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements

# store flux, sigma for all orders
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
print('storing flux, sigma of binaries to dataframes')
for order_n in order_numbers:    
	# lists to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []
	# order index for wavelength re-scaling
	order_idx = order_n - 1

	# get order data for all stars in training set
	for i in range(len(known_binaries)):
		# load file data
		row = known_binaries.iloc[i]
		filename = '{}/{}_adj.fits'.format(
            shifted_path,  
            row.obs_id) # .replace('rj','ij')) # for i chip
		id_starname_list.append(row.id_starname) # save star name for column

		# load spectrum from file
		# and resample to unclipped HIRES wavelength scale
		# (since flux, sigma arrays get clipped post-wavelet filtering)
		KOI_spectrum = read_fits(filename)
		rescaled_order = KOI_spectrum.rescale(original_wav_data[order_idx])

		# process for Cannon training
		flux_norm, sigma_norm = dwt.load_spectrum(
		    rescaled_order, 
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
flux_path = '{}/known_binary_flux_dwt.csv'.format(df_path)
sigma_path = '{}/known_binary_sigma_dwt.csv'.format(df_path)

flux_df.to_csv(flux_path, index=False)
sigma_df.to_csv(sigma_path, index=False)
print('wavelet-filtered binary spectra saved to:')
print(flux_path)
print(sigma_path)

# compute metrics for binary sample and save to dataframe
print('computing metrics for binary sample:')
metric_keys = ['fit_chisq', 'training_density', 'binary_fit_chisq', 'delta_chisq', 'delta_BIC', 'f_imp']
binary_label_keys = ['teff1', 'logg1', 'feh12', 'vsini1', 'rv1', 'teff2', 'logg2', 'vsini2' , 'rv2']
binary_label_keys = ['cannon_'+i for i in binary_label_keys]
metric_data = []
for star in flux_df.columns[1:]:
	print(star)
	# load flux, sigma
	flux = flux_df[star]
	sigma = sigma_df[star]
	# create spectrum object
	spec = Spectrum(
		flux, 
		sigma, 
		order_numbers, 
		cannon_model)
	# calculate metrics
	spec.fit_single_star()
	spec.fit_binary()
	# store metrics in dataframe
	keys = ['id_starname'] + metric_keys + binary_label_keys
	values = [star] + [spec.fit_chisq, spec.training_density, \
	spec.binary_fit_chisq, spec.delta_chisq, spec.delta_BIC, spec.f_imp] + \
	spec.binary_fit_cannon_labels.tolist()
	metric_data.append(dict(zip(keys, values)))
# convert metric data to dataframe
metric_df = pd.DataFrame(metric_data)
metric_path = 'data/metric_dataframes/known_binary_metrics.csv'
metric_df.to_csv(metric_path)
print('known binary metrics saved to {}'.format(metric_path))


