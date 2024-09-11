"""
Loads labels + HIRES spectra for the single star validation sets.

(1) saves metrics for subset of training set stars that are likely single
	*these stars were part of the Kraus 2016 survey, but no unresolved (<0.8arcsec)
	companion was identified.
(2) saves spectra + metrics for stars with HIRES spectra where no 
	companions was found in Raghavan et al. (2010)

"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.spectrum import read_fits
from spectrum import Spectrum
import spectrum_utils
import thecannon as tc
import pandas as pd
import numpy as np
import dwt

# ============== save metrics for single star validation sample from training set ==========================

# print('saving single star sample...')
# # training set stars
# training_set = pd.read_csv('./data/cannon_models/rchip/adopted_orders_dwt/cannon_labels.csv')

# # full sample from Kraus 2016
# kraus2016_full_sample = pd.read_csv('./data/literature_data/Kraus2016/Kraus2016_Table1.csv', delim_whitespace=True)
# kraus2016_full_sample.insert(1,'id_starname', [i.replace('KOI-', 'K0') for i in kraus2016_full_sample ['KOI']])
# print('{} stars in full Kraus 2016 sample (Table 1)'.format(len(kraus2016_full_sample)))

# # stars from training set in Kraus 2016 sample
# kraus2016_singles = training_set[training_set['id_starname'].isin(kraus2016_full_sample['id_starname'])]
# print('{} training set stars from Kraus 2016 sample with no <0.8arcsec companion'.format(len(kraus2016_singles)))

# # save to .csv
# kraus2016_singles_filename = './data/metric_dataframes/kraus2016_single_metrics.csv'
# kraus2016_singles.to_csv(kraus2016_singles_filename)
# print('')

# ============== save metrics for single star validation sample from Raghavan 2010 ==========================

# define paths to spectrum files + labels
df_path = './data/spectrum_dataframes'
shifted_path = './data/raghavan2010_singles_spectra_shifted'

# path to names of Raghavan 2010 singles
known_singles = pd.read_csv('./data/literature_data/Raghavan2010_singles_obs_ids.csv')

print('saving single star sample...')
# filter fluxes with wavelet decomposition
filter_wavelets=True
# store orders in relevant Cannon model
order_numbers = [i for i in np.arange(1,17,1).tolist() if i not in [8, 11, 16]]
model_path = './data/cannon_models/rchip/adopted_orders_dwt'
cannon_model = tc.CannonModel.read(model_path+'/cannon_model.model')

# load original wavelength data for rescaling
# this is from the KOI-1 original r chip file
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') 
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements

# store flux, sigma for all orders
flux_df = pd.DataFrame()
sigma_df = pd.DataFrame()
print('storing flux, sigma of single stars from Raghavan 2010 to dataframes')
for order_n in order_numbers:    
	# lists to store data
	id_starname_list = []
	flux_list = []
	sigma_list = []
	# order index for wavelength re-scaling
	order_idx = order_n - 1

	# get order data for all stars in training set
	for i in range(len(known_singles)):
		# load file data
		row = known_singles.iloc[i]
		filename = '{}/{}_adj.fits'.format(
            shifted_path,  
            row.observation_id.replace('j','rj')) # .replace('rj','ij')) # for i chip
		id_starname = row.resolvable_name.replace(' ', '')
		id_starname_list.append(id_starname) # save star name for column
		print(id_starname)

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
flux_path = '{}/Raghavan2010_singles_flux_dwt.csv'.format(df_path)
sigma_path = '{}/Raghavan2010_singles_sigma_dwt.csv'.format(df_path)

flux_df.to_csv(flux_path, index=False)
sigma_df.to_csv(sigma_path, index=False)
print('wavelet-filtered spectra saved to:')
print(flux_path)
print(sigma_path)

# compute metrics for sample sample and save to dataframe
print('computing metrics for single star sample:')
metric_keys = ['fit_chisq', 'training_density', 'binary_fit_chisq', 'delta_chisq', 'delta_BIC', 'f_imp']
single_label_keys = ['teff1', 'logg1', 'feh12', 'vsini1', 'rv1', 'teff2', 'logg2', 'vsini2' , 'rv2']
single_label_keys = ['cannon_'+i for i in single_label_keys]
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
	keys = ['id_starname'] + metric_keys + single_label_keys
	values = [star] + [spec.fit_chisq, spec.training_density, \
	spec.binary_fit_chisq, spec.delta_chisq, spec.delta_BIC, spec.f_imp] + \
	spec.binary_fit_cannon_labels.tolist()
	metric_data.append(dict(zip(keys, values)))

# convert metric data to dataframe
metric_df = pd.DataFrame(metric_data)
metric_path = './data/metric_dataframes/Raghavan2010_singles_metrics.csv'
metric_df.to_csv(metric_path)
print('Raghavan 2010 single star metrics saved to {}'.format(metric_path))
print('')