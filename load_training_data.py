from astropy.io import fits
import spectrum_dwt
import pandas as pd
import numpy as np
import glob

# load table with CKS properties from CKS website
cks_stars = pd.read_csv(
    './data/literature_data/cks_physical_merged_with_fileroots.csv',
    dtype={'spectrum_fileroot': str}) # retains trailing zeros in spectrum fileroot
print(len(cks_stars), ' table entries from CKS website')

# remove duplicate targets
cks_stars = cks_stars.drop_duplicates('id_kic')
print(len(cks_stars), 'remaining after removing duplicate entries for multi-planet hosts')

# require finite training set labels
training_set_labels = ['cks_steff', 'cks_slogg', 'cks_smet', 'cks_svsini']
cks_stars = cks_stars.dropna(subset=training_set_labels)
print(len(cks_stars), 'with finite training set labels')

# require main sequence stars
cks_stars = cks_stars.query('cks_slogg > 4')
print(len(cks_stars), 'remaining after requiring logg>4')

# remove stars with low SNR
low_sigma_idx_to_remove = []
path = './data/cks-spectra_shifted_resampled_r/'
for i in range(len(cks_stars)):

    # load file data
    row = cks_stars.iloc[i]
    row_id_starname = row.id_starname.replace('K','k')
    row_spectrum_fileroot = str(row.spectrum_fileroot)

    filename = path + 'cks-{}_rj{}_adj_resampled.fits'.format(
        row_id_starname, row_spectrum_fileroot)
    file = fits.open(filename)[1].data

    # compute average pixel error, remove if >3%
    sigma_avg = np.mean(file['serr'])
    if sigma_avg >= 0.03:
        low_sigma_idx_to_remove.append(i)
cks_stars = cks_stars.drop(cks_stars.index[low_sigma_idx_to_remove])
print(len(cks_stars), ' after removing spectra with > 3 percent flux errors')

# write training + test set labels to .csv files
id_starname_list = []
flux_list = []
sigma_list = []

for i in range(len(cks_stars)):

	# load file data
    row = cks_stars.iloc[i]
    row_id_starname = row.id_starname.replace('K','k')
    row_spectrum_fileroot = str(row.spectrum_fileroot)

    filename = path + 'cks-{}_rj{}_adj_resampled.fits'.format(
        row_id_starname, row_spectrum_fileroot)
    file = fits.open(filename)[1].data
    
    # store flux, sigma
    flux = file['s']
    sigma = file['serr']

    # remove nans from flux, sigma
    # note: this needs to happen here so that the Cannon
    # always returns flux values for all wavelengths
    flux = np.nan_to_num(flux, nan=1)
    sigma = np.nan_to_num(sigma, nan=1)

    # slice + normalize flux for wavelet decomposition
    # (also require even number of elements)
    flux_norm = flux[spectrum_dwt.wavedec_idx][:-1] - 1
    sigma_norm = sigma[spectrum_dwt.wavedec_idx][:-1]

    level_min, level_max = 1,8
    waverec_levels = np.arange(level_min,level_max+1,1)
    flux_waverec = spectrum_dwt.flux_waverec(flux_norm, 'sym5', waverec_levels)
    flux_waverec += 1 # normalize to 1 for training
    
    # save to lists
    flux_list.append(flux_waverec) 
    sigma_list.append(sigma_norm)
    id_starname_list.append(row.id_starname)
    
flux_df = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
sigma_df = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

flux_df.to_csv('./data/hires_spectra_dataframes/training_flux.csv')
sigma_df.to_csv('./data/hires_spectra_dataframes/training_sigma.csv')
cks_stars.to_csv('./data/label_dataframes/training_labels.csv')
print('training flux and sigma saved to .csv files')
print('training labels saved to .csv file')

# write training + test data to fits files
flux_filename = './data/cannon_training_data/training_flux.fits'
sigma_filename = './data/cannon_training_data/training_sigma.fits' 
flux_arr = flux_df.to_numpy().T
sigma_arr = sigma_df.to_numpy().T

fits.HDUList([fits.PrimaryHDU(flux_arr)]).writeto(flux_filename, overwrite=True)
print('training flux array saved to {}'.format(flux_filename))

fits.HDUList([fits.PrimaryHDU(sigma_arr)]).writeto(sigma_filename, overwrite=True)
print('training sigma array saved to {}'.format(sigma_filename))










