"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_hires_fits
from astropy.io import fits
import spectrum_dwt
import pandas as pd
import numpy as np
import glob

# define paths to load and store spectrum files
original_path = './data/cks-spectra/'
shifted_resampled_path = './data/cks-spectra_shifted_resampled_r/'
df_path = './data/cks-spectra_dataframes'
fits_path = './data/cannon_training_data'

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
for i in range(len(cks_stars)):

    # load file data
    row = cks_stars.iloc[i]
    row_id_starname = row.id_starname.replace('K','k')
    row_spectrum_fileroot = str(row.spectrum_fileroot)

    filename = original_path + 'cks-{}_rj{}.fits'.format(
        row_id_starname, row_spectrum_fileroot)
    target = read_hires_fits(filename)

    # compute average pixel error, remove if >3%
    snr = np.nanmean(target.s)/np.nanmean(target.serr)
    if snr < 20:
        low_sigma_idx_to_remove.append(i)
cks_stars = cks_stars.drop(cks_stars.index[low_sigma_idx_to_remove])
print(len(cks_stars), ' after removing spectra with per pixel SNR < 20')
cks_stars.to_csv('./data/label_dataframes/training_labels.csv')
print('training labels saved to .csv file')

# re-write code below to save data for particular order
def write_training_set_to_file(order_idx):

    # order numbers are not zero-indexed
    order_n = order_idx + 1

    # lists to store data
    id_starname_list = []
    flux_list = []
    sigma_list = []

    # get order data for all stars in training set
    for i in range(len(cks_stars)):

        # load file data
        row = cks_stars.iloc[i]
        row_id_starname = row.id_starname.replace('K','k')
        row_spectrum_fileroot = str(row.spectrum_fileroot)

        filename = shifted_resampled_path + 'order{}/cks-{}_rj{}.fits'.format(
            order_n,row_id_starname, row_spectrum_fileroot)
        file = fits.open(filename)[1].data
        
        # store flux, sigma
        flux_norm = file['s']
        sigma_norm = file['serr']
        w_order = file['w']

        # remove nans from flux, sigma
        # note: this needs to happen here so that the Cannon
        # always returns flux values for all wavelengths
        finite_idx = ~np.isnan(flux_norm)
        if np.sum(finite_idx) != len(flux_norm):
            flux_norm = np.interp(w_order, w_order[finite_idx], flux_norm[finite_idx])
        sigma_norm = np.nan_to_num(sigma_norm, nan=1)

        # require even number of elements
        if len(flux_norm) %2 != 0:
            flux_norm = flux_norm[:-1]
            sigma_norm = sigma_norm[:-1]

        # require even number of elements
        if len(flux_norm) %2 != 0:
            flux_norm = flux_norm[:-1]
            sigma_norm = sigma_norm[:-1]

        # compute wavelet transform of flux
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

    # write training data to .csv files
    flux_df.to_csv('{}/training_flux_order{}.csv'.format(df_path, order_n))
    sigma_df.to_csv('{}/training_sigma_order{}.csv'.format(df_path, order_n))
    print('training flux and sigma saved to .csv files')

    # write training data to fits files
    flux_filename = '{}/training_flux_order{}.fits'.format(fits_path, order_n)
    sigma_filename = '{}/training_sigma_order{}.fits'.format(fits_path, order_n)

    flux_arr = flux_df.to_numpy().T
    sigma_arr = sigma_df.to_numpy().T

    fits.HDUList([fits.PrimaryHDU(flux_arr)]).writeto(flux_filename, overwrite=True)
    print('training flux array saved to {}'.format(flux_filename))

    fits.HDUList([fits.PrimaryHDU(sigma_arr)]).writeto(sigma_filename, overwrite=True)
    print('training sigma array saved to {}'.format(sigma_filename))


# write training set data to files for all 16 orders
for order_idx in range(0, 16):
    write_training_set_to_file(order_idx)








