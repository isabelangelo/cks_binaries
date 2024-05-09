"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_hires_fits
from astropy.io import fits
import dwt
import pandas as pd
import numpy as np
import glob

# ============ load literature data ==================================================

# define paths to load and store spectrum files
original_path = './data/cks-spectra/'
shifted_resampled_path = './data/cks-spectra_shifted_resampled_r/'
df_path = './data/cannon_training_data'

# function to load known tables with known binaries
k16_path = './data/literature_data/Kraus2016/'
def load_kraus_table(filename):
    table = pd.read_csv(k16_path+filename, delim_whitespace=True)
    table['KOI'] = table['KOI'].str.replace('-A-C', '')
    table['KOI'] = table['KOI'].str.replace('-A-B', '')
    table['KOI'] = table['KOI'].str.replace('-B-C', '')
    return table[['KOI', 'sep_mas', 'sep_err']]

# clip size to clip 5% each end of order (in pixels)
# used to generate training set
order_clip = 200

# write clipped wavelength data to reference file
original_w_filename = './data/cks-spectra/cks-k00002_rj122.92.fits' # can be any r chip file
w_data = fits.open(original_w_filename)[2].data[:, :-1] # require even number of elements
w_data = w_data[:, order_clip:-1*order_clip] # clip 5% on each side
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
fits.HDUList([fits.PrimaryHDU(w_data)]).writeto(reference_w_filename, overwrite=True)
print('clipped reference wavlength saved to {}'.format(reference_w_filename))

# ============ clean training set + write labels to file  =========================================

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

# remove unresolved binaries from Kraus 2016
# combine companions identified from different methods
kraus_binaries = pd.concat([
    load_kraus_table('Kraus2016_Table3.csv'), # NRM
    load_kraus_table('Kraus2016_Table5.csv'),  # aperture photometry
    load_kraus_table('Kraus2016_Table6.csv')]) # multi-PSF fitting
# query targets where any separation within reported uncertainties
# falls within hires slit width of 0.8arcsec
hires_slit_width = 800 # 0.8arcsec = 800mas
kraus_binaries = kraus_binaries.query('sep_mas - sep_err < @hires_slit_width')
# add column with starnames to match training set table
kraus_binaries['id_starname'] = kraus_binaries['KOI'].str.replace('KOI-', 'K0')
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(kraus_binaries['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kraus 2016')

# write to .csv file
cks_stars.to_csv('./data/label_dataframes/training_labels.csv')
print('training labels saved to .csv file')

# ============ write training flux, sigma to files  ================================================

# re-write code below to save data for particular order
def single_order_training_data(order_idx, filter_wavelets=True):

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

        if filter_wavelets:
            # compute wavelet transform of flux
            level_min, level_max = 1,8
            waverec_levels = np.arange(level_min,level_max+1,1)
            flux_waverec = dwt.flux_waverec(flux_norm, 'sym5', waverec_levels)
            flux_waverec += 1 # normalize to 1 for training
            flux_norm = flux_waverec

        # clip order on each end
        flux_norm = flux_norm[order_clip:-1*order_clip]
        sigma_norm = sigma_norm[order_clip:-1*order_clip]

        # save to lists
        flux_list.append(flux_norm) 
        sigma_list.append(sigma_norm)
        id_starname_list.append(row.id_starname)

    # store flux, sigma data
    flux_df_n = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
    sigma_df_n = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

    # store order number
    flux_df_n.insert(0, 'order_number', order_n)
    sigma_df_n.insert(0, 'order_number', order_n)

    return flux_df_n, sigma_df_n

# write wavelet filtered training set flux, sigma to files
flux_df_dwt = pd.DataFrame()
sigma_df_dwt = pd.DataFrame()
for order_idx in range(0, 16):
    flux_df_n, sigma_df_n = single_order_training_data(order_idx)
    flux_df_dwt = pd.concat([flux_df_dwt, flux_df_n])
    sigma_df_dwt = pd.concat([sigma_df_dwt, sigma_df_n])
flux_df_dwt.to_csv('{}/training_flux_dwt.csv'.format(df_path), index=False)
sigma_df_dwt.to_csv('{}/training_sigma_dwt.csv'.format(df_path), index=False)
print('wavelet-filtered training flux and sigma saved to .csv files')

# write training set flux, sigma for all orders to single file
# write wavelet filtered training set flux, sigma to files
flux_df_original = pd.DataFrame()
sigma_df_original = pd.DataFrame()
for order_idx in range(0, 16):
    flux_df_n, sigma_df_n = single_order_training_data(order_idx, filter_wavelets=False)
    flux_df_original = pd.concat([flux_df_original, flux_df_n])
    sigma_df_original = pd.concat([sigma_df_original, sigma_df_n])
flux_df_original.to_csv('{}/training_flux_original.csv'.format(df_path), index=False)
sigma_df_original.to_csv('{}/training_sigma_original.csv'.format(df_path), index=False)
print('training flux and sigma pre wavelet filter saved to .csv files')



