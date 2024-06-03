"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_hires_fits
from astropy.io import fits
import dwt
import pandas as pd
import numpy as np
import glob

# function to remove extraneous columns from dataframe
def trimmed(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ============ load literature data ==================================================

# define paths to spectrum files + labels
original_path = './data/cks-spectra/'
shifted_resampled_path = './data/cks-spectra_shifted_resampled_r/'
df_path = './data/cannon_training_data/'
label_path = './data/label_dataframes/'

# load tables with known binaries from Kraus 2016
k16_path = './data/literature_data/Kraus2016/'
def load_kraus_table(filename):
    table = pd.read_csv(k16_path+filename, delim_whitespace=True)
    table['KOI'] = table['KOI'].str.replace('-A-C', '')
    table['KOI'] = table['KOI'].str.replace('-A-B', '')
    table['KOI'] = table['KOI'].str.replace('-B-C', '')
    return table[['KOI', 'sep_mas', 'sep_err']]
# combine companions identified from different methods
kraus_binaries = pd.concat([
    load_kraus_table('Kraus2016_Table3.csv'), # NRM
    load_kraus_table('Kraus2016_Table5.csv'),  # aperture photometry
    load_kraus_table('Kraus2016_Table6.csv')]) # multi-PSF fitting
# add column to match training set table
kraus_binaries['id_starname'] = kraus_binaries['KOI'].str.replace('KOI-', 'K0')

# load table with known binaries from Kolbl 2015
kolbl_binaries = pd.read_csv('./data/literature_data/Kolbl2015_Table9.csv', skiprows=1, 
                 delimiter=' ', names=['koi','Teff_A', 'pm_A', 'Teff_A_err', 'Teff_B', 
                 'pm_B', 'Teff_B_err','FB_over_FA', 'pm_FB_over_FA', 'FB_over_FA_err', 
                 'dRV', 'planetary_data_n', 'planetary_data_status'])
# add column to match training set table
kolbl_binaries.insert(0, 'id_starname', ['K'+(i).zfill(5) for i in kolbl_binaries.koi])

# write clipped wavelength data to reference file
original_w_filename = './data/cks-spectra/cks-k00001_rj122.742.fits' # can be any r chip file
w_data = read_hires_fits(original_w_filename).w[:,:-1] # require even number of elements
w_data = w_data[:, dwt.order_clip:-1*dwt.order_clip] # clip 5% on each side
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
fits.HDUList([fits.PrimaryHDU(w_data)]).writeto(reference_w_filename, overwrite=True)
print('clipped reference wavlength saved to {}'.format(reference_w_filename))

# ============ clean training set + write labels to file  ===================================

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

# remove unresolved spectral binaries from Kraus 2016
# query targets where any separation within reported uncertainties
# falls within hires slit width of 0.8arcsec
hires_slit_width = 800 # 0.8arcsec = 800mas
kraus_binaries = kraus_binaries.query('sep_mas - sep_err < @hires_slit_width')
# write names + labels for binaries to .csv file
kraus2016_binaries = cks_stars[cks_stars['id_starname'].isin(kraus_binaries['id_starname'])]
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(kraus_binaries['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kraus 2016')

# remove unresolved SB2s from Kolbl 2015
kolbl2015_binaries = cks_stars[cks_stars['id_starname'].isin(kolbl_binaries['id_starname'])]
cks_stars = cks_stars[~cks_stars['id_starname'].isin(kolbl_binaries['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kolbl 2015')

# write binaries to file
kolbl2015_binaries['source'] = ['Kolbl2015']*len(kolbl2015_binaries)
kraus2016_binaries['source'] = ['Kraus2016']*len(kraus2016_binaries)
known_binaries = pd.concat((
    trimmed(kraus2016_binaries),
    trimmed(kolbl2015_binaries)))
known_binaries.to_csv(label_path+'known_binary_labels.csv', index=False)

# remove stars with processing errors
processing_err_starnames = np.array(['K02864']) 
cks_stars = cks_stars[~cks_stars.id_starname.isin(processing_err_starnames)]
print(len(cks_stars), ' remain after removing spectra with shifting/processing errors')

# write to .csv file
trimmed(cks_stars).to_csv(label_path+'training_labels.csv', index=False)
print('training labels saved to .csv file')

# ============ write training flux, sigma to files  ================================================

# re-write code below to save data for particular order
def single_order_training_data(order_idx, filter_wavelets=True):

    # order numbers are not zero-indexed
    order_n = order_idx + 1

    # places to store data
    id_starname_list = []
    flux_arr = np.array([])
    sigma_arr = np.array([])

    # get order data for all stars in training set
    for i in range(len(cks_stars)):

        # load file data
        row = cks_stars.iloc[i]
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
flux_df_dwt.to_csv('{}training_flux_dwt.csv'.format(df_path), index=False)
sigma_df_dwt.to_csv('{}training_sigma_dwt.csv'.format(df_path), index=False)
print('wavelet-filtered training flux and sigma saved to .csv files')

# write original training set flux, sigma to files
flux_df_original = pd.DataFrame()
sigma_df_original = pd.DataFrame()
for order_idx in range(0, 16):
    flux_df_n, sigma_df_n = single_order_training_data(order_idx, filter_wavelets=False)
    flux_df_original = pd.concat([flux_df_original, flux_df_n])
    sigma_df_original = pd.concat([sigma_df_original, sigma_df_n])
flux_df_original.to_csv('{}/training_flux_original.csv'.format(df_path), index=False)
sigma_df_original.to_csv('{}/training_sigma_original.csv'.format(df_path), index=False)
print('training flux and sigma pre wavelet filter saved to .csv files')



