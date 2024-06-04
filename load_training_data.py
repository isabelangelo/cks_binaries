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
df_path = './data/cannon_training_data'
label_path = './data/label_dataframes/'

# load binary data from Kraus 2016
# function to load known tables with known binaries
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
# query targets where any separation within reported uncertainties
# falls within hires slit width of 0.8arcsec
hires_slit_width = 800 # 0.8arcsec = 800mas
kraus_binaries = kraus_binaries.query('sep_mas - sep_err < @hires_slit_width')
# add column with starnames to match training set table
kraus_binaries['id_starname'] = kraus_binaries['KOI'].str.replace('KOI-', 'K0')

# load binary data from Kolbl 2016
# load + reformat discovered binares in Table 9
colnames = ['KOI','Teff_A', 'pm_Teff_A','Teff_A_err','Teff_B', 'pm_Teff_B','Teff_B_err','FB_over_FA',
'pm_FB_over_FA','FB_over_FA_err','dRV','Planetary_Data_N','Planetary_Data_Status']
colnames_to_keep = ['KOI','Teff_A','Teff_A_err','Teff_B','Teff_B_err', 'FB_over_FA',
'FB_over_FA_err','dRV','Planetary_Data_N','Planetary_Data_Status']
kolbl_binaries = pd.read_csv('./data/literature_data/Kolbl2015_Table9.csv', 
    delimiter=' ', skiprows=1, names=colnames)[colnames_to_keep]
id_starnames = ['K0'+str(value).zfill(4) for value in kolbl_binaries.KOI]
# add column with starnames to match training set table
kolbl_binaries['id_starname'] = id_starnames

# write clipped wavelength data to reference file
original_w_filename = './data/cks-spectra/cks-k00001_rj122.742.fits' # can be any r chip file
w_data = read_hires_fits(original_w_filename).w[:,:-1] # require even number of elements
w_data = w_data[:, dwt.order_clip:-1*dwt.order_clip] # clip 5% on each side
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

    # compute average pixel error, remove if snr<20
    # note: I needed to use the photon noise instead of 
    # target.serr, since those errors are unreasonably small
    snr = np.nanmean(target.s)/np.nanmean(np.sqrt(target.s))
    if snr < 20:
        low_sigma_idx_to_remove.append(i)
cks_stars = cks_stars.drop(cks_stars.index[low_sigma_idx_to_remove])
print(len(cks_stars), ' after removing spectra with per pixel SNR < 20')

# remove unresolved binaries from Kraus 2016
kraus2016_binaries = cks_stars[cks_stars['id_starname'].isin(kraus_binaries['id_starname'])]
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(kraus_binaries['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kraus 2016')

# remove unresolved binaries from Kolbl 2015
kolbl2015_binaries = cks_stars[cks_stars['id_starname'].isin(kolbl_binaries['id_starname'])]
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(kolbl_binaries['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kolbl 2015')

# remove KOI-2864, which seems to have some RV pipeline processing errors
cks_stars = cks_stars[~cks_stars.id_starname.isin(['K02864'])]
print(len(cks_stars), ' after removing stars with processing errors')

# write to .csv file
#trimmed(cks_stars).to_csv(label_path+'training_labels.csv', index=False)
print('training labels saved to .csv file')

# write binaries to file, perserving the source information
kraus2016_binaries['source']='Kraus2016'
kolbl2015_binaries['source']='Kolbl2015'
known_binaries = pd.concat((kraus2016_binaries, kolbl2015_binaries))
trimmed(known_binaries).to_csv(label_path+'known_binary_labels.csv', index=False)
print('saved binary labels to .csv')

# ============ write training flux, sigma to files  ================================================

# re-write code below to save data for particular order
def single_order_training_data(order_idx, filter_wavelets=True):

    # order numbers are not zero-indexed
    order_n = order_idx + 1

    # places to store data
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



