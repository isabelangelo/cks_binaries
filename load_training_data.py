"""
Loads labels + HIRES spectra for the Cannon training and test sets.
"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.spectrum import read_fits
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
original_path = './data/cks-spectra'
shifted_path = './data/cks-spectra_shifted'
df_path = './data/cannon_training_data'
label_path = './data/label_dataframes'

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
k16_tbl = pd.concat([
    load_kraus_table('Kraus2016_Table3.csv'), # NRM
    load_kraus_table('Kraus2016_Table5.csv'),  # aperture photometry
    load_kraus_table('Kraus2016_Table6.csv')]) # multi-PSF fitting

# query targets where any separation within reported uncertainties
# falls within hires slit width of 0.8arcsec
hires_slit_width = 800 # 0.8arcsec = 800mas
k16_tbl = k16_tbl.query('sep_mas - sep_err < @hires_slit_width')

# add column with starnames to match training set table
k16_tbl['id_starname'] = k16_tbl['KOI'].str.replace('KOI-', 'K0')

# load binary data from Kolbl 2016
# load + reformat discovered binares in Table 9
colnames = ['KOI','Teff_A', 'pm_Teff_A','Teff_A_err','Teff_B', 'pm_Teff_B','Teff_B_err','FB_over_FA',
'pm_FB_over_FA','FB_over_FA_err','dRV','Planetary_Data_N','Planetary_Data_Status']
colnames_to_keep = ['KOI','Teff_A','Teff_A_err','Teff_B','Teff_B_err', 'FB_over_FA',
'FB_over_FA_err','dRV','Planetary_Data_N','Planetary_Data_Status']
k15_tbl = pd.read_csv('./data/literature_data/Kolbl2015_Table9.csv', 
    delimiter=' ', skiprows=1, names=colnames)[colnames_to_keep]
id_starnames = ['K0'+str(value).zfill(4) for value in k15_tbl.KOI]

# add column with starnames to match training set table
k15_tbl['id_starname'] = id_starnames

# write clipped wavelength data to reference file
original_wav_file = read_hires_fits('./data/cks-spectra/rj122.742.fits') # KOI-1 original r chip file
original_wav_data = original_wav_file.w[:,:-1] # require even number of elements
wav_data = original_wav_data[:, dwt.order_clip:-1*dwt.order_clip] # clip 5% on each side
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
fits.HDUList([fits.PrimaryHDU(wav_data)]).writeto(reference_w_filename, overwrite=True)
print('clipped reference wavlength saved to {}'.format(reference_w_filename))

# ============ clean training set data  =========================================

# load table with CKS properties from CKS website
cks_stars = pd.read_csv('./data/label_dataframes/cks_stars.csv') 
print(len(cks_stars), ' table entries from CKS + CKS-cool')

# re-format vsini column
cks_stars['cks_vsini'] = cks_stars['cks_vsini'].replace('--', np.nan)
cks_stars['cks_vsini'] = cks_stars['cks_vsini'].astype(float)
# set nan vsini for cool stars to 2±2 km/s
cks_stars = cks_stars.fillna(value={"cks_vsini": 2.0, "cks_vsini_err": 2.0})

# temporary: remove stars from cks-cool
cks_stars = cks_stars.query('cks_teff>=4200')
print(len(cks_stars), ' after removing stars with Teff<4200 (TEMPORARY)')

# remove duplicate targets
cks_stars = cks_stars.drop_duplicates('id_starname')
print(len(cks_stars), 'remaining after removing duplicate entries')

# require finite training set labels
training_set_labels = ['cks_teff', 'cks_logg', 'cks_feh', 'cks_vsini']
cks_stars = cks_stars.dropna(subset=training_set_labels)
print(len(cks_stars), 'with finite training set labels')

# remove stars with low SNR
low_sigma_idx_to_remove = []
for i in range(len(cks_stars)):

    # load file data
    row = cks_stars.iloc[i]
    filename = '{}/{}.fits'.format(
        original_path, 
        # row.obs_id.replace('rj','ij')) for i chip
    target = read_hires_fits(filename)

    # compute average pixel error, remove if snr<20
    # note: I needed to use the photon noise instead of 
    # target.serr, since those errors are unreasonably small
    snr = np.nanmean(target.s)/np.nanmean(np.sqrt(target.s))
    if snr < 20:
        low_sigma_idx_to_remove.append(i)
cks_stars = cks_stars.drop(cks_stars.index[low_sigma_idx_to_remove])
print(len(cks_stars), ' after removing spectra with per pixel SNR < 20')

# TEMPORARY: remove stars with i chip shifing errors
# shifting_error_ids = np.array([
#     'K00006', 'K00176', 'K00201', 'K00297', 'K00301', 'K00308',
#     'K00312', 'K00523', 'K00659', 'K00673', 'K00710', 'K01282',
#     'K01444', 'K01806', 'K01922', 'K01984', 'K02109', 'K02195',
#     'K02250', 'K02260', 'K02358', 'K02623', 'K02749', 'K03060',
#     'K03065', 'K03122', 'K03158', 'K03315', 'K03425', 'K03943',
#     'K04157', 'K04159', 'K04215', 'K04323', 'K04367', 'K04505',
#     'K04588', 'K04601', 'K04716', 'K04771', 'K04822', 'K05236',
#     'KIC11187332', 'GL570B'], dtype=object)
# cks_stars = cks_stars[~cks_stars.id_starname.isin(shifting_error_ids)]
# print(len(cks_stars), ' after removing stars with specmatch shifting errors')


# ============ store binaries in separate files  =========================================

# remove unresolved binaries from Kraus 2016
kraus2016_binaries = cks_stars[cks_stars['id_starname'].isin(k16_tbl['id_starname'])]
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(k16_tbl['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kraus 2016')

# remove unresolved binaries from Kolbl 2015
kolbl2015_binaries = cks_stars[cks_stars['id_starname'].isin(k15_tbl['id_starname'])]
# update training labels
cks_stars = cks_stars[~cks_stars['id_starname'].isin(k15_tbl['id_starname'])]
print(len(cks_stars), ' after removing unresolved binaries from Kolbl 2015')

# remove KOI-2864, which seems to have some RV pipeline processing errors
cks_stars = cks_stars[~cks_stars.id_starname.isin(['K02864'])]
print(len(cks_stars), ' after removing stars with processing errors')
print(cks_stars.cks_vsini.min())

# ============ write tables to files  =========================================

import pdb;pdb.set_trace()

# write to .csv file
trimmed(cks_stars).to_csv(label_path+'/training_labels.csv', index=False)
print('training labels saved to .csv file')

# write binaries to file, perserving the source information
kraus2016_companions = pd.merge(k16_tbl, kraus2016_binaries)
trimmed(kraus2016_companions).to_csv(
    label_path+'/kraus2016_binary_labels.csv', index=False)
print('{} CKS targets ({} <0.8" companions) from Kraus 2016 saved to .csv'.format(
    len(kraus2016_binaries), len(kraus2016_companions)))

kolbl2015_companions = pd.merge(k15_tbl, kolbl2015_binaries)
trimmed(kolbl2015_companions).to_csv(
    label_path+'/kolbl2015_binary_labels.csv', index=False)
print('{} CKS targets ({} SB2 copmanions) from Kolbl 2015 saved to .csv'.format(
    len(kolbl2015_binaries), len(kolbl2015_companions)))

# ============ write training flux, sigma to files  ================================================

# save data for particular order
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
        filename = '{}/{}_adj.fits'.format(
            shifted_path, 
            # row.obs_id.replace('rj','ij')) for i chip
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

    return flux_df_n, sigma_df_n

# write wavelet filtered training set flux, sigma to files
flux_df_dwt = pd.DataFrame()
sigma_df_dwt = pd.DataFrame()
for order_idx in range(0, 10):
    flux_df_n, sigma_df_n = single_order_training_data(order_idx)
    flux_df_dwt = pd.concat([flux_df_dwt, flux_df_n])
    sigma_df_dwt = pd.concat([sigma_df_dwt, sigma_df_n])
flux_df_dwt.to_csv('{}/training_flux_dwt.csv'.format(df_path), index=False)
sigma_df_dwt.to_csv('{}/training_sigma_dwt.csv'.format(df_path), index=False)
print('wavelet-filtered training flux and sigma saved to .csv files')

# write original training set flux, sigma to files
flux_df_original = pd.DataFrame()
sigma_df_original = pd.DataFrame()
for order_idx in range(0, 10):
    flux_df_n, sigma_df_n = single_order_training_data(order_idx, filter_wavelets=False)
    flux_df_original = pd.concat([flux_df_original, flux_df_n])
    sigma_df_original = pd.concat([sigma_df_original, sigma_df_n])
flux_df_original.to_csv('{}/training_flux_original.csv'.format(df_path), index=False)
sigma_df_original.to_csv('{}/training_sigma_original.csv'.format(df_path), index=False)
print('training flux and sigma pre wavelet filter saved to .csv files')



