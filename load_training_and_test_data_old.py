import pandas as pd
import numpy as np
import glob
from astropy.io import fits

# load table with CKS properties from CKS website
cks_stars = pd.read_csv('./data/cks_physical_merged.csv')
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


# # remove targets from Kraus 2016 sample
# kraus2016_targets = pd.read_csv('./CKS_binaries_old/data_files/Kraus2016_Table1.csv',
#                                delim_whitespace=True)
# kraus2016_target_id_starnames = [i.replace('OI-', '0') for i in kraus2016_targets.KOI.to_numpy()]

# kraus2016_target_idx_to_remove = []
# for i in range(len(cks_stars)):
#     row = cks_stars.iloc[i]
#     if row.id_starname in kraus2016_target_id_starnames:
#         kraus2016_target_idx_to_remove.append(i)
# cks_stars = cks_stars.drop(cks_stars.index[kraus2016_target_idx_to_remove])
# print(len(cks_stars), ' remaining after removing Kraus 2016 targets')

# store paths to fluxes
def fileroot(id_starname):
    glob_filepath = './data/cks-spectra_shifted_resampled_i/'
    glob_filename = 'cks-{}_ij*.fits'.format(id_starname.replace('K', 'k'))
    filename = glob.glob(glob_filepath+glob_filename)[0]
    fileroot = filename.split('/')[3][:-19]
    return fileroot
cks_stars['spectrum_fileroot'] = [fileroot(i) for i in cks_stars.id_starname.to_numpy()]

# remove stars with low SNR
low_sigma_idx_to_remove = []
path = './data/cks-spectra_shifted_resampled_i/'
for i in range(len(cks_stars)):
    row = cks_stars.iloc[i]
    filename = path + row.spectrum_fileroot + '_adj_resampled.fits'
    file = fits.open(filename)[1].data
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

	# store flux, sigma from initial files
    row = cks_stars.iloc[i]
    filename = path + row.spectrum_fileroot + '_adj_resampled.fits'
    file = fits.open(filename)[1].data
    flux = file['s']
    sigma = file['serr']

    # remove nans from flux, sigma
    # note: this needs to happen here so that the Cannon
    # always returns flux values for all wavelengths
    flux = np.nan_to_num(flux, nan=1)
    sigma = np.nan_to_num(sigma, nan=1)
    
    # save to lists
    flux_list.append(np.array(flux))
    sigma_list.append(np.array(sigma))
    id_starname_list.append(row.id_starname)
    
flux_df = pd.DataFrame(dict(zip(id_starname_list, flux_list)))
sigma_df = pd.DataFrame(dict(zip(id_starname_list, sigma_list)))

flux_df.to_csv('./training_flux.csv')
sigma_df.to_csv('./training_sigma.csv')
cks_stars.to_csv('./training_labels.csv')
print('training flux and sigma saved to .csv files')
print('training labels saved to training_labels.csv')

# write training + test data to fits files
flux_filename = 'training_flux.fits'
sigma_filename = 'training_sigma.fits' 
flux_arr = flux_df.to_numpy().T
sigma_arr = sigma_df.to_numpy().T

fits.HDUList([fits.PrimaryHDU(flux_arr)]).writeto(flux_filename, overwrite=True)
print('training flux array saved to {}'.format(flux_filename))

fits.HDUList([fits.PrimaryHDU(sigma_arr)]).writeto(sigma_filename, overwrite=True)
print('training sigma array saved to {}'.format(sigma_filename))







