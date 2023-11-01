"""
This code updates cks_physical_merged.csv to include a column with the 
spectrum fileroot so that targets can be matches to CKS spectra.
The information is stored in the 'spectrum_fileroot' column
of cks_physical_merged_with_fileroots.csv
"""
import glob
from astropy.io import fits
import pandas as pd

# load spectrum filenames
cks_filenames = glob.glob('./data/cks-spectra/*.fits')

# load table with CKS properties from CKS website
cks_stars = pd.read_csv('./data/literature_data/cks_physical_merged.csv')
cks_stars = cks_stars.drop_duplicates('id_kic')

# store target names from file headers, file roots
id_starnames = []
spectrum_fileroots = []
for filename in cks_filenames:
    file = fits.open(filename)
    id_starname = file[0].header['TARGNAME']
    id_starname = id_starname.replace('CK', 'K')
    id_starname = id_starname.replace('k', 'K')
    spectrum_fileroot = filename.split('j')[1].split('.fits')[0]
    id_starnames.append(id_starname)
    spectrum_fileroots.append(spectrum_fileroot)


# test to confirm that all the stars in cks_stars have spectra!
n_not_found = 0
for id_starname in cks_stars.id_starname.to_numpy():
    if id_starname not in id_starnames:
        print('WARNING: target ID {} in table missing from spectra:'.format(
        id_starname))
        n_not_found +=1
        
if n_not_found==0:
    print('test passed, spectrum identified for all stars in CKS table')

# generate dataframe containing fileroot information
fileroot_df = pd.DataFrame({'id_starname': id_starnames, 
                                   'spectrum_fileroot': spectrum_fileroots})
fileroot_df = fileroot_df.drop_duplicates()

# merge with original table and save to file
cks_stars_with_fileroots = pd.merge(cks_stars, fileroot_df, on='id_starname')
cks_stars_with_fileroots.to_csv('./data/literature_data/cks_physical_merged_with_fileroots.csv')