"""
This code combines stars from CKS and CKS-cool, 
saves their labels to a file 
and transfers the spectra from cadence to the local machine
"""
from astropy.table import Table
from astropy.io import fits
import specmatchemp.library
import pandas as pd
import os
import pexpect

# table with CKS stars
cks_main_stars_path = '../cks_binaries/data/literature_data/Petigura2017_Table5.fits'
cks_main_stars = Table(fits.open(cks_main_stars_path)[1].data).to_pandas()
print(len(cks_main_stars), 'stars from CKS')

# remove KOI-2864, which seems to have some RV pipeline processing errors
cks_main_stars = cks_main_stars[~cks_main_stars.Name.str.contains('KOI-02864')]
print(len(cks_main_stars), ' after removing KOI-02864 due to processing errors')

# table with CKS-cool stars
lib = specmatchemp.library.read_hdf()
min_teff = 3300 # secondary with >1% flux contribution
max_teff = cks_main_stars['Teff'].min()
cks_cool_stars = lib.library_params.query('@min_teff<Teff & Teff<@max_teff')
print(len(cks_cool_stars), 'stars from CKS-cool')

# store password for rsync
password = "CPS<3RVs"

# rename columns of CKS sample
cks_cols_to_keep = ['Name', 'Obs','Teff', 'e_Teff', 'logg', 'e_logg', \
                    '[Fe/H]', 'e_[Fe/H]','vsini', 'e_vsini']
cks_main_stars = cks_main_stars[cks_cols_to_keep].rename(
    columns={
    "Name": "id_starname", 
    "Obs": "obs_id",
    "Teff": "cks_teff",
    "e_Teff": "cks_teff_err",
    "logg":"cks_logg",
    "e_logg": "cks_logg_err", 
    "[Fe/H]": "cks_feh",
    "e_[Fe/H]": "cks_feh_err",
    "vsini": "cks_vsini",
    "e_vsini": "cks_vsini_err"})
cks_main_stars['sample'] = ['cks'] * len(cks_main_stars)
# re-format star names to be consistent with filenames
cks_main_stars.id_starname = [i.replace('KOI-', 'K').replace(' ', '') for i in cks_main_stars.id_starname]

# rename columns of CKS-cool
cks_cool_cols_to_keep = ['cps_name', 'lib_obs','Teff', 'u_Teff', 'logg', 'u_logg', \
                    'feh', 'u_feh','vsini']
cks_cool_stars = cks_cool_stars[cks_cool_cols_to_keep].rename(
    columns={
    "cps_name": "id_starname", 
    "lib_obs": "obs_id",
    "Teff": "cks_teff",
    "u_Teff": "cks_teff_err",
    "logg": "cks_logg", 
    "u_logg": "cks_logg_err",
    "feh": "cks_feh",
    "u_feh": "cks_feh_err",
    "vsini": "cks_vsini"})
cks_cool_stars['sample'] = ['cks-cool'] * len(cks_cool_stars)

# combine samples for training set
cks_stars = pd.concat([cks_main_stars, cks_cool_stars], ignore_index=True)
# re-format obs ids
cks_stars.obs_id = [i.replace(' ','') for i in cks_stars.obs_id]

# save to file
cks_stars_filename = './data/label_dataframes/cks_stars.csv'
cks_stars.to_csv(cks_stars_filename)
print('table with CKS + CKS-cool stars ({} total) saved to {}'.format(
    len(cks_stars),
    cks_stars_filename))

# function to run rsync command
def run_rsync(command):
    # run the command using pexpect
    program = pexpect.spawn(command)
    program.expect("observer@cadence.caltech.edu's password:")
    program.sendline(password)
    program.expect(pexpect.EOF)

# copy over CKS stars
for index, row in cks_stars.iterrows():
    # filenames for rsync command
    obs_ids = [row.obs_id.replace('rj','bj'), row.obs_id, row.obs_id.replace('rj','ij')]
    for obs_id in obs_ids:
        obs_filename = obs_id+'.fits'
        if os.path.exists('./data/cks-spectra/'+obs_filename):
            print('{} already in ./data/cks-spectra/'.format(obs_filename))
            pass
        else:
            # write command
            command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/cks-spectra/{}".format(
                obs_filename,
                obs_filename)
            run_rsync(command)
    print('copied {} b,r,i chip spectra to ./data/cks-spectra/'.format(row.id_starname))

# copy over Kepler-1656 spectra to test wavelet filtering
for obs_n in ['rj351.570', 'rj487.76']:
    obs_ids = [obs_n.replace('rj','bj'), obs_n, obs_n.replace('rj','ij')]
    for obs_id in obs_ids:
        obs_filename = obs_id+'.fits'
        if os.path.exists('./data/kepler1656_spectra/'+obs_filename):
            print('{} already in ./data/kepler1656-spectra/'.format(obs_filename))
            pass
        else:
            # write command
            command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/kepler1656-spectra/{}".format(
                obs_filename,
                obs_filename)
            run_rsync(command)
    print('copied Kepler-1656 b,r,i chip spectra to ./data/kepler1656-spectra/')




