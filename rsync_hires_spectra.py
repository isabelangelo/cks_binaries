import specmatchemp.library
import pandas as pd
import os
import pexpect

# table with CKS stars
cks_stars = pd.read_csv(
    './data/literature_data/cks_physical_merged_with_fileroots.csv',
    dtype={'spectrum_fileroot': str}) # retains trailing zeros in spectrum fileroot

# table with CKS-cool stars
lib = specmatchemp.library.read_hdf()
min_teff = 3100 # corresponding to q=0.2 for Sun-like star
max_teff = cks_stars.cks_steff.min()
cks_cool_stars = lib.library_params.query('@min_teff<Teff & Teff<@max_teff')

# store password for rsync
password = "CPS<3RVs"

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
    fits_filename = 'rj{}.fits'.format(row.spectrum_fileroot)
    object_name = row.id_starname
    if os.path.exists('./data/hires-spectra_r/'+object_name+'.fits'):
        print('{} already in ./data/hires-spectra_r/'.format(object_name))
        pass
    else:
        # write command
        command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/hires-spectra_r/{}.fits".format(
            fits_filename,
            object_name)
        run_rsync(command)
        print('copied {} to ./data/hires-spectra_r/'.format(object_name))

# copy over CKS-cool stars
for index, row in cks_cool_stars.iterrows():
    # filenames for rsync command
    fits_filename = row.lib_obs+'.fits'
    object_name = row.cps_name
    if os.path.exists('./data/hires-spectra_r/'+object_name+'.fits'):
        print('{} already in ./data/hires-spectra_r/'.format(object_name))
        pass
    else:
        # write command
        command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/hires-spectra_r/{}.fits".format(
            fits_filename,
            object_name)
        run_rsync(command)
        print('copied {} to ./data/hires-spectra_r/'.format(object_name))


# copy over Kepler-1656 spectra to test wavelet filtering
for fits_filename in ['rj351.570.fits', 'rj487.76.fits']:
    object_name = 'K00367'
    object_filename = './data/kepler1656_spectra/{}_{}'.format(object_name, fits_filename)
    if os.path.exists(object_filename):
        print('{} already in ./data/kepler1656_spectra/'.format(object_name))
        pass
    else:
        command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} {}".format(
            fits_filename,
            object_filename)
        run_rsync(command)
        print('copied {} to ./data/kepler1656_spectra/'.format(object_name))



