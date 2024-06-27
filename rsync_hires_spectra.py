import specmatchemp.library
import pandas as pd
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

# copy over CKS stars
for index, row in cks_stars.iterrows():
    # filenames for rsync command
    fits_filename = 'rj{}.fits'.format(row.spectrum_fileroot)
    object_name = row.id_starname
    # write command
    command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/hires-spectra/{}.fits".format(
        fits_filename,
        object_name)
    if index>3:
        break
    # run the command using pexpect
    program = pexpect.spawn(command)
    program.expect("observer@cadence.caltech.edu's password:")
    program.sendline(password)
    program.expect(pexpect.EOF)
    print('copied {} to ./data/hires-spectra/'.format(object_name))

# copy over CKS-cool stars
for index, row in cks_cool_stars.iterrows():
    # filenames for rsync command
    fits_filename = row.lib_obs+'.fits'
    object_name = row.cps_name
    # write command
    command = "rsync observer@cadence.caltech.edu:/mir3/iodfitsdb/{} ./data/hires-spectra/{}.fits".format(
        fits_filename,
        object_name)
    if index>234:
        break
    # run the command using pexpect
    program = pexpect.spawn(command)
    program.expect("observer@cadence.caltech.edu's password:")
    program.sendline(password)
    program.expect(pexpect.EOF)
    print('copied {} to ./data/hires-spectra/'.format(object_name))