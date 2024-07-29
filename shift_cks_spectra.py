"""
This code uses Specmatch-Emp to shift all the raw CKS r chip spectra 
and rescale them onto the original HIRES wavelength scale. 
"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.specmatch import SpecMatch
from specmatchemp import SPECMATCHDIR
import glob
import os

# paths to store shifted spectra
shifted_resampled_path = './data/cks-spectra_shifted'
kepler1656_path = './data/kepler1656_spectra'

# iterate over CKS spectra and shift
spectrum_ids = [i[20:-5] for i in glob.glob('./data/cks-spectra/ij*.fits')]
for spectrum_id in spectrum_ids:
    input_path = './data/cks-spectra'
    output_path = './data/cks-spectra_shifted'
    command = 'smemp shift -d {} -o {} {}'.format(
        input_path, 
        output_path, 
        spectrum_id)
    os.system(command)

# shift kepler-1656 spectra for wavelet diagnostics
print('shifting and registering Kepler-1656 spectra for diagnostics')
for filename in glob.glob('./data/kepler1656-spectra/ij*.fits'):
    spectrum_id = filename[27:-5]
    input_path = './data/kepler1656-spectra'
    output_path = './data/kepler1656-spectra'
    command = 'smemp shift -d {} -o {} {}'.format(
        input_path, 
        output_path, 
        spectrum_id)
    os.system(command)