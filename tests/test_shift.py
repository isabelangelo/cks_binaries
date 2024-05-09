import glob
import numpy as np
import specmatchemp.library
from specmatchemp import spectrum
from specmatchemp.specmatch import SpecMatch

print('loading library')

# load specmatch library
lib = specmatchemp.library.read_hdf()

print('gathering filenames')

# shift spectrum
spectrum_filenames = glob.glob('../data/cks-spectra/*rj*.fits')

def shift_spectrum(spectrum_filename):
	hires_spectrum = spectrum.read_hires_fits(spectrum_filename)
	sm_hires = SpecMatch(hires_spectrum, lib)
	sm_hires.shift()

print('shifting spectra')

shift_spectrum(spectrum_filenames[0])
# shift_spectrum(spectrum_filenames[1])

