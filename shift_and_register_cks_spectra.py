"""
This code uses Specmatch-Emp to shift all the raw CKS r chip spectra 
and rescale them onto the SpecMatch-Emp library wavelength scale. 
This wavelength scale goes from λ = 4990–6410 Å and is  uniform in 
Δlog λ.
"""
import glob
import numpy as np
import specmatchemp.library
from specmatchemp import spectrum
from specmatchemp.specmatch import SpecMatch

# load specmatch library
lib = specmatchemp.library.read_hdf()

# iterate over HIRES spectra
spectrum_filenames = glob.glob('./data/cks-spectra/*rj*.fits')
for filename in spectrum_filenames:
	# load spectrum
	hires_spectrum = spectrum.read_hires_fits(filename)

	# create specmatch object + shift
	sm_hires = SpecMatch(hires_spectrum, lib)
	sm_hires.shift()
	shifted_spec = sm_hires.target

	# extend spectrum to correct size for rescaling
	extended_w = np.linspace(shifted_spec.w[0], shifted_spec.w[-1], len(lib.wav))
	extended_spec =  shifted_spec.extend(extended_w)

	# resample spectrum onto library wavelength
	resampled_spec = extended_spec.rescale(lib.wav)

	# save to file
	resampled_filename = './data/cks-spectra_shifted_resampled_r/' + resampled_spec.name +'_resampled.fits'
	resampled_spec.to_fits(resampled_filename)
	print('saved resampled spectrum saved to {}'.format(resampled_filename))