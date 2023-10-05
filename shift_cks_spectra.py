import glob
import subprocess
import specmatchemp.spectrum
import numpy as np
from astropy.io import fits

# shift spectra with specmatch command line interface
spectrum_paths = glob.glob('./data/cks-spectra/*ij*.fits')
for spectrum_path in spectrum_paths:
	subprocess.run(['smemp','shift','-d','./data/cks-spectra','-o','./data/cks-spectra_shifted_i','-f',spectrum_path])

print('')
print('finished shifting spectra')
print('resampling spectra...')

# load wavelength scale
w_to_resample_to = fits.open('./data/w_to_resample_to_i_chip.fits')[0].data

# resample spectra onto uniform wavelength scale
shifted_spectrum_paths = glob.glob('./data/cks-spectra_shifted_i/*ij*_adj.fits')
for spectrum_path in shifted_spectrum_paths:
	# load spectrum
	shifted_spec = specmatchemp.spectrum.read_fits(spectrum_path, wavlim=None)
	# extend to correct size and rscale to correct wavelengths
	ext_spec =  shifted_spec.extend(np.linspace(shifted_spec.w[0], shifted_spec.w[-1], len(w_to_resample_to)))
	resampled_spec = ext_spec.rescale(w_to_resample_to)
	# save to file
	resampled_filename = './data/cks-spectra_shifted_resampled_i/' + resampled_spec.name+'_resampled.fits'
	resampled_spec.to_fits(resampled_filename)
	print('saved resampled spectrum saved to {}'.format(resampled_filename))
