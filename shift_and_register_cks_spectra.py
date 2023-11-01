"""
This code shifts all the raw CKS r chip spectra and registers them 
onto a new wavelength scale. The spectra are first shifted with 
specmatch. Then, since the orders are not separated in wavelength 
space, the wavelength is sampled uniformly between the maximum 
wavelength range spanned by all spectra (with a 1 angstrom pad), 
preserving the length of the entire spectrum.
The spectra are then resampled onto the computed wavelength array
with specmatch.
"""
from astropy.io import fits
import numpy as np
import glob
import subprocess
import specmatchemp.spectrum
import numpy as np
from astropy.io import fits

# shift spectra with specmatch command line interface
# spectrum_paths = glob.glob('./data/cks-spectra/*rj*.fits')
# for spectrum_path in spectrum_paths:
# 	subprocess.run(['smemp','shift','-d','./data/cks-spectra','-o','./data/cks-spectra_shifted_r','-f',spectrum_path])

# print('')
# print('finished shifting spectra')

# information about r chip spectra
shifted_filenames = glob.glob('./data/cks-spectra_shifted_r/*rj*.fits')
n_orders = 16 # HIRES r chip
n_pixels = 4021 # pixels per order in HIRES spectrum

# compute wavelength solution to resample to
spectrum_start = []
spectrum_end = [] 
for filename in shifted_filenames:
	file_data = fits.open(filename)[1].data
	w = file_data['w']
	s = file_data['s']
	spectrum_start.append(w[0])
	spectrum_end.append(w[-1])

# compute maximum wavelength range spanned by spectrum
# padded by 0.1angstrom to avoid interpolating between nan and finite fluxes
spectrum_min = np.max(spectrum_start) + 0.1
spectrum_max = np.min(spectrum_end) - 0.1
print('full wavelength range : ', spectrum_min, spectrum_max)

# sample wavelength values to interpolate to
w_to_resample_to = np.linspace(spectrum_min, spectrum_max, n_pixels*n_orders)

# write to fits file
w_filename = 'w_to_resample_to_r_chip.fits'
w_hdu = fits.PrimaryHDU(w_to_resample_to)
w_hdul = fits.HDUList([w_hdu])
w_hdul.writeto(w_filename, overwrite=True)
print('wavelength array saved to {}'.format(w_filename))


# resample spectra onto the same wavelength scape
print('resampling spectra...')

# load wavelength scale
# w_to_resample_to = fits.open('./data/w_to_resample_to_r_chip.fits')[0].data

# resample spectra onto uniform wavelength scale
shifted_spectrum_paths = glob.glob('./data/cks-spectra_shifted_r/*rj*.fits')
for spectrum_path in shifted_spectrum_paths:
	# load spectrum
	shifted_spec = specmatchemp.spectrum.read_fits(spectrum_path, wavlim=None)
	# extend to correct size and rscale to correct wavelengths
	ext_spec =  shifted_spec.extend(np.linspace(shifted_spec.w[0], shifted_spec.w[-1], len(w_to_resample_to)))
	resampled_spec = ext_spec.rescale(w_to_resample_to)
	# save to file
	resampled_filename = './data/cks-spectra_shifted_resampled_r/' + resampled_spec.name+'_resampled.fits'
	resampled_spec.to_fits(resampled_filename)
	print('saved resampled spectrum saved to {}'.format(resampled_filename))