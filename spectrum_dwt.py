# to do : change plots to use gridspec

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pywt

# load data
w_full = fits.open('./data/w_to_resample_to_i_chip.fits')[0].data
idx = (w_full>6670) & (w_full<6785)
w = w_full[idx]
example_flux = fits.open(
    './data/kepler1656_spectra/CK00367_2019_ij351.570_adj_resampled.fits')[1].data['s'][idx]

# normalize flux + require even number of elements
# (this may be replaced with an interpolation to accommodate more orders)
w = w[:-1]
example_flux = example_flux[:-1] - 1

# define tranform function inputs
wt_kwargs = {'mode':'zero', 'axis':-1}


# select wavelet type
# note: use pywt.wavelist(kind='discrete') to see wavelet options
# wavelet_wavedec = 'sym2' # chooses 2.2
# print('selected wavelet {}'.format(wavelet_wavedec))

# n_wavedec_levels = pywt.dwt_max_level(len(example_flux), wavelet_wavedec)
# print('maximum levels allowed in decomposition is {}'.format(n_wavedec_levels))

# function to compute inverse transform of single level
def flux_waverec(flux, wavelet_wavedec, levels):
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	coeffs = pywt.wavedec(flux, wavelet_wavedec, level = max_level, **wt_kwargs)
	# set coefficients to zero for all other levels
	all_levels = range(0, max_level+1) # iterate over all levels, including last
	for level in all_levels:
		if level not in levels:
			coeffs[level] = np.zeros_like(coeffs[level])
	# perform wavelet recomposition
	flux_waverec = pywt.waverec(coeffs, wavelet_wavedec, **wt_kwargs)
	return flux_waverec


# plot the difference between the original + inverse transform
def plot_flux_waverec_residuals(flux, wavelet_wavedec, object_name):
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	all_level_flux_waverec = flux_waverec(flux, np.arange(0, max_level+1))
	diff = flux - all_level_flux_waverec
	print(np.mean(abs(diff)))
	plt.figure(figsize=(15,5))
	plt.subplot(211);plt.title(object_name)
	plt.plot(w, flux, color='k', label='original spectrum')
	plt.plot(w, all_level_flux_waverec, 'r--', label='IDWT(wavelet coefficients)')
	plt.ylabel('normalized flux');plt.legend(frameon=False)
	plt.subplot(212)
	plt.plot(w, diff, 'k-')
	plt.xlabel('wavelength (angstrom)')
	plt.ylabel('residuals')
	path = './figures/{}_waverec_residuals.png'.format(wavelet_wavedec)
	plt.savefig(path, dpi=150)

# plot different orders
def plot_flux_waverec_levels(flux, wavelet_wavedec, object_name):
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	n_levels = max_level + 1
	fig, axes = plt.subplots(n_levels+1, 1, sharex=True, sharey=False, 
		figsize=(7,8), tight_layout=True)
	plt.rcParams['font.size']=8
	axes[0].plot(w, flux, color='k', lw=0.5)
	axes[0].text(6760, 0.1, 'original signal')
	axes[0].set_ylim(-0.6,0.4)
	for level in range(0, n_levels):
		level_flux_waverec = flux_waverec(flux, [level])
		axes[n_levels - level].plot(w, level_flux_waverec, 'k-', lw=0.5)
		#axes[n_levels - level].set_title('level = {}'.format(level), fontsize=8)
	fig.suptitle(object_name)
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')
	plt.subplots_adjust(hspace=0.4)
	plt.savefig('/Users/isabelangelo/Desktop/wavelet_levels_2019_bior22.png')
	plt.show()


flux_2019 = fits.open(
    './data/kepler1656_spectra/CK00367_2019_ij351.570_adj_resampled.fits')[1].data['s']
flux_2019 = flux_2019[idx][:-1] - 1

flux_2022 = fits.open(
    './data/kepler1656_spectra/CK00367_2022_ij487.76_adj_resampled.fits')[1].data['s']
flux_2022 = flux_2022[idx][:-1] - 1

# plot the difference between 2019 and 2022
def plot_waverec_level_diff(wavelet_wavedec):
	max_level = pywt.dwt_max_level(len(flux), wavelet_wavedec)
	fig, axes = plt.subplots(max_level+1, 2, sharex=True, sharey=False, 
	                         figsize=(14,10), tight_layout=True)

	for level in range(max_level+1):
	    level_2019 = flux_waverec(flux_2019, [level])
	    level_2022 = flux_waverec(flux_2022, [level])
	    axis_n = max_level-level
	    axes[axis_n, 0].plot(w, level_2019, color='orangered', alpha=0.7)
	    axes[axis_n, 0].plot(w, level_2022, color='cornflowerblue', alpha=0.7)
	    axes[axis_n, 0].set_ylim(-0.2,0.2)
	    axes[axis_n, 0].text(6715, 0.22, 'level = {}'.format(level))
	    if level == max_level:
	        axes[axis_n, 0].text(6670, 0.07, '2019', color='orangered')
	        axes[axis_n, 0].text(6670, -0.15, '2022', color='cornflowerblue')
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')


	for level in range(max_level+1):
	    level_2019 = flux_waverec(flux_2019, [level])
	    level_2022 = flux_waverec(flux_2022, [level])
	    level_diff = level_2019 - level_2022
	    rms = "{:0.2e}".format(np.sqrt(np.sum(level_diff**2)/len(level_diff)))
	    axis_n = max_level-level
	    axes[axis_n, 1].plot(w, level_diff, 'k-', alpha=0.7)
	    axes[axis_n, 1].set_ylim(-0.05,0.05)
	    axes[axis_n, 1].text(6705, 0.06, 'level = {}, rms = {}'.format(level, rms), color='firebrick')
	fig.supxlabel('wavelength (nm)')
	fig.supylabel('flux')

	axes[0,0].set_title('IDWT of orginal signal\n at each decomposition level', pad=30)
	axes[0,1].set_title('IDWT difference (2019 - 2022)', pad=30)
	fig.suptitle('wavelet = {}'.format(wavelet_wavedec))
	path = './figures/{}_level_diff.png'.format(wavelet_wavedec)
	plt.savefig(path, dpi=150)

plot_flux_waverec_residuals(flux_2019, wavelet_wavedec)
