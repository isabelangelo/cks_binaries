# to do : there are some variables that are defined twice, 
# I need to make sure that's fixed
# I think this will be solved when I write an object
# to do : I think I should work in scale, not fourier equivalent period, since it's more directly 
# related to the wavelet transform
# to do: experiement with no de-trending
# to do : if I end up using this a lot, I should write a HIRESSPectrum
# object that stores all of this information

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
import matplotlib.gridspec as gridspec

# load data for all sections
# note: changing this might also require changing wavelet parameters
# in wavelet tranform function (e.g., dj, s0, J)
wavelet_to_use = wavelet.MexicanHat()

def compute_wavelet_transform(w, flux):
	# store variables with same names as tutorial
	t = w
	dt = w[1]-w[0]
	t0 = w[0]
	dat = flux

	# normalize data
	p = np.polyfit(t - t0, dat, 1)
	dat_notrend = dat - np.polyval(p, t - t0)
	std = dat_notrend.std()  # Standard deviation
	var = std ** 2  # Variance
	dat_norm = dat_notrend / std  # Normalized dataset
	N = dat.size

	# The next step is to define some parameters of our wavelet analysis. We
	# select the mother wavelet, in this case the Morlet wavelet with
	# :math:`\omega_0=6`.
	mother = wavelet_to_use
	dj = 1 / 12  # Twelve sub-octaves per octaves
	s0 = 1 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
	J = 12 / dj  # 11 powers of two with dj sub-octaves (same as swt)
	alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

	# compute wavelet transform data
	wavelet_transform = wavelet.cwt(dat_norm, dt, dj, s0, J,
                                                      mother)
	wave, scales, freqs, coi, fft, fftfreqs = wavelet_transform
	iwave = wavelet.icwt(wave, scales, dt, dj, mother)

	# return relevant components
	return t, dat_norm, wavelet_transform, iwave


def plot_inverse_transform(w, flux, sup_title_str):
	# compute wavelet transform, inverse transform
	t, dat_norm, wavelet_transform, iwave = compute_wavelet_transform(w, flux)
	# make figure
	plt.figure(figsize=(10,7))
	plt.subplot(311)
	plt.plot(t, dat_norm, 'k', linewidth=1.5, label='original spectrum')
	plt.plot(t, iwave, 'r--', linewidth=1, label='inverse wavelet transform')
	plt.title(sup_title_str)
	plt.ylabel('normalized flux')
	plt.legend(frameon=False)
	plt.subplot(312)
	plt.plot(t, dat_norm, 'k', linewidth=1.5)
	plt.plot(t, iwave, 'r', linewidth=1, alpha=0.5)
	plt.ylim(-1,1)
	plt.subplot(313)
	plt.plot(t, dat_norm - iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
	plt.ylabel('original - inverse transform');plt.xlabel('wavelength (angstrom)')
	plt.subplots_adjust(hspace=0)
	plt.savefig('/Users/isabelangelo/Desktop/'+sup_title_str+'_inverse_transform.png', 
		dpi=300, bbox_inches='tight')


def plot_wavelet_coefficients(w, flux, sup_title_str):
	# compute wavelet transform, inverse transform
	t, dat_norm, wavelet_transform, iwave = compute_wavelet_transform(w, flux)
	wave, scales, freqs, coi, fft, fftfreqs = wavelet_transform
	mother = wavelet_to_use

	# information for contour map
	dt = t[1]-t[0];dat=flux;N=dat.size;dj = 1 / 12
	power = (np.abs(wave)) ** 2 # normalized wavelet spectrum
	fft_power = np.abs(fft) ** 2 #  Fourier power spectrum
	period = 1 / freqs # Fourier equivalent periods for each wavelet scale.
	alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise
	signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
	                                         significance_level=0.95,
	                                         wavelet=mother)
	sig95 = np.ones([1, N]) * signif[:, None]
	sig95 = power / sig95 # power is significant where power/sig95>1
	levels = np.logspace(-4,4,base=2, num=50) # levels for onour plot

	# Prepare the figure
	figprops = dict(figsize=(10, 20), dpi=72)
	fig = plt.figure(**figprops)
	outer = gridspec.GridSpec(4, 1, wspace=0.2, hspace=0.4, height_ratios = [1,3,5,5])

	# First sub-plot, the original time series anomaly and inverse wavelet
	# transform.
	ax = plt.Subplot(fig, outer[0])
	ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
	ax.plot(t, dat_norm, 'k', linewidth=1.5)
	ax.set_title('{}'.format(sup_title_str))
	ax.set_ylabel(r'{} [{}]'.format(r'$f_{\lambda}$', 'counts'))
	fig.add_subplot(ax)

	# Second sub-plot, the global wavelet and Fourier power spectra and theoretical
	# noise spectra. Note that period scale is logarithmic.
	bx = plt.Subplot(fig, outer[1], sharex=ax)
	bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
	        extend='both', cmap=plt.cm.bone_r)
	extent = [t.min(), t.max(), 0, max(period)]
	bx.contour(t, np.log2(period), sig95, [-99, 1], colors='orangered', linewidths=2,
	           extent=extent)
	bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
	                           t[:1] - dt, t[:1] - dt]),
	        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
	                           np.log2(period[-1:]), [1e-9]]),
	        'k', alpha=0.8)
	#bx.set_title('{} Wavelet Power Spectrum ({})'.format(r'$f_{\lambda}$', mother.name))
	bx.set_title('{} Wavelet Power Spectrum'.format(r'$f_{\lambda}$'))
	bx.set_ylabel('Fouerier Equivalent Period (nm)')
	bx.set_ylim(np.log2(period).min(), np.log2(period).max())
	Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
	                           np.ceil(np.log2(period.max())))
	bx.set_yticks(np.log2(Yticks))
	bx.set_yticklabels(Yticks)
	fig.add_subplot(bx)

	# Third and Fourth sub-plot, slices of coefficients in scale space
	cx = plt.Subplot(fig, outer[2], sharex=ax);fig.add_subplot(cx);cx.set_yticks([])
	dx = plt.Subplot(fig, outer[3], sharex=ax);fig.add_subplot(dx);dx.set_yticks([])

	period_slice_edges = 2 ** np.arange(np.ceil(np.log2(period.min())),
	                       np.ceil(np.log2(period.max())),1)
	n_wavelet_orders = len(period_slice_edges)-1

	inner_cx = gridspec.GridSpecFromSubplotSpec(n_wavelet_orders, 1, subplot_spec=cx, wspace=0.1, hspace=0)
	inner_dx = gridspec.GridSpecFromSubplotSpec(n_wavelet_orders, 1, subplot_spec=dx, wspace=0.1, hspace=0)
	for i in range(n_wavelet_orders):
	    period_start = period_slice_edges[i]
	    period_stop = period_slice_edges[i+1]

	    period_slice_idx = (period >= period_start) & (period <= period_stop)
	    iwave_slice = wavelet.icwt(wave[period_slice_idx], scales[period_slice_idx], dt, dj, mother)
	    period_slice = period[period_slice_idx]

	    cx_i = plt.Subplot(fig, inner_cx[n_wavelet_orders-i-1])
	    cx_i.plot(t, iwave_slice, 'b-')
	    fig.add_subplot(cx_i)
	    cx_i.set_xticks([]);cx_i.set_yticks([])
	    slice_str = '{} - {} nm'.format(
	        np.round(period_start,1),
	        np.round(period_stop,1))
	    cx_i.set_ylabel(slice_str, rotation=0, labelpad=55)
	    cx_i.set_yticks([])
	    
	    dx_i = plt.Subplot(fig, inner_dx[n_wavelet_orders-i-1])
	    dx_i.plot(t, dat_norm - iwave_slice, 'b-')
	    fig.add_subplot(dx_i)
	    dx_i.set_ylabel(slice_str, rotation=0, labelpad=55)
	    dx_i.set_xticks([])

	plt.savefig('/Users/isabelangelo/Desktop/'+sup_title_str+'.png', dpi=300, bbox_inches='tight')
	


# let me try to plot this for one that already works
training_flux = pd.read_csv('./training_flux.csv')
w_full = fits.open('./data/w_to_resample_to_i_chip.fits')[0].data
idx = (w_full>6670) & (w_full<6785)
w = w_full[idx]
flux_KOI123 = training_flux['K00123'].to_numpy()[idx]
flux_KOI155 = training_flux['K00155'].to_numpy()[idx]
# plot_wavelet_coefficients(w, flux_KOI123,'KOI-123')
# plot_wavelet_coefficients(w, flux_KOI155,'KOI-155')
# plot_inverse_transform(w, flux_KOI123,'KOI-123')
# plot_inverse_transform(w, flux_KOI155,'KOI-155')

# # next I want to make these for the two Kepler-1656 nights.
flux_2019 = fits.open('./data/kepler1656_spectra/CK00367_2019_ij351.570_adj_resampled.fits')[1].data['s'][idx]
flux_2022 = fits.open('./data/kepler1656_spectra/CK00367_2022_ij487.76_adj_resampled.fits')[1].data['s'][idx]
# plot_wavelet_coefficients(w, flux_2019,'11-09-2019')
# plot_wavelet_coefficients(w, flux_2022,'09-04-2022')


