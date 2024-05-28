import thecannon as tc
import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
from scipy.optimize import leastsq

# load wavelength data
reference_w_filename = './data/cannon_training_data/cannon_reference_w.fits'
w_data = fits.open(reference_w_filename)[0].data

# define wavelength limits for masks
sodium_wmin, sodium_wmax = 5889, 5897
max_v_shift = 30*u.km/u.s 
telluric_wmin = (6270*u.angstrom*(1-max_v_shift/c.c)).value
telluric_wmax = (6310*u.angstrom*(1+max_v_shift/c.c)).value

class Spectrum(object):
    """
    HIRES spectrum object
    
    Args:
        flux (np.array): flux of object, post-wavelet transform
        sigma (np.array): flux errors of object
        order_numbers (int or array-like): order numbers to include, 1-16 for HIRES r chip
                        e.g., 4, [1,2,6,15,16]
        cannon_model (tc.CannonModel): Cannon model object to use to model spectrum
    """
    def __init__(self, flux, sigma, order_numbers, cannon_model):
        
        # store spectrum information
        self.flux = np.array(flux)
        self.sigma = np.array(sigma)

        # convert single order to list
        if type(order_numbers)==int:
            self.order_numbers = [order_numbers]
        else:
            self.order_numbers = order_numbers
        
        # store order wavelength
        self.wav = w_data[[i-1 for i in self.order_numbers]].flatten()
        
        # compute wavelength mask
        sodium_mask = np.where((self.wav>sodium_wmin) & (self.wav<sodium_wmax))[0]
        telluric_mask = np.where((self.wav>telluric_wmin) & (self.wav<telluric_wmax))[0]
        self.mask = np.array(list(sodium_mask) + list(telluric_mask))
        
        # store cannon model information
        self.cannon_model = cannon_model
        training_data = self.cannon_model.training_set_labels
        self.training_density_kde = stats.gaussian_kde(training_data.T)
        
    def fit_single_star(self):
        """
        Run the test step on the spectra
        (similar to the Cannon 2 test step, 
        but we mask the sodium + telluric lines)
        Args:
            flux (np.array): normalized flux data
            sigma (np.array): flux error data
            single_star_model (tc.CannonModle): Cannon model to use for fitting
        """
        # mask out sodium, telluric features
        sigma_for_fit = self.sigma.copy()
        if len(self.mask)>0:
            sigma_for_fit[self.mask] = np.inf
            
        # single star model goodness-of-fit
        def residuals(param):
            """
            per-pixel chi-squared for a given set of Cannon labels
                Args:
                    param (np.array): teff, logg, met, vsini values
                Returns:
                    resid (np.array): per pixel chi-squared
            """
            # re-parameterize from log(vbroad) to vbroad for Cannon
            cannon_param = param.copy()
            cannon_param[-1] = 10**cannon_param[-1]
            # compute chisq
            model = self.cannon_model(cannon_param)
            weights = 1/np.sqrt(sigma_for_fit**2+self.cannon_model.s2)
            resid = weights * (model - self.flux)
            return resid
        
        # initial labels from cannon model
        initial_labels = self.cannon_model._fiducials.copy()
        # re-parameterize from vbroad to log(vroad) in optimizer
        initial_labels[-1] = np.log10(initial_labels[-1]) 
        # perform fit
        result = leastsq(residuals,x0=initial_labels)
        self.fit_cannon_labels = result[0].copy()
        # re-parameterize from log(vbroad) to vbroad
        self.fit_cannon_labels[-1] = 10**self.fit_cannon_labels[-1] 
        # compute metrics associated with best-fit labels
        # note: I need to use the logvbroad parameterization for chi-squared
        # since it's calculated useing residuals()
        self.fit_chisq = np.sum(residuals(result[0])**2)
        self.training_density = self.training_density_kde(self.fit_cannon_labels)[0]
        # compute residuals of best-fit model
        self.model_flux = self.cannon_model(self.fit_cannon_labels)
        self.residuals = self.flux - self.model_flux

    # temporary function to visualize the fit
    def plot_fit(self, order_n1=None, order_n2=None):

        if order_n1 is None:
            order_n1 = 4
        if order_n2 is None:
            order_n2 = 7

        self.fit_single_star()
        min1, max1 = w_data[order_n1-1].min(),w_data[order_n1-1].max()
        min2, max2 = w_data[order_n2-1].min(),w_data[order_n2-1].max()
        log_chisq = np.log10(self.fit_chisq)

        fig, axes = plt.subplots(3,1, figsize=(17,15))
        axes[0].errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
                         ecolor='#E8E8E8', elinewidth=4, zorder=0)
        axes[0].plot(self.wav, self.model_flux, '-', alpha=0.8, lw=1.5, color='#2D36FD')
        axes[0].plot(self.wav, self.residuals*2, 'k-', lw=1)
        axes[0].text(5050,1.2, r'log $\chi^2$ = {}'.format(log_chisq.round(2)))
        axes[0].axvspan(min1, max1, color='cyan', alpha=0.1)
        axes[0].axvspan(min2, max2, color='r', alpha=0.1)
        axes[0].set_xlim(self.wav[0],self.wav[-1])

        axes[1].errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
                         ecolor='#E8E8E8', elinewidth=4, zorder=0)
        axes[1].plot(self.wav, self.model_flux, '-', alpha=0.8, lw=1.5, color='#2D36FD')
        axes[1].plot(self.wav, self.residuals*2, 'k-', lw=1)
        axes[1].set_xlim(min1, max1)

        axes[2].errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
                         ecolor='#E8E8E8', elinewidth=4, zorder=0)
        axes[2].plot(self.wav, self.model_flux, '-', alpha=0.8, lw=1.5, color='#2D36FD')
        axes[2].plot(self.wav, self.residuals*2, 'k-', lw=1)
        axes[2].set_xlim(min2, max2)

        # plot telluric mask
        mask_kwgs={'color':'lightgrey','alpha':0.5}
        axes[0].axvspan(telluric_wmin, telluric_wmax, **mask_kwgs)
        axes[1].axvspan(telluric_wmin, telluric_wmax, **mask_kwgs)
        axes[2].axvspan(telluric_wmin, telluric_wmax, **mask_kwgs)
        plt.show()









