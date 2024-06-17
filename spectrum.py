import thecannon as tc
import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
from scipy.optimize import leastsq
from spectrum_utils import *

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
            # re-parameterize from log(vsini) to vsini for Cannon
            cannon_param = param.copy()
            cannon_param[-1] = 10**cannon_param[-1]
            # compute chisq
            model = self.cannon_model(cannon_param)
            weights = 1/np.sqrt(sigma_for_fit**2+self.cannon_model.s2)
            resid = weights * (model - self.flux)
            return resid
        
        # initial labels from cannon model
        initial_labels = self.cannon_model._fiducials.copy()
        # re-parameterize from vsini to log(vroad) in optimizer
        initial_labels[-1] = np.log10(initial_labels[-1]) 
        # perform fit
        result = leastsq(residuals,x0=initial_labels)
        self.fit_cannon_labels = result[0].copy()
        # re-parameterize from log(vsini) to vsini
        self.fit_cannon_labels[-1] = 10**self.fit_cannon_labels[-1] 
        # compute metrics associated with best-fit labels
        # note: I need to use the logvsini parameterization for chi-squared
        # since it's calculated using residuals()
        self.fit_chisq = np.sum(residuals(result[0])**2)
        self.training_density = self.training_density_kde(self.fit_cannon_labels)[0]
        # compute residuals of best-fit model
        self.model_flux = self.cannon_model(self.fit_cannon_labels)
        self.residuals = self.flux - self.model_flux

    def fit_binary(self):
        # mask out sodium, telluric features
        sigma_for_fit = self.sigma.copy()
        if len(self.mask)>0:
            sigma_for_fit[self.mask] = np.inf
            
        # binary model goodness-of-fit
        def residuals(param):
            """
            per-pixel chi-squared for a given set of primary + secondary star labels
                Args:
                    param1 (np.array): primary star [teff, logg, met, vsini, RV]
                    param2 (np.array): secondary star [teff, logg, met, vsini, RV]
                Returns:
                    resid (np.array): per pixel chi-squared
            """
            
            # store primary, secondary parameters
            cannon_param1 = param[:5].copy()
            cannon_param2 = param[5:].copy()
            
            # prevent model from regions where flux ratio can't be interpolated
            if 2450>cannon_param1[0] or 34000<cannon_param1[0]:
                return np.inf*np.ones(len(self.flux))
            elif 2450>cannon_param2[0] or 34000<cannon_param2[0]:
                return np.inf*np.ones(len(self.flux))
            
            # re-parameterize from log(vsini) to vsini for Cannon
            cannon_param1[-1] = 10**cannon_param1[-1]
            cannon_param2[-1] = 10**cannon_param2[-1]
            
            # compute relative flux based on temperature
            W1, W2 = flux_weights(cannon_param1[0], cannon_param2[0], self.wav)
            
            # compute single star models for both components
            flux1 = self.cannon_model(cannon_param1[:-1])
            flux2 = self.cannon_model(cannon_param2[:-1])
            
            # shift flux1, flux2 according to drv
            delta_w1 = self.wav * cannon_param1[-1]/speed_of_light_kms
            delta_w2 = self.wav * cannon_param2[-1]/speed_of_light_kms
            flux1_shifted = np.interp(self.wav, self.wav + delta_w1, flux1)
            flux2_shifted = np.interp(self.wav, self.wav + delta_w2, flux2)
            
            # compute weighted sum of primary, secondary
            model = W1*flux1_shifted + W2*flux2_shifted
            
            # compute chisq
            weights = 1/np.sqrt(sigma_for_fit**2+self.cannon_model.s2)
            resid = weights * (model - self.flux)
            return resid
        
        # function to run optimizer with specified initial conditions
        def optimizer(initial_teff):
            
            # determine initial labels
            fiducial_labels = self.cannon_model._fiducials[1:].tolist()
            initial_primary_labels = [initial_teff[0]] + fiducial_labels + [0]
            initial_secondary_labels = [initial_teff[1]] + fiducial_labels + [0]
            
            # re-parameterize from vsini to log(vroad) for optimizer
            initial_primary_labels[3] = np.log10(initial_primary_labels[3])
            initial_secondary_labels[3] = np.log10(initial_secondary_labels[3])
            initial_labels = initial_primary_labels + initial_secondary_labels

            # perform least-sqaures fit
            result = leastsq(residuals,x0=initial_labels)
            fit_labels = result[0]
            binary_fit_chisq = np.sum(residuals(fit_labels)**2)
            
            # re-parameterize from log(vsini) to vsini
            binary_fit_cannon_labels = fit_labels.copy()
            binary_fit_cannon_labels[3] = 10**binary_fit_cannon_labels[3]
            binary_fit_cannon_labels[8] = 10**binary_fit_cannon_labels[8]
            
            return binary_fit_cannon_labels, binary_fit_chisq
        
        # run optimizers, store fit with lowest chi2
        lowest_global_chi2 = np.inf    
        binary_fit_cannon_labels = None

        for initial_teff in initial_teff_arr:
            results = optimizer(initial_teff)
            print(results[0][1], results[0][6], results[1])
            if results[1] < lowest_global_chi2:
                lowest_global_chi2 = results[1]
                binary_fit_cannon_labels = np.array(results[0])

        # assert that the primary is the brighter star
        if fit_cannon_labels[0]<fit_cannon_labels[5]:
            primary_fit_cannon_labels = fit_cannon_labels[5:]
            secondary_fit_cannon_labels = fit_cannon_labels[:5]
            binary_fit_cannon_labels = primary_fit_cannon_labels + secondary_fit_cannon_labels
            
        # store metrics for binary fit
        self.binary_fit_cannon_labels = binary_fit_cannon_labels.copy()
        # re-parameterize from log(vsini) to vsini
        self.binary_fit_cannon_labels[3] = 10**self.binary_fit_cannon_labels[3] 
        self.binary_fit_cannon_labels[8] = 10**self.binary_fit_cannon_labels[8] 
        # compute metrics associated with best-fit labels
        # note: I need to use the logvsini parameterization for chi-squared
        # since it's calculated using residuals()
        self.binary_fit_chisq = np.sum(residuals(binary_fit_cannon_labels)**2)

    # temporary function to visualize the fit
    def plot_fit(self, zoom_order=14):
        self.fit_single_star()
        plt.figure(figsize=(15,10))
        plt.rcParams['font.size']=15
        plt.subplots_adjust(hspace=0)
        plt.subplot(211)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8, lw=1.5)
        plt.plot(self.wav, self.residuals-1, 'k-')
        plt.xlim(self.wav[0],self.wav[-1])
        plt.ylabel('normalized flux')

        plt.subplot(212)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8, lw=1.5)
        plt.plot(self.wav, self.residuals-1, 'k-')
        plt.xlim(w_data[zoom_order-1][0], w_data[zoom_order-1][-1])
        plt.xlabel('wavelength (angstrom)');plt.ylabel('normalized flux')





