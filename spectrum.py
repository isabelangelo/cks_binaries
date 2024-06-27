import thecannon as tc
import numpy as np
import lmfit
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import leastsq
from spectrum_utils import *

class Spectrum(object):
    """
    HIRES Spectrum object
    
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
        self.wav = wav_data[[i-1 for i in self.order_numbers]].flatten()
        
        # compute wavelength mask
        sodium_mask = np.where((self.wav>sodium_wmin) & (self.wav<sodium_wmax))[0]
        telluric_mask = np.where((self.wav>telluric_wmin) & (self.wav<telluric_wmax))[0]
        self.mask = np.array(list(sodium_mask) + list(telluric_mask))

        # mask out sodium, telluric features
        self.sigma_for_fit = self.sigma.copy()
        if len(self.mask)>0:
            self.sigma_for_fit[self.mask] = np.inf
        
        # store cannon model information
        self.cannon_model = cannon_model
        training_data = self.cannon_model.training_set_labels
        self.training_density_kde = stats.gaussian_kde(training_data.T)
        
    def fit_single_star(self):
        """ Run the test step on the ra (similar to the Cannon 2 
        test step, but we mask the sodium + telluric lines)"""
            
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
            weights = 1/np.sqrt(self.sigma_for_fit**2+self.cannon_model.s2)
            resid = weights * (model - self.flux)

            return resid
        
        # initial labels from cannon model
        initial_labels = self.cannon_model._fiducials.copy()
        # re-parameterize from vsini to log(vroad) in optimizer
        initial_labels[-1] = np.log10(initial_labels[-1]) 
        # perform fit
        result = leastsq(residuals,x0=initial_labels, full_output=True)
        self.fit_cannon_labels = result[0].copy()
        # re-parameterize from log(vsini) to vsini
        self.fit_cannon_labels[-1] = 10**self.fit_cannon_labels[-1] 
        # compute metrics associated with best-fit labels
        # note: I need to use the logvsini parameterization for chi-squared
        # since it's calculated using residuals()
        self.fit_chisq = np.sum(result[2]["fvec"]**2)
        self.training_density = self.training_density_kde(self.fit_cannon_labels)[0]
        # compute residuals of best-fit model
        self.model_flux = self.cannon_model(self.fit_cannon_labels)
        self.model_residuals = self.flux - self.model_flux

    def fit_binary(self):
        """
        Perform binary model fit to HIRES spectrum and compute model flux
        and chi-squared associated with best-fit binary model
        Asserts that the primary is the brighter star in the fit
        """

        # map wavelengths in each order to indices in flattened array
        # this speeds up interpolation step in shift()
        wav_idx = [np.nonzero(np.isin(self.wav, wav_order))[0] for wav_order in wav_data]

        # store names of binary parameter keys
        # to access when calling cannon model
        cannon_param1_keys = ['teff1','logg1','feh1','vsini1','rv1']
        cannon_param2_keys = ['teff2','logg2','feh2','vsini2','rv2']

        # compute weights for chisq calcultion
        weights = 1/np.sqrt(self.sigma_for_fit**2+self.cannon_model.s2)

        # kwgsfor local optimizer (l-bfgs-b)
        lbfgsb_options = {
            'maxiter': 100,  # Maximum number of iterations
            'ftol': 1e-8,     # Function value tolerance
            'gtol': 1e-5,     # Gradient norm tolerance
            'eps': 1e-8       # Step size for numerical gradient approximation
        }

        def shift(wav, flux, rv_shift):
            """Shift flux according to input RV"""
            delta_wav = wav * rv_shift/speed_of_light_kms
            flux_shifted = np.empty(flux.shape)
            for i in range(len(wav_data)):
                w_order = wav_data[i]
                idx = wav_idx[i]
                if len(idx)>0:
                    order_flux_shifted = np.interp(
                        wav[idx], 
                        wav[idx] + delta_wav[idx], 
                        flux[idx])
                    flux_shifted[idx] = order_flux_shifted
            return flux_shifted

        def binary_model(cannon_param1, cannon_param2, wav, cannon_model):
            """Calculate binary model associated with set of parameters
            a particular set of model parameters"""

            # compute relative flux based on temperature
            W1, W2 = flux_weights(cannon_param1[0], cannon_param2[0], wav)

            # compute single star models for both components
            flux1 = cannon_model(cannon_param1[:-1])
            flux2 = cannon_model(cannon_param2[:-1])

            # shift flux1, flux2 according to drv
            flux1_shifted = shift(wav, flux1, cannon_param1[-1])
            flux2_shifted = shift(wav, flux2, cannon_param2[-1])

            # compute weighted sum of primary, secondary
            model = W1*flux1_shifted + W2*flux2_shifted

            return model


        def residuals(params, wav, flux, cannon_model):
            """
            per-pixel chi-squared for a given set of primary + secondary star labels
                Args:
                    param (np.array): [teff1, logg1, met1, vsini1, RV1, 
                    teff2, logg2, met2, vsini2, RV2] 
                    (1 denotes primary, 2 denotes secondary)
                Returns:
                    resid (np.array): per pixel chi-squared
            """
            # store primary, secondary parameters
            cannon_param1 = [params[i].value for i in cannon_param1_keys]
            cannon_param2 = [params[i].value for i in cannon_param2_keys]

            # compute chisq
            model = binary_model(cannon_param1, cannon_param2, wav, cannon_model)
            resid = weights * (model - flux)

            return resid
        
        # fit single star model to inform initial conditions
        #self.fit_single_star()
        logg_init, feh_init, vsini_init = self.fit_cannon_labels[1:]

        # perform coarse brute search for ballpark teff1, teff2
        brute_params = lmfit.Parameters()
        brute_params.add('teff1', min=4500, max=6500, brute_step=100)
        brute_params.add('logg1', value=logg_init, vary=False)
        brute_params.add('feh1', value=feh_init, vary=False)
        brute_params.add('vsini1', value=vsini_init, vary=False)
        brute_params.add('rv1', value=0, vary=False)
        brute_params.add('teff2', min=4500, max=6500, brute_step=100)
        brute_params.add('logg2', value=logg_init, vary=False)
        brute_params.add('feh2', value=feh_init, vary=False)
        brute_params.add('vsini2', value=vsini_init, vary=False)
        brute_params.add('rv2', value=0, vary=False)
        chisq_args = (self.wav, self.flux, self.cannon_model)
        op_brute = lmfit.minimize(residuals, brute_params, args=chisq_args, 
            method='brute', calc_covar=False)
        brute_teff = (op_brute.params['teff1'].value, op_brute.params['teff2'].value)
        
        # perform localized search at minimum from brute search
        local_params = lmfit.Parameters()
        local_params.add('teff1', min=4500, max=6500, value=max(brute_teff), vary=True)
        local_params.add('logg1', min=2.8, max=5, value=logg_init, vary=True)
        local_params.add('feh1', min=-1, max=1, value=feh_init, vary=True)
        local_params.add('vsini1', min=0, max=30, value=vsini_init, vary=True)
        local_params.add('rv1', min=-10, max=10, value=0, vary=True)    
        local_params.add('teff2', min=4500, max=6500, value=min(brute_teff), vary=True)
        local_params.add('logg2', min=2.8, max=5, value=logg_init, vary=True)
        local_params.add('feh2', min=-1, max=1, value=feh_init, vary=True)
        local_params.add('vsini2', min=0, max=30, value=vsini_init, vary=True)
        local_params.add('rv2', min=-10, max=10, value=0, vary=True)
        op_lbfgsb = lmfit.minimize(residuals, local_params, args=chisq_args, 
            method='lbfgsb', calc_covar=False, options=lbfgsb_options)
        
        # compute labels, residuals of best-fit model
        self.binary_fit_cannon_labels = list(op_lbfgsb.params.valuesdict().values())
        self.binary_model_flux = binary_model(
            self.binary_fit_cannon_labels[:5], 
            self.binary_fit_cannon_labels[:-5],
            self.wav, 
            self.cannon_model)
        self.binary_model_residuals = self.flux - self.binary_model_flux
        # compute metrics associated with best-fit labels
        self.binary_fit_chisq = op_lbfgsb.chisqr
        self.delta_chisq = self.fit_chisq - self.binary_fit_chisq
        

    # temporary function to visualize the fit
    def plot_fit(self, zoom_order=14):
        self.fit_single_star()
        plt.figure(figsize=(10,7))
        plt.rcParams['font.size']=15
        plt.subplots_adjust(hspace=0)
        plt.subplot(211)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
            ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8, lw=1.5)
        plt.plot(self.wav, self.model_residuals-1, 'k-')
        plt.xlim(self.wav[0],self.wav[-1])
        plt.ylabel('normalized flux')

        plt.subplot(212)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
            ecolor='#E8E8E8', elinewidth=4, zorder=0)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8, lw=1.5)
        plt.plot(self.wav, self.model_residuals-1, 'k-')
        plt.xlim(wav_data[zoom_order-1][0], wav_data[zoom_order-1][-1])
        plt.xlabel('wavelength (angstrom)');plt.ylabel('normalized flux')

    def plot_binary(self, zoom_min=5150, zoom_max=5190):
        self.fit_single_star()
        self.fit_binary()
        # create figure
        plt.figure(figsize=(10,7))
        plt.rcParams['font.size']=10
        # top panel: spectrum with single star, binary fits
        plt.subplot(311);plt.ylabel('wavelet-filtered\nflux')
        plt.plot(self.wav, self.flux, 'k-', label='data')
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.7, label='best-fit single star')
        plt.plot(self.wav, self.binary_model_flux, 'c-', alpha=0.7, label='best-fit binary')
        plt.axvspan(telluric_wmin, telluric_wmax, color='lightgrey', alpha=0.6)
        plt.legend(ncols=3, loc='upper left')
        # middel panel: spectrum with single star, binary fits, zoomed in
        plt.subplot(312);plt.ylabel('wavelet-filtered\nflux')
        plt.plot(self.wav, self.flux, 'k-', label='data')
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.7, label='best-fit single star')
        plt.plot(self.wav, self.binary_model_flux, 'c-', alpha=0.7, label='best-fit binary')
        plt.legend(ncols=3, loc='upper left')
        plt.xlim(zoom_min, zoom_max);plt.ylim(-0.7,0.4)
        # middle panel: residuals of single star, binary fits, zoomed in
        plt.subplot(313);plt.ylabel('residuals')
        single_resid = 'single fit\n'+'$\chi^2$={}'.format(int(self.fit_chisq))
        binary_resid = 'binary fit\n'+'$\chi^2$={}'.format(int(self.binary_fit_chisq))
        plt.plot(self.wav, self.model_residuals, 'r-', alpha=0.7, label=single_resid)
        plt.plot(self.wav, self.binary_model_residuals, 'c-', alpha=0.7, label=binary_resid)
        plt.legend(ncols=2, labelcolor='linecolor', loc='lower left')
        plt.xlim(zoom_min, zoom_max)
        plt.xlabel('wavelength (angstrom)')
        plt.show()

# test plots
# import pandas as pd
# import thecannon as tc
# binary_flux = pd.read_csv('./data/spectrum_dataframes/known_binary_flux_dwt.csv')
# binary_sigma = pd.read_csv('./data/spectrum_dataframes/known_binary_sigma_dwt.csv')
# model = tc.CannonModel.read('./data/cannon_models/rchip_orders_11-12_omitted_dwt/rchip_orders_11-12_omitted_dwt.model')
# order_numbers = [i for i in range(1,17) if i not in (11,12)]
# spec = Spectrum(
#     binary_flux['K00289'], 
#     binary_sigma['K00289'],
#     order_numbers,
#     model)
# spec.fit_single_star()
# spec.fit_binary()
# print(spec.binary_fit_chisq)
# print(binary_fit_chisq)
# Spectrum(
#     binary_flux['K00291'], 
#     binary_sigma['K00291'],
#     order_numbers,
#     model).plot_binary()



