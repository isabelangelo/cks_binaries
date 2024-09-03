import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import leastsq
from scipy.optimize import brute
from scipy.optimize import least_squares
from spectrum_utils import *

# for testing purposes
import time

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

        # store spectrum information, mask telluric features
        self.flux = np.array(flux)
        self.sigma = np.array(sigma)

        # convert single order to list
        if type(order_numbers)==int:
            self.order_numbers = [order_numbers]
        else:
            self.order_numbers = order_numbers
        
        # store order wavelength
        self.wav = wav_data[[i-1 for i in self.order_numbers]].flatten()

        # compute telluric mask
        self.mask = np.empty_like(self.flux).astype(bool)
        self.mask.fill(True)
        for n, row in mask_table_cut.iterrows():
            start = row['minw']
            end = row['maxw']
            self.mask[(self.wav>start) & (self.wav<end)] = False
        
        # store cannon model information
        self.cannon_model = cannon_model
        training_data = self.cannon_model.training_set_labels
        self.training_density_kde = stats.gaussian_kde(training_data.T)

        # per pixel weights for chisq calculation
        self.err2 = self.sigma**2+self.cannon_model.s2
        self.weights = 1/np.sqrt(self.err2)

        # quantities for BIC calculation
        self._L_err_term = np.log(np.sqrt(2*np.pi*self.err2))
        self._n_pixels = np.sum(self.mask)
        
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
            model = self.cannon_model(param)
            resid = self.weights * (model - self.flux)
            resid = resid[self.mask]

            return resid
        
        # initial labels from cannon model
        initial_labels = self.cannon_model._fiducials.copy()
        
        # perform fit
        result = leastsq(residuals,x0=initial_labels, full_output=True)
        self.fit_cannon_labels = result[0].copy()
        
        # compute metrics associated with best-fit labels
        # note: I need to use the logvsini parameterization for chi-squared
        # since it's calculated using residuals()
        self.fit_chisq = np.sum(result[2]["fvec"]**2)
        self.training_density = self.training_density_kde(self.fit_cannon_labels)[0]
        
        # residuals of best-fit model
        self.model_flux = self.cannon_model(self.fit_cannon_labels)
        self.model_residuals = self.flux - self.model_flux
        
        # BIC of best-fit model evaluated at non-masked pixels
        L_data_term = self.model_residuals**2/self.err2
        ln_L = (-1/2)*np.sum(L_data_term[self.mask] - self._L_err_term[self.mask])
        self.fit_BIC = len(self.fit_cannon_labels)*np.log(self._n_pixels) - 2*ln_L


    def fit_binary(self, save_chisq_surface_to = None):
        """
        Perform binary model fit to HIRES spectrum and compute model flux
        and chi-squared associated with best-fit binary model
        Asserts that the primary is the brighter star in the fit
        """
        # initial conditions of binary set by single star
        # for labels other than Teff
        # note: we parameterize vsini in log space to avoid vsini<0
        teff_init, logg_init, feh_init, vsini_init = self.fit_cannon_labels

        # define minima and maxima for brute search
        # based on model interpolation bounds
        label_arr = self.cannon_model.training_set_labels.T
        teff_min, teff_max = label_arr[0].min(), label_arr[0].max()
        teff_ratio_min, teff_ratio_max = teff_min/teff_max, 1
        rv_min, rv_max = -10, 10

        def binary_model(cannon_param1, cannon_param2, wav, cannon_model):
            """Calculate binary model associated with set of parameters
            a particular set of model parameters"""

            # compute relative flux based on temperature
            W1, W2 = flux_weights(cannon_param1[0], cannon_param2[0], wav)

            # compute single star models for both components
            flux1 = cannon_model(cannon_param1[:-1])
            flux2 = cannon_model(cannon_param2[:-1])

            # shift flux1, flux2 according to drv
            delta_w1 = wav * cannon_param1[-1]/speed_of_light_kms
            delta_w2 = wav * cannon_param2[-1]/speed_of_light_kms    
            flux1_shifted = np.interp(wav, wav + delta_w1, flux1)
            flux2_shifted = np.interp(wav, wav + delta_w2, flux2)

            # compute weighted sum of primary, secondary
            model = W1*flux1_shifted + W2*flux2_shifted

            return model


        def residuals(params, wav, flux, cannon_model):
            """
            per-pixel log(chi-squared) for a given set of primary + secondary star labels
                Args:
                    param (np.array): [teff1, logg1, met1, vsini1, RV1, 
                    teff_ratio, logg2, met2, vsini2, RV2] 
                    (1 denotes primary, 2 denotes secondary)
                Returns:
                    log_chisq (np.array): per pixel chi-squared
            """
            # store primary, secondary parameters
            cannon_param1 = params[:5]
            cannon_param2 = params[[5,6,2,7,8]]

            # re-parameterize from teff2/teff1 to teff2
            cannon_param2[0] = cannon_param2[0]*cannon_param1[0]
            
            # prevent model from regions outside of Cannon training set
            if 4200>cannon_param1[0] or 7000<cannon_param1[0]:
                return np.inf*np.ones(len(flux[self.mask]))
            elif 4200>cannon_param2[0] or 7000<cannon_param2[0]:
                return np.inf*np.ones(len(flux[self.mask]))
            else:
                # compute chisq
                model = binary_model(cannon_param1, cannon_param2, wav, cannon_model)
                resid = self.weights * (model - flux)
                resid = resid[self.mask]
                #print(cannon_param1[0].round(2), (cannon_param2[0]/cannon_param1[0]).round(4), sum(resid**2))
                #print(cannon_param1[0].round(2), (cannon_param2[0]/cannon_param1[0]).round(4), len(resid))
                return resid

        # wrapper function to fix non-teff labels in brute search
        def residuals_wrapper(teff_params, wav, flux, cannon_model):
            """
            Wrapper function that computes residuals while only varying teff1, teff_ratio,
            used only for coarse brute search
            """
            params = np.array([teff_params[0], logg_init, feh_init, vsini_init, 0, \
                      teff_params[1], logg_init, vsini_init, 0])
            sum_resid2 = sum(residuals(params, wav, flux, cannon_model)**2)
            return sum_resid2

        # perform coarse brute search for ballpark teff1, teff_ratio
        # based on El-Badry 2018a Figure 2, 
        # teff1 is bound by the training set, teff_ratio=0-1
        # but we return chisq=inf for teff2 outside training set
        #t0_brute = time.time()
        teff_ranges = (
            slice(teff_min, teff_max, 100), # teff1
            slice(teff_ratio_min, teff_ratio_max, 0.01)) # teff_ratio
        chisq_args = (self.wav, self.flux, self.cannon_model)
        op_brute = brute(residuals_wrapper, teff_ranges, chisq_args, finish=None) 
        teff1_init, teff_ratio_init = op_brute
        #print('time for brute search: {} seconds'.format(time.time()-t0_brute))
        #print('from brute search, teff1={}K, teff2/teff1={}'.format(teff1_init, teff_ratio_init))

        # perform localized search at minimum from brute search
        t0_local = time.time()
        initial_labels = np.array([teff1_init, logg_init, feh_init, vsini_init, 0, \
                      teff_ratio_init, logg_init, vsini_init, 0])
        op_leastsq = leastsq(residuals, initial_labels, 
            args=chisq_args, full_output=True, ftol=1e-6)
        #print('time for local optimizer: {} seconds'.format(time.time()-t0_local))

        # chisq metrics associated with best-fit labels
        self.binary_fit_chisq = sum(op_leastsq[2]['fvec']**2) # convert from log
        self.delta_chisq = self.fit_chisq - self.binary_fit_chisq
        #print('best-fit single star chisq:', self.fit_chisq)
        #print('best-fit binary chisq:', self.binary_fit_chisq)
        #print('')

        # compute labels, residuals of best-fit model
        self.binary_fit_cannon_labels = op_leastsq[0]

        # re-parameterize from teff2/teff1 to teff2
        self.binary_fit_cannon_labels[5] *= self.binary_fit_cannon_labels[0]

        # store best-fit binary model
        self.binary_model_flux = binary_model(
            self.binary_fit_cannon_labels[:5], 
            self.binary_fit_cannon_labels[[5,6,2,7,8]],
            self.wav, 
            self.cannon_model)
        self.binary_model_residuals = self.flux - self.binary_model_flux

        # BIC of best-fit binary model evaluated at non-masked pixels
        L_data_term = self.binary_model_residuals**2/self.err2
        ln_L = (-1/2)*np.sum(L_data_term[self.mask] - self._L_err_term[self.mask])
        self.binary_fit_BIC = len(self.binary_fit_cannon_labels)*np.log(self._n_pixels) - 2*ln_L
        self.delta_BIC = self.fit_BIC - self.binary_fit_BIC

        # fractional improvement parameter
        # see El-Badry et al. (2018)
        f_imp_numerator = (abs(self.model_residuals) - abs(self.binary_model_residuals))/self.sigma
        f_imp_denominator = abs(self.model_flux - self.binary_model_flux)/self.sigma
        self.f_imp = sum(f_imp_numerator[self.mask])/sum(f_imp_denominator[self.mask])

        # TEMPORARY: plot for testing optimizer
        if save_chisq_surface_to is not None:
            # define parameter space to explore
            teff1 = np.linspace(4200,7000,100)
            teff_ratio = np.linspace(0.6,1,100)
            param_ranges = (teff_ratio.min(), teff_ratio.max(), teff1.min(), teff1.max())

            # array to store binary chisq values
            binary_chisq = np.ones((len(teff1), len(teff_ratio)))
            for i in range(len(teff1)):
                for j in range(len(teff_ratio)):
                    teff_params = (teff1[i], teff_ratio[j])
                    chisq = residuals_wrapper(teff_params, self.wav, self.flux, self.cannon_model)
                    binary_chisq[i,j] = chisq

            # plot the surface and save to file
            plt.figure(figsize=(10,7))
            plt.imshow(np.log10(binary_chisq), 
                       extent=param_ranges, 
                       aspect='auto', origin='lower', 
                       vmin=4.4,vmax=5, cmap='Spectral')
            # optimizer outputs
            fit_label = 'optimizer, chisq={}'.format(int(self.binary_fit_chisq))
            plt.plot( 
                self.binary_fit_cannon_labels[5]/self.binary_fit_cannon_labels[0], 
                self.binary_fit_cannon_labels[0], 'w*', ms=15, mec='k', label=fit_label)
            # true minimum
            true_label = 'grid minimum, chisq={}\n single star chisq={}'.format(
                int(np.min(binary_chisq)), int(self.fit_chisq))
            teff1_min_idx, teff_ratio_min_idx = np.argwhere(binary_chisq == np.min(binary_chisq))[0]
            plt.plot(teff_ratio[teff_ratio_min_idx], teff1[teff1_min_idx], 'b*', ms=15, mec='k', label=true_label)
            # label axes + save figure
            plt.colorbar(label=r'log $\chi^2_{\rm binary}$')
            plt.xlabel('teff2/teff1');plt.ylabel('teff1 (K)')
            plt.xlim(0.57,1)
            plt.legend()
            figure_path = './data/binary_optimizer_validation_plots/{}.png'.format(save_chisq_surface_to)
            plt.savefig(figure_path)
            print('saved file to {}'.format(figure_path))


    # temporary function to visualize the fit
    def plot_fit(self, zoom_min=5400, zoom_max=5420):
        self.fit_single_star()
        plt.figure(figsize=(15,10))
        plt.rcParams['font.size']=15
        plt.subplots_adjust(hspace=0)
        plt.subplot(211)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
            ecolor='#E8E8E8', elinewidth=4, zorder=0, linewidth=2)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8)
        plt.plot(self.wav, self.model_residuals-1, 'k-')
        plt.axvspan(self.wav[~self.mask][0], self.wav[~self.mask][-1], 
            alpha=0.2, color='lightgrey')
        plt.xlim(self.wav[0],self.wav[-1])
        plt.ylabel('normalized flux')
        plt.xticks([])

        plt.subplot(212)
        plt.errorbar(self.wav, self.flux, yerr=self.sigma, color='k', 
            ecolor='#E8E8E8', elinewidth=4, zorder=0, linewidth=2)
        plt.plot(self.wav, self.model_flux, 'r-', alpha=0.8)
        plt.plot(self.wav, self.model_residuals-1, 'k-')
        plt.axvspan(self.wav[~self.mask][0], self.wav[~self.mask][-1], 
            alpha=0.2, color='lightgrey')
        plt.xlim(zoom_min, zoom_max)
        plt.xlabel('wavelength (angstrom)');plt.ylabel('normalized flux')

    def plot_binary(self, zoom_min=5150, zoom_max=5190):
        # TO DO: add telluric mask to plot
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



