# once this is done, I'll update cannon_model_diagnostics
# (1) replace model test step code
# (2) remove redundant import statements

import thecannon as tc
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from scipy.optimize import leastsq

# define clip size for each order (in pixels)
# used to generate training set
order_clip = 200

# load wavelength data
w_filename = './data/cks-spectra/cks-k00002_rj122.92.fits' # can be any r chip file
w_data = fits.open(w_filename)[2].data[:, :-1] # require even number of elements
w_data = w_data[:, order_clip:-1*order_clip] # clip 5% on each side

# define wavelength limits for masks
sodium_wmin, sodium_wmax = 5889.5, 5896.5
max_v_shift = 30*u.km/u.s 
telluric_wmin = (6270*u.angstrom*(1-max_v_shift/c.c)).value
telluric_wmax = (6310*u.angstrom*(1+max_v_shift/c.c)).value

class SingleOrderSpectrum(object):
    def __init__(self, flux, sigma, order_number):
        # store spectrum information
        self.flux = np.array(flux[order_clip:-1*order_clip])
        self.sigma = np.array(sigma[order_clip:-1*order_clip])
        self.order_number = order_number
        # store order wavelength
        self.w = w_data[order_number-1]
        # compute wavelength mask
        sodium_mask = np.where((self.w>sodium_wmin) & (self.w<sodium_wmax))[0]
        telluric_mask = np.where((self.w>telluric_wmin) & (self.w<telluric_wmax))[0]
        self.mask = np.array(list(sodium_mask) + list(telluric_mask))
        # store relevant cannon model
        model_path = './data/cannon_models/rchip_order{}.model'.format(order_number)
        self.cannon_model = tc.CannonModel.read(model_path)
    
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
        self.chisq_single = np.sum(residuals(fit_labels)**2)
        # re-parameterize from log(vbroad) to vbroad
        self.cannon_labels = result[0].copy()
        self.cannon_labels[-1] = 10**self.cannon_labels[-1]