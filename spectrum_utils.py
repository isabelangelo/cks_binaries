import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as c
from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d

__all__ = ["initial_teff_arr", "flux_weights", "teff2radius", "speed_of_light_kms"]


# initial Teff values for binary model optimizer
teff_grid = np.arange(4000,8000,1000)
initial_teff_arr = [(x, y) for x in teff_grid for y in teff_grid if x>=y]

# speed of light for wavelength calculation
speed_of_light_kms = c.c.to(u.km/u.s).value

# temperature to radius conversion for binary model
pm2013 = pd.read_csv('./data/literature_data/PecautMamajek_table.csv', 
                     skiprows=22, delim_whitespace=True).replace('...',np.nan)
teff_pm2013 = np.array([float(i) for i in pm2013['Teff']])
R_pm2013 = np.array([float(i) for i in pm2013['R_Rsun']])
mass_pm2013 = np.array([float(i) for i in pm2013['Msun']])

valid_mass = ~np.isnan(mass_pm2013)
teff2radius = interp1d(teff_pm2013[valid_mass], R_pm2013[valid_mass])


# function to compute flux weights 
# of primary, secondary in binary model
def flux_weights(teff1, teff2, wav):
    """
    Returns un-normalized relative fluxes,
    based on blackbody curve * R^2
    """
    
    # blackbody functions
    bb1 = BlackBody(temperature=teff1*u.K)
    bb2 = BlackBody(temperature=teff2*u.K)
    # evaluate blackbody at model wavelengths
    bb1_curve = bb1(wav*u.AA).value
    bb2_curve = bb2(wav*u.AA).value

    # calculate unweighted flux contributions
    W1 = bb1_curve*teff2radius(teff1)**2
    W2 = bb2_curve*teff2radius(teff2)**2
    
    # normalize weights to sum to 1
    W_sum = W1 + W2
    W1 /= W_sum
    W2 /= W_sum

    return W1, W2