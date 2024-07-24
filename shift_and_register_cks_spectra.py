"""
This code uses Specmatch-Emp to shift all the raw CKS r chip spectra 
and rescale them onto the original HIRES wavelength scale. 
"""
from specmatchemp.spectrum import read_hires_fits
from specmatchemp.specmatch import SpecMatch
from specmatchemp import SPECMATCHDIR
import specmatchemp.library
import glob
import os

# paths to store shifted spectra
shifted_resampled_path = './data/cks-spectra_shifted_resampled_i'
kepler1656_path = './data/kepler1656_spectra'

# load specmatch library for reference spectra
lib = specmatchemp.library.read_hdf()

def shift_and_save_orders(path, filename):
    
    # load spectrum + create SpecMatch object
    KOI_spectrum = read_hires_fits(
        filename, 
        maskfile = os.path.join(SPECMATCHDIR, 'hires_telluric_mask.csv'))
    sm_KOI = SpecMatch(KOI_spectrum, lib)
    w_to_resample_to = sm_KOI.target_unshifted.w
    
    # shift spectrum
    shifted_spectrum = sm_KOI.shift()
    
    rescaled_orders=[]
    # rescale to each order and save
    for order_idx in range(w_to_resample_to.shape[0]):
        # resample order to original wavelength scale
        w_order = w_to_resample_to[order_idx]
        rescaled_order = shifted_spectrum.rescale(w_order)
        # write to file
        shifted_resampled_filename = '{}/order{}/{}.fits'.format(
            path,
            str(order_idx+1),
            filename.split('/')[-1].replace('.fits', ''))
        rescaled_order.to_fits(shifted_resampled_filename)

# create directories for the different orders
for order_idx in range(10):
    order_n = order_idx+1 # order numbers are not zero-indexed
    order_path = '{}/order{}'.format(shifted_resampled_path, order_n)
    order_path_kepler1656 = '{}/order{}'.format(kepler1656_path, order_n)
    if not os.path.exists(order_path):
        os.mkdir(order_path)
    if not os.path.exists(order_path_kepler1656):
        os.mkdir(order_path_kepler1656)
    else:
        pass

# iterate over CKS spectra
print('shifting + registering training set spectra')
spectrum_filenames = glob.glob('./data/cks-spectra_i/*.fits')
for filename in spectrum_filenames:
    shift_and_save_orders(shifted_resampled_path, filename)

# shift kepler-1656 spectra for wavelet diagnostics
print('shifting and registering Kepler-1656 spectra for diagnostics')
kepler1656_filenames = glob.glob(kepler1656_path + '/*.fits')
for filename in kepler1656_filenames:
    shift_and_save_orders(kepler1656_path, filename)
