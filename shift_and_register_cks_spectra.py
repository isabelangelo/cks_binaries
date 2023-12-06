"""
This code uses Specmatch-Emp to shift all the raw CKS r chip spectra 
and rescale them onto the original HIRES wavelength scale. 
"""
from specmatchemp import SHIFT_REFERENCES
from specmatchemp.library import read_hdf as read_sm_lib
from specmatchemp.shift import shift, bootstrap_shift
from specmatchemp.spectrum import read_fits, read_hires_fits
from specmatchemp.spectrum import Spectrum
from specmatchemp.specmatch import SpecMatch
import numpy as np
import glob
import os

# load specmatch library
lib = read_sm_lib()

# load specmatch reference spectra for shifting
ref_specs = [read_fits(os.path.join('/Users/isabelangelo/.specmatchemp/shifted_spectra/',
             r[0] + '_adj.fits')) for r in SHIFT_REFERENCES]

# create directories for the different orders
for order_idx in range(16):
    order_n = order_idx+1 # order numbers are not zero-indexed
    order_path = './data/cks-spectra_shifted_resampled_r/order{}'.format(order_n)
    print(order_path)
    if not os.path.exists(order_path):
        os.mkdir(order_path)
    else:
        pass

# # iterate over CKS spectra
# print('shifting + registering training set spectra')
# spectrum_filenames = glob.glob('./data/cks-spectra/*rj*.fits')
# shifted_resampled_path = './data/cks-spectra_shifted_resampled_r'
# for filename in spectrum_filenames:

#     # load target to shift
#     target = read_hires_fits(filename)

#     # extract order for bootstrap shift
#     bootstrap_order = target.cut(5120, 5200)

#     # shift single order to determine best reference
#     bootstrap_shift_data = {}
#     shifted_bs_order = bootstrap_shift(bootstrap_order, ref_specs, store=bootstrap_shift_data)
#     best_ref_spec = ref_specs[bootstrap_shift_data['shift_reference']]

#     # shift + register all orders
#     for order_idx in range(target.w.shape[0]):
#         # order numbers are not zero-indexed
#         order_n = order_idx+1 
#         order = Spectrum(target.w[order_idx], target.s[order_idx], target.serr[order_idx], target.mask[order_idx])
#         shifted_order = shift(order, best_ref_spec)

#         # extend spectrum to correct size for rescaling
#         w_to_resample_to = target.w[order_idx]
#         extended_w = np.linspace(shifted_order.w[0], shifted_order.w[-1], len(w_to_resample_to))
#         extended_order =  shifted_order.extend(extended_w)

#         # resample spectrum onto library wavelength
#         resampled_order = extended_order.rescale(w_to_resample_to)
        
#         # write to file
#         fileroot = filename.split('/')[-1].replace('.fits', '')
#         shifted_resampled_filename = '{}/order{}/{}.fits'.format(
#         	shifted_resampled_path,
#         	str(order_n),
#         	fileroot)
#         resampled_order.to_fits(shifted_resampled_filename)
#     print('saved resampled spectrum saved to {}/'.format(shifted_resampled_path))
# print('saved all training set spectra to files')

print('shifting and registering Kepler-1656 spectra for diagnostics')
# also shift ang register Kepler-1656 spectra for DWT diagnostics
kepler1656_path = './data/kepler1656_spectra'
kepler1656_filenames = glob.glob(kepler1656_path + '/*rj*.fits')
kepler1656_raw_spectra_filenames = [f for f in kepler1656_filenames if 'order' not in f]
for filename in kepler1656_raw_spectra_filenames:
    # load target to shift
    import pdb
    target = read_hires_fits(filename)

    # extract order for bootstrap shift
    bootstrap_order = target.cut(5120, 5200)

    # shift single order to determine best reference
    bootstrap_shift_data = {}
    shifted_bs_order = bootstrap_shift(bootstrap_order, ref_specs, store=bootstrap_shift_data)
    best_ref_spec = ref_specs[bootstrap_shift_data['shift_reference']]

    # shift + register all orders
    for order_idx in range(target.w.shape[0]):
        # order numbers are not zero-indexed
        order_n = order_idx+1 

        order = Spectrum(target.w[order_idx], target.s[order_idx], target.serr[order_idx], target.mask[order_idx])
        shifted_order = shift(order, best_ref_spec)

        # extend spectrum to correct size for rescaling
        w_to_resample_to = target.w[order_idx][1:-1]
        extended_w = np.linspace(shifted_order.w[0], shifted_order.w[-1], len(w_to_resample_to))
        extended_order =  shifted_order.extend(extended_w)

        # resample spectrum onto library wavelength
        resampled_order = extended_order.rescale(w_to_resample_to)

        # write to file
        fileroot = filename.split('/')[-1].replace('.fits', '')
        shifted_resampled_filename = '{}/{}_order{}.fits'.format(
        	kepler1656_path,
        	fileroot,
        	str(order_n))
        resampled_order.to_fits(shifted_resampled_filename)
    print('saved resampled spectrum saved to {}/'.format(shifted_resampled_filename))


