import spectrum
import pandas as pd
import thecannon as tc

# validation samples 
training_flux = pd.read_csv('./data/cannon_training_data/training_flux_adopted_orders_dwt.csv')
training_sigma = pd.read_csv('./data/cannon_training_data/training_sigma_adopted_orders_dwt.csv')
binary_flux = pd.read_csv('./data/spectrum_dataframes/known_binary_flux_dwt.csv')
binary_sigma = pd.read_csv('./data/spectrum_dataframes/known_binary_sigma_dwt.csv')

# model, order numbers for spectrum object
model = tc.CannonModel.read('./data/cannon_models/rchip/adopted_orders_dwt/cannon_model.model')
order_numbers = [i for i in range(1,17) if i not in (2,3,12)]

#################### chisq surface plots to validate binary model ####################

def save_chisq_surface(optimizer_str, id_starname, type='training'):
	print(id_starname)
	if type=='training':
		spec = spectrum.Spectrum(
		    training_flux[id_starname], 
		    training_sigma[id_starname],
		    order_numbers,
		    model)
	elif type=='binary':
		spec = spectrum.Spectrum(
		    binary_flux[id_starname], 
		    binary_sigma[id_starname],
		    order_numbers,
		    model)
	spec.fit_single_star()
	spec.fit_binary(save_chisq_surface_to=optimizer_str + '/' + id_starname)

# K01781: prone to delta_chisq<0 
save_chisq_surface('leastsq', 'K01781', type='training')

# K00387: binary with Teff2=3000-4200 prone to delta_chisq<0 
save_chisq_surface('leastsq', 'K00387', type='binary')

# KOI-289: q~1 binary
save_chisq_surface('leastsq', 'K00289', type='binary')

# KOI-112: q~0.7 binary
save_chisq_surface('leastsq', 'K00112', type='binary')

# KOI-1: training set star
save_chisq_surface('leastsq', 'K00001', type='training')

# KOI-41: training set star
save_chisq_surface('leastsq', 'K00041', type='training')







