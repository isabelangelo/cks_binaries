"""
The purpose of this code is to load various samples of stars
for analysis. It also loads information for the adopted cannon model
to use in spectrum.Spectrum.
"""
import pandas as pd
import thecannon as tc

# load data + inputs for Spectrum object
training_flux = pd.read_csv('./data/cannon_training_data/training_flux_adopted_orders_dwt.csv')
training_sigma = pd.read_csv('./data/cannon_training_data/training_sigma_adopted_orders_dwt.csv')
binary_flux = pd.read_csv('./data/spectrum_dataframes/known_binary_flux_dwt.csv')
binary_sigma = pd.read_csv('./data/spectrum_dataframes/known_binary_sigma_dwt.csv')
adopted_model = tc.CannonModel.read('./data/cannon_models/rchip/adopted_orders_dwt/cannon_model.model')
adopted_order_numbers = [i for i in range(1,17) if i not in [2,12]]