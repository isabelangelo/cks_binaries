from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import thecannon as tc
from cannon_model_diagnostics import *

def train_cannon_model(order_numbers, model_suffix):
    """
    Trains a Cannon model using all the orders specified in order_numbers
    order_numbers (list): order numbers to train on, 1-16 for HIRES r chip
                        e.g., [1,2,6,15,16]
    model suffix (str): file ending for Cannon model 
                        (for example, 'order4' will save to rchip_order4.model)
    """

    # path to save model files to, 
    # should be descriptive of current model to be trained
    model_fileroot = 'rchip_{}.model'.format(model_suffix)

    # define training set labels
    training_labels = ['cks_steff', 'cks_slogg', 'cks_smet','cks_svsini']

    # Load the table containing the training set labels
    training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
    training_set = training_set_table[training_labels]

    # store training flux, sigma
    training_data_path = './data/cannon_training_data/'
    normalized_flux = np.array([])
    normalized_ivar = np.array([])
    training_arrs_created = False

    for order_n in order_numbers:
        normalized_flux_filename = training_data_path + 'training_flux_order{}.fits'.format(order_n)
        normalized_sigma_filename = training_data_path + 'training_sigma_order{}.fits'.format(order_n)

        normalized_flux_n = fits.open(normalized_flux_filename)[0].data
        normalized_sigma_n = fits.open(normalized_sigma_filename)[0].data
        normalized_ivar_n = 1/normalized_sigma_n**2

        if training_arrs_created != True:
            normalized_flux = normalized_flux_n
            normalized_ivar = normalized_ivar_n
            training_arrs_created = True

        else:
            normalized_flux = np.hstack((normalized_flux, normalized_flux_n))
            normalized_ivar = np.hstack((normalized_ivar, normalized_ivar_n))

    # Create a vectorizer that defines our model form.
    vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

    # Create the model that will run in parallel using all available cores.
    model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                           vectorizer=vectorizer)
    # train model
    model_path = './data/cannon_models/'
    model_filename = model_path + model_fileroot
    model.train()
    print('finished training cannon model')
    model.write(model_filename, include_training_set_spectra=True, overwrite=True)
    print('model written to {}'.format(model_filename))

    # generate one-to-one plots
    print('generating one-to-one diagnostic plot using leave-one-out cross-validation...')
    training_df_path = './data/cks-spectra_dataframes/'
    plot_one_to_one_leave1out(
        order_numbers, 
        training_set.to_pandas(), 
        model_path + model_fileroot + '_one_to_one.png',
        model_suffix)

# # train cannon models + save stats for all 16 individual orders
# for order_n in range(1, 17):
#     train_cannon_model([order_n], 'order{}'.format(order_n))

# # train cannon model + save stats for all 16 orders combined
# all_orders_list = np.arange(1,17,1).tolist()
# train_cannon_model(all_orders_list, 'all_orders')

# train cannon model + save stats for all orders except 11+12
no_sodium_list = [i for i in np.arange(1,17,1).tolist() if i not in [11,12]]
train_cannon_model(no_sodium_list, 'orders_11-12_omitted')






