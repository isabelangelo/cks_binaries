from astropy.table import Table
from astropy.io import fits
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import thecannon as tc
from cannon_model_diagnostics import *

# define training set labels
training_labels = ['cks_teff', 'cks_logg', 'cks_feh','cks_vsini']

# Load the table containing the training set labels
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

# Load the dataframe containing the training set flux, sigma
training_data_path = './data/cannon_training_data/'
training_flux_original = pd.read_csv(training_data_path+'training_flux_original.csv')
training_sigma_original = pd.read_csv(training_data_path+'training_sigma_original.csv')
training_flux_dwt = pd.read_csv(training_data_path+'training_flux_dwt.csv')
training_sigma_dwt = pd.read_csv(training_data_path+'training_sigma_dwt.csv')


def train_cannon_model(order_numbers, model_suffix, filter_type='dwt', 
    save_training_data=False, save_binary_metrics=False):
    """
    Trains a Cannon model using all the orders specified in order_numbers
    order_numbers (list): order numbers to train on, 1-16 for HIRES r chip
                        e.g., [1,2,6,15,16]
    model_suffix (str): file ending for Cannon model 
                        (for example, 'order4' will save to rchip_order4.model)
    filter_type (str): if 'dwt', model is trained on wavelet filtered data.
                       if 'original', model is trained on SpecMatch-Emp output data.
    """

    # determine dataframe that contains training data
    if filter_type=='dwt':
        flux_df = training_flux_dwt
        sigma_df = training_sigma_dwt
    else:
        flux_df = training_flux_original
        sigma_df = training_sigma_original

    # store training flux, sigma for selected orders
    # note: for flux, sigma, we index at 1 to exclude order_number column
    training_flux_df = flux_df[flux_df['order_number'].isin(order_numbers)]
    training_sigma_df = sigma_df[sigma_df['order_number'].isin(order_numbers)]
    normalized_flux = training_flux_df.to_numpy()[:,1:].T
    normalized_sigma = training_sigma_df.to_numpy()[:,1:].T
    normalized_ivar = 1/normalized_sigma**2

    # save training data to a .csv
    if save_training_data:
        flux_path = '{}training_flux_{}.csv'.format(
            training_data_path,model_suffix)
        sigma_path = '{}training_sigma_{}.csv'.format(
            training_data_path,model_suffix)
        training_flux_df.to_csv(flux_path, index=False)
        training_sigma_df.to_csv(sigma_path, index=False)

    # Create a vectorizer that defines our model form.
    vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

    # Create the model that will run in parallel using all available cores.
    model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                           vectorizer=vectorizer)

    # train and store model
    model_path = './data/cannon_models/rchip/{}/'.format(model_suffix)
    os.mkdir(model_path)
    model_filename = model_path + 'cannon_model.model'
    model.train()
    print('finished training cannon model')
    model.write(model_filename, include_training_set_spectra=True, overwrite=True)
    print('model written to {}'.format(model_filename))

    # generate one-to-one plots
    print('generating one-to-one diagnostic plot using leave-one-out cross-validation...')
    training_df_path = './data/cks-spectra_dataframes/'
    plot_one_to_one_leave1out(
        order_numbers, 
        training_set_table, 
        model_path + 'one_to_one.png',
        model_suffix,
        save_binary_metrics=save_binary_metrics)

####### train cannon models on original and wavelet-filtered spectra #######

# # all individual orders
# for order_n in range(1, 17):
#     train_cannon_model([order_n], 'order{}_dwt'.format(order_n))

# # all orders combined
# all_orders_list = np.arange(1,17,1).tolist()
# train_cannon_model(all_orders_list, 'all_orders_dwt')

# all orders except 2,3,12 + save training data
# plus the same version with the original unfiltered flux for comparison
adopted_orders = [i for i in np.arange(1,17,1).tolist() if i not in [2,12]]
train_cannon_model(adopted_orders, 'adopted_orders_dwt', save_training_data=True, save_binary_metrics=True)
train_cannon_model(adopted_orders, 'adopted_orders_original', filter_type='original')

# copy training set data to adopted orders directory
# to be used in iterative cleaning.
adopted_order_path = './data/cannon_models/rchip/adopted_orders_dwt/'
copyfile(
    training_data_path+'training_flux_adopted_orders_dwt.csv',
    adopted_order_path+'training_flux.csv')
copyfile(
    training_data_path+'training_sigma_adopted_orders_dwt.csv',
    adopted_order_path+'training_sigma.csv')
copyfile(
    './data/label_dataframes/training_labels.csv',
    adopted_order_path+'training_labels.csv')


