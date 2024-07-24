from astropy.table import Table
from astropy.io import fits
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


def train_cannon_model(order_numbers, model_suffix, filter_type='dwt', save_training_data=False):
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
    model_path = './data/cannon_models/ichip/{}/'.format(model_suffix)
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
        training_set.to_pandas(), 
        model_path + 'one_to_one.png',
        model_suffix)

####### train cannon models with wavelet filters #######
# all individual orders
for order_n in range(2, 3):
    train_cannon_model([order_n], 'order{}_dwt'.format(order_n))
    train_cannon_model([order_n], 'order{}_original'.format(order_n), filter_type='original')

# all orders combined
# all_orders_list = np.arange(1,11,1).tolist()
# train_cannon_model(all_orders_list, 'all_orders_dwt')

# all orders except 11+12 + save training data
# no_sodium_list = [i for i in np.arange(1,17,1).tolist() if i not in [2, 11,12]]
# train_cannon_model(no_sodium_list, 'orders_2.11.12_omitted_dwt', save_training_data=True)

###### train cannon models with original spectra #######
# all individual orders
# for order_n in range(1, 11):
#     train_cannon_model([order_n], 'order{}_original'.format(order_n), filter_type='original')

# all orders combined
# all_orders_list = np.arange(1,11,1).tolist()
# train_cannon_model(all_orders_list, 'all_orders_original', filter_type='original')

# orders except 11+12, without wavelet filtering
# no_sodium_list = [i for i in np.arange(1,17,1).tolist() if i not in [2, 11,12]]
# train_cannon_model(no_sodium_list, 'orders_2.11.12_omitted_original', filter_type='original')






