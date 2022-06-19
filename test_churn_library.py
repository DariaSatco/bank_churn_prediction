"""
This is module with unit tests for churn_library.py methods
"""

import os
import glob
import logging

import pandas as pd
import numpy as np

from sklearn.datasets import make_moons
from churn_library import (import_data,
                           perform_eda,
                           encoder_helper,
                           perform_feature_engineering,
                           train_models)


def create_test_df():
    '''
    Helper function to create test dataframe to run different tests
    '''
    test_df = pd.DataFrame(columns=['A', 'B', 'Attrition_Flag', 'test_target'],
                           data=[[1.5, 'one', 'Existing Customer', 0],
                                 [4.6, 'one', 'Attrited Customer', 1],
                                 [7.1, 'two', 'Existing Customer', 0],
                                 [1., 'one', 'Attrited Customer', 1]])
    return test_df


def test_import_data_completness():
    '''
    Test if size of imported data is correct
    '''
    test_df = import_data("./data/bank_data.csv")
    try:
        assert test_df.shape == (10127, 22)
        logging.info('SUCCESS: size of data read with import_data is correct')
    except AssertionError as err:
        logging.error(
            'ERROR: size of data loaded via import_data is different from expected')
        raise err


def test_import_data_columns_filter():
    '''
    Test case of pre-filtered columns
    '''
    keep_cols = ['CLIENTNUM']
    test_df = import_data("./data/bank_data.csv", keep_cols)
    try:
        assert list(test_df.columns) == keep_cols
        logging.info(
            'SUCCESS: keep_cols were correctly kept while reading data with import_data')
    except AssertionError as err:
        logging.error(
            'ERROR: columns of output dataframe after import_data do not match the keep_cols list')
        raise err


def test_eda_file_created():
    '''
    Test that file was successfully created after perform eda called
    '''
    # create test dataframe
    test_df = create_test_df()

    # create tmp and report folders if not exists
    os.makedirs('./tmp', exist_ok=True)
    os.makedirs('./tmp/report', exist_ok=True)

    # remove old report if already exists
    if os.path.isfile('./tmp/report/test_eda.html'):
        os.remove('./tmp/report/test_eda.html')

    # run EDA
    perform_eda(test_df, './tmp/report/test_eda.html')

    # check if file was successfully created
    try:
        assert os.path.exists('./tmp/report/test_eda.html')
        logging.info(
            'SUCCESS: EDA test report was sucessfully created by running perform_eda.')
    except AssertionError as err:
        logging.error('ERROR: EDA report file creation failed!')
        raise err


def test_encoder_helper_columns_created():
    '''
    Test that encoder_helper is creating new columns with correct names
    '''
    # create test dataframe
    test_df = create_test_df()
    # encode category columns
    test_df = encoder_helper(test_df,
                             category_lst=['B', 'Attrition_Flag'],
                             target_col='test_target',
                             response='_test_encoder')
    try:
        set_enc_cols = set(['B_test_encoder', 'Attrition_Flag_test_encoder'])
        set_all_cols = set(test_df.columns)
        assert set_enc_cols.issubset(set_all_cols)
        logging.info(
            'SUCCESS: encoder_helper is adding new columns to dataframe with correct names')
    except AssertionError as err:
        logging.error('ERROR: failed to add new columns with the right names!')
        raise err


def test_encoder_helper_encoded_vals():
    '''
    Test that encoded values are correctly calculated
    '''
    # create test dataframe
    test_df = create_test_df()
    # encode category columns
    test_df = encoder_helper(test_df,
                             category_lst=['B', 'Attrition_Flag'],
                             target_col='test_target',
                             response='_test_encoder')
    ref_array = np.round([[0.66666667, 0.],
                          [0.66666667, 1.],
                          [0., 0.],
                          [0.66666667, 1.]], 2)
    try:
        assert np.array_equal(np.round(
            test_df[['B_test_encoder', 'Attrition_Flag_test_encoder']].values, 2), ref_array)
        logging.info('SUCCESS: values encoded by encoder_helper are correct')
    except AssertionError as err:
        logging.error('ERROR: encoder_helper generated incorrect encoding!')
        raise err


def test_perform_feature_split_ratio():
    '''
    Test that perform_feature_engineering is generating correct splits
    '''
    test_df = create_test_df()
    params = {'target_col': 'test_target',
              'test_ratio': 0.5,
              'cat_columns': ['B'],
              'quant_columns': ['A']}

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        test_df, params)
    try:
        assert len(y_test) / len(test_df) == 0.5
        logging.info(
            'SUCCESS: perform_feature_engineering output split validated')
    except AssertionError as err:
        logging.error(
            f'ERROR: split ratio is incorrect! Expected {params["split_ratio"]}, \
				but got {len(y_test)/len(test_df)}')
        raise err


def test_train_models_outputs_generated():
    '''
    Test that train_models is generating all necessary files
    '''
    # clean tmp folder
    files_list = [f for f in glob.glob('./tmp/*') if not os.path.isdir(f)]
    for file in files_list:
        os.remove(file)
    assert len([f for f in glob.glob('./tmp/*') if not os.path.isdir(f)]) == 0

    # generate dataset
    X, y = make_moons(n_samples=100, noise=0.1)
    X_df = pd.DataFrame(X, columns=['X', 'Y'])
    y_df = pd.DataFrame(y, columns=['target'])
    test_df = pd.concat([X_df, y_df], axis=1)

    # prepare train/test
    params = {'target_col': 'target',
              'test_ratio': 0.2,
              'cat_columns': [],
              'quant_columns': ['X', 'Y'],
              'save_model_dir': './tmp/',
              'save_results_dir': './tmp/',
              'save_eda_dir': './tmp/',
              'RF_param_grid': {'n_estimators': [10, 20],
                                'max_depth': [4, 5]}
              }
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        test_df, params)

    # call train_models
    train_models(X_train, X_test, y_train, y_test, params)

    # check that all files were successfully created
    counter = 0
    try:
        assert os.path.isfile('./tmp/rfc_model.pkl')
    except AssertionError:
        logging.error('ERROR: File ./tmp/rfc_model.pkl was not created')
        counter += 1

    try:
        assert os.path.isfile('./tmp/logistic_model.pkl')
    except AssertionError:
        logging.error('ERROR: File ./tmp/logistic_model.pkl was not created')
        counter += 1

    try:
        assert os.path.isfile('./tmp/roc_curves.png')
    except AssertionError:
        logging.error('ERROR: File ./tmp/roc_curves.png was not created')
        counter += 1

    try:
        assert os.path.isfile('./tmp/feature_importance.png')
    except AssertionError:
        logging.error(
            'ERROR: File ./tmp/feature_importance.png was not created')
        counter += 1

    try:
        assert os.path.isfile('./tmp/rf_classification_report.png')
    except AssertionError:
        logging.error(
            'ERROR: File ./tmp/rf_classification_report.png was not created')
        counter += 1

    try:
        assert os.path.isfile('./tmp/lr_classification_report.png')
    except AssertionError:
        logging.error(
            'ERROR: File ./tmp/lr_classification_report.png was not created')
        counter += 1

    try:
        assert counter == 0
        logging.info(
            'SUCCESS: all necessary files were succesfully generated!')
    except AssertionError as err:
        logging.error('ERROR: Some of the files were not generated')
        raise err


if __name__ == "__main__":

    logging.basicConfig(
        filename='./logs/churn_library_tests.log',
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    test_import_data_completness()
    test_import_data_columns_filter()
    test_eda_file_created()
    test_encoder_helper_columns_created()
    test_encoder_helper_encoded_vals()
    test_perform_feature_split_ratio()
    test_train_models_outputs_generated()
