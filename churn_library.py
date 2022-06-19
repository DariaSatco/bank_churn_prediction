"""
This is a module with necessary methods for customer churn prediction model
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import yaml
import joblib
import logging

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

import matplotlib.pyplot as plt
import sweetviz as sv


logging.basicConfig(
    filename='./logs/churn_library_main.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def load_params(config_file_pth: str) -> Dict:
    '''
    Read parameters from yaml config file

    Args:
        config_file_pth (string): path to the yaml file with configuration

    Returns:
        parameters (dictionary): dictionary witn parameters
    '''
    try:
        with open(config_file_pth) as config_file:
            parameters = yaml.safe_load(config_file)
        logging.info(f'Loaded parameters from {config_file_pth}')
        return parameters
    except FileNotFoundError as err:
        logging.error("ERROR: YAML file read failed. The file {config_file_pth} wasn't found")
        raise err


def import_data(pth: str, 
                keep_cols: List[str] = None) -> pd.DataFrame:
    '''
    Returns dataframe for the csv found at pth

    Args:
        pth (string)    : path to the csv file
        keep_cols (list): list of columns to read from file

    Returns:
        df (DataFrame): pandas dataframe from file
    '''	
    try:
        df = pd.read_csv(pth, usecols = keep_cols)
        logging.info(f"Dataframe loaded from {pth}: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError as err:
        logging.error("ERROR: CSV file read failed. The file {pth} wasn't found")
        raise err
    except ValueError as err:
        logging.error('ERROR: List of columns to keep from csv file do not match file content.')
        raise err
    

def perform_eda(df: pd.DataFrame, 
                save_to: str) -> None:
    '''
    Perform eda using sweetviz functionality to generate HTML
    with an overview of data
    
    Args:
        df (DataFrame)   : dataframe to analyze
        save_to (string) : path to save output file
    
    Returns:
        None
    '''
    analysis = sv.analyze([df, 'input_data'])
    analysis.show_html(save_to)
    logging.info(f'Saved EDA to {save_to}')


def compare_churn_vs_stayed(df: pd.DataFrame,
                            churn_col: str,
                            save_to: str) -> None:
    '''
    Run comparison by each feature between Churned and Not churned
    customers using sweetviz functionality to generate
    HTML report

    Args:
        df (DataFrame)   : dataframe to analyze
        churn_col (str)  : name of column with 1/0 values corresponding to
                           Churned / Not churned
        save_to (string) : path to save output file

    Returns:
        None
    '''
    report = sv.compare_intra(df, df[churn_col]==1, ["Churn", "Stayed"])
    report.show_html(save_to)
    logging.info(f'Saved Churn vs stayed data comparison to {save_to}')


def build_target(df: pd.DataFrame,
                 parameters: Dict) -> pd.DataFrame:
    '''
    Createst target column as new column in input dataframe. Drops existing column
    Attrition_Flag used for target creation

    Args:
        df (DataFrame)   : input dataframe
        parameters (Dict): parameters dictionary

    Returns:
        DataFrame with target column added
    '''
    target_col = parameters['target_col']
    try:
        df[target_col] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        df = df.drop(columns=['Attrition_Flag'])
        logging.info('Generated target column with name' + parameters['target_col'])
        return df
    except KeyError as err:
        logging.error('ERROR: While build_target run Attrition_Flag column was not found in dataframe!')
        raise err
   
    
def encoder_helper(df: pd.DataFrame, 
                   category_lst: List[str],
                   target_col: str, 
                   response: str = '') -> pd.DataFrame:
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category. If response argument is not given
    encoded features will be saved with same column names, alternatively
    there will be created new columns with names = original column name + response

    Args:
        df (DataFrame)     : input dataset
        category_lst (list): list of columns that contain categorical features
        target_col (string): name of binary (1/0) target column
        response (string)  : string of response name [optional argument that could 
                be used for naming variables or index y column]

    Returns:
        df (DataFrame): pandas dataframe with new columns for
    '''
    for col in category_lst:
        df[col + response] = df.groupby(col)[target_col].transform('mean')
        logging.info(f'{col + response} was added to the dataframe by encoding original column {col}')
    return df
    

def perform_feature_engineering(df: pd.DataFrame, 
                                parameters: Dict,
                                response: str = '') -> Tuple:
    '''
    Prepares data for model training:
        * features and target values 
        * train/test split

    Args:
        df (DataFrame)     : input data
        parameters (Dict)  : dictionary with parameters
        response (string)  : string of response name [optional argument that could 
                be used for naming variables or index y column]

    Returns:
        X_train (DataFrame) : X training data
        X_test (DataFrame)  : X testing data
        y_train (DataFrame) : y training data
        y_test (DataFrame)  : y testing data
    '''
    y = df[parameters['target_col']].copy()

    df = encoder_helper(df, parameters['cat_columns'], parameters['target_col'],
                        response=response)
    x_cols = parameters['quant_columns'] + [col + response for col in parameters['cat_columns']]
    X = df[x_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = parameters['test_ratio'], 
                                                        random_state = 42)
    logging.info(f'Prepared train/test dataset split with {1-parameters["test_ratio"]}/{parameters["test_ratio"]} ratio')                                                     
    return X_train, X_test, y_train, y_test



def classification_report_image(model: sklearn.base.BaseEstimator,
                                X_train: pd.DataFrame, 
                                X_test: pd.DataFrame, 
                                y_train: pd.DataFrame, 
                                y_test: pd.DataFrame,
                                model_name: str,
                                output_pth: str) -> None:
    '''
    Run classification report and saves it into image with output_pth

    Args:
        model (BaseEstimator) : model object
        X_train (DataFrame)   : X training data
        X_test (DataFrame)    : X testing data
        y_train (DataFrame)   : y training data
        y_test (DataFrame)    : y testing data
        model_name (string)   : name of the model to put on plot
        output_pth (string)   : path to the output file to save plot

    Returns
        None
    '''
    # calculate predictions
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)

    fig = plt.figure(figsize=(5,5))
    
    # generate train scores
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)), {'fontsize': 10}, fontproperties = 'monospace')
    
    # generate test scores
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)), {'fontsize': 10}, fontproperties = 'monospace')
    
    # kill X and Y axises
    plt.axis('off')

    # save image
    fig.savefig(output_pth, dpi=fig.dpi, format='png', bbox_inches='tight')
    logging.info(f'Calssification report plot for {model_name} saved to {output_pth}')



def roc_curves(model_list: List,
               X_test: pd.DataFrame,
               y_test: pd.DataFrame,
               output_pth: str) -> None:
    '''
    Creates and stores ROC Curve plots in output_pth for all models
    from model_list

    Args:
        model_list (list)   : list of model objects
        X_test (DataFrame)  : test features table
        y_test (DataFrame)  : test target table
        output_pth (string) : file path to save image

    Returns:
        None
    '''
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for model in model_list:
        plot_roc_curve(model, X_test, y_test, ax=ax, alpha=0.8)
    fig.savefig(output_pth, dpi=fig.dpi, format='png', bbox_inches='tight')
    logging.info(f'ROC curve plot saved to {output_pth}')
    


def feature_importance_plot(model: sklearn.base.BaseEstimator,
                            X_data: pd.DataFrame, 
                            output_pth: str) -> None:
    '''
    Creates and stores the feature importances in output_pth
    
    Args:
        model (BaseEstimetor): model object containing feature_importances_
        X_data (DataFrame)   : pandas dataframe of X values
        output_pth (string)  : path to store the figure

    Returns:
        None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(10,7))

    # Create plot title
    plt.title("Feature Importance")
    plt.xlabel('Importance')

    # Add bars
    plt.barh(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.yticks(range(X_data.shape[1]), names)

    # save plot to output_pth
    fig.savefig(output_pth, dpi=fig.dpi, format='png', bbox_inches='tight')
    logging.info(f'Feature importance plot saved to {output_pth}')
    


def train_models(X_train: pd.DataFrame, 
                 X_test: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 y_test: pd.DataFrame, 
                 parameters: Dict) -> None:
    '''
    Train models and store model results: images + scores, and store models
    
    Args:
        X_train (DataFrame): X training data
        X_test (DataFrame) : X testing data
        y_train (DataFrame): y training data
        y_test (DataFrame) : y testing data
        parameters (Dict)  : parameters in dictionary format
    
    Returns:
        None
    '''
    # initialize models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # parameters for RF tuning
    param_grid = parameters['RF_param_grid']

    # fit RF
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # fit logistic regression
    lrc.fit(X_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, parameters['save_model_dir'] + 'rfc_model.pkl')
    logging.info('Saced RF best classifier to ' + parameters['save_model_dir'] + 'rfc_model.pkl')

    joblib.dump(lrc, parameters['save_model_dir'] + 'logistic_model.pkl')
    logging.info('Saved Logistic regression model to ' + parameters['save_model_dir'] + 'logistic_model.pkl')

    # create and save plots
    roc_curves([cv_rfc.best_estimator_, lrc], X_test, y_test, parameters['save_results_dir'] + 'roc_curves.png')
    feature_importance_plot(cv_rfc.best_estimator_, X_train, parameters['save_results_dir'] + 'feature_importance.png')

    # save score reports
    classification_report_image(cv_rfc.best_estimator_, X_train, X_test, 
    y_train, y_test, 'Random Forest', parameters['save_results_dir'] + 'rf_classification_report.png')
    classification_report_image(lrc, X_train, X_test, y_train, y_test, 
    'Logistic Regression', parameters['save_results_dir'] + 'lr_classification_report.png')


def load_model(model_path: str):
    '''
    Load model from model_path

    Args:
        model_path (string) : path where model was saved

    Returns:
        model object
    '''
    model = joblib.load(model_path) 
    return model


if __name__ == '__main__':
    
    # load parameters
    parameters = load_params('config.yaml')

    # read dataset
    columns_to_read = parameters['cat_columns'] + parameters['quant_columns'] + ['Attrition_Flag']
    input_data = import_data(parameters['dataset_path'], keep_cols = columns_to_read)

    # generate EDA report
    perform_eda(input_data, parameters['save_eda_dir'] + 'sample_data_overview.html')

    # create target column
    preprocessed_data = build_target(input_data, parameters)

    # generate Churn vs Stayed data overview
    compare_churn_vs_stayed(preprocessed_data, parameters['target_col'], parameters['save_eda_dir'] + 'Churn vs stayed.html')

    # prepare train/test data
    X_train, X_test, y_train, y_test = perform_feature_engineering(preprocessed_data, parameters)

    # train models and save outcomes
    train_models(X_train, X_test, y_train, y_test, parameters)
    logging.info('SUCCESS: Trained models and saved results')