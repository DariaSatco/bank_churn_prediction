# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity. 

## Project Description
**Objective**: predict churn for bank customers based on their demographical and bank activity features. 

This is binary classification problem with imbalanced classes (84% of 0's VS 16% of 1's). The dataset
includes 10127 samples with 20 features (10 numerical and 10 categorical).

Main pipeline `churn_library.py` includes the following steps:
* EDA of input data using `sweetviz` library. EDA report is generated and saved to `images/eda`
* Preprocessing of categorical features by target-encoding approach
* Training Logistic regression and Random Forest classifier and comparing them via ROC curves and ROC-AUC scores
* Saving pre-trained models to `models`
* Saving feature importance, ROC-curves and classification reports to `images/results`

Main pipeline methods are covered by unit tests in `test_churn_library.py`

All parameters needed to run pipeline are kept in `config.yaml`:
```yaml
dataset_path: ./data/bank_data.csv      # file with data
save_model_dir: ./models/               # output path to save pretrained models
save_results_dir: ./images/results/     # output path to save model training results
save_eda_dir: ./images/eda/             # output path to save EDA report

cat_columns:                            # list of categorical columns to use as features
  - Gender
  - Education_Level
  - Marital_Status
  - Income_Category
  - Card_Category

quant_columns:                          # list of numerical columns to use as features
  - Customer_Age
  - Dependent_count
  - Months_on_book
  - Total_Relationship_Count
  - Months_Inactive_12_mon
  - Contacts_Count_12_mon
  - Credit_Limit
  - Total_Revolving_Bal
  - Avg_Open_To_Buy
  - Total_Amt_Chng_Q4_Q1
  - Total_Trans_Amt
  - Total_Trans_Ct
  - Total_Ct_Chng_Q4_Q1
  - Avg_Utilization_Ratio

target_col: Churn                     # name of column where binary (1/0) target will be kept 
test_ratio: 0.3                       # share of test in train/test split

RF_param_grid:                        # parameters grid for Random Forest GridSearch
  n_estimators: [200, 500]
  max_features: ['auto', 'sqrt']
  max_depth: [4, 5, 100]
  criterion: ['gini', 'entropy']
```

## Running Files
* Create and activate environment, install requirements 
```
conda create --name your_environment_name python
conda activate your_environment_name
pip install -r requirements.txt
```
* Check `config.yaml`. You can keep default configuration or adapt paths and parameters by your choice.
* Run main pipeline
```
python churn_library.py
```
* If you are doing further changes in methods, run tests to check that pipeline is not broken. Tests can be run via `pytest`
```
pytest test_churn_library.py
```
or via main
```
python test_churn_library.py
```
* Logs of pipeline runs are kept in `logs` folder



