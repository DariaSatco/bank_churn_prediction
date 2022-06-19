import os
import logging
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_library_tests.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def create_test_df():
	'''
	Helper function to create test dataframe to run different tests
	'''
	test_df = pd.DataFrame(columns = ['A', 'B', 'Attrition_Flag', 'test_target'],
                           data = [[1.5, 'one', 'Existing Customer', 0], 
						   			[4.6, 'one', 'Attrited Customer', 1], 
									[7.1, 'two', 'Existing Customer', 0], 
									[1., 'one', 'Attrited Customer', 1]])
	return test_df


def test_import_data_completness():
	'''
	Test if size of imported data is correct
	'''
	df = import_data("./data/bank_data.csv")
	try:
		assert df.shape == (10127, 22)
		logging.info('SUCCESS: size of data read with import_data is correct')
	except AssertionError as err:
		logging.error('ERROR: size of data loaded via import_data is different from expected')
		raise err


def test_import_data_columns_filter():
	'''
	Test case of pre-filtered columns
	'''
	keep_cols = ['CLIENTNUM']
	df = import_data("./data/bank_data.csv", keep_cols)
	try:
		assert list(df.columns) == keep_cols
		logging.info('SUCCESS: keep_cols were correctly kept while reading data with import_data')
	except AssertionError as err:
		logging.error('ERROR: columns of output dataframe after import_data do not match the keep_cols list')
		raise err


def test_eda_file_created():
	'''
	Test that file was successfully created after perform eda called
	'''
	# create test dataframe
	test_df = create_test_df()

	# remove old report if already exists
	if os.path.isfile('./tmp/test_eda.html'):
		os.remove('./tmp/test_eda.html')
	
	# run EDA
	perform_eda(test_df, './tmp/test_eda.html')

	# check if file was successfully created
	try:
		assert os.path.exists('./tmp/test_eda.html')
		logging.info('SUCCESS: EDA test report was sucessfully created by running perform_eda.')
	except AssertionError as err:
		logging.error('ERROR: EDA report file creation failed!')
		raise err
	


def test_encoder_helper_columns_created():
	'''
	Test that encoder_helper is creating new columns with correct names
	'''
	test_df = create_test_df()
	test_df = encoder_helper(test_df, 
							category_lst = ['B', 'Attrition_Flag'],
							target_col = 'test_target', 
							response = '_test_encoder')
	try:
		set_enc_cols = set(['B_test_encoder', 'Attrition_Flag_test_encoder'])
		set_all_cols = set(test_df.columns)
		assert set_enc_cols.issubset(set_all_cols)
		logging.info('SUCCESS: encoder_helper is adding new columns to dataframe with correct names')
	except AssertionError as err:
		logging.error('ERROR: failed to add new columns with the right names!')
		raise err
	

def test_encoder_helper_encoded_vals():
	'''
	Test that encoded values are correctly calculated
	'''
	test_df = create_test_df()
	test_df = encoder_helper(test_df, 
							category_lst = ['B', 'Attrition_Flag'],
							target_col = 'test_target', 
							response = '_test_encoder')
	ref_array = np.round([[0.66666667, 0.],
       					  [0.66666667, 1.],
       					  [0., 0.],
						  [0.66666667, 1.]], 2)
	try:
		assert np.array_equal(np.round(test_df[['B_test_encoder', 'Attrition_Flag_test_encoder']].values, 2), ref_array)
		logging.info('SUCCESS: values encoded by encoder_helper are correct')
	except AssertionError as err: 
		logging.error('ERROR: encoder_helper generated incorrect encoding!')
		raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








