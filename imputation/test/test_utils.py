import pytest
import pandas as pd
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_nan_count, get_missing_val_percentage, get_dfs

def test_get_nan_count_mix_nan():
    # Creating a sample dataframe for testing
    data = {
        'hba1c_12m': [7.2, None, None, 6.5],
        'ldl_12m': [None, 3.5, 2.7, 2.9],
        'hdl_12m': [1.1, None, None, None],
        'bmi_12m': [24.5, 26.1, None, 2.2],
        'other_column': [1, 2, 3, None]  # This column should be ignored
    }
    df = pd.DataFrame(data)

    # Expected DataFrame for NaN counts
    expected_nan_info = pd.DataFrame({
        'Feature': ['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m'],
        'NaN Count': [2, 1, 3, 1]
    })

    # Call the function and get the result
    nan_info = get_nan_count(df)

    # Assert that the returned DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(nan_info.reset_index(drop=True), expected_nan_info.reset_index(drop=True))

def test_get_nan_count_empty_dataframe():
    # Creating an empty dataframe
    df = pd.DataFrame(columns=['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m', 'other_column'])

    # Expected DataFrame for NaN counts (all counts should be 0)
    expected_nan_info = pd.DataFrame({
        'Feature': ['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m'],
        'NaN Count': [0, 0, 0, 0]
    })
    nan_info = get_nan_count(df)
    pd.testing.assert_frame_equal(nan_info.reset_index(drop=True), expected_nan_info.reset_index(drop=True))

def test_get_nan_count_all_none_nan():
    data = {
        'hba1c_12m': [7.2, 6.9, 6.5],
        'ldl_12m': [3.5, 2.9, 3.2],
        'hdl_12m': [1.1, 0.9, 1.2],
        'bmi_12m': [24.5, 26.1, 25.0],
        'other_column': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    expected_nan_info = pd.DataFrame({
        'Feature': ['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m'],
        'NaN Count': [0, 0, 0, 0]
    })
    nan_info = get_nan_count(df)
    pd.testing.assert_frame_equal(nan_info.reset_index(drop=True), expected_nan_info.reset_index(drop=True))

@pytest.mark.xfail
def test_get_nan_count_to_fail():
    data = {
        'hba1c_12m': [7.2, None, 6.5],
        'ldl_12m': [None, None, None],
        'hdl_12m': [None, 0.9, None],
        'bmi_12m': [24.5, 26.1, 25.0],
        'other_column': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    expected_nan_info = pd.DataFrame({
        'Feature': ['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m'],
        'NaN Count': [1, 2, 2, 0]
    })
    nan_info = get_nan_count(df)
    pd.testing.assert_frame_equal(nan_info.reset_index(drop=True), expected_nan_info.reset_index(drop=True))

def test_get_missing_val_percentage_mixed_nan():
    # Creating a sample dataframe with no NaN values
    data = {
        'col1': [1, None, 3],
        'col2': [None, 5, None],
        'col3': [7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Expected output should be zero for all columns
    expected_output = pd.Series({'col1': 33.33, 'col2': 66.67, 'col3': 0.0})
    result = get_missing_val_percentage(df)
    pd.testing.assert_series_equal(result.round(2), expected_output)
    
def test_get_missing_val_percentage_all_non_nan():
    # Creating a sample dataframe with no NaN values
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    }
    df = pd.DataFrame(data)

    # Expected output should be zero for all columns
    expected_output = pd.Series({'col1': 0.0, 'col2': 0.0, 'col3': 0.0})
    result = get_missing_val_percentage(df)
    pd.testing.assert_series_equal(result, expected_output)
    
def test_get_missing_val_percentage_empty_dataframe():
    # Creating an empty dataframe
    df = pd.DataFrame(columns=['col1', 'col2', 'col3'])
    expected_output = pd.Series({'col1': float('nan'), 'col2': float('nan'), 'col3': float('nan')})
    result = get_missing_val_percentage(df)
    pd.testing.assert_series_equal(result.isna(), expected_output.isna())
    
def test_get_dfs_with_valid_input():
    data = {
        'drug_class': [3, 4, 1, 2, 3],
        'bmi': [' ', '30.5', '22.0', '24.5', '27.5'],
        'sp': ['1', '0', '1', ' ', ' '],
        'ika': ['1.2', '2.3', '3.1', '4.0', '5.5'],
        'smoking': ['1', '0', '1', '0', ' ']
    }
    df_original = pd.DataFrame(data)
    result_df = get_dfs(df_original)

    # Expected DataFrame after filtering and conversion
    expected_data = {
        'drug_class': [3, 4, 3],
        'bmi': [np.NaN, 30.5, 27.5],
        'sp': [1, 0, np.NaN],
        'ika': [1.2, 2.3, 5.5],
        'smoking': [1, 0, np.NaN]
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_get_dfs_with_no_matching_drug_classes():
    # Create a sample DataFrame with no matching drug classes
    data = {
        'drug_class': [1, 2, 5],
        'bmi': ['25.0', '30.5', '22.0'],
        'sp': ['1', '0', '1'],
        'ika': ['1.2', '2.3', '3.1'],
        'smoking': ['1', '0', ' ']
    }
    df_original = pd.DataFrame(data)
    result_df = get_dfs(df_original)
    expected_data = {
        'drug_class': pd.Series(dtype='int'),  
        'bmi': pd.Series(dtype='float'),       
        'sp': pd.Series(dtype='float'),          
        'ika': pd.Series(dtype='float'),       
        'smoking': pd.Series(dtype='float')     
    }
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

@pytest.mark.xfail
def test_get_dfs_with_wrong_conversion():
    data = {
        'drug_class': [3, 4, 1, 2, 3],
        'bmi': [' ', '30.5', '22.0', '24.5', '27.5'],
        'sp': ['1', '0', '1', ' ', ' '],
        'ika': ['1.2', '2.3', '3.1', '4.0', '5.5'],
        'smoking': ['1', '0', '1', '0', ' ']
    }
    df_original = pd.DataFrame(data)
    result_df = get_dfs(df_original)

    expected_data = {
        'drug_class': [3, 4, 3],
        'bmi': [np.NaN, 30.5, 27.5],
        'sp': ['1', '0', np.NaN],
        'ika': [1.2, 2.3, 5.5],
        'smoking': [1, 0, np.NaN]
    }
    expected_df = pd.DataFrame(expected_data)
    
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)
