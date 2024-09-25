import pytest
import pandas as pd
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import preprocess, countUsers, get_model_name

def test_preprocess():
    # Test case 1:
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'days_hba1c': [30, 40, 50],
        'hdl_12m': [1.0, 2.0, 3.0],
        'bmi_12m': [25, 30, 55],
        'response_variable': [0.5, 0.7, 0.6],
    })
    response_variable_list = ['response_variable']
    result = preprocess(df, response_variable_list)
    expected_shape = (2, 3)  # 2 rows after filtering and 3 columns because id column and days_hba1c columns will drop during the preprocessing
    assert result.shape == expected_shape

    # Test case 2: with missing values
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'days_hba1c': [30, np.nan, 50],
        'hdl_12m': [1.0, 2.0, 3.0],
        'bmi_12m': [25, 30, 55],
        'response_variable': [0.5, np.nan, 0.6],
    })
    response_variable_list = ['response_variable']
    result = preprocess(df, response_variable_list)
    expected_shape = (1, 3)  # 1 row should remain
    assert result.shape == expected_shape
    
    # Test case 3: with out of range days_hba1c
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'days_hba1c': [30, 400, 50],
        'hdl_12m': [1.0, 2.0, 2.2],
        'bmi_12m': [44, 30, 40],
        'response_variable': [0.5, 0.7, 0.6],
    })
    response_variable_list = ['response_variable']
    result = preprocess(df, response_variable_list)
    expected_shape = (2, 3)
    assert result.shape == expected_shape
    
    # Test case 4: dopping columns
    df = pd.DataFrame({
        'id': [1, 2],
        'days_hba1c': [30, 40],
        'hdl_12m': [1.0, 2.0],
        'bmi_12m': [25, 30],
        'response_variable': [0.5, 0.7],
        'date_hba_bl_6m': ['2020-01-01', '2020-02-01'],
    })
    response_variable_list = ['response_variable']
    result = preprocess(df, response_variable_list)
    expected_columns = {'hdl_12m', 'bmi_12m', 'response_variable'}
    assert set(result.columns) == expected_columns

    # Test case 5: Exceeding bmi or hdl threshold
    df = pd.DataFrame({
        'id': [1, 2],
        'days_hba1c': [30, 40],
        'hdl_12m': [2.0, 4.0],
        'bmi_12m': [55, 30],
        'response_variable': [0.5, 0.7],
    })
    response_variable_list = ['response_variable']
    result = preprocess(df, response_variable_list)
    expected_shape = (0, 3)  # Should be empty after removing rows
    assert result.shape == expected_shape

def test_count_users():
    # Test case 1: with matching drug class
    data = {
        'drug_class': [1, 2, 3, 1, 2, 2],
        'user_id': [101, 102, 103, 104, 105, 106]
    }
    df = pd.DataFrame(data)
    # Test for drug_class = 1, 2 and 3
    result_1 = countUsers(1, df)
    result_2 = countUsers(2, df)
    result_3 = countUsers(3, df)
    expected_result_1 = 2  # There are two users with drug_class 1
    expected_result_2 = 3  # There are three users with drug_class 2
    expected_result_3 = 1  # There are one users with drug_class 3
    
    assert result_1 == expected_result_1
    assert result_2 == expected_result_2
    assert result_3 == expected_result_3
    
    # Test case 2: withn no matching drug class
    data = {
        'drug_class': [1, 2, 3, 1, 2, 2],
        'user_id': [101, 102, 103, 104, 105, 106]
    }
    df = pd.DataFrame(data)
    # test for drug class 5
    results = countUsers(5, df)
    expected_results = 0
    assert results == expected_results
    
    # Test case 3: with empty dataframe
    df_empty = pd.DataFrame(columns=['drug_class', 'user_id'])
    result = countUsers(1, df_empty)
    expected_result = 0 
    assert result == expected_result

def test_get_model_name():
    # Test case 1: Logistic regression
    model = LogisticRegression()
    expected_name = 'LogisticRegression'  # The expected name after processing
    result = get_model_name(model)
    assert result == expected_name, f"Expected '{expected_name}', but got '{result}'"

    # Test case 2: Randomforest
    model = RandomForestClassifier()
    expected_name = 'RandomForestClassifier'  # The expected name after processing
    result = get_model_name(model)
    assert result == expected_name, f"Expected '{expected_name}', but got '{result}'"

    # Test case 3: generic model
    class CustomModel:
        pass

    model = CustomModel()
    expected_name = 'CustomModel'
    result = get_model_name(model)
    assert result == expected_name, f"Expected '{expected_name}', but got '{result}'"
