import pytest
import pandas as pd
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import preprocess, countUsers, get_model_name, get_scores, get_outliers, pred_all, find_lowest_respponse_value,\
    find_highest_respponse_value, find_closest_to_42, get_strata, check_aggreement, get_perc, min_max_normalize, get_stats
from constants import SGLT_VALUE, DPP_VALUE

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

class MockModel:
    def predict(self, X):
        return X.sum(axis=1)  # Predict based on sum of features
    
    def score(self, X, Y):
        return 0.95  # Mock score value for training set
    
def test_get_scores():
    X_train = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    Y_train = pd.Series([5, 7, 9])
    
    X_test = pd.DataFrame({
        'feature1': [1, 2],
        'feature2': [4, 5]
    })
    Y_test = pd.Series([5, 7])
    
    # Mock model
    model = MockModel()

    # Initialize result dictionaries
    model_results = {}
    model_results_drugs = {}

    # Test case 1
    pred, model_results, model_results_drugs, score = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
    assert isinstance(pred, (np.ndarray, pd.Series)), "Predictions should be a numpy array or pandas series."
    assert score == r2_score(Y_test, X_test.sum(axis=1)), "R² score should match the expected value."
    assert 'MockModel' in model_results, "Model results should be updated."
    assert model_results['MockModel'] == score, "R² score should be added to model_results."

    # Test case 2: with empty data
    X_train = pd.DataFrame({'feature1': [], 'feature2': []})
    Y_train = pd.Series([])
    
    X_test = pd.DataFrame({'feature1': [], 'feature2': []})
    Y_test = pd.Series([])

    # Run the function and expect it to fail gracefully
    with pytest.raises(ValueError):
        pred, model_results, model_results_drugs, score = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)

def test_get_outliers():
    # Test case 1: no outliers
    Y = pd.DataFrame({
        'var1': [10, 15, 20, 25],
        'var2': [30, 35, 40, 45]
    })
    predictions = np.array([
        [10, 30],
        [15, 35],
        [302, 502],
        [203, 602]
    ])
    outliers = get_outliers(Y, predictions)
    assert outliers == [], "Expected no outliers when predictions match actual values."


def test_pred_all():
    # Mock model with a predict method
    mock_model = Mock()
    
    # Configure the mock to return a specific value when predict is called
    mock_model.predict.return_value = [0.8]

    # Test case 1: Test sglt
    row = pd.DataFrame({'feature1': [5.1], 'feature2': [3.5], 'drug_class': [SGLT_VALUE]})
    pred_sglt_, pred_dpp_ = pred_all(mock_model, row.iloc[0], SGLT_VALUE)
    assert pred_sglt_ == 0.8
    assert pred_dpp_ == 0.8
    assert row.iloc[0]['drug_class'] == SGLT_VALUE

    # Test case 2: test DPP    
    # Configure the mock to return a specific value when predict is called
    mock_model.predict.return_value = [0.9]

    # Sample row for prediction
    row = pd.DataFrame({'feature1': [6.2], 'feature2': [3.1], 'drug_class': [DPP_VALUE]})
    pred_sglt_, pred_dpp_ = pred_all(mock_model, row.iloc[0], DPP_VALUE)
    assert pred_sglt_ == 0.9
    assert pred_dpp_ == 0.9
    assert row.iloc[0]['drug_class'] == DPP_VALUE

    # Test case 3: Invalid drug class
    row = pd.DataFrame({'feature1': [6.2], 'feature2': [3.1], 'drug_class': [99]})

    # Check if ValueError is raised for an invalid drug class
    with pytest.raises(ValueError, match="No drug class for given input"):
        pred_all(mock_model, row.iloc[0], 99)

def test_find_lowest_respponse_value():
    # Test Case 1: where SGLT has the lower predicted response value
    pred_sglt = 0.5
    pred_dpp = 0.7
    min_difference, drug_class = find_lowest_respponse_value(pred_sglt, pred_dpp)
    assert min_difference == pred_sglt
    assert drug_class == SGLT_VALUE

    # Test Case 2: where DPP has the lower predicted response value
    pred_sglt = 0.8
    pred_dpp = 0.3
    min_difference, drug_class = find_lowest_respponse_value(pred_sglt, pred_dpp)
    assert min_difference == pred_dpp
    assert drug_class == DPP_VALUE

    # Test Case 3: where both predicted values are equal
    pred_sglt = 0.6
    pred_dpp = 0.6

    min_difference, drug_class = find_lowest_respponse_value(pred_sglt, pred_dpp)
    assert min_difference == pred_sglt  # or pred_dpp, since they're equal
    assert drug_class == SGLT_VALUE     # SGLT is picked if they're the same

    # Test Case 4: with edge values (e.g., one very small and one very large)
    pred_sglt = 0.0
    pred_dpp = 100.0
    min_difference, drug_class = find_lowest_respponse_value(pred_sglt, pred_dpp)
    assert min_difference == pred_sglt
    assert drug_class == SGLT_VALUE

def test_find_highest_respponse_value():
    # Test Case 1: where SGLT has the higher predicted response value
    pred_sglt = 0.8
    pred_dpp = 0.6
    max_difference, drug_class = find_highest_respponse_value(pred_sglt, pred_dpp)
    assert max_difference == pred_sglt
    assert drug_class == SGLT_VALUE

    # Test Case 2: where DPP has the higher predicted response value
    pred_sglt = 0.4
    pred_dpp = 0.9
    max_difference, drug_class = find_highest_respponse_value(pred_sglt, pred_dpp)
    assert max_difference == pred_dpp
    assert drug_class == DPP_VALUE

    # Test Case 3: where both predicted values are equal
    pred_sglt = 0.8
    pred_dpp = 0.8
    max_difference, drug_class = find_highest_respponse_value(pred_sglt, pred_dpp)
    assert max_difference == pred_sglt  # or pred_dpp, since they're equal
    assert drug_class == SGLT_VALUE     # SGLT is picked if they're the same

    # Test Case 4: with edge values (e.g., one very small and one very large)
    pred_sglt = 100.0
    pred_dpp = 0.0
    max_difference, drug_class = find_highest_respponse_value(pred_sglt, pred_dpp)
    assert max_difference == pred_sglt
    assert drug_class == SGLT_VALUE
    
def test_find_closest_to_42():
    # Test Case 1: where the SGLT value is closer to 42
    pred_sglt = 40
    pred_dpp = 50
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    assert closest_value == pred_sglt
    assert drug_class == SGLT_VALUE

    # Test Case 2: where the DPP value is closer to 42
    pred_sglt = 30
    pred_dpp = 45
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    assert closest_value == pred_dpp
    assert drug_class == DPP_VALUE

    # Test Case 3: where both predicted values are equidistant from 42
    pred_sglt = 41
    pred_dpp = 43
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    # Assert that SGLT is selected by convention when both are equidistant
    assert closest_value == pred_sglt  # Should return SGLT if both are equidistant
    assert drug_class == SGLT_VALUE

    # Test Case 4: where the DPP value is exactly 42
    pred_sglt = 50
    pred_dpp = 42
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    assert closest_value == pred_dpp
    assert drug_class == DPP_VALUE

    # Test Case 5: where the SGLT value is exactly 42
    pred_sglt = 42
    pred_dpp = 35
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    # Assert that the function picks SGLT since it is exactly 42
    assert closest_value == pred_sglt
    assert drug_class == SGLT_VALUE

    # Test Case 5: with extreme values (e.g., one very far from 42)
    pred_sglt = 1000
    pred_dpp = 10
    closest_value, drug_class = find_closest_to_42(pred_sglt, pred_dpp)
    assert closest_value == pred_dpp
    assert drug_class == DPP_VALUE

def test_get_strata():
    
    # Sample DataFrame
    data = {
        'drug_col': [1, 0, 1, 0, 1],
        'other_feature': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)
    # Test case 1: basic test
    result = get_strata(df, 'drug_col', SGLT_VALUE)
    expected = df[df['drug_col'] == SGLT_VALUE]
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    # Test case 2: empty df
    empty_df = pd.DataFrame(columns=['drug_col', 'other_feature'])
    result = get_strata(empty_df, 'drug_col', SGLT_VALUE)
    expected = pd.DataFrame(columns=['drug_col', 'other_feature'])
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    # Test case 3: invalid column name
    with pytest.raises(KeyError):
        get_strata(df, 'invalid_col', SGLT_VALUE)

def test_check_aggreement():
    data = {
    'drug_class': [1, 0, 1, 0, 1],
    'variable_name': [1, 0, 1, 0, 1],
    'other_feature': [10, 20, 30, 40, 50]
    }
    df = pd.DataFrame(data)

    # Test case 1: basic functionality
    concordant, discordant_ = check_aggreement(df, discordant=0, data=df, variable_name='variable_name')
    expected_concordant = df[df['variable_name'] == df['drug_class']]
    expected_discordant = df[df['drug_class'] == 0]
    pd.testing.assert_frame_equal(concordant.reset_index(drop=True), expected_concordant.reset_index(drop=True))
    pd.testing.assert_frame_equal(discordant_.reset_index(drop=True), expected_discordant.reset_index(drop=True))

    # Test case 2: for an empty DataFrame
    empty_df = pd.DataFrame(columns=['drug_class', 'variable_name', 'other_feature'])
    concordant, discordant_ = check_aggreement(empty_df, discordant=0, data=empty_df, variable_name='variable_name')
    expected_concordant = pd.DataFrame(columns=empty_df.columns)
    expected_discordant = pd.DataFrame(columns=empty_df.columns)
    pd.testing.assert_frame_equal(concordant.reset_index(drop=True), expected_concordant)
    pd.testing.assert_frame_equal(discordant_.reset_index(drop=True), expected_discordant)

def test_get_perc():
    # Test case 1: 
    variable_1 = np.array([1, 2, 3, 4, 5])
    variable_2 = np.array([5, 4, 3, 2, 1])
    
    expected_mean = 0.0  # Corrected mean
    expected_std = np.sqrt(8)  # Corrected standard deviation ≈ 2.8284

    mean, std = get_perc(variable_1, variable_2)
    
    assert np.isclose(mean, expected_mean), f"Expected mean: {expected_mean}, got: {mean}"
    assert np.isclose(std, expected_std), f"Expected std: {expected_std}, got: {std}"

    # Test case 2: for when both variables are the same
    variable_1 = np.array([2, 2, 2, 2])
    variable_2 = np.array([2, 2, 2, 2])
    
    expected_mean = 0.0
    expected_std = 0.0

    mean, std = get_perc(variable_1, variable_2)
    
    assert np.isclose(mean, expected_mean), f"Expected mean: {expected_mean}, got: {mean}"
    assert np.isclose(std, expected_std), f"Expected std: {expected_std}, got: {std}"

    # Test case 3: for negative numbers
    variable_1 = np.array([-1, -2, -3])
    variable_2 = np.array([-3, -2, -1])
    
    expected_mean = 0.0  # Corrected mean
    expected_std = np.sqrt(8/3)  # Corrected standard deviation

    mean, std = get_perc(variable_1, variable_2)
    
    assert np.isclose(mean, expected_mean), f"Expected mean: {expected_mean}, got: {mean}"
    assert np.isclose(std, expected_std), f"Expected std: {expected_std}, got: {std}"
    
    # Test case 4: for arrays of different sizes
    variable_1 = np.array([1, 2, 3])
    variable_2 = np.array([1, 2])

    with pytest.raises(ValueError, match="operands could not be broadcast together"):
        get_perc(variable_1, variable_2)

def test_min_max_normalize():
    # Test case 1: Normal case with positive integers
    arr = np.array([1, 2, 3, 4, 5])
    expected_normalized = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

    # Test case 2: Normal case with negative and positive values
    arr = np.array([-2, -1, 0, 1, 2])
    expected_normalized = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

    # Test case 3: All elements are the same
    arr = np.array([5, 5, 5, 5])
    expected_normalized = np.array([0.0, 0.0, 0.0, 0.0])  # Handle division by zero case
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

    # Test case 4: Empty array
    arr = np.array([])
    expected_normalized = np.array([])  # Should return an empty array
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

    # Test case 5: Array with NaN values
    arr = np.array([1, 2, np.nan, 4, 5])
    expected_normalized = np.array([0.0, 0.25, np.nan, 0.75, 1.0])
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

    # Test case 6: Single element array
    arr = np.array([10])
    expected_normalized = np.array([0.0])  # Should return 0.0 as the only value
    normalized = min_max_normalize(arr)
    np.testing.assert_array_almost_equal(normalized, expected_normalized)

def test_get_stats():
    # Sample test data
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, None, 50],
        'C': [None, None, None, None, None],  # All NaN
    }

    df = pd.DataFrame(data)
    # Test case 1: Normal case with integers
    mean, median = get_stats(df, 'A')
    assert mean == 3.0, "Mean of A should be 3.0"
    assert median == 3.0, "Median of A should be 3.0"

    # Test case 2: Normal case with floats and NaN values
    mean, median = get_stats(df, 'B')
    assert mean == 27.5, "Mean of B should be 27.5"
    assert median == 25.0, "Median of B should be 20.0"

    # Test case 3: All NaN values
    mean, median = get_stats(df, 'C')
    assert pd.isna(mean), "Mean of C should be NaN"
    assert pd.isna(median), "Median of C should be NaN"

    # Test case 4: Check behavior with a non-existing column
    try:
        get_stats(df, 'D')
    except KeyError as e:
        assert str(e) == "'D'", "Should raise KeyError for non-existing column"

