import pandas as pd
import os
import sys
import numpy as np
import pytest

from unittest import mock
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import ensemble_based_on_majority, calculate_accuracy, find_optimal_threshold

def test_ensemble_based_on_majority():
    # Test case 1: Simple majority with no tie
    hba1c_label = 'hba1c'
    pred_label = 'ensemble_pred'
    
    df1 = pd.DataFrame({
        'ldl': [1, 1, 0, 0, 1],
        'hdl': [0, 0, 0, 1, 1],
        'bmi': [0, 1, 0, 0, 1],
        'hba1c': [0, 1, 1, 0, 1],
    })
    expected_output1 = pd.Series([0, 1, 0, 0, 1], name=pred_label)  # Majority based on the input data
    result1 = ensemble_based_on_majority(df1, hba1c_label, pred_label)
    pd.testing.assert_series_equal(result1, expected_output1)

    # Test case 2: Majority with a tie that uses hba1c as tie-breaker
    df2 = pd.DataFrame({
        'ldl': [1, 1, 0, 0, 0],
        'hdl': [0, 0, 1, 1, 1],
        'bmi': [1, 1, 1, 0, 0],
        'hba1c': [0, 1, 0, 0, 1],
    })
    expected_output2 = pd.Series([0, 1, 0, 0, 1], name=pred_label) 
    result2 = ensemble_based_on_majority(df2, hba1c_label, pred_label)
    pd.testing.assert_series_equal(result2, expected_output2)

    # Test case 3: All drugs predict DPP (0)
    df3 = pd.DataFrame({
        'ldl': [0, 0, 0, 0],
        'hdl': [0, 0, 0, 0],
        'bmi': [0, 0, 0, 0],
        'hba1c': [1, 1, 1, 1], 
    })
    expected_output3 = pd.Series([0, 0, 0, 0], name=pred_label)  # All are DPP (0)
    result3 = ensemble_based_on_majority(df3, hba1c_label, pred_label)
    pd.testing.assert_series_equal(result3, expected_output3)

    # Test case 4: With empty DataFrame
    df4 = pd.DataFrame(columns=['ldl', 'hdl', 'bmi', 'hba1c'])
    expected_output4 = pd.Series([], dtype=int, name=pred_label)  # Expecting an empty series
    result4 = ensemble_based_on_majority(df4, hba1c_label, pred_label)
    pd.testing.assert_series_equal(result4, expected_output4)

    # Test case 5: All drugs predict SGLT (1)
    df5 = pd.DataFrame({
        'ldl': [1, 1, 1, 1],
        'hdl': [1, 1, 1, 1],
        'bmi': [1, 1, 1, 1],
        'hba1c': [0, 0, 0, 0],
    })
    expected_output5 = pd.Series([1, 1, 1, 1], name = pred_label)  # All should predict SGLT (1)
    result5 = ensemble_based_on_majority(df5, hba1c_label, pred_label)
    pd.testing.assert_series_equal(result5, expected_output5)

def test_calculate_accuracy():
    # Test case 1: with all correct predictions
    df = pd.DataFrame({
        'true_label': [1, 0, 1, 1, 0],
        'pred_label': [1, 0, 1, 1, 0]
    })
    
    expected_accuracy = 1.0
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"

    # Test case 2: with all incorrect predictions
    df = pd.DataFrame({
        'true_label': [1, 0, 1, 1, 0],
        'pred_label': [0, 1, 0, 0, 1]
    })
    expected_accuracy = 0.0  # All predictions are incorrect
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"

    # Test case 3: with some correct and some incorrect predictions
    df = pd.DataFrame({
        'true_label': [1, 0, 1, 1, 0],
        'pred_label': [1, 1, 1, 0, 0]
    })
    expected_accuracy = 0.6  # 3 out of 5 predictions are correct
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"
    
    # Test case 4: with an empty DataFrame
    df = pd.DataFrame(columns=['true_label', 'pred_label'])
    expected_accuracy = 0.0  # No predictions to evaluate
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"

    # Test case 5: with a single prediction
    df = pd.DataFrame({
        'true_label': [1],
        'pred_label': [1]
    })
    expected_accuracy = 1.0  # Only one correct prediction
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"
    
    # Test case 6: with different data types (string labels)
    df = pd.DataFrame({
        'true_label': ['A', 'B', 'A', 'B'],
        'pred_label': ['A', 'B', 'B', 'A']
    })
    expected_accuracy = 0.5  # 2 out of 4 predictions are correct
    result = calculate_accuracy(df, 'true_label', 'pred_label')
    assert result == expected_accuracy, f"Expected {expected_accuracy}, but got {result}"

def test_find_optimal_threshold():
    # Test case 1
    actual_values = pd.Series([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    weighted_sum = pd.Series([0.1, 0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.3, 0.85, 0.15])
    # Mock plt.show to prevent plotting during the test
    with mock.patch("matplotlib.pyplot.show"):
        result = find_optimal_threshold(actual_values, weighted_sum)
    fpr, tpr, thresholds = roc_curve(actual_values, weighted_sum)
    optimal_idx = np.argmax(tpr - fpr)
    expected_optimal_threshold = thresholds[optimal_idx]
    assert np.isclose(result, expected_optimal_threshold), f"Expected {expected_optimal_threshold}, but got {result}"
    
    # Test case 2: All actual values are the same (no positive class)
    actual_values_same = pd.Series([0, 0, 0, 0, 0])
    weighted_sum_same = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    # Mock plt.show to prevent plotting during the test
    with pytest.raises(ValueError, match="Both classes"):
        find_optimal_threshold(actual_values_same, weighted_sum_same)
        
    # Test case 3: Constant predicted probabilities (no distinction between classes)
    actual_values_constant = pd.Series([0, 1, 0, 1, 0])
    weighted_sum_constant = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5])
    with mock.patch("matplotlib.pyplot.show"):
        result_constant = find_optimal_threshold(actual_values_constant, weighted_sum_constant)
    # Since all predictions are the same, the ROC curve would be flat, and any threshold could be considered
    assert result_constant == 0.5, f"Expected threshold of 0.5, but got {result_constant}"

    # Test case 4: Predictions with clear threshold
    actual_values_perfect = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    weighted_sum_perfect = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    with mock.patch("matplotlib.pyplot.show"):
        result_perfect = find_optimal_threshold(actual_values_perfect, weighted_sum_perfect)
    # Assert the threshold is close to 0.5 with a tolerance for differences as the predictions are clean
    assert  np.isclose(result_perfect, 0.5, atol=0.1), f"Expected threshold of ~0.5 for perfect predictions, but got {result_perfect}"