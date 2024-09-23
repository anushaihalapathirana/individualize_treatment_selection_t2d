import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, roc_auc_score
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from helper import calculate_count_diff

def ensemble_based_on_majority(df_drug, hba1c_label, pred_label):
    
    """
    Applies a majority vote approach to determine the ensemble drug prediction based on multiple assigned drug features.

    Args:
        df_drug (DataFrame): A DataFrame containing the assigned drug predictions for multiple features (e.g., hba1c, LDL, HDL, BMI).
        hba1c_label (str): The column name corresponding to the hba1c drug prediction, used as a tie-breaker when necessary.
        pred_label (str): The name of the column to store the ensemble drug prediction.

    Returns:
        Series: A Series containing the ensemble drug prediction for each row, where:
            - If the sum of assigned drug predictions equals half the number of features, the value of `hba1c_label` is used as the tie-breaker.
            - If the sum of assigned drug predictions is greater than half the number of features (SGLT_VALUE = 1, DPP_VALUE = 0),
            the ensemble prediction is SGLT, otherwise DPP.

    Example:
        For a DataFrame with four assigned drug columns, if two out of four drugs are predicted, the method will use the hba1c value as the tie-breaker.
        Otherwise, it will assign the majority predicts the drug.
    """
    
    df = df_drug.copy()
    num_columns = df.shape[1]
    row_sums = df.sum(axis=1)
    df[pred_label] = np.where(row_sums == (num_columns / 2), df[hba1c_label], (row_sums > (num_columns / 2)).astype(int))
    return df[pred_label]

def calculate_accuracy(df, true_label, pred_label):
    
    """
    Calculates the accuracy of predictions by comparing the true labels with the predicted labels.

    Args:
        df (DataFrame): A DataFrame containing the true and predicted labels.
        true_label (str): The column name representing the true labels in the DataFrame.
        pred_label (str): The column name representing the predicted labels in the DataFrame.

    Returns:
        float: The accuracy of the predictions, calculated as the ratio of correct predictions to the total number of predictions.
    """
    
    correct_predictions = (df[true_label] == df[pred_label]).sum()
    total_predictions = df.shape[0]
    accuracy = correct_predictions / total_predictions
    return accuracy


def find_optimal_threshold(actual_values, weighted_sum):
    
    """
    Finds the optimal threshold for a binary classification model using the ROC curve and Youden's J statistic.
    The function also plots the ROC curve and highlights the optimal threshold.

    Args:
        actual_values (Series): The true binary labels.
        weighted_sum (Series): The predicted scores or probabilities from the model.

    Returns:
        float: The optimal threshold value based on Youden's J statistic, which maximizes the difference 
               between the true positive rate (TPR) and false positive rate (FPR).
               
    Visualization:
        - The ROC curve shows the trade-off between the TPR and FPR at different threshold levels.
        - The optimal threshold is marked with a red dot, representing the best balance between sensitivity and specificity.

    """
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(actual_values, weighted_sum)
    roc_auc = roc_auc_score(actual_values, weighted_sum)

    # Find the optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plotting the ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', label=f'Optimal Threshold: {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return optimal_threshold
      
def calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                          response_variable, baseline_val, predicted_change, variable_name):
    
    """
    Calculates and prints the difference between actual and predicted changes in a given variable over a 12-month period for concordant patient groups.

    Args:
        concordant_dpp (DataFrame): DataFrame containing patients in the DPP group whose treatment predictions are concordant with actual outcomes.
        discordant_dpp_sglt (DataFrame): DataFrame containing patients who were predicted to receive DPP treatment but were actually treated with SGLT.
        concordant_sglt (DataFrame): DataFrame containing patients in the SGLT group whose treatment predictions are concordant with actual outcomes.
        discordant_sglt_dpp (DataFrame): DataFrame containing patients who were predicted to receive SGLT treatment but were actually treated with DPP.
        response_variable (str): The name of the variable representing the 12-month outcome (e.g., 'hba1c_12m', 'ldl_12m').
        baseline_val (str): The name of the variable representing the baseline value (e.g., 'hba1c_bl_6m', 'ldl').
        predicted_change (str): The name of the column representing the predicted change in the response variable.
        variable_name (str): The name of the variable being analyzed (e.g., 'hba1c', 'ldl', 'hdl', 'bmi') for reporting in the print statement.

    Returns:
        None: This function calculates and prints the total number of patients who showed improvement in the given variable 
              (actual vs. predicted) for both concordant SGLT and DPP patient groups.
    """
    
    concordant_sglt_actual, concordant_sglt_pred, sglt_greater_than_bl_actual, sglt_greater_than_bl_pred = calculate_count_diff(concordant_sglt, 
                                                                                                                                response_variable, baseline_val, predicted_change)
    concordant_dpp_actual, concordant_dpp_pred, dpp_greater_than_bl_actual, dpp_greater_than_bl_pred = calculate_count_diff(concordant_dpp, response_variable, baseline_val, predicted_change)
                                                                                                                
    print("The number of patients who showed improvement over 12-month with ", variable_name," change (observed vs predicted)", concordant_sglt_actual + concordant_dpp_actual , ':', concordant_sglt_pred + concordant_dpp_pred)
