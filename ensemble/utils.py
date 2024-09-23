import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, roc_auc_score
from tabulate import tabulate

def ensemble_based_on_majority(df_drug, hba1c_label, pred_label):
    df = df_drug.copy()
    num_columns = df.shape[1]
    row_sums = df.sum(axis=1)
    df[pred_label] = np.where(row_sums == (num_columns / 2), df[hba1c_label], (row_sums > (num_columns / 2)).astype(int))
    return df[pred_label]

def calculate_accuracy(df, true_label, pred_label):
    correct_predictions = (df[true_label] == df[pred_label]).sum()
    # Calculate the total number of predictions
    total_predictions = df.shape[0]
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


def find_optimal_threshold(actual_values, weighted_sum):
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

def check_aggreement(df, discordant_1, data, variable_name):
    
    concordant_glp = pd.DataFrame(columns=data.columns)
    discordant_df_1 = pd.DataFrame(columns=data.columns)

    concordant = df[df[variable_name] == df['drug_class']]
    discordant_df_1 = df[df['drug_class'] == discordant_1]
    
    return concordant, discordant_df_1

def get_concordant_discordant(dpp_strata,sglt_strata, data, dpp_strata_actual, sglt_strata_actual, variable_name):

    sglt_val = 1
    dpp_val = 0
    # discordant_dpp_sglt = received SGLT actually but model assigned DPP
    # discordant_sglt_dpp = received DPP in real life but our model assigned SGLT
    
    concordant_dpp, discordant_dpp_sglt = check_aggreement(dpp_strata, sglt_val, data, variable_name)

    concordant_sglt, discordant_sglt_dpp = check_aggreement(sglt_strata, dpp_val, data, variable_name)

    print(" =========== Total number of samples assigned by the model VS Total number of samples in original test data")
    print('DPP samples ', concordant_dpp.shape[0]+discordant_dpp_sglt.shape[0],  dpp_strata_actual.shape[0])
    print('SGLT samples ', concordant_sglt.shape[0]+discordant_sglt_dpp.shape[0], sglt_strata_actual.shape[0])
    print('\n')
   
    
    concordant_dpp_count = concordant_dpp.shape[0]
    discordant_dpp_sglt_count = discordant_dpp_sglt.shape[0]
    concordant_sglt_count = concordant_sglt.shape[0]
    discordant_sglt_dpp_count = discordant_sglt_dpp.shape[0]

    if((concordant_dpp_count + discordant_dpp_sglt_count != 0) & (concordant_sglt_count + discordant_sglt_dpp_count !=0)):
    # Calculate percentages
        concordant_dpp_percentage = (concordant_dpp_count / (concordant_dpp_count + discordant_dpp_sglt_count)) * 100
        concordant_sglt_percentage = (concordant_sglt_count / (concordant_sglt_count + discordant_sglt_dpp_count)) * 100
        discordant_dpp_sglt_percentage = (discordant_dpp_sglt_count / (concordant_dpp_count + discordant_dpp_sglt_count)) * 100
        discordant_sglt_dpp_percentage = (discordant_sglt_dpp_count / (concordant_sglt_count + discordant_sglt_dpp_count)) * 100
    else:
        concordant_dpp_percentage = 1
        concordant_sglt_percentage = 1
        discordant_dpp_sglt_percentage=1
        discordant_sglt_dpp_percentage =1
    # Data for the table
    data = [
        ["Concordant", "SGLT","SGLT", concordant_sglt_count, f"{concordant_sglt_percentage:.2f}%"],
        ["Discordant", "DPP", "SGLT", discordant_sglt_dpp_count, f"{discordant_dpp_sglt_percentage:.2f}%"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", concordant_dpp_count, f"{concordant_dpp_percentage:.2f}%"],
        ["Discordant", "SGLT", "DPP", discordant_dpp_sglt_count, f"{discordant_sglt_dpp_percentage:.2f}%"],
    ]

    # Print the table
    print(tabulate(data, headers=["Category","Real value", "Predicted value",  "Count", "Percentage of Predicted cases"]))
    print('\n')
    
    return ( concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp)

def get_perc(variable_1, variable_2):
    normal = 42.0
    std = (variable_1-variable_2).std()
    mean = (variable_1-variable_2).mean()
    return mean, std
    
def percentage_change_original_data(dpp_strata_actual, sglt_strata_actual, baseline_val, response_variable):
    # Calculate percentages for each category
    sglt_percentage, sglt_std = get_perc(sglt_strata_actual[response_variable], sglt_strata_actual[baseline_val])
    dpp_percentage, dpp_std = get_perc(dpp_strata_actual[response_variable], dpp_strata_actual[baseline_val])
    

    # Data for the table
    data = [
        ["SGLT", f"{sglt_percentage:.2f}", f"{sglt_std:.2f}"],
        ["DPP", f"{dpp_percentage:.2f}", f"{dpp_std:.2f}"]
    ]

    # Print the table
    headers = ["Category", "Mean Percentage Change from Baseline (original dataset)", "standard deviation of the percentage change from Baseline (original dataset)"]
    print(tabulate(data, headers=headers))
    
def calculate_count_diff(data, response_variable, baseline_val, predicted_change ):
    # Use vectorized operations to compare entire columns at once
    
    real_change = (data[response_variable] - data[baseline_val])
    pred_change = (data[predicted_change] - data[baseline_val])
    
    count_actual = (real_change > pred_change).sum()
    count_pred = (real_change < pred_change).sum()
    
    greater_than_bl_actual = (real_change>0).sum()
    greater_than_bl_pred = (pred_change>0).sum()
    
    return count_actual, count_pred, greater_than_bl_actual, greater_than_bl_pred
    
def calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                          response_variable, baseline_val, predicted_change, variable_name):
    
    concordant_sglt_actual, concordant_sglt_pred, sglt_greater_than_bl_actual, sglt_greater_than_bl_pred = calculate_count_diff(concordant_sglt, 
                                                                                                                                response_variable, baseline_val, predicted_change)
    concordant_dpp_actual, concordant_dpp_pred, dpp_greater_than_bl_actual, dpp_greater_than_bl_pred = calculate_count_diff(concordant_dpp, response_variable, baseline_val, predicted_change)
                                                                                                                
    print("The number of patients who showed improvement over 12-month with ", variable_name," change (observed vs predicted)", concordant_sglt_actual + concordant_dpp_actual , ':', concordant_sglt_pred + concordant_dpp_pred)
    
    
    