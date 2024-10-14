import pandas as pd
import numpy as np
import yaml
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import read_data, preprocess, get_stats
from constants import COMMON_VARIABLE_PATH, TEST_PATH_WO_LDL_IMPUTATION, DATA_STATS_FILE_LOCATION,\
    TRAIN_PATH_WO_LDL_IMPUTATION, DATA_WITHOUT_DATES
class Stats:
    
    """This class contains calulations of the stats (mean and median) for the selected features in the final model and outcomes in both 
    the final dataset used for prediction modeling (post-imputation) and the original dataset.
    
    Keywords:
    Original:
    This dataset contains the initial data before any modifications.
   
    After Imputation:
    This dataset is created after imputing the response variable. In this step, filtered only drug classes 3 and 4, columns with more than the threshold of missing values were removed, correlated features were eliminated, and the data was filtered based on baseline HbA1c levels and eGFR levels.
   
    Final Dataset:
    In this version, samples of HbA1c measurements within the period of 21 days to 1 year were considered, and rows with missing values for the LDL response variable were dropped.
    
    """
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the full path to the CSV file
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)
    
        # Define the relative path to the CSV file from the script's directory
        relative_path_to_data_wo_dates = os.path.join("..", DATA_WITHOUT_DATES)
        relative_path_to_impute_train_data_wo_ldl = os.path.join("..", TRAIN_PATH_WO_LDL_IMPUTATION)
        relative_path_to_impute_test_data_wo_ldl = os.path.join("..", TEST_PATH_WO_LDL_IMPUTATION)
        relative_path_to_data_stats = os.path.join("..", DATA_STATS_FILE_LOCATION)
        
        # Get the absolute path to the CSV file
        self.file_path_original_data = os.path.abspath(os.path.join(self.script_directory, relative_path_to_data_wo_dates))
        self.file_path_train_data = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_train_data_wo_ldl))
        self.file_path_test_data = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_test_data_wo_ldl))
        self.file_path_data_stats = os.path.abspath(os.path.join(self.script_directory, relative_path_to_data_stats))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        
    def get_data_stats(self):
        # load original data, before imputed data
        original_df = read_data(self.file_path_original_data,  sep = ';',decimal = ',', encoding = 'utf-8', engine ='python')
        original_df = original_df.replace(' ', np.NaN)
        convert = original_df.select_dtypes('object').columns
        original_df.loc[:, convert] = original_df[convert].apply(pd.to_numeric, downcast='float', errors='coerce')
        original_df = original_df.astype(float)

        # load after imputed data - delete columns with more than threshold NaN, remove correlated features,
        #  filter by hba1c baseline levels and egfr levels
        after_impt_X_train = read_data(self.file_path_train_data)
        after_impt_X_test = read_data(self.file_path_test_data)
    
        # load final preprocessed data - drop hba1c measured period 21 days to 1 yr, drop NAN for ldl 
        final_X_train = preprocess(after_impt_X_train, self.response_variable_list)
        final_X_test = preprocess(after_impt_X_test, self.response_variable_list)
        
        data_ = {'original': original_df, 'AI_X_train': after_impt_X_train,
        'AI_X_test': after_impt_X_test, 'Final_X_Train': final_X_train, 
        'Final_X_Test': final_X_test}

        # for key, value in data.items():
        #     print(f"\nKey: {key}")
        #     print(f"Sample size: {value.shape}")
        #     for i in variables:
        #         mean, median = get_stats(value, i)
        #         print(f"{i}: mean {mean}, median: {median}")
        
        data = {'original': original_df, 
        'After_Imputation': pd.concat([after_impt_X_train, after_impt_X_test], ignore_index=True),
        'Final': pd.concat([final_X_train, final_X_test], ignore_index=True)}
        variables = ['P_Krea', 'bmi', 'eGFR', 'gluk', 'hba1c_bl_18m', 'hba1c_bl_6m',\
        'hdl', 'ika', 'ldl', 'obese', 't2d_dur_y', 'trigly', 'hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m']
        
        # Create an empty DataFrame to hold all the results
        results = pd.DataFrame()

        # Loop through the datasets and collect the stats for each variable
        for key, value in data.items():
            stats = {'Dataset': key}
            for var in variables:
                mean, median = get_stats(value, var)
                stats[f"{var}_mean"] = mean
                stats[f"{var}_median"] = median
            # Convert the stats dictionary to a DataFrame and concatenate it with the results DataFrame
            stats_df = pd.DataFrame([stats])
            results = pd.concat([results, stats_df], ignore_index=True)

        # save the results in a csv file
        results.to_csv(self.file_path_data_stats, index=False)

if __name__ == "__main__":
    dataStats = Stats()
    dataStats.get_data_stats()
