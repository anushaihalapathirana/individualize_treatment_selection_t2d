import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import TRAIN_PATH, TEST_PATH, BMI_PATH, HDL_PATH, LDL_PATH, HBA1C_PATH, \
    TRAIN_PATH_WO_LDL_IMPUTATION, TEST_PATH_WO_LDL_IMPUTATION
from preprocess import ImputationPreprocessing
from hdl import ImputationHDL
from ldl import ImputationLDL
from hba1c import ImputationHbA1c
from bmi import ImputationBMI
from utils import get_dfs, read_data, missing_value_prediction

class Main():

    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, TRAIN_PATH)
        self.file_path_X_test = os.path.join(self.script_directory, TEST_PATH)
        
        self.file_path_bmi = os.path.join(self.script_directory, BMI_PATH)
        self.file_path_hdl = os.path.join(self.script_directory, HDL_PATH)
        self.file_path_ldl = os.path.join(self.script_directory, LDL_PATH)
        self.file_path_hba1c = os.path.join(self.script_directory, HBA1C_PATH)
        
        # Define the relative path to the CSV file from the script's directory
        relative_path_to_impute_train_data_wo_ldl = os.path.join("..", TRAIN_PATH_WO_LDL_IMPUTATION)
        relative_path_to_impute_test_data_wo_ldl = os.path.join("..", TEST_PATH_WO_LDL_IMPUTATION)
        
        # Get the absolute path to the CSV file
        self.file_path_imputed_train = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_train_data_wo_ldl))
        self.file_path_imputed_test = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_test_data_wo_ldl))
        
    def impute_data(self):

        imp = ImputationPreprocessing()
        df = imp.read_data()
        df, X_train, X_test, Y_train, Y_test, X, Y = imp.preprocess(df, 0.25)
        
        imputeBMI = ImputationBMI()
        df = read_data(imputeBMI.file_path_X_train)
        df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeBMI.preprocess_data(df)
        print('df_missing_val shape : ', df_missing_val.shape)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeBMI.model_training(X_train, Y_train, X_test, Y_test)
        missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeBMI.file_path_bmi_imputed, 'bmi_12m')
    
        imputeHba1c = ImputationHbA1c()
        df = read_data(imputeHba1c.file_path_X_train)
        df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeHba1c.preprocess_data(df)
        print('df_missing_val shape : ', df_missing_val.shape)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHba1c.model_training(X_train, Y_train, X_test, Y_test)
        missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeHba1c.file_path_hba1c_imputed, 'hba1c_12m')
    
        imputeHDL = ImputationHDL()
        df = read_data(imputeHDL.file_path_X_train)
        df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeHDL.preprocess_data(df)
        print('df_missing_val shape : ', df_missing_val.shape)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHDL.model_training(X_train, Y_train, X_test, Y_test)
        missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeHDL.file_path_hdl_imputed, 'hdl_12m')
    
        # imputeLDL = ImputationLDL()
        # df = read_data(imputeLDL.file_path_X_train)
        # df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeLDL.preprocess_data(df)
        # print('df_missing_val shape : ', df_missing_val.shape)
        # original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeLDL.model_training(X_train, Y_train, X_test, Y_test)
        # missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeLDL.file_path_ldl_imputed, 'ldl_12m')
    
       
    def delete_files(self):
        # Delete all the csv files
        file_paths_to_delete = [self.file_path_ldl, self.file_path_hba1c, self.file_path_hdl, self.file_path_bmi,
                                self.file_path_X_train, self.file_path_X_test]

        for file_path in file_paths_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully.")
            else:
                print(f"File {file_path} does not exist.")

    def combine_data(self):
        df_X_train = read_data(self.file_path_X_train)
        df_X_test = read_data(self.file_path_X_test)
        
        # List of file paths to read
        file_paths = [self.file_path_ldl, self.file_path_hba1c, self.file_path_hdl, self.file_path_bmi]

        # Initialize empty DataFrames
        df_ldl = pd.DataFrame()
        df_hba1c = pd.DataFrame()
        df_hdl = pd.DataFrame()
        df_bmi = pd.DataFrame()

        # Iterate over the list and read each file if it exists
        for file_name in file_paths:
            if os.path.exists(file_name):
                if 'ldl' in file_name:
                    df_ldl = read_data(file_name)
                elif 'hba1c' in file_name:
                    df_hba1c = read_data(file_name)
                elif 'hdl' in file_name:
                    df_hdl = read_data(file_name)
                elif 'bmi' in file_name:
                    df_bmi = read_data(file_name)
                    
                print(f"File {file_name} does not exist.")
                
        print("original shape df_X_train: ", np.shape(df_X_train))
        print("original shape df_X_test: ", np.shape(df_X_test))

        df_X_train = get_dfs(df_X_train)
        df_X_test = get_dfs(df_X_test)

        print(df_bmi.shape)
        print(df_hba1c.shape)
        print(df_hdl.shape)
        print(df_ldl.shape)
        print(df_X_train.shape)
        print(df_X_test.shape)

        result_df =  df_X_train.copy()
        updates = {
            'ldl_12m': df_ldl,
            'bmi_12m': df_bmi,
            'hba1c_12m': df_hba1c,
            'hdl_12m': df_hdl
        }

        for col, df_update in updates.items():
            if not df_update.empty:
                result_df = result_df.drop([f'{col}'], axis=1)
                result_df[col] = df_update[col]

        print(result_df.shape)
        
        print(result_df[['id','hba1c_12m', 'ldl_12m','hdl_12m','bmi_12m']].isna().sum())
        print(df_X_test[['id','hba1c_12m', 'ldl_12m','hdl_12m','bmi_12m']].isna().sum())
        # Save combined data
        result_df.to_csv(self.file_path_imputed_train,index=True)
        df_X_test.to_csv(self.file_path_imputed_test,index=True)
    
if __name__ == "__main__":
    print("Initialte imputation...")
    main = Main()
    main.delete_files()
    main.impute_data()
    main.combine_data()