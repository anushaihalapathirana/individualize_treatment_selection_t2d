import pandas as pd
import numpy as np
import random
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgboost.sklearn import XGBRegressor

from constants import COMMON_VARIABLE_PATH, HDL_PATH, SEED, TRAIN_PATH
from helper import cross_val, get_scores, read_data
from utils import preprocess, remove_outliers, missing_value_prediction
   
class ImputationHDL:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, TRAIN_PATH)
        self.file_path_hdl_imputed = os.path.join(self.script_directory, HDL_PATH)
        self.file_path_common_variables = os.path.abspath(os.path.join(self.script_directory, COMMON_VARIABLE_PATH))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = ['hdl_12m']
        self.target_variable = 'hdl_12m'
        self.correlated_variables = self.common_data['correlated_variables']
    
    def preprocess_data(self, df):
        variables_to_drop = ['ldl_12m', 'hba1c_12m', 'bmi_12m', 'days_ldl', 'init_year']
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = preprocess(df, 0.25, self.target_variable, variables_to_drop)
    
        random.seed(SEED)
        
        selected_features = ['drug_class', 'MD_RCT_mmol_mol', 'hba1c_bl_18m', 'ldl', 'hdl', 'gluk', 'met_oad0']
            
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        # remove outliers
        X_train, X_test, Y_train, Y_test = remove_outliers(X_train, X_test, Y_train, Y_test, self.response_variable_list)
        
        return df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features

    def model_training(self, X_train, Y_train, X_test, Y_test):
        train = X_train.copy()
        train[self.response_variable_list] = Y_train[self.response_variable_list].copy()
        
        model_results = {}
        model_results_drugs = {}
        
        model = XGBRegressor(
            n_estimators=40, 
            eta=0.05, 
            subsample=0.9, 
            colsample_bytree=1,
            alpha=0.1,
            max_depth = 10,
            max_leaves = 8,
            learning_rate =0.1
        )
        
        model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
        # fit the model
        model.fit(X_train, Y_train)
        
        # summarize prediction
        original_data_pred, model_results, model_results_drugs_ori, score_ori = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
        return original_data_pred, model_results, model_results_drugs_ori, score_ori, model
        
    
if __name__ == "__main__":
    imputeHDL = ImputationHDL()
    df = read_data(imputeHDL.file_path_X_train)
    df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeHDL.preprocess_data(df)
    print('df_missing_val shape : ', df_missing_val.shape)
    original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHDL.model_training(X_train, Y_train, X_test, Y_test)
    missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeHDL.file_path_hdl_imputed, 'hdl_12m')
    
    
    
