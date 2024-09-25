import pandas as pd
import numpy as np
import random
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.neural_network import MLPRegressor

from constants import COMMON_VARIABLE_PATH, HBA1C_PATH, SEED, TRAIN_PATH
from helper import cross_val, get_scores
from utils import preprocess, remove_outliers, read_data, missing_value_prediction

class ImputationHbA1c:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, TRAIN_PATH)
        self.file_path_hba1c_imputed = os.path.join(self.script_directory, HBA1C_PATH)
        self.file_path_common_variables = os.path.abspath(os.path.join(self.script_directory, COMMON_VARIABLE_PATH))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = ['hba1c_12m']
        self.target_variable = 'hba1c_12m'
        self.correlated_variables = self.common_data['correlated_variables']
        
    def preprocess_data(self, df):
        variables_to_drop = ['ldl_12m', 'bmi_12m', 'hdl_12m', 'days_ldl', 'init_year']
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = preprocess(df, 0.25, self.target_variable, variables_to_drop)
    
        random.seed(SEED)
        
        selected_features = ['hba1c_bl_6m', 'insulin', 'hba1c_bl_18m', 'sum_diab_drugs', 'MD_RCT_mmol_mol', 'drug_class', 'dpp4', 't2d_dur_y', 'gluk', 'concordant_dis', 'met_oad0', 'trigly', 'hyperten', 'ika']
            
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
        model = MLPRegressor(random_state=123, max_iter=2000,hidden_layer_sizes = 16,learning_rate= 'adaptive')

        model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
        # fit the model
        model.fit(X_train, Y_train)
        
        # summarize prediction
        original_data_pred, model_results, model_results_drugs_ori, score_ori = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
        return original_data_pred, model_results, model_results_drugs_ori, score_ori, model
        
    
if __name__ == "__main__":
    imputeHba1c = ImputationHbA1c()
    df = read_data(imputeHba1c.file_path_X_train)
    df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeHba1c.preprocess_data(df)
    print('df_missing_val shape : ', df_missing_val.shape)
    original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHba1c.model_training(X_train, Y_train, X_test, Y_test)
    missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeHba1c.file_path_hba1c_imputed, 'hba1c_12m')
    
    
    
    
    
