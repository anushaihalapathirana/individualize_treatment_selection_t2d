import pandas as pd
import numpy as np
import random
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from catboost import CatBoostRegressor

from constants import COMMON_VARIABLE_PATH, LDL_PATH, SEED, TRAIN_PATH
from utils import preprocess, remove_outliers, cross_val, get_scores, read_data, missing_value_prediction

class ImputationLDL:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, TRAIN_PATH)
        self.file_path_ldl_imputed = os.path.join(self.script_directory, LDL_PATH)
        self.file_path_common_variables = os.path.abspath(os.path.join(self.script_directory, COMMON_VARIABLE_PATH))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = ['ldl_12m']
        self.target_variable = 'ldl_12m'
        self.correlated_variables = self.common_data['correlated_variables']
    
    def preprocess_data(self, df):
        variables_to_drop = ['bmi_12m', 'hba1c_12m', 'hdl_12m', 'days_ldl', 'init_year']
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = preprocess(df, 0.25, self.target_variable, variables_to_drop)
        
        random.seed(SEED)
        
        selected_features = ['hba1c_bl_6m', 'ika', 'ldl', 'insulin', 'sum_diab_drugs', 'hyperten', 'chd', 'cvd_comp', 'obese', 'C02A', 'C10A'] 
            
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # remove outliers
        X_train, X_test, Y_train, Y_test = remove_outliers(X_train, X_test, Y_train, Y_test, self.response_variable_list)
        return df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features

        
    def model_training(self, X_train, Y_train, X_test, Y_test):
        train = X_train.copy()
        train[self.response_variable_list] = Y_train[self.response_variable_list].copy()
        
        model_results = {}
        
        model1 = XGBRegressor(
            n_estimators=20, 
            eta=0.04, 
            subsample=0.6, 
            colsample_bytree=0.9,
            alpha=0.4,
            max_depth = 12,
            max_leaves = 10,
            learning_rate =0.15)

        model3 = CatBoostRegressor(iterations=50,learning_rate=0.1, depth=6)

        model2 = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=123)

        model = VotingRegressor([('xgb', model1), ('rfr', model2), ('catboost', model3)])
        
        model = cross_val(model, train , X_train, Y_train, self.response_variable_list)
        # fit the model
        model.fit(X_train, Y_train)
        
        # summarize prediction
        original_data_pred, model_results, model_results_drugs_ori, score_ori = get_scores(model, X_test, Y_test, X_train, Y_train)
        return original_data_pred, model_results, model_results_drugs_ori, score_ori, model
        
    
if __name__ == "__main__":
    imputeLDL = ImputationLDL()
    df = read_data(imputeLDL.file_path_X_train)
    df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeLDL.preprocess_data(df)
    print('df_missing_val shape : ', df_missing_val.shape)
    original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeLDL.model_training(X_train, Y_train, X_test, Y_test)
    missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original, imputeLDL.file_path_ldl_imputed, 'ldl_12m')
    
    
