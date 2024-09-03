import pandas as pd
import numpy as np
import random
import yaml
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from xgboost.sklearn import XGBRegressor

from constants import COMMON_VARIABLE_PATH, BMI_PATH, SEED, TRAIN_PATH, SGLT_VALUE, DPP_VALUE,\
    ORIGINAL_DPP_VALUE, ORIGINAL_SGLT_VALUE
from utils import preprocess, print_sample_count, outlier_detect, cross_val, get_scores
    
class ImputationBMI:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, TRAIN_PATH)
        self.file_path_bmi_imputed = os.path.join(self.script_directory, BMI_PATH)
        self.file_path_common_variables = os.path.abspath(os.path.join(self.script_directory, COMMON_VARIABLE_PATH))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = ['bmi_12m']
        self.target_variable = 'bmi_12m'
        self.correlated_variables = self.common_data['correlated_variables']
        
    def read_data(self):
        """Read training data file

        Returns:
            df: dataframe
        """
        df = pd.read_csv(self.file_path_X_train, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python', index_col=0)
        return df
    
    def preprocess_data(self, df):
        variables_to_drop = ['ldl_12m', 'hba1c_12m', 'hdl_12m', 'days_ldl', 'init_year']
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = preprocess(df, 0.25, self.target_variable)
        df = df.drop(variables_to_drop, axis=1)
        X_train = X_train.drop(variables_to_drop, axis=1)
        X_test = X_test.drop(variables_to_drop, axis=1)
        
        print_sample_count(df, ORIGINAL_DPP_VALUE, ORIGINAL_SGLT_VALUE, label = 'preprocessed')
        print_sample_count(X_train, DPP_VALUE, SGLT_VALUE, label = 'training')
        print_sample_count(X_test, DPP_VALUE, SGLT_VALUE, label = 'test')
    
        random.seed(SEED)
        
        selected_features = ['sp', 'ika', 't2d_dur_y', 'pkrea_luo', 'bmi', 'dpp4', 'hdl', 'trigly', 'obese'] # kbest 10
            
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        return df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features

    def remove_outliers(self, X_train, X_test, Y_train, Y_test):
        ################# OUTLIER ################
        print('Shape of training data before removing outliers:', np.shape(X_train))
        print('Shape of test data before removing outliers:', np.shape(X_test))
            
        out_train, out_test = outlier_detect(X_train, Y_train, X_test, Y_test)
        
        train_ = X_train.copy()
        train_[self.response_variable_list] = Y_train.values
            
        test_ = X_test.copy()
        test_[self.response_variable_list] = Y_test.values
            
        train_ = pd.DataFrame(train_.drop(out_train, axis = 0))
        test_ = pd.DataFrame(test_.drop(out_test, axis = 0))
            
        Y_train = train_[self.response_variable_list]
        X_train = train_.drop(self.response_variable_list, axis=1)
            
        Y_test = test_[self.response_variable_list]
        X_test = test_.drop(self.response_variable_list, axis=1)
            
        print('Shape of training data after removing outliers:', np.shape(X_train))
        print('Shape of test data after removing outliers:', np.shape(X_test))
        
        return X_train, X_test, Y_train, Y_test
    
    def model_training(self, X_train, Y_train, X_test, Y_test):
        train = X_train.copy()
        train[self.response_variable_list] = Y_train[self.response_variable_list].copy()
        
        model_results = {}
        
        model = XGBRegressor(
                n_estimators=70, 
                eta=0.06, 
                subsample=0.9, 
                colsample_bytree=0.8,
                alpha=0.04,
                max_depth = 15,
                max_leaves = 5,
                learning_rate =0.1
            )
        
        model = cross_val(model, train , X_train, Y_train, self.response_variable_list)
        # fit the model
        model.fit(X_train, Y_train)
        
        # summarize prediction
        original_data_pred, model_results, model_results_drugs_ori, score_ori = get_scores(model, X_test, Y_test, X_train, Y_train)
        return original_data_pred, model_results, model_results_drugs_ori, score_ori, model

    def missing_value_prediction(self, model, df_missing, df_original, selected_features, df_missing_val_original):
        df_missing_val = df_missing[selected_features]
        mv_pred_test_numpy = model.predict(df_missing_val)
        print('Length of mv pred test numpy array: ', len(mv_pred_test_numpy))
        df_missing_val_original['bmi_12m'] = mv_pred_test_numpy
        print('Shape of df_missing_val_original bmi_12m: ', df_missing_val_original['bmi_12m'])
        result_df = pd.concat([df_original, df_missing_val_original])
        # Save file
        result_df.to_csv(self.file_path_bmi_imputed, index=True)
        print(result_df[['bmi_12m']])
        
    
if __name__ == "__main__":
    imputeBMI = ImputationBMI()
    df = imputeBMI.read_data()
    df, X_train, X_test, Y_train, Y_test, df_missing_val, df_missing_val_original, df_original, selected_features = imputeBMI.preprocess_data(df)
    print('df_missing_val shape : ', df_missing_val.shape)
    X_train, X_test, Y_train, Y_test = imputeBMI.remove_outliers(X_train, X_test, Y_train, Y_test)
    original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeBMI.model_training(X_train, Y_train, X_test, Y_test)
    
    imputeBMI.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
    
    
    