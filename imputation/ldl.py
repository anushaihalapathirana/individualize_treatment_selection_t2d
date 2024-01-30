import pandas as pd
import numpy as np
import random
import yaml
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from helpers import print_sample_count, get_features_kbest, outlier_detect, cross_val, get_scores
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor
from constants import X_TRAIN_PATH, COMMON_VARIABLE_PATH, LDL_PATH, SEED

class ImputationLDL:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, X_TRAIN_PATH)
        self.file_path_ldl_imputed = os.path.join(self.script_directory, LDL_PATH)
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = ['ldl_12m']
        self.correlated_variables = self.common_data['correlated_variables']
        
        
    def read_data(self):
        """Read training data file

        Returns:
            df: dataframe
        """
        df = pd.read_csv(self.file_path_X_train, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python', index_col=0)
    
        return df
    
    def preprocess(self, df, test_size):
        
        """Further preprocess data (Focusing response variable as LDL)

        Args:
            df : dataframe
            test_size (float): size of the test data. This use to split the data into training and test dataset.
        
        Returns: 
            df : Preprocessed dataframe
            X_train, X_test, Y_train, Y_test : After train and test split
            X, Y : X and Y before train test split
            scaler : Scalar object. Used later to descale
            df_missing_val, df_missing_val_original : dataframes with missing values of ldl_12m
            df_original : original dataframe
        """
        
        print('Shape of data :', np.shape(df))
        
        df_missing_val = df[df['ldl_12m'].isnull()]
        df_missing_val_original = df_missing_val.copy()
        
        # remove rows with missing 'response variable'
        df = df.dropna(how='any', subset = self.response_variable_list)
        df_original = df.copy()
        print('Shape of data after excluding missing response:', np.shape(df))
        
        date_cols = ['date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl', 'date_12m', 'date_n1',
                 'date_ldl_12m', 'date_bmi_12m', 'date_hdl_12m']

        df = df.drop(date_cols, axis=1)
        df_missing_val = df_missing_val.drop(date_cols, axis=1)
        
        # select time interval
        start = 21
        end = 365 #426
        df = df[(df['days_hba1c'] >= start) & (df['days_hba1c'] <= end)]
        print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
        
        df = df.astype(float)
        df_missing_val = df_missing_val.astype(float)

        # split data
        random.seed(SEED)
        # Save original data set
        original = df
        Y = df[self.response_variable_list]
        X = df.drop(self.response_variable_list, axis=1)
        random.seed(SEED)
        
        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=123)
        
        df_missing_val = df_missing_val.drop(self.response_variable_list, axis=1)
        
        # data imputation
        original_X_train = X_train
        original_X_test = X_test
        original_df_missing_val = df_missing_val
        random.seed(SEED)
        
        # Impute all the other features, Except response variable
        imputer = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        df_missing_val = imputer.transform(df_missing_val)
        
        X_train = pd.DataFrame(X_train, columns = original_X_train.columns, index=original_X_train.index)
        X_test = pd.DataFrame(X_test, columns = original_X_train.columns, index=original_X_test.index)
        df_missing_val = pd.DataFrame(df_missing_val, columns = original_df_missing_val.columns, index=original_df_missing_val.index)
        
        columns_to_skip_normalization = []
        # List of columns to normalize
        columns_to_normalize = [col for col in X_train.columns if col not in columns_to_skip_normalization]

        # scale data
        scaler = MinMaxScaler()
        
        X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
        X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
        df_missing_val[columns_to_normalize] = scaler.transform(df_missing_val[columns_to_normalize])
        
        return df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original
    
    def feature_selection(self, df, X_train, Y_train, X_test):
        X_test_ = pd.DataFrame(X_test)
        X_train_ = pd.DataFrame(X_train)

        X_train = X_train.drop(['init_year'], axis = 1)
        X_test = X_test.drop(['init_year'], axis = 1)
        
        # print drug sample count in preprocessed data, training data and test data
        print_sample_count(df, X_train_, X_test_)
        
        random.seed(SEED)
        
        feats = get_features_kbest(X_train, Y_train, 10)
        selected_features=feats
                
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        number_of_features = len(selected_features)
        print(selected_features)
        return X_train, X_test, selected_features
    
    def remove_outliers(self, X_train, Y_train, X_test, Y_test):
        
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
        model_results_drugs = {}
        
        model = XGBRegressor(n_estimators=50, eta=0.01, subsample=0.5, colsample_bytree=0.8, alpha=0.1,
                max_depth = 5, max_leaves = 6, learning_rate =0.1)
        
        model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list, n_splits=3)
        model.fit(X_train, Y_train)
        # make a prediction
        yhat = model.predict(X_test)
        # summarize prediction
        original_data_pred, model_results, model_results_drugs_ori, score_ori = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
        return original_data_pred, model_results, model_results_drugs_ori, score_ori, model

    def missing_value_prediction(self, model, df_missing, df_original, selected_features, df_missing_val_original):
        df_missing_val = df_missing[selected_features]
        mv_pred_test_numpy = model.predict(df_missing_val)
        print('Length of mv pred test numpy array: ', len(mv_pred_test_numpy))
        df_missing_val_original['ldl_12m'] = mv_pred_test_numpy
        print('Shape of df_missing_val_original ldl_12m: ', df_missing_val_original['ldl_12m'])
        result_df = pd.concat([df_original, df_missing_val_original])
        # Save file
        result_df.to_csv(self.file_path_ldl_imputed, index=True)
        print(result_df[['ldl_12m']])
        
    
if __name__ == "__main__":
    imputeLDL = ImputationLDL()
    df = imputeLDL.read_data()
    df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = imputeLDL.preprocess(df, 0.25)
    print('df_missing_val shape : ', df_missing_val.shape)
    X_train, X_test, selected_features = imputeLDL.feature_selection(df, X_train, Y_train, X_test)
    X_train, X_test, Y_train, Y_test = imputeLDL.remove_outliers(X_train, Y_train, X_test, Y_test)
    original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeLDL.model_training(X_train, Y_train, X_test, Y_test)
    imputeLDL.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
    
    
    
