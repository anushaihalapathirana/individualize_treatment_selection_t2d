import pandas as pd
import numpy as np
import random
import os
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.regression.linear_model as sm
from skrebate import ReliefF
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from sklearn.multioutput import MultiOutputRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper import outlier_detect, get_model_name

def read_data(file_path):
        """Read training data file

        Returns:
            df: dataframe
        """
        df = pd.read_csv(file_path, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python', index_col=0)
        return df

def get_nan_count(df):
    """Print NaN count in selected columns

        Args:
            df : dataframe
            
        Return: 
            nan_info: Dataframe consists with nan count
    """
    selected_columns = df[['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m']].columns
    # Count NaN values in selected columns
    nan_counts = df[selected_columns].isna().sum()
    nan_info = pd.DataFrame({'Feature': selected_columns, 'NaN Count': nan_counts})
    return nan_info
    
def get_missing_val_percentage(df):
    return (df.isnull().sum()* 100 / len(df))

def get_dfs(df_orginal):
    
    df_orginal = df_orginal[
                (df_orginal['drug_class'] == 3) |
                (df_orginal['drug_class'] == 4) ]

    # replace ' ' as NaN
    df_orginal = df_orginal.replace(' ', np.NaN)
    print('Shape of data after removing other drug types:', np.shape(df_orginal))

        # filter by bmi
    df_orginal['bmi'] = df_orginal['bmi'].astype(float)
    df_orginal['sp'] = df_orginal['sp'].astype(float)
    df_orginal['ika'] = df_orginal['ika'].astype(float)
    df_orginal['smoking'] = df_orginal['smoking'].astype(float)
    return df_orginal


def preprocess(df, test_size, target_variable, variables_to_drop):
    
    """Further preprocess data (Focusing response variable as BMI)

        Args:
            df : dataframe
            test_size (float): size of the test data. This use to split the data into training and test dataset.
            target_variable: target variable name (string)
            
        Returns: 
            df : Preprocessed dataframe
            X_train, X_test, Y_train, Y_test : After train and test split
            X, Y : X and Y before train test split
            scaler : Scalar object. Used later to descale
            df_missing_val, df_missing_val_original : dataframes with missing values of bmi_12m
            df_original : original dataframe
    """
        
    print('Shape of data :', np.shape(df))
    
    df_missing_val = df[df[target_variable].isnull()]
    df_missing_val_original = df_missing_val.copy()
    
    response_variable_list = [target_variable]
    
    # remove rows with missing 'response variable'
    df = df.dropna(how='any', subset = response_variable_list)
    df_original = df.copy()
    print('Shape of data after excluding missing response:', np.shape(df))
    
    date_cols = ['date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl',
                 'date_12m', 'date_n1',
                 'date_ldl_12m',
                 'date_bmi_12m',
                 'date_hdl_12m']

    df = df.drop(date_cols, axis=1)
    df_missing_val = df_missing_val.drop(date_cols, axis=1)
    
    # select time interval
    
    start = 21
    end = 365 #426
    df = df[(df['days_hba1c'] >= start) & (df['days_hba1c'] <= end)]
    print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
    
    df = df.astype(float)
    df_missing_val = df_missing_val.astype(float)
    
    if (target_variable == 'bmi_12m'):
        bmi_greater_less_5= 50 
        mask_bmi = df[target_variable] > bmi_greater_less_5  
        df = df.drop(df[mask_bmi].index, axis = 0)
        
    
    if (target_variable == 'hdl_12m'):
        hdl_greater_than_5= 2.5  
        mask_hdl = df[target_variable] > hdl_greater_than_5  
        df = df.drop(df[mask_hdl].index, axis = 0)
        
    
    # drop other response variables
    df = df.drop(variables_to_drop, axis=1)
    df_missing_val = df_missing_val.drop(variables_to_drop, axis = 1)
    
    # split data
    random.seed(42)
    # Save original data set
    original = df
    Y = df[response_variable_list]
    X = df.drop(response_variable_list, axis=1)
    random.seed(42)
    
    # Split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=123)
    
    df_missing_val = df_missing_val.drop(response_variable_list, axis=1)
    
    # data imputation
    original_X_train = X_train
    original_X_test = X_test
    original_df_missing_val = df_missing_val
    random.seed(42)
    imputer = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
    # imputeX = KNNImputer(missing_values=np.nan, n_neighbors = 3, weights='distance')
    # imputeX = IterativeImputer(max_iter=5, random_state=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    df_missing_val = imputer.transform(df_missing_val)
    
    X_train = pd.DataFrame(X_train, columns = original_X_train.columns, index=original_X_train.index)
    X_test = pd.DataFrame(X_test, columns = original_X_train.columns, index=original_X_test.index)
    df_missing_val = pd.DataFrame(df_missing_val, columns = original_df_missing_val.columns, index=original_df_missing_val.index)
    
    #     columns_to_skip_normalization = ['drug_class']
    columns_to_skip_normalization = []
    # List of columns to normalize
    columns_to_normalize = [col for col in X_train.columns if col not in columns_to_skip_normalization]

    # scale data
    scaler = MinMaxScaler()
    select = {}
    
    X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    df_missing_val[columns_to_normalize] = scaler.transform(df_missing_val[columns_to_normalize])
    
    
    return df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original
 
def countUsers(drug_id, df):
    df_ = df.apply(lambda x : True
                if x['drug_class'] == drug_id else False, axis = 1)
    number_of_rows = len(df_[df_ == True].index)
    return number_of_rows

def print_sample_count(df, dpp_val, sglt_val, label = ''):
    print('==== sample count in '+ label +' data =======')
    print(' number of dpp4 : ', countUsers(dpp_val, df))
    print(' number of sglt2 : ', countUsers(sglt_val, df))
    
def remove_outliers(X_train, X_test, Y_train, Y_test, response_variable_list):
        ################# OUTLIER ################
        print('Shape of training data before removing outliers:', np.shape(X_train))
        print('Shape of test data before removing outliers:', np.shape(X_test))
            
        out_train, out_test = outlier_detect(X_train, Y_train, X_test, Y_test)
        
        train_ = X_train.copy()
        train_[response_variable_list] = Y_train.values
            
        test_ = X_test.copy()
        test_[response_variable_list] = Y_test.values
            
        train_ = pd.DataFrame(train_.drop(out_train, axis = 0))
        test_ = pd.DataFrame(test_.drop(out_test, axis = 0))
            
        Y_train = train_[response_variable_list]
        X_train = train_.drop(response_variable_list, axis=1)
            
        Y_test = test_[response_variable_list]
        X_test = test_.drop(response_variable_list, axis=1)
            
        print('Shape of training data after removing outliers:', np.shape(X_train))
        print('Shape of test data after removing outliers:', np.shape(X_test))
        
        return X_train, X_test, Y_train, Y_test

def missing_value_prediction(model, df_missing, df_original, selected_features, df_missing_val_original, file_path, target_variable):
    df_missing_val = df_missing[selected_features]
    mv_pred_test_numpy = model.predict(df_missing_val)
    print('Length of mv pred test numpy array: ', len(mv_pred_test_numpy))
    df_missing_val_original[target_variable] = mv_pred_test_numpy
    print('Shape of df_missing_val_original '+ target_variable +': ', df_missing_val_original[target_variable])
    result_df = pd.concat([df_original, df_missing_val_original])
    # Save file
    result_df.to_csv(file_path, index=True)
    print(result_df[[target_variable]])