import pandas as pd
import numpy as np
import random
from matplotlib.pyplot import pie, axis, show
import seaborn as sns
import missingno as msno
from scipy import stats
import matplotlib.pyplot as plt
import yaml
import os 
import sys

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import statsmodels.regression.linear_model as sm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, CompoundKernel
import sklearn_relief as sr
from skrebate import ReliefF
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as ltb
from sklearn.svm import SVR
from scipy.stats import ks_2samp
from tabulate import tabulate

from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE


from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.exceptions import DataConversionWarning

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import COMMON_VARIABLE_PATH, SEED, TESTING_DATA_FILE_PATH, TRAINING_DATA_FILE_PATH
from helpers import cross_val, get_model_name, get_scores, get_features_kbest, get_features_ref, \
    get_features_ref_multiout, get_features_relieff, outlier_detect
class Exp:
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the full path to the CSV file
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)
        self.file_path_test_data = os.path.join(self.script_directory, TESTING_DATA_FILE_PATH)
        self.file_path_train_data = os.path.join(self.script_directory, TRAINING_DATA_FILE_PATH)
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        self.correlated_variables = self.common_data['correlated_variables']
        
    def preprocess(self, df):
        """Further preprocessing

        Args:
            df : dataframe

        Returns:
            df : preprocessed dataframe
        """
        # remove rows with missing 'response variable'
        df = df.dropna(how='any', subset = self.response_variable_list)
        print('Shape of data after excluding missing response:', np.shape(df))

        df = df.drop('id', axis=1)
    
        date_cols = ['date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl', 'date_12m', 'date_n1',
                    'date_ldl_12m', 'date_bmi_12m', 'date_hdl_12m']
        df = df.drop(date_cols, axis=1)
        
        # select time interval and filter by that interval
        start = 21
        end = 365 #426
        df = df[(df['days_hba1c'] >= start) & (df['days_hba1c'] <= end)]
        print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
        
        # drop the columns calculated to find the days from baseline to response date
        df = df.drop(['days_hba1c', 'days_bmi', 'days_hdl', 'days_ldl'], axis=1)

        hdl_greater_than_2_5= 2.5  # Replace this with your desired threshold
        mask_hdl = df['hdl_12m'] > hdl_greater_than_2_5 
        df = df.drop(df[mask_hdl].index, axis = 0)

        bmi_greater_greater_50= 50  # Replace this with your desired threshold
        mask_bmi = df['bmi_12m'] > bmi_greater_greater_50
        df = df.drop(df[mask_bmi].index, axis = 0)
        
        df = df.astype(float)
        return df
    
    def get_test_train_data(self, X_train_df, X_test_df):
        """Impute and scale train and test data

        Args:
            X_train_df: data with imputed response variables
            X_test_df: data without imputed response variables

        Returns:
            original: preprocessed data set
            X_train, X_test, Y_train, Y_test: training and test data
            X, Y: input and output variables without dividing to train and test sets
            scaler: scaler instance
        """
        # split data
        random.seed(SEED)
        # Save original data set
        original = pd.concat([X_train_df, X_test_df], ignore_index=False)
        
        Y = original[self.response_variable_list]
        X = original.drop(self.response_variable_list, axis=1)
        
        Y_train = X_train_df[self.response_variable_list]
        X_train = X_train_df.drop(self.response_variable_list, axis=1)
        
        Y_test = X_test_df[self.response_variable_list]
        X_test = X_test_df.drop(self.response_variable_list, axis=1)
        random.seed(SEED)
        
        # imputate missing values in input variables
        original_X_train = X_train.copy()
        original_X_test = X_test.copy()
        random.seed(SEED)
        print('X_train shape before imputation: ', X_train.shape)
        
        imputer = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
        # imputeX = KNNImputer(missing_values=np.nan, n_neighbors = 3, weights='distance')
        # imputeX = IterativeImputer(max_iter=5, random_state=0)
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        print('X_train shape after imputation: ',X_train.shape)
        
        X_train = pd.DataFrame(X_train, columns = original_X_train.columns, index=original_X_train.index)
        X_test = pd.DataFrame(X_test, columns = original_X_train.columns, index=original_X_test.index)
        
        columns_to_skip_normalization = []
        columns_to_normalize = [col for col in X_train.columns if col not in columns_to_skip_normalization]

        # scale data 
        # scaler = StandardScaler()
        scaler = MinMaxScaler()
        select = {}
        
        # random oversampling 
        combined_df = pd.concat([X_train, Y_train], axis=1)
        X_oversamp = combined_df.drop(['drug_class'], axis = 1)
        Y_oversamp = combined_df['drug_class']
        random.seed(SEED)
        ros = RandomOverSampler(random_state=0)
        #smote = SMOTE()
        random.seed(SEED)
        X_resampled, y_resampled = ros.fit_resample(X_oversamp, Y_oversamp)
        print('..... Over sampling .....')
        print(set(Y_oversamp))
        print(sorted(Counter(Y_oversamp).items()))
        print(sorted(Counter(y_resampled).items()))
        combined = pd.concat([X_resampled, y_resampled], axis=1)
        
        X_train = combined.drop(self.response_variable_list, axis = 1)
        Y_train = combined[self.response_variable_list]
        
        X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
        X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
        
        print('Shape of training data after oversampling', X_train.shape)

        return original, X_train, X_test, Y_train, Y_test, X, Y, scaler

    def train_models(self, model, X_test, Y_test, X_train, Y_train, train,scaler,X_test_original):
        model_results = {}
        model_results_drugs = {}
        if str(get_model_name(model)) == 'Sequential':
            model.compile(optimizer='adam', loss='mean_squared_error')
            model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
            model = model.fit(X_train, Y_train, epochs=250, batch_size=16, verbose=0)
        else:
            model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
            model = model.fit(X_train, Y_train)
        data_pred, model_results, model_results_drugs, score = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
        
        return model_results, model


    def run(self, algo, i):
        df_X_train = pd.read_csv(self.file_path_train_data, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python',index_col=0)
        df_X_test = pd.read_csv(self.file_path_test_data, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python',index_col=0)
        
        X_train_ = self.preprocess(df_X_train)
        X_test_ = self.preprocess(df_X_test)
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler = self.get_test_train_data(X_train_, X_test_)

        X_test_original = X_test.copy()

        X_test_ = pd.DataFrame(X_test)
        X_train_ = pd.DataFrame(X_train)

        X_train = X_train.drop(['init_year'], axis = 1)
        X_test = X_test.drop(['init_year'], axis = 1)

        selected_features = []
        items = ['drug_class']
        
        random.seed(SEED) 
        if algo == 'kbest':
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                random.seed(SEED)
                feats = get_features_kbest(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'relieff':
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                random.seed(42)
                feats = get_features_relieff(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'refMulti':
            random.seed(42)
            selected_list = get_features_ref_multiout(X_train, Y_train, i)
        elif algo=='ref':
            random.seed(42)
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                feats = get_features_ref(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        else:
            # selected_list = ['Lower_MD_mmol_mol', 'bmi', 'comb_comp_enn', 'cvd_comp' ,'drug_class', 'eGFR',
                    #'gluk', 'hba1c_bl_6m', 'hba1c_prev_1y', 'hdl', 'ika', 'insulin', 'kol', 'ldl',
                    #'n_of_dis', 'obese', 'sp', 'sum_diab_drugs', 't2d_dur_y', 'trigly']
       
            selected_list = ['P_Krea', 'T2D_bloodcirc', 'bmi', 'chd', 'comb_comp_enn', 'cvd_comp', 'dg406',
                'dg602' ,'diab_retinop', 'drug_class', 'eGFR', 'gluk', 'hba1c_bl_6m', 'hba1c_prev_1y', 'hdl',
                'ika', 'insulin', 'kol', 'ldl', 'n_of_dis', 'obese', 'sp', 'sum_diab_drugs' ,'t2d_dur_y' ,'trigly']

        if algo == 'kbest' or algo == 'relieff' or algo == 'ref' :
            selected_list = sum(selected_features, [])
        
        for item in items:
            if item not in selected_list:
                selected_list.extend([item])

        # remove duplicate in selected feature list
        selected_list = np.unique(selected_list)
        number_of_features = len(selected_list)
        print('\n\n')
        print('Selected feature list: ', selected_list)
        X_train_selected = X_train[selected_list]
        X_test_selected = X_test[selected_list]

        ################# DROP OUTLIERS ################
        print('Shape of training data before removing outliers:', np.shape(X_train_selected))
        print('Shape of test data before removing outliers:', np.shape(X_test_selected))
        
        out_train, out_test = outlier_detect(X_train_selected, Y_train, X_test_selected, Y_test)
        #out_train=[14, 415, 744, 829, 916, 967, 1386, 279, 1225, 1332, 1656, 321, 426, 480, 779, 887, 1046, 1121, 1588, 82, 104, 359, 518, 847, 945, 978, 980, 1095, 1256, 1396, 1423, 1705]
                #[14, 415, 744, 829, 916, 967, 1386, 279, 1225, 1656, 321, 426, 480, 779, 887, 945, 1046, 1180, 1588, 82, 98, 104, 359, 711, 847, 980, 1095, 1180, 1256, 1263, 1423, 1705]
        #out_test= [6014, 1520]
        
        train_ = X_train_selected.copy()
        train_[self.response_variable_list] = Y_train.values
        
        test_ = X_test_selected.copy()
        test_[self.response_variable_list] = Y_test.values
        
        train_ = pd.DataFrame(train_.drop(out_train, axis = 0))
        test_ = pd.DataFrame(test_.drop(out_test, axis = 0))
        
        Y_train = train_[self.response_variable_list]
        X_train_selected = train_.drop(self.response_variable_list, axis=1)
        
        Y_test = test_[self.response_variable_list]
        X_test_selected = test_.drop(self.response_variable_list, axis=1)
        
        print('Shape of training data after removing outliers:', np.shape(X_train_selected))
        print('Shape of test data after removing outliers:', np.shape(X_test_selected))

        ##########################################
        
        train = X_train_selected.copy()
        train[self.response_variable_list] = Y_train.values

        ##################### TRAIN MODEL ###################
        rfr = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=123)

        random.seed(SEED)
        #wrapper = MultiOutputRegressor(catboost)
        model = RegressorChain(rfr, order=[0,1,2,3])

        random.seed(SEED) 
        model_results, model = self.train_models(model, X_test_selected, Y_test, X_train_selected, Y_train, train, scaler, X_test_original)
        return model_results, model, X_test_selected, Y_test, X_train_selected, Y_train, train, scaler, X_test_original

    
if __name__ == "__main__":
    print("Initialte model...")
    exp = Exp()
    
    model_results, model, X_test, Y_test, X_train, Y_train, train, scaler, X_test_original = exp.run('hc',8)
    table = []
    for model_, score in model_results.items():
        table.append([model_, score])

    table_str = tabulate(table, headers=['Model', 'Test R2 Score'], tablefmt='grid')
    print(table_str)
        

        
