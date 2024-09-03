import pandas as pd
import numpy as np
import random
from matplotlib.pyplot import pie, axis, show
import seaborn as sns
import missingno as msno
from scipy import stats
import matplotlib.pyplot as plt
import yaml

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
import statsmodels.regression.linear_model as sm
from collections import Counter

from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE

from sklearn.multioutput import RegressorChain, MultiOutputRegressor


def get_nan_count(df):
    """Print NaN count in selected columns

        Args:
            df : dataframe
    """
    selected_columns = df[['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m']].columns
    # Count NaN values in selected columns
    nan_counts = df[selected_columns].isna().sum()
    nan_info = pd.DataFrame({'Feature': selected_columns, 'NaN Count': nan_counts})
    print("\n NaN counts in resonse variables:")
    print(nan_info)
    print()
    
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
    df_orginal['sp'] = df_orginal['sp'].astype(int)
    df_orginal['ika'] = df_orginal['ika'].astype(float)
    df_orginal['smoking'] = df_orginal['smoking'].astype(float)
    return df_orginal


def preprocess(df, test_size, target_variable):
    
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
    df2 = df.apply(lambda x : True
                if x['drug_class'] == drug_id else False, axis = 1)
    number_of_rows = len(df2[df2 == True].index)
    return number_of_rows

def print_sample_count(df, dpp_val, sglt_val, label = ''):
    print('==== sample count in '+ label +' data =======')
    print(' number of dpp4 : ', countUsers(dpp_val, df))
    print(' number of sglt2 : ', countUsers(sglt_val, df))
    
def get_features_kbest(X_train, Y_train, k):
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    # Fit the selector to your data and transform the feature matrix
    X_selected = selector.fit_transform(X_train, Y_train)

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get the selected feature names
    selected_features = X_train.columns[selected_indices]
    selected_features = selected_features.to_list()
    return selected_features


def get_features_ref(X_train, Y_train, k=3): 
    random.seed(42)
    model = MultiOutputRegressor(RandomForestRegressor(random_state = 123))
#     model = RandomForestRegressor(random_state = 123)
    model.fit(X_train, Y_train)  # Fit the model before using RFE
    base_estimator = RandomForestRegressor(random_state=123)
    rfe = RFE(estimator=base_estimator, n_features_to_select=k)  
    X_selected = rfe.fit_transform(X_train, Y_train)
    selected_indices = rfe.get_support(indices=True)
    selected_features = [feature_name for feature_name in X_train.columns[selected_indices]]
    return selected_features

def get_features_ref_single(X_train, Y_train, k=3): 
    random.seed(42)  # Fit the model before using RFE
    model = RandomForestRegressor(random_state=123)
    rfe = RFE(estimator=model, n_features_to_select=k)  
    X_selected = rfe.fit_transform(X_train, Y_train)
    selected_indices = rfe.get_support(indices=True)
    selected_features = [feature_name for feature_name in X_train.columns[selected_indices]]
    return selected_features

def get_features_relieff(X_train, Y_train, k):
    best_n = k
    X_train_array = X_train.to_numpy()
    y_train_array = Y_train.to_numpy()
    fs = ReliefF()

    # Perform feature selection on the training data
    fs.fit(X_train_array, y_train_array)

    feat_dict = {}
    for feature_name, feature_score in zip(X_train.columns,
                                               fs.feature_importances_):
        feat_dict[feature_name] = feature_score

    # sort and get most important features
    feat_names = []
    sorted_feat_dict = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)

    best = sorted_feat_dict[: best_n]
    for i in best:
        feat_names.append(i[0])
    return feat_names


def outlier_detect(X_train, Y_train, X_test, Y_test):
    # Fit the model for each output in Y_train
    models = []
    predictions_train = []
    #X_train = X_train.apply(pd.to_numeric)
    #Y_train = pd.to_numeric(Y_train)
    for col in Y_train.columns:
        model = sm.OLS(Y_train[col], X_train).fit()
        predictions_train.append(model.predict(X_train))
        models.append(model)

    # Make predictions for each output in Y_test
    predictions = np.column_stack([model.predict(X_test) for model in models])

    # Print out the statistics for each output
    #for i, model in enumerate(models):
    #    print(f"Summary for Output {i + 1}:")
    #    display(model.summary())

    # Check for outliers in training set
    out_train = []
    for i, col in enumerate(Y_train.columns):
        error = Y_train[col] - predictions_train[i]
        stdres = (error - np.mean(error)) / np.std(error)
        c = stdres.abs() > 4
        #display(Counter(c))
        index_outlier = np.where(c == True)
        #display(index_outlier)
        index = stdres.index
        for j in range(len(c)):
            if c.iloc[j] == True:
                #print(f"Output {i + 1}, Train, Outlier Index: {index[j]}")
                out_train.append(index[j])

    # Check for outliers in testing set
    out_test = []
    for i, col in enumerate(Y_test.columns):
        error = Y_test[col] - predictions[:, i]
        stdres = (error - np.mean(error)) / np.std(error)
        c = stdres.abs() > 4
        #display(Counter(c))
        index_outlier = np.where(c == True)
        #display(index_outlier)
        index = stdres.index
        for j in range(len(c)):
            if c.iloc[j] == True:
                #print(f"Output {i + 1}, Test, Outlier Index: {index[j]}")
                out_test.append(index[j])

    print("Training set outliers:", out_train)
    print("Testing set outliers:", out_test)
    
    return out_train, out_test

def cross_val(model, train, X_train, Y_train, response_variable_list, n_splits=10):
    
    dfs = []
    acc_arr = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    i = 1
    for train_index, test_index in kf.split(train, Y_train):
        X_train1 = train.iloc[train_index].loc[:, X_train.columns]
        X_test1 = train.iloc[test_index].loc[:,X_train.columns]
        y_train1 = train.iloc[train_index].loc[:,response_variable_list]
        y_test1 = train.iloc[test_index].loc[:,response_variable_list]
        
        #Train the model
        model.fit(X_train1, y_train1) #Training the model
        
        # auc cal
        y_scores = model.predict(X_test1)
        score = model.score(X_test1, y_test1)
        acc_arr.append(score)
        
        # How many occurrences appear in the train set
        s_train = y_train1.apply(lambda col: col.value_counts()).transpose()
        s_train.columns = [f"train {i}_" + str(col) for col in s_train.columns]
        
        s_test = y_test1.apply(lambda col: col.value_counts()).transpose()
        s_test.columns = [f"test {i}_" + str(col) for col in s_test.columns]
        
        df = pd.concat([s_train, s_test], axis=1, sort=False)
        dfs.append(df)

        i += 1

    variance = np.var(acc_arr, ddof=1)
    
    print("Cross validation variance" , variance)
    print("Cross validation mean score" , sum(acc_arr) / len(acc_arr))
    return model

def get_model_name(model):
    model_name = str(type(model)).split('.')[-1][:-2]
    return model_name


model_results = {}
model_results_drugs = {}

def get_scores(model, X_test, Y_test, X_train, Y_train, name = ''):
    preds = model.predict(X_test)
    score = model.score(X_test,Y_test)
    
    # Calculate R2 scores for each target
    r2_train = model.score(X_train, Y_train)
    r2_test = model.score(X_test, Y_test)
        
    print(f'R2 score Training :', r2_train)
    print(f'R2 score Testing :', r2_test)
        
    rmse = np.sqrt(mean_squared_error(Y_test, preds))
    print(f"RMSE (Target): {rmse}")
        
    if not name.strip():
        model_results[str(get_model_name(model))] = score
    else:
        model_results_drugs[str(get_model_name(model)+'_'+name)] = score
        
    return preds, model_results, model_results_drugs, score