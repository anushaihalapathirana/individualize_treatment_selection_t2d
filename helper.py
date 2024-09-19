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

from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE


from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.exceptions import DataConversionWarning

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
    """
    selected_columns = df[['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m']].columns
    # Count NaN values in selected columns
    nan_counts = df[selected_columns].isna().sum()
    nan_info = pd.DataFrame({'Feature': selected_columns, 'NaN Count': nan_counts})
    print("\n NaN counts in resonse variables:")
    print(nan_info)
    print()
    
def preprocess(df, response_variable_list):

    # remove rows with missing 'response variable'
    df = df.dropna(how='any', subset = response_variable_list)
    print('Shape of data after excluding missing response:', np.shape(df))

    df = df.drop('id', axis=1)
   
    
    date_cols = ['date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl',
                 'date_12m', 'date_n1',
                 'date_ldl_12m',
                 'date_bmi_12m',
                 'date_hdl_12m']
    df = df.drop(date_cols, axis=1)
    
    # select time interval
    start = 21
    end = 365 #426
    df = df[(df['days_hba1c'] >= start) & (df['days_hba1c'] <= end)]
    
    print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
    
    df = df.drop(['days_hba1c', 'days_bmi', 'days_hdl', 'days_ldl'], axis=1)

    hdl_greater_than_5= 2.5  # Replace this with your desired threshold
    mask_hdl = df['hdl_12m'] > hdl_greater_than_5  # Replace 'column_name' with the actual column you want to filter
    df = df.drop(df[mask_hdl].index, axis = 0)

    bmi_greater_less_5= 50  # Replace this with your desired threshold
    mask_bmi = df['bmi_12m'] > bmi_greater_less_5  # Replace 'column_name' with the actual column you want to filter
    df = df.drop(df[mask_bmi].index, axis = 0)
    
    df = df.astype(float)
    return df

    
def get_test_train_data(X_train_df, X_test_df, response_variable_list):
    
    # split data
    random.seed(42)
    # Save original data set
    original = pd.concat([X_train_df, X_test_df], ignore_index=False)
    
    Y = original[response_variable_list]
    X = original.drop(response_variable_list, axis=1)
    
    Y_train = X_train_df[response_variable_list]
    X_train = X_train_df.drop(response_variable_list, axis=1)
    
    Y_test = X_test_df[response_variable_list]
    X_test = X_test_df.drop(response_variable_list, axis=1)
    random.seed(42)
    
    
    # data imputation
    original_X_train = X_train.copy()
    original_X_test = X_test.copy()
    random.seed(42)
    
    print('X_train shape: ',X_train.shape)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
    # imputeX = KNNImputer(missing_values=np.nan, n_neighbors = 3, weights='distance')
    # imputeX = IterativeImputer(max_iter=5, random_state=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print(X_train.shape)
    
    
    X_train = pd.DataFrame(X_train, columns = original_X_train.columns, index=original_X_train.index)
    X_test = pd.DataFrame(X_test, columns = original_X_train.columns, index=original_X_test.index)
    
    #     columns_to_skip_normalization = ['drug_class']
    columns_to_skip_normalization = []
    # List of columns to normalize
    columns_to_normalize = [col for col in X_train.columns if col not in columns_to_skip_normalization]

    # scale data 
#     scaler = StandardScaler()
    scaler = MinMaxScaler()
    select = {}
#     X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
#     X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    
    # random oversampling 
    combined_df = pd.concat([X_train, Y_train], axis=1)
    X_oversamp = combined_df.drop(['drug_class'], axis = 1)
    Y_oversamp = combined_df['drug_class']
    random.seed(42)
    ros = RandomOverSampler(random_state=0)
    #smote = SMOTE()
    random.seed(42)
    X_resampled, y_resampled = ros.fit_resample(X_oversamp, Y_oversamp)
    print(set(Y_oversamp))
    print(sorted(Counter(Y_oversamp).items()))
    print(sorted(Counter(y_resampled).items()))
    combined = pd.concat([X_resampled, y_resampled], axis=1)
    
    X_train = combined.drop(response_variable_list, axis = 1)
    Y_train = combined[response_variable_list]
    
    X_test_before_scale = X_test.copy()
    X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    
    #print('Shape of training data after oversampling', np.shape(X_train))
    
    
    return original, X_train, X_test, Y_train, Y_test, X, Y, scaler, X_test_before_scale


def countUsers(drug_id, df):
    df2 = df.apply(lambda x : True
                if x['drug_class'] == drug_id else False, axis = 1)
    number_of_rows = len(df2[df2 == True].index)
    return number_of_rows


def get_features_kbest(X_train, Y_train,i):
    random.seed(42)
    np.random.seed(42)
    selector = SelectKBest(score_func=mutual_info_regression, k=i)
    # Fit the selector to your data and transform the feature matrix
    X_selected = selector.fit_transform(X_train, Y_train)

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Get the selected feature names
    selected_features = X_train.columns[selected_indices]
    selected_features = selected_features.to_list()
    return selected_features


def get_features_ref(X_train, Y_train,i): 
    random.seed(42)
    model = RandomForestRegressor(random_state = 123)
    rfe = RFE(estimator=model, n_features_to_select=i)  
    X_selected = rfe.fit_transform(X_train, Y_train)
    selected_indices = rfe.get_support(indices=True)
    selected_features = [feature_name for feature_name in X_train.columns[selected_indices]]
    return selected_features

def get_features_ref_multiout(X_train, Y_train, k=12): 
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


def get_features_relieff(X_train, Y_train,i):
    best_n = i
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

def get_model_name(model):
    model_name = str(type(model)).split('.')[-1][:-2]
    return model_name



def cross_val(model, train, X_test, Y_test, X_train, Y_train, response_variable_list, n_splits=3):
    dfs = []
    acc_arr = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    i = 1
    random.seed(42)
    for train_index, test_index in kf.split(train, Y_train):
        X_train1 = train.iloc[train_index].loc[:, X_train.columns]
        X_test1 = train.iloc[test_index].loc[:,X_train.columns]
        y_train1 = train.iloc[train_index].loc[:,response_variable_list]
        y_test1 = train.iloc[test_index].loc[:,response_variable_list]
        
        if (get_model_name(model)=='Sequential'):
            random.seed(42)
            model.fit(X_train1, y_train1,epochs=250, batch_size=16, verbose=0)
        else:
            #Train the model
            random.seed(42)
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


def get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs, name = ''):
    pred = model.predict(X_test)
    score = r2_score(Y_test, pred)
    r2_train = model.score(X_train, Y_train)        
    print(f'R2 score Training :', r2_train)
    r_squared = r2_score(Y_test, pred)
    print(f"R2 score Testing: {r_squared:.4f}")

    rmse = np.sqrt(mean_squared_error(Y_test, pred))
    print("RMSE: %f" % (rmse))
    if not name.strip():
        model_results[str(get_model_name(model))] = r2_score(Y_test, pred)  
    else:
        model_results_drugs[str(get_model_name(model)+'_'+name)] = r2_score(Y_test, pred)
    return pred, model_results, model_results_drugs, score

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



def train_models(model, X_test, Y_test, X_train, Y_train, train,scaler,X_test_original, response_variable_list):
    model_results = {}
    model_results_drugs = {}
    if str(get_model_name(model)) == 'Sequential':
        model.compile(optimizer='adam', loss='mean_squared_error')
        model = cross_val(model, train, X_test, Y_test, X_train, Y_train, response_variable_list)
        model = model.fit(X_train, Y_train, epochs=250, batch_size=16, verbose=0)
    else:
        model = cross_val(model, train, X_test, Y_test, X_train, Y_train, response_variable_list)
        model = model.fit(X_train, Y_train)
    data_pred, model_results, model_results_drugs, score = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)
    
    return model_results, model
    
    
def print_val(name, pred_sglt, pred_dpp):
    print(name)
    print(pred_sglt)
    print(pred_dpp)
    

def pred_all(model, row, drug_class):
    sglt_val = 1
    dpp_val = 0
    if drug_class == sglt_val:
        pred_sglt_ = model.predict(row.values[None])[0]
        row['drug_class'] = dpp_val
        pred_dpp_ = model.predict(row.values[None])[0]
#         print_val('SGLT', pred_sglt, pred_dpp)
        
    elif drug_class == dpp_val:
        pred_dpp_ = model.predict(row.values[None])[0]
        row['drug_class'] = sglt_val
        pred_sglt_ = model.predict(row.values[None])[0]
#         print_val('DPP', pred_sglt, pred_dpp)
        
    else:
        print('Worng drug class')
    return pred_sglt_, pred_dpp_

def find_lowest_respponse_value(pred_sglt, pred_dpp):
    sglt_val = 1
    dpp_val = 0
    values = [pred_sglt, pred_dpp]
    max_index = values.index(min(values))
    max_difference = [pred_sglt, pred_dpp][max_index]
    drug_class = [sglt_val, dpp_val][max_index]
    return max_difference, drug_class

def find_highest_respponse_value(pred_sglt, pred_dpp):
    sglt_val = 1
    dpp_val = 0
    values = [pred_sglt, pred_dpp]
    max_index = values.index(max(values))
    max_difference = [pred_sglt, pred_dpp][max_index]
    drug_class = [sglt_val, dpp_val][max_index]
    return max_difference, drug_class

#### new change
def find_closest_to_42(pred_sglt, pred_dpp):
    sglt_val = 1
    dpp_val = 0
    values = [pred_sglt, pred_dpp]
    drug_classes = [sglt_val, dpp_val]
    max_index = min(range(len(values)), key=lambda i: abs(values[i] - 42.0))
    closest_value = values[max_index]
    drug_class = drug_classes[max_index]
    return closest_value, drug_class

def predict_drug_classes(model, X_test, Y_train):
    X = X_test.copy()
    X_test_copy = X_test.copy()
    X_test_copy['assigned_drug_hba1c'] = np.nan
    X_test_copy['predicted_change_hba1c'] = np.nan
    X_test_copy['assigned_drug_ldl'] = np.nan
    X_test_copy['predicted_change_ldl'] = np.nan
    X_test_copy['assigned_drug_hdl'] = np.nan
    X_test_copy['predicted_change_hdl'] = np.nan
    X_test_copy['assigned_drug_bmi'] = np.nan
    X_test_copy['predicted_change_bmi'] = np.nan
        
    assigned_drug_class_list = [np.nan] * Y_train.shape[1]
    max_change_list = [np.nan] * Y_train.shape[1]
        
    for index, row in X.iterrows():
        drug_class = row['drug_class']

        pred_original = model.predict(row.values[None])[0]
        pred_sglt, pred_dpp = pred_all(model, row, drug_class) 

        for j in range(Y_train.shape[1]):
            if (Y_train.iloc[:,j].name == 'hdl_12m'):
                temp_max_change, temp_assigned_drug_class = find_highest_respponse_value(pred_sglt[j], pred_dpp[j])
            else:
                temp_max_change, temp_assigned_drug_class = find_lowest_respponse_value(pred_sglt[j], pred_dpp[j])
                
            max_change_list[j] = temp_max_change
            assigned_drug_class_list[j] = temp_assigned_drug_class
                
        X_test_copy.at[index, 'assigned_drug_hba1c'] = assigned_drug_class_list[0]
        X_test_copy.at[index, 'predicted_change_hba1c'] = max_change_list[0]

        X_test_copy.at[index, 'assigned_drug_ldl'] = assigned_drug_class_list[1]
        X_test_copy.at[index, 'predicted_change_ldl'] = max_change_list[1]

        X_test_copy.at[index, 'assigned_drug_hdl'] = assigned_drug_class_list[2]
        X_test_copy.at[index, 'predicted_change_hdl'] = max_change_list[2]

        X_test_copy.at[index, 'assigned_drug_bmi'] = assigned_drug_class_list[3]
        X_test_copy.at[index, 'predicted_change_bmi'] = max_change_list[3]
    return X_test_copy
    
def print_strata_stats(dpp_strata_actual, sglt_strata_actual, dpp_strata_hba1c, sglt_strata_hba1c, dpp_strata_ldl, sglt_strata_ldl,
                       dpp_strata_hdl, sglt_strata_hdl, dpp_strata_bmi, sglt_strata_bmi):
    print(' Sample count in test data')
    print(' number of dpp4 samples in test dataset : ', dpp_strata_actual.shape[0])
    print(' number of sglt2 samples in test dataset : ', sglt_strata_actual.shape[0])

    print(' \n Assigned sample count: HBA1C')

    print(' number of dpp4 assigned : ', dpp_strata_hba1c.shape[0])
    print(' number of sglt2 assigned : ', sglt_strata_hba1c.shape[0])

    print(' \n Assigned sample count: LDL')

    print(' number of dpp4 assigned : ', dpp_strata_ldl.shape[0])
    print(' number of sglt2 assigned : ', sglt_strata_ldl.shape[0])

    print(' \n Assigned sample count: HDL')

    print(' number of dpp4 assigned : ', dpp_strata_hdl.shape[0])
    print(' number of sglt2 assigned : ', sglt_strata_hdl.shape[0])

    print(' \n Assigned sample count: BMI')

    print(' number of dpp4 assigned : ', dpp_strata_bmi.shape[0])
    print(' number of sglt2 assigned : ', sglt_strata_bmi.shape[0])

def get_strata(df, drug_col, drug_value):
        """
        Extracts a subset of the DataFrame where the values in a specified column match a given drug value.
        
        Args:
            df (DataFrame): The input DataFrame from which to extract a subset.
            drug_col (int): The column in the DataFrame representing the assigned drug or drug-related feature.
            drug_value (int): The value to match in the specified drug column (e.g., DPP_VALUE, SGLT_VALUE).

        Returns:
            DataFrame: A subset of the input DataFrame where the values in 'drug_col' match 'drug_value'.
        """
        return df[df[drug_col] == drug_value]
    
def check_aggreement(df, discordant_1, data, variable_name):
    
    concordant_glp = pd.DataFrame(columns=data.columns)
    discordant_df_1 = pd.DataFrame(columns=data.columns)

    concordant = df[df[variable_name] == df['drug_class']]
    discordant_df_1 = df[df['drug_class'] == discordant_1]
    
    return concordant, discordant_df_1

def get_concordant_discordant(dpp_strata,sglt_strata, data, dpp_strata_actual, sglt_strata_actual, variable_name):

    sglt_val = 1
    dpp_val = 0
    # discordant_dpp_sglt = received SGLT actually but model assigned DPP
    # discordant_sglt_dpp = received DPP in real life but our model assigned SGLT
    
    concordant_dpp, discordant_dpp_sglt = check_aggreement(dpp_strata, sglt_val, data, variable_name)

    concordant_sglt, discordant_sglt_dpp = check_aggreement(sglt_strata, dpp_val, data, variable_name)

    print(" =========== Total number of samples assigned by the model VS Total number of samples in original test data")
    print('DPP samples ', concordant_dpp.shape[0]+discordant_dpp_sglt.shape[0],  dpp_strata_actual.shape[0])
    print('SGLT samples ', concordant_sglt.shape[0]+discordant_sglt_dpp.shape[0], sglt_strata_actual.shape[0])
    print('\n')
   
    # Your data
    concordant_dpp_count = concordant_dpp.shape[0]
    discordant_dpp_sglt_count = discordant_dpp_sglt.shape[0]
    concordant_sglt_count = concordant_sglt.shape[0]
    discordant_sglt_dpp_count = discordant_sglt_dpp.shape[0]

    if((concordant_dpp_count + discordant_dpp_sglt_count != 0) & (concordant_sglt_count + discordant_sglt_dpp_count !=0)):
    # Calculate percentages
        concordant_dpp_percentage = (concordant_dpp_count / (concordant_dpp_count + discordant_dpp_sglt_count)) * 100
        concordant_sglt_percentage = (concordant_sglt_count / (concordant_sglt_count + discordant_sglt_dpp_count)) * 100
        discordant_dpp_sglt_percentage = (discordant_dpp_sglt_count / (concordant_dpp_count + discordant_dpp_sglt_count)) * 100
        discordant_sglt_dpp_percentage = (discordant_sglt_dpp_count / (concordant_sglt_count + discordant_sglt_dpp_count)) * 100
    else:
        concordant_dpp_percentage = 1
        concordant_sglt_percentage = 1
        discordant_dpp_sglt_percentage=1
        discordant_sglt_dpp_percentage =1
    # Data for the table
    data = [
        ["Concordant", "SGLT","SGLT", concordant_sglt_count, f"{concordant_sglt_percentage:.2f}%"],
        ["Discordant", "DPP", "SGLT", discordant_sglt_dpp_count, f"{discordant_dpp_sglt_percentage:.2f}%"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", concordant_dpp_count, f"{concordant_dpp_percentage:.2f}%"],
        ["Discordant", "SGLT", "DPP", discordant_dpp_sglt_count, f"{discordant_sglt_dpp_percentage:.2f}%"],
    ]

    # Print the table
    print(tabulate(data, headers=["Category","Real value", "Predicted value",  "Count", "Percentage of Predicted cases"]))
    print('\n')
    
    return ( concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp)

def print_change_mean(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable):
    # calculate average response. best average should be in concordant group
    print('-------- Average Change --------')

    # Calculate means for each category
    concordant_sglt_mean = concordant_sglt[response_variable].mean()
    discordant_sglt_dpp_mean = discordant_sglt_dpp[response_variable].mean()
    concordant_dpp_mean = concordant_dpp[response_variable].mean()
    discordant_dpp_sglt_mean = discordant_dpp_sglt[response_variable].mean()

    # Data for the table
    data = [
        ["Concordant", "SGLT", "SGLT", concordant_sglt_mean],
        ["Discordant", "DPP", "SGLT", discordant_sglt_dpp_mean],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", concordant_dpp_mean],
        ["Discordant", "SGLT", "DPP", discordant_dpp_sglt_mean],
        
        
    ]

    # Print the table
    headers = ["Category","Real value", "Predicted value", "Average"]
    print(tabulate(data, headers=headers))
    print('\n')

def get_perc(variable_1, variable_2):
    normal = 42.0
    std = (variable_1-variable_2).std()
    mean = (variable_1-variable_2).mean()
    return mean, std
    
def percentage_change_original_data(dpp_strata_actual, sglt_strata_actual, baseline_val, response_variable):
    # Calculate percentages for each category
    sglt_percentage, sglt_std = get_perc(sglt_strata_actual[response_variable], sglt_strata_actual[baseline_val])
    dpp_percentage, dpp_std = get_perc(dpp_strata_actual[response_variable], dpp_strata_actual[baseline_val])
    

    # Data for the table
    data = [
        ["SGLT", f"{sglt_percentage:.2f}%", f"{sglt_std:.2f}%"],
        ["DPP", f"{dpp_percentage:.2f}%", f"{dpp_std:.2f}%"]
    ]

    # Print the table
    headers = ["Category", "Mean Percentage Change from Baseline (original dataset)", "standard deviation of the percentage change from Baseline (original dataset)"]
    print(tabulate(data, headers=headers))
    
# Percentage Change= [(HbA1c Change/HbA1c Baseline) * 100].mean()
def calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable, baseline_val):

    # Calculate percentages for each category
    concordant_sglt_percentage, concordant_sglt_std = get_perc(concordant_sglt[response_variable], concordant_sglt[baseline_val])
    discordant_sglt_dpp_percentage, discordant_sglt_dpp_std = get_perc(discordant_sglt_dpp[response_variable], discordant_sglt_dpp[baseline_val])
    concordant_dpp_percentage, concordant_dpp_std = get_perc(concordant_dpp[response_variable], concordant_dpp[baseline_val])
    discordant_dpp_sglt_percentage, discordant_dpp_sglt_std = get_perc(discordant_dpp_sglt[response_variable], discordant_dpp_sglt[baseline_val])

    sglt_diff = concordant_sglt_percentage - discordant_sglt_dpp_percentage
    dpp_diff = concordant_dpp_percentage - discordant_dpp_sglt_percentage
    # Data for the table
    data = [
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_percentage:.2f}%",  f"{concordant_sglt_std:.2f}%", f"{sglt_diff:.2f}%"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_percentage:.2f}%", f"{discordant_sglt_dpp_std:.2f}%", ''],
        
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_percentage:.2f}%",  f"{concordant_dpp_std:.2f}%", f"{dpp_diff:.2f}"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_percentage:.2f}%", f"{discordant_dpp_sglt_std:.2f}%", ''],
    ]

    # Print the table
    headers = ["Category","Real value", "Predicted value", "Mean % Change from Baseline", "std", 'treatment difference']
    print(tabulate(data, headers=headers))
    
    
def calculate_percentage_change_othre_responses(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable1, response_variable2, response_variable3,
            baseline_val1,baseline_val2, baseline_val3,
            label1, label2, label3):
    
    # Calculate percentages for each response
    concordant_sglt_v1, concordant_sglt_std_v1 = get_perc(concordant_sglt[response_variable1], concordant_sglt[baseline_val1])
    discordant_sglt_dpp_v1, discordant_sglt_dpp_std_v1 = get_perc(discordant_sglt_dpp[response_variable1], discordant_sglt_dpp[baseline_val1])
    concordant_dpp_v1, concordant_dpp_std_v1 = get_perc(concordant_dpp[response_variable1], concordant_dpp[baseline_val1])
    discordant_dpp_sglt_v1, discordant_dpp_sglt_std_v1 = get_perc(discordant_dpp_sglt[response_variable1], discordant_dpp_sglt[baseline_val1])

    concordant_sglt_v2, concordant_sglt_std_v2 = get_perc(concordant_sglt[response_variable2], concordant_sglt[baseline_val2])
    discordant_sglt_dpp_v2, discordant_sglt_dpp_std_v2 = get_perc(discordant_sglt_dpp[response_variable2], discordant_sglt_dpp[baseline_val2])
    concordant_dpp_v2, concordant_dpp_std_v2 = get_perc(concordant_dpp[response_variable2], concordant_dpp[baseline_val2])
    discordant_dpp_sglt_v2, discordant_dpp_sglt_std_v2 = get_perc(discordant_dpp_sglt[response_variable2], discordant_dpp_sglt[baseline_val2])

    concordant_sglt_v3, concordant_sglt_std_v3 = get_perc(concordant_sglt[response_variable3], concordant_sglt[baseline_val3])
    discordant_sglt_dpp_v3, discordant_sglt_dpp_std_v3 = get_perc(discordant_sglt_dpp[response_variable3], discordant_sglt_dpp[baseline_val3])
    concordant_dpp_v3, concordant_dpp_std_v3 = get_perc(concordant_dpp[response_variable3], concordant_dpp[baseline_val3])
    discordant_dpp_sglt_v3, discordant_dpp_sglt_std_v3 = get_perc(discordant_dpp_sglt[response_variable3], discordant_dpp_sglt[baseline_val3])

    data = [
        [label1,'','','',''],
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v1:.2f}%",  f"{concordant_sglt_std_v1:.2f}%"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v1:.2f}%", f"{discordant_sglt_dpp_std_v1:.2f}%"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v1:.2f}%",  f"{concordant_dpp_std_v1:.2f}%"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v1:.2f}%", f"{discordant_dpp_sglt_std_v1:.2f}%"],
        ['','','','',''],
        
        [label2,'','','',''],
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v2:.2f}%",  f"{concordant_sglt_std_v2:.2f}%"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v2:.2f}%", f"{discordant_sglt_dpp_std_v2:.2f}%"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v2:.2f}%",  f"{concordant_dpp_std_v2:.2f}%"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v2:.2f}%", f"{discordant_dpp_sglt_std_v2:.2f}%"],
        ['','','','',''],
        
        [label3,'','','',''],
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v3:.2f}%",  f"{concordant_sglt_std_v3:.2f}%"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v3:.2f}%", f"{discordant_sglt_dpp_std_v3:.2f}%"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v3:.2f}%",  f"{concordant_dpp_std_v3:.2f}%"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v3:.2f}%", f"{discordant_dpp_sglt_std_v3:.2f}%"],
        ['','','','',''],
    ]

    # Print the table
    headers = ["Category","Real value", "Predicted value", "Mean Change", "standard deviation of the change"]
    print(tabulate(data, headers=headers))
    
    
def calculate_count_diff(data, response_variable, baseline_val, predicted_change ):
    # Use vectorized operations to compare entire columns at once
    
    real_change = (data[response_variable] - data[baseline_val])
    pred_change = (data[predicted_change] - data[baseline_val])
    
    count_actual = (real_change > pred_change).sum()
    count_pred = (real_change < pred_change).sum()
    
    greater_than_bl_actual = (real_change>0).sum()
    greater_than_bl_pred = (pred_change>0).sum()
    
    return count_actual, count_pred, greater_than_bl_actual, greater_than_bl_pred
    
def calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                          response_variable, baseline_val, predicted_change):
    concordant_sglt_actual, concordant_sglt_pred, sglt_greater_than_bl_actual, sglt_greater_than_bl_pred = calculate_count_diff(concordant_sglt, 
                                                                                                                                response_variable, baseline_val, predicted_change)
    concordant_dpp_actual, concordant_dpp_pred, dpp_greater_than_bl_actual, dpp_greater_than_bl_pred = calculate_count_diff(concordant_dpp,
                                                                                                                            response_variable, baseline_val, predicted_change)

    # Data for the table
    data = [
        ["SGLT", f"{concordant_sglt_actual}",  f"{concordant_sglt_pred}"],
        ["DPP", f"{concordant_dpp_actual}",  f"{concordant_dpp_pred}"],
    ]

    # Print the table
    print("\n\n# of samples\n")
    headers = ["Category", "# of samples that have lower hba1c in real data", "# of samples that have lower hba1c in predicted data"]
    print(tabulate(data, headers=headers))
    print("\n")
    print("The number of patients who took SGLT and had a higher 12-month HbA1c change than baseline in the real data: ", sglt_greater_than_bl_actual)
    print("The number of patients who took SGLT and had a higher 12-month HbA1c change than baseline in the predicted data: ", sglt_greater_than_bl_pred)
    print("The number of patients who took DPP and had a higher 12-month HbA1c change than baseline in the real data: ", dpp_greater_than_bl_actual)
    print("The number of patients who took SGLT and had a higher 12-month HbA1c change than baseline in the predicted data: ", dpp_greater_than_bl_pred) 
    

def check_distribution(df, df_act, response_variable, predicted_change):
    # Find outliers using z-score
    out_test = []
    error = df_act[response_variable] - df[predicted_change]
    stdres = (error - np.mean(error)) / np.std(error)
    c = stdres.abs() > 5
    index_outlier = np.where(c == True)
    index = stdres.index
    for j in range(len(c)):
        if c.iloc[j] == True:
                #print(f"Output {i + 1}, Test, Outlier Index: {index[j]}")
            out_test.append(index[j])
    
    z_scores_col1 = (df_act[response_variable] - np.mean(df_act[response_variable])) / np.std(df_act[response_variable])
    outliers_col1 = df_act[abs(z_scores_col1) > 3]
    outliers_act = outliers_col1.index.to_list()
    
    z_scores_col2 = (df[predicted_change] - np.mean(df[predicted_change])) / np.std(df[predicted_change])
    outliers_col2 = df[abs(z_scores_col2) > 3]
    outliers_pred = outliers_col2.index.to_list()
 
    
    return outliers_act, outliers_pred
    
def plot_scatter(df, df_act, df2, df_act2, baseline_val, predicted_change, response_variable):
    # Set up the figure with two rows and two columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Visualize box plots in the first row
    axs[0, 0].boxplot([df_act[response_variable], df[predicted_change]], labels=[response_variable, predicted_change])
    axs[0, 0].set_title('Observed and Predicted values for DPP')
    axs[0, 0].set_ylabel('Value')

    axs[0, 1].boxplot([df_act2[response_variable], df2[predicted_change]], labels=[response_variable, predicted_change])
    axs[0, 1].set_title('Observed and Predicted values for SGLT')
    axs[0, 1].set_ylabel('Value')
    
    # Scatter plot for the first set of DataFrames in the first row
    axs[1, 0].scatter(df_act[baseline_val], df_act[response_variable], color='blue', label='Observed')
    axs[1, 0].scatter(df[baseline_val], df[predicted_change], color='brown', label='Predicted')

    # Fit lines for the scatter plots in the first row
    actual_line = np.polyfit(df_act[baseline_val], df_act[response_variable], 1)
    predicted_line = np.polyfit(df[baseline_val], df[predicted_change], 1)

    axs[1, 0].plot(df_act[baseline_val], np.polyval(actual_line, df_act[baseline_val]), color='blue', linestyle='--', label='Observed')
    axs[1, 0].plot(df[baseline_val], np.polyval(predicted_line, df[baseline_val]), color='brown', linestyle='--', label='Predicted')

    # Set labels and legend for the first row
    axs[1, 0].set_xlabel(baseline_val)
    axs[1, 0].set_ylabel(response_variable)
    axs[1, 0].legend()

    # Scatter plot for the second set of DataFrames in the second row
    axs[1, 1].scatter(df_act2[baseline_val], df_act2[response_variable], color='blue', label='Observed')
    axs[1, 1].scatter(df2[baseline_val], df2[predicted_change], color='brown', label='Predicted')

    # Fit lines for the scatter plots in the second row
    actual_line2 = np.polyfit(df_act2[baseline_val], df_act2[response_variable], 1)
    predicted_line2 = np.polyfit(df2[baseline_val], df2[predicted_change], 1)

    axs[1, 1].plot(df_act2[baseline_val], np.polyval(actual_line2, df_act2[baseline_val]), color='blue', linestyle='--', label='Observed')
    axs[1, 1].plot(df2[baseline_val], np.polyval(predicted_line2, df2[baseline_val]), color='brown', linestyle='--', label='Predicted')

    # Set labels and legend for the second row
    axs[1, 1].set_xlabel(baseline_val)
    axs[1, 1].set_ylabel(response_variable)
    axs[1, 1].legend()

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot as an image file for inclusion in the research paper
    #plt.savefig('scatter_plot_research_paper.png', dpi=300)

    # Show the plot
    plt.show()


def plot_scatter_with_CI(df, df_act, df2, df_act2, baseline_val, predicted_change, response_variable):
    # Set up the figure with two rows and two columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Visualize box plots in the first row
    axs[0, 0].boxplot([df_act[response_variable], df[predicted_change]], labels=[response_variable, predicted_change])
    axs[0, 0].set_title('Observed and Predicted values for DPP')
    axs[0, 0].set_ylabel('Value')

    axs[0, 1].boxplot([df_act2[response_variable], df2[predicted_change]], labels=[response_variable, predicted_change])
    axs[0, 1].set_title('Observed and Predicted values for SGLT')
    axs[0, 1].set_ylabel('Value')
    
    # Scatter plot for the first set of DataFrames in the first row
    sns.regplot(x=df_act[baseline_val], y=df_act[response_variable], ax=axs[1, 0], color='blue', scatter=False, label='Observed', ci=95)
    sns.regplot(x=df[baseline_val], y=df[predicted_change], ax=axs[1, 0], color='brown', scatter=False, label='Predicted', ci=95)

    # Set labels and legend for the first row
    axs[1, 0].set_xlabel(baseline_val)
    axs[1, 0].set_ylabel(response_variable)
    axs[1, 0].legend()

    # Scatter plot for the second set of DataFrames in the second row
    sns.regplot(x=df_act2[baseline_val], y=df_act2[response_variable], ax=axs[1, 1], color='blue', scatter=False, label='Observed', ci=95)
    sns.regplot(x=df2[baseline_val], y=df2[predicted_change], ax=axs[1, 1], color='brown', scatter=False, label='Predicted', ci=95)

    # Set labels and legend for the second row
    axs[1, 1].set_xlabel(baseline_val)
    axs[1, 1].set_ylabel(response_variable)
    axs[1, 1].legend()

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot as an image file for inclusion in the research paper
    plt.savefig('scatter_plot_research_paper.png', dpi=300)

    # Show the plot
    plt.show()


def drug_class_visualization(df, df_act, df2, df_act2, response_variable, predicted_change, assigned_drug, baseline_val):
    # glp strata predicted by the model
    
    df_ = df[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
    df_2 = df2[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
    
    # glp strata in real
    df_act_ = df_act[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
    df_act_2 = df_act2[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]

    outliers_act, outliers_pred = check_distribution(df_, df_act_, response_variable, predicted_change)
    outliers_act2, outliers_pred2 = check_distribution(df_2, df_act_2, response_variable, predicted_change)
    
    df_ = df_.drop(outliers_pred)
    df_act_ = df_act_.drop(outliers_act)
    df_2 = df_2.drop(outliers_pred2)
    df_act_2 = df_act_2.drop(outliers_act2)

    plot_scatter(df_, df_act_,df_2,df_act_2, baseline_val, predicted_change, response_variable)
    
    df_new = df.drop(outliers_pred)
    return df_new

def drug_class_outlier_remove(df, df_act, response_variable, predicted_change, assigned_drug, baseline_val):
    # glp strata predicted by the model
    df_ = df[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
    # glp strata in real
    df_act_ = df_act[[response_variable
                      , predicted_change, 'drug_class', assigned_drug, baseline_val]]

    outliers_act, outliers_pred = check_distribution(df_, df_act_, response_variable, predicted_change)
    
    df_ = df_.drop(outliers_pred)
    df_act_ = df_act_.drop(outliers_act)
    df_new = df.drop(outliers_pred)
    
    return df_new

def save_data_for_ensemble(X_train_original, Y_train, X_test_original, Y_test, file_path):
    train_data = pd.concat([X_train_original, Y_train], axis=1)
    # Concatenate X_test and Y_test horizontally
    test_data = pd.concat([X_test_original, Y_test], axis=1)

    result = pd.concat([train_data, test_data], axis=0)
    
    result.to_csv(file_path)

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = (arr - min_val) / (max_val - min_val)
    return normalized

def get_feature_importance(model, X, file_path):
    feature_importances = []
    for i, regressor in enumerate(model.estimators_):
        if hasattr(regressor, 'feature_importances_'):
            fi_norm = min_max_normalize(regressor.feature_importances_[:X.shape[1]])
            feature_importances.append(fi_norm)
            
        else:
            print(f"Regressor for output {i} does not have feature importances")
            
    stacked_feature_importances = np.vstack(feature_importances)
    average_feature_importances = np.mean(stacked_feature_importances, axis=0)
    importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': average_feature_importances
    }).sort_values(by='Importance', ascending=False)

    importances_df.to_csv(file_path, index = False)
    
    return importances_df

def get_feature_importance_for_voting_regressor(model, X, file_path):
    feature_importances = []
    
    for i, val in enumerate(model.estimators_):
    # Handle VotingRegressor directly
        if hasattr(model.estimators_[i], 'named_estimators_'):
            for j, regressor in model.estimators_[i].named_estimators_.items():
                # normalize it because different regressors in different scales
                fi_norm = min_max_normalize(model.estimators_[i].named_estimators_[j].feature_importances_[:X.shape[1]])
                feature_importances.append(fi_norm)

    # Calculate average feature importances across all base regressors
    if feature_importances:
        stacked_feature_importances = np.vstack(feature_importances)
        average_feature_importances = np.mean(stacked_feature_importances, axis=0)
        
        importances_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': average_feature_importances
        }).sort_values(by='Importance', ascending=False)
        
        # Save to CSV (optional)
        importances_df.to_csv(file_path, index=False) # file_path = FEATURE_IMPORTANCE_DF_LOCATION
        
        return importances_df
    else:
        print("No feature importances found for any regressor.")
        return None