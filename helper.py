import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as sm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from skrebate import ReliefF
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.multioutput import MultiOutputRegressor

from constants import SEED, SGLT_VALUE, DPP_VALUE


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

    """
    Further preprocesses the given DataFrame by performing the following steps:

    1. Removes rows with missing values in any of the specified response variables.
    2. Filters the DataFrame to include only rows where 'days_hba1c' is between 21 and 365 (inclusive).
    3. Drops id column and columns related to time intervals and dates.
    4. Removes rows where 'hdl_12m' is greater than 2.5 and 'bmi_12m' is greater than 50.
    5. Converts all remaining columns to float type.

    Args:
        df (DataFrame): The DataFrame to be preprocessed.
        response_variable_list (list): A list of target column names.

    Returns:
        DataFrame: The preprocessed DataFrame.
    """
    
    # remove rows with missing 'response variable'
    df = df.dropna(how='any', subset = response_variable_list)
    print('Shape of data after excluding missing response:', np.shape(df))
    
    # select time interval
    start = 21
    end = 365 #426
    df = df[(df['days_hba1c'] >= start) & (df['days_hba1c'] <= end)]
    print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
    
    del_cols = ['id', 'date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl',
                 'date_12m', 'date_n1', 'date_ldl_12m', 'date_bmi_12m', 'date_hdl_12m',
                 'days_hba1c', 'days_bmi', 'days_hdl', 'days_ldl']
    df = df.drop(del_cols, axis=1)
    
    # Define thresholds
    hdl_threshold = 2.5
    bmi_threshold = 50
    mask = (df['hdl_12m'] > hdl_threshold) | (df['bmi_12m'] > bmi_threshold) # Create combined mask for rows to be removed
    df = df[~mask] # Drop rows based on the combined mask
    
    df = df.astype(float)
    return df

    
def get_test_train_data(X_train_df, X_test_df, response_variable_list):
    
    """
    Prepare training and testing data by performing the following steps:

    1. Combine the training and testing datasets to handle missing values and scaling consistently.
    2. Separate features and response variables for both training and testing datasets.
    3. Impute missing values in features using the most frequent strategy.
    4. Scale the features using Min-Max scaling, except for specified columns.
    5. Perform random oversampling on the training data to balance the response variable distribution.

    Args:
        X_train_df (DataFrame): The training DataFrame.
        X_test_df (DataFrame): The testing DataFrame.
        response_variable_list (list): A list of column names of target variables.

    Returns:
        tuple: A tuple containing:
            - original (DataFrame): The combined original dataset (training + testing).
            - X_train (DataFrame): The processed training features after imputation, scaling, and oversampling.
            - X_test (DataFrame): The processed testing features after imputation and scaling.
            - Y_train (Series): The training response variables.
            - Y_test (Series): The testing response variables.
            - X (DataFrame): The features from the original combined dataset.
            - Y (DataFrame): The response variables from the original combined dataset.
            - scaler (Object): The scaler used for normalizing the features.
            - X_test_before_scale (DataFrame): The testing features before scaling.
    """
    
    # split data
    random.seed(SEED)
    # Save original data set
    original = pd.concat([X_train_df, X_test_df], ignore_index=False)
    
    Y = original[response_variable_list]
    X = original.drop(response_variable_list, axis=1)
    
    Y_train = X_train_df[response_variable_list]
    X_train = X_train_df.drop(response_variable_list, axis=1)
    
    Y_test = X_test_df[response_variable_list]
    X_test = X_test_df.drop(response_variable_list, axis=1)
    random.seed(SEED)
    
    # data imputation
    original_X_train = X_train.copy()
    original_X_test = X_test.copy()
    random.seed(SEED)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy = "most_frequent")
    # imputeX = KNNImputer(missing_values=np.nan, n_neighbors = 3, weights='distance')
    # imputeX = IterativeImputer(max_iter=5, random_state=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print('X_train shape after imputation: ', X_train.shape)
    
    X_train = pd.DataFrame(X_train, columns = original_X_train.columns, index=original_X_train.index)
    X_test = pd.DataFrame(X_test, columns = original_X_train.columns, index=original_X_test.index)

    # columns_to_skip_normalization = ['drug_class']
    columns_to_skip_normalization = []
    # List of columns to normalize
    columns_to_normalize = [col for col in X_train.columns if col not in columns_to_skip_normalization]

    # scale data 
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    select = {}
    # X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    # X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    
    # random oversampling 
    combined_df = pd.concat([X_train, Y_train], axis=1)
    X_oversamp = combined_df.drop(['drug_class'], axis = 1)
    Y_oversamp = combined_df['drug_class']
    random.seed(SEED)
    ros = RandomOverSampler(random_state=0)
    #smote = SMOTE()
    random.seed(SEED)
    X_resampled, y_resampled = ros.fit_resample(X_oversamp, Y_oversamp)
    print('\n Shape of the data after oversampling')
    print(set(Y_oversamp))
    print(sorted(Counter(Y_oversamp).items()))
    print(sorted(Counter(y_resampled).items()))
    combined = pd.concat([X_resampled, y_resampled], axis=1)
    
    X_train = combined.drop(response_variable_list, axis = 1)
    Y_train = combined[response_variable_list]
    
    X_test_before_scale = X_test.copy()
    X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])
    
    return original, X_train, X_test, Y_train, Y_test, X, Y, scaler, X_test_before_scale

def countUsers(drug_id, df):
    
    """
    Counts the number of users in the DataFrame who are associated with a specific drug class.

    Args:
        drug_id (int): The drug class identifier to filter the users.
        df (DataFrame): The DataFrame containing user data with a column 'drug_class'.

    Returns:
        int: The number of users associated with the specified drug class.
    """
    
    df2 = df.apply(lambda x : True
                if x['drug_class'] == drug_id else False, axis = 1)
    number_of_rows = len(df2[df2 == True].index)
    return number_of_rows


def get_features_kbest(X_train, Y_train,i):
    
    """
    Selects the top `i` features from the training dataset using the `SelectKBest` method with mutual 
    information as the scoring function.

    Args:
        X_train (DataFrame): The training data.
        Y_train (Series): The training target variable.
        i (int): The number of top features to select.

    Returns:
        list: A list of the names of the top `i` selected features.
    """
    
    random.seed(SEED)
    np.random.seed(SEED)
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
    
    """
    Selects the top `i` features from the training dataset using Recursive Feature Elimination (RFE) with a 
    RandomForestRegressor.

    This function utilizes RFE to recursively remove features and identify the top `i` features based on their
    importance scores from a RandomForestRegressor model.

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.Series): The training target variable.
        i (int): The number of top features to select.

    Returns:
        list: A list of the names of the top `i` selected features.
    """
    
    random.seed(SEED)
    model = RandomForestRegressor(random_state = 123)
    rfe = RFE(estimator=model, n_features_to_select=i)  
    X_selected = rfe.fit_transform(X_train, Y_train)
    selected_indices = rfe.get_support(indices=True)
    selected_features = [feature_name for feature_name in X_train.columns[selected_indices]]
    return selected_features

def get_features_ref_multiout(X_train, Y_train, k=12):
    
    """
    Selects the top `k` features from the training dataset using Recursive Feature Elimination (RFE) 
    with a MultiOutputRegressor.

    This function first trains a `MultiOutputRegressor` with a `RandomForestRegressor` as the base estimator 
    on the training data. It then uses RFE to identify and return the names of the top `k` features that contribute
    most to predicting the target variables.

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.DataFrame): The training target variables DataFrame with multiple outputs.
        k (int, optional): The number of top features to select. Defaults to 12.

    Returns:
        list: A list of the names of the top `k` selected features.
    """
    
    random.seed(SEED)
    model = MultiOutputRegressor(RandomForestRegressor(random_state = 123))
    model.fit(X_train, Y_train)  # Fit the model before using RFE
    base_estimator = RandomForestRegressor(random_state=123)
    rfe = RFE(estimator=base_estimator, n_features_to_select=k)  
    X_selected = rfe.fit_transform(X_train, Y_train)
    selected_indices = rfe.get_support(indices=True)
    selected_features = [feature_name for feature_name in X_train.columns[selected_indices]]
    return selected_features


def get_features_relieff(X_train, Y_train,i):
    
    """
    Selects the top `i` features from the training dataset using the ReliefF algorithm.

    This function uses the ReliefF algorithm to evaluate feature importance and returns the names of the top `i` features
    with the highest importance scores.

    Args:
        X_train (pd.DataFrame): The training data.
        Y_train (pd.Series): The training target variable.
        i (int): The number of top features to select.

    Returns:
        list: A list of the names of the top `i` selected features, ordered by their importance scores.
    """
    
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
    
    """
    This function retrieves the name of the model class by converting the model instance's type to a string,
    and then parsing the class name from the string representation.

    Args:
        model (object): An instance of a machine learning model.

    Returns:
        str: The name of the model class.
    """
    
    model_name = str(type(model)).split('.')[-1][:-2]
    return model_name

def cross_val(model, train, X_test, Y_test, X_train, Y_train, response_variable_list, n_splits=3):
    
    """
    Performs cross-validation on the given model using KFold splitting, and evaluates its performance.

    This function splits the training data into `n_splits` folds, trains the model on each training fold,
    evaluates it on the corresponding test fold, and computes performance metrics. The function also
    calculates and prints the variance and mean score of the model across all folds.

    Args:
        model (object): The machine learning model to be evaluated. It should have `fit`, `predict`, and `score` methods.
        train (DataFrame): The training dataset containing features and response variables.
        X_test (DataFrame): The test data.
        Y_test (Series): The test target variables.
        X_train (DataFrame): The training data.
        Y_train (pd.Series): The training target variables.
        response_variable_list (list): A list of column names in `train` that are considered response variables.
        n_splits (int, optional): The number of folds for cross-validation. Defaults to 3.

    Returns:
        object: The trained model after cross-validation.
    """
    
    dfs = []
    acc_arr = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    i = 1
    random.seed(SEED)
    
    for train_index, test_index in kf.split(train, Y_train):
        X_train1 = train.iloc[train_index].loc[:, X_train.columns]
        X_test1 = train.iloc[test_index].loc[:,X_train.columns]
        y_train1 = train.iloc[train_index].loc[:,response_variable_list]
        y_test1 = train.iloc[test_index].loc[:,response_variable_list]
        
        # Train the model
        if (get_model_name(model)=='Sequential'):
            # Use for NN model
            random.seed(SEED)
            model.fit(X_train1, y_train1,epochs=250, batch_size=16, verbose=0)
        else:
            random.seed(SEED)
            model.fit(X_train1, y_train1) 
        
        # calculate the performace matrics
        y_scores = model.predict(X_test1)
        score = model.score(X_test1, y_test1)
        acc_arr.append(score)
        
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
    
    """
    Evaluates and prints the performance of a machine learning model using various metrics.

    This function predicts outcomes on the test set using the provided model, calculates the R² score and RMSE,
    and updates the provided results dictionaries with these metrics.
    
    Args:
        model (object): The machine learning model to be evaluated. It should have `predict` and `score` methods.
        X_test (DataFrame): The test data.
        Y_test (Series): The test target variables.
        X_train (DataFrame): The training data.
        Y_train (Series): The training target variables.
        model_results (dict): A dictionary to store R² scores of models.
        model_results_drugs (dict): A dictionary to store R² scores of models with additional identifiers.
        name (str, optional): An optional string to append to the model name for storing results. Defaults to an empty string.

    Returns:
        tuple: A tuple containing:
            - pred (ndarray): The predicted values for the test set.
            - model_results (dict): The updated dictionary of R² scores for models.
            - model_results_drugs (dict): The updated dictionary of R² scores for models with additional identifiers.
            - score (float): The R² score of the model on the test set.
    """
    
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

def get_outliers(Y, predictions):
    """
    Detects outliers based on standardized residuals.
    
    Args:
        Y (pd.DataFrame): The target variable DataFrame (either training or testing).
        predictions (np.ndarray): The predicted values corresponding to Y.

    Returns:
        list: A list of indices where outliers are detected.
    """
    outliers = []
    for i, col in enumerate(Y.columns):
        actual = Y[col].values  # Convert to numpy array
        pred = predictions[:, i] if isinstance(predictions, np.ndarray) else predictions[i]  # Handle train/test
        error = actual - pred
        stdres = (error - np.mean(error)) / np.std(error)  # Standardized residuals
        
        # Identify outliers based on the threshold
        outliers.extend(Y.index[abs(stdres) > 4])
    
    return outliers

def outlier_detect(X_train, Y_train, X_test, Y_test):
    
    """
    Detects outliers in the training and testing datasets based on residuals from OLS regression models.

    This function fits an Ordinary Least Squares (OLS) regression model for each response variable in `Y_train`,
    makes predictions on both the training and testing datasets, and identifies outliers based on standardized residuals.
    Outliers are defined as data points with standardized residuals exceeding an absolute value of 4.

    Args:
        X_train (DataFrame): The training data.
        Y_train (DataFrame): The training target variables.
        X_test (DataFrame): The testing features data.
        Y_test (DataFrame): The testing target variables.

    Returns:
        tuple: A tuple containing:
            - out_train (list): A list of indices of outliers in the training set.
            - out_test (list): A list of indices of outliers in the testing set.
    """
    
    # Fit the model for each output in Y_train
    models = []
    predictions_train = []
    for col in Y_train.columns:
        model = sm.OLS(Y_train[col], X_train).fit()
        predictions_train.append(model.predict(X_train))
        models.append(model)

    # Make predictions for each output in Y_test
    predictions = np.column_stack([model.predict(X_test) for model in models])
    
    # Detect outliers in the training set
    out_train = get_outliers(Y_train, predictions_train)
    # Detect outliers in the testing set
    out_test = get_outliers(Y_test, predictions)

    print("Training set outliers:", out_train)
    print("Testing set outliers:", out_test)
    
    return out_train, out_test

def train_models(model, X_test, Y_test, X_train, Y_train, train, scaler, X_test_original, response_variable_list):
    
    """
    Trains a given model using cross-validation and fits it to the training data, 
    while calculating and storing performance results.

    This function checks the type of model and applies the appropriate training procedure.
    It uses cross-validation for evaluation, compiles the model if it’s a sequential model, 
    and fits the model to the training data.

    Args:
        model (object): The machine learning model to be trained. Can be a sequential model (e.g., Keras) or a traditional ML model.
        X_test (DataFrame): Test dataset.
        Y_test (DataFrame): Target variables for the test dataset.
        X_train (DataFrame): Training dataset.
        Y_train (DataFrame): Target variable(s) for the training dataset.
        train (DataFrame): Combined training dataset used for cross-validation.
        scaler (object): Scaler used for normalization of the data.
        X_test_original (DataFrame): Unscaled version of the test dataset.
        response_variable_list (list): List of target variable names to be used for training and prediction.

    Returns:
        tuple: A tuple containing:
            - model_results (dict): A dictionary containing the performance metrics (e.g., R² scores) of the model for different targets.
            - model (object): The trained model after fitting on the training data.
    """
    
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

def pred_all(model, row, drug_class):
    
    """
    Predicts outcomes based on the specified drug class using the given model.

    This function takes a model and a data row, and based on the specified drug class,
    it makes predictions for both SGLT and DPP drug classes.

    Args:
        model (object): The trained machine learning model used for making predictions.
        row (DataFrame): A single row of input features for prediction.
        drug_class (int): An integer indicating the current drug class (0 for DPP, 1 for SGLT).

    Returns:
        tuple: A tuple containing:
            - pred_sglt_ (ndarray): The predicted values for the SGLT drug class.
            - pred_dpp_ (ndarray): The predicted values for the DPP drug class.
        
    Raises:
        ValueError: If an invalid drug class is provided.
    """
    
    if drug_class == SGLT_VALUE:
        pred_sglt_ = model.predict(row.values[None])[0]
        row['drug_class'] = DPP_VALUE
        pred_dpp_ = model.predict(row.values[None])[0]
        
    elif drug_class == DPP_VALUE:
        pred_dpp_ = model.predict(row.values[None])[0]
        row['drug_class'] = SGLT_VALUE
        pred_sglt_ = model.predict(row.values[None])[0]
    else:
        print('Worng drug class')
        raise ValueError(f"No drug class for given input")
    return pred_sglt_, pred_dpp_

def find_lowest_respponse_value(pred_sglt, pred_dpp):
    
    """
    Determines the lowest predicted response value between SGLT and DPP drug classes.

    This function takes the predicted response values for both SGLT and DPP drug classes,
    identifies which one is lower, and returns that value along with the corresponding drug class.
    This method used to take decisions on hba1c, ldl and bmi

    Args:
        pred_sglt (float): The predicted response value for the SGLT drug class.
        pred_dpp (float): The predicted response value for the DPP drug class.

    Returns:
        tuple: A tuple containing:
            - min_difference (float): The lowest predicted response value between SGLT and DPP.
            - drug_class (int): An integer representing the drug class corresponding to the lowest value 
                                (1 for SGLT, 0 for DPP).
    """
    
    values = [pred_sglt, pred_dpp]
    min_index = values.index(min(values))
    min_difference = [pred_sglt, pred_dpp][min_index]
    drug_class = [SGLT_VALUE, DPP_VALUE][min_index]
    return min_difference, drug_class

def find_highest_respponse_value(pred_sglt, pred_dpp):
    
    """
    Determines the highest predicted response value between SGLT and DPP drug classes.

    This function takes the predicted response values for both SGLT and DPP drug classes,
    identifies which one is higher, and returns that value along with the corresponding drug class.
    This method used to take decisions on hdl

    Args:
        pred_sglt (float): The predicted response value for the SGLT drug class.
        pred_dpp (float): The predicted response value for the DPP drug class.

    Returns:
        tuple: A tuple containing:
            - max_difference (float): The highest predicted response value between SGLT and DPP.
            - drug_class (int): An integer representing the drug class corresponding to the highest value 
                                (1 for SGLT, 0 for DPP).
    """
    
    values = [pred_sglt, pred_dpp]
    max_index = values.index(max(values))
    max_difference = [pred_sglt, pred_dpp][max_index]
    drug_class = [SGLT_VALUE, DPP_VALUE][max_index]
    return max_difference, drug_class

#### new change
def find_closest_to_42(pred_sglt, pred_dpp):
    
    """
    Finds the predicted response value closest to 42 between SGLT and DPP drug classes. This method only use for hba1c.
    But in this experiment, we did not use this method

    This function takes the predicted response values for both SGLT and DPP drug classes,
    determines which value is closest to 42, and returns that value along with the corresponding drug class.

    Args:
        pred_sglt (float): The predicted response value for the SGLT drug class.
        pred_dpp (float): The predicted response value for the DPP drug class.

    Returns:
        tuple: A tuple containing:
            - closest_value (float): The predicted response value closest to 42.
            - drug_class (int): An integer representing the drug class corresponding to the closest value 
                                (1 for SGLT, 0 for DPP).
    """
    
    values = [pred_sglt, pred_dpp]
    drug_classes = [SGLT_VALUE, DPP_VALUE]
    max_index = min(range(len(values)), key=lambda i: abs(values[i] - 42.0))
    closest_value = values[max_index]
    drug_class = drug_classes[max_index]
    return closest_value, drug_class

def predict_drug_classes(model, X_test, Y_train):
    
    """
    Predicts drug classes and associated changes in response variables for a given test dataset.

    This function utilizes a trained model to make predictions for different drug classes (SGLT and DPP)
    based on the input test data. It assigns the drug class that results in the highest or lowest predicted
    change in response variables (HbA1c, LDL, HDL, and BMI) for each patient in the test dataset.

    Args:
        model (object): The trained model used for making predictions.
        X_test (DataFrame): The test dataset containing features, including drug class information.
        Y_train (DataFrame): The training dataset containing target variables.

    Returns:
        DataFrame: A copy of the input test dataset with assigned drug classes and predicted change for all response
        variables
    """
    X_test_copy = X_test.copy()
    
    response_vars = ['hba1c', 'ldl', 'hdl', 'bmi']
    for var in response_vars:
        X_test_copy[f'assigned_drug_{var}'] = np.nan
        X_test_copy[f'predicted_change_{var}'] = np.nan

    # Iterate over each row in the test set
    for index, row in X_test.iterrows():
        drug_class = row['drug_class']
        pred_sglt, pred_dpp = pred_all(model, row, drug_class)

        # Process predictions for each response variable
        for j, var in enumerate(response_vars):
            if var == 'hdl':
                max_change, assigned_drug_class = find_highest_respponse_value(pred_sglt[j], pred_dpp[j])
            else:
                max_change, assigned_drug_class = find_lowest_respponse_value(pred_sglt[j], pred_dpp[j])

            X_test_copy.at[index, f'assigned_drug_{var}'] = assigned_drug_class
            X_test_copy.at[index, f'predicted_change_{var}'] = max_change

    return X_test_copy
    
def print_strata_stats(dpp_strata_actual, sglt_strata_actual, dpp_strata_hba1c, sglt_strata_hba1c, dpp_strata_ldl, sglt_strata_ldl,
                       dpp_strata_hdl, sglt_strata_hdl, dpp_strata_bmi, sglt_strata_bmi):
    
    """
    Prints statistics about sample counts in test data and assigned samples for various response variables.

    This function displays the number of samples in the test dataset for two drug classes (DPP4 and SGLT2),
    along with the counts of samples assigned for HBA1C, LDL, HDL, and BMI for both drug classes.

    Args:
        dpp_strata_actual (DataFrame): Samples in the test dataset for the DPP4 class.
        sglt_strata_actual (DataFrame): Samples in the test dataset for the SGLT2 class.
        dpp_strata_hba1c (DataFrame): Assigned samples for HBA1C for the DPP4 class.
        sglt_strata_hba1c (DataFrame): Assigned samples for HBA1C for the SGLT2 class.
        dpp_strata_ldl (DataFrame): Assigned samples for LDL for the DPP4 class.
        sglt_strata_ldl (DataFrame): Assigned samples for LDL for the SGLT2 class.
        dpp_strata_hdl (DataFrame): Assigned samples for HDL for the DPP4 class.
        sglt_strata_hdl (DataFrame): Assigned samples for HDL for the SGLT2 class.
        dpp_strata_bmi (DataFrame): Assigned samples for BMI for the DPP4 class.
        sglt_strata_bmi (DataFrame): Assigned samples for BMI for the SGLT2 class.
    """
    
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
    
    """
    Checks the agreement between drug class assignments and specified variable values in the dataset.

    This function separates the dataset into concordant and discordant subsets based on the specified 
    variable and the drug class. It returns two DataFrames: one containing rows where the variable matches 
    the drug class (concordant) and another containing rows that are discordant with respect to the specified 
    discordant drug class.

    Args:
        df (DataFrame): The input DataFrame containing drug class assignments and other relevant data.
        discordant_1 (str): The drug class to be considered as discordant.
        data (DataFrame): A DataFrame used to ensure the concordant DataFrame has the same columns.
        variable_name (str): The name of the column in `df` to check for agreement with the drug class.

    Returns:
        tuple: A tuple containing:
            - concordant (DataFrame): DataFrame of rows where the variable matches the drug class.
            - discordant_df_1 (DataFrame): DataFrame of rows that are discordant with respect to `discordant_1`.
    """
    
    discordant_df_1 = pd.DataFrame(columns=data.columns)

    concordant = df[df[variable_name] == df['drug_class']]
    discordant_df_1 = df[df['drug_class'] == discordant_1]
    
    return concordant, discordant_df_1

def get_concordant_discordant(dpp_strata,sglt_strata, data, dpp_strata_actual, sglt_strata_actual, variable_name):

    """
    Analyzes the concordance and discordance between predicted drug classes and actual drug classes in the dataset.

    This function compares the assigned drug classes (DPP and SGLT) to the actual drug classes in the provided strata,
    calculates counts and percentages of concordant and discordant samples, and prints a summary table with the results.
    It also checks for agreements using the `check_aggreement` function.

    Args:
        dpp_strata (DataFrame): The DataFrame containing the DPP-strata data to analyze.
        sglt_strata (DataFrame): The DataFrame containing the SGLT-strata data to analyze.
        data (DataFrame): The dataset containing drug class assignments and other relevant variables.
        dpp_strata_actual (DataFrame): The actual DPP-strata data for comparison.
        sglt_strata_actual (DataFrame): The actual SGLT-strata data for comparison.
        variable_name (str): The name of the column used to check for agreement with the drug class.

    Returns:
        tuple: A tuple containing:
            - concordant_dpp (DataFrame): DataFrame of concordant DPP samples.
            - discordant_dpp_sglt (DataFrame): DataFrame of discordant DPP samples that were assigned SGLT.
            - concordant_sglt (DataFrame): DataFrame of concordant SGLT samples.
            - discordant_sglt_dpp (DataFrame): DataFrame of discordant SGLT samples that were assigned DPP.
    """
    
    # discordant_dpp_sglt = received SGLT actually but model assigned DPP
    # discordant_sglt_dpp = received DPP in real life but our model assigned SGLT
    
    concordant_dpp, discordant_dpp_sglt = check_aggreement(dpp_strata, SGLT_VALUE, data, variable_name)

    concordant_sglt, discordant_sglt_dpp = check_aggreement(sglt_strata, DPP_VALUE, data, variable_name)

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
    
    return (concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp)

def print_change_mean(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable):
    
    """
    Prints the average change in the response variable for concordant and discordant drug class predictions.

    This function calculates and displays the mean of the specified response variable for each of the following groups:
    concordant DPP, discordant DPP assigned SGLT, concordant SGLT, and discordant SGLT assigned DPP. The results are
    displayed in a tabulated format for easy comparison of the average changes between these groups.

    Args:
        concordant_dpp (DataFrame): DataFrame containing samples where DPP was predicted and matched the actual class.
        discordant_dpp_sglt (DataFrame): DataFrame containing samples where DPP was predicted but SGLT was the actual class.
        concordant_sglt (DataFrame): DataFrame containing samples where SGLT was predicted and matched the actual class.
        discordant_sglt_dpp (DataFrame): DataFrame containing samples where SGLT was predicted but DPP was the actual class.
        response_variable (str): The name of the response variable (column) for which to calculate the mean change.

    Returns:
        None: This function prints the average change for each group and does not return a value.
    """
    
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
    
    """
    Calculates the mean and standard deviation of the difference between two variables.

    This function computes the mean and standard deviation of the difference between two input variables.

    Args:
        variable_1 (ndarray): The first variable.
        variable_2 (ndarray): The second variable.

    Returns:
        tuple: A tuple containing:
            - mean (float): The mean of the difference between `variable_1` and `variable_2`.
            - std (float): The standard deviation of the difference between `variable_1` and `variable_2`.
    """
    
    std = (variable_1-variable_2).std()
    mean = (variable_1-variable_2).mean()
    return mean, std
    
def percentage_change_original_data(dpp_strata_actual, sglt_strata_actual, baseline_val, response_variable):
    
    """
    Calculates and prints the mean percentage change and standard deviation of the response variable 
    from the baseline value for both SGLT and DPP drug categories.

    This function compares the original test data for two drug categories (SGLT and DPP), calculating 
    the mean and standard deviation of the percentage change in a response variable from its baseline value. 
    It displays the results in a table format.

    Args:
        dpp_strata_actual (DataFrame): The actual data for DPP drug class.
        sglt_strata_actual (DataFrame): The actual data for SGLT drug class.
        baseline_val (str): The name of the column representing the baseline value (e.g., 'baseline_hba1c').
        response_variable (str): The name of the response variable for which the percentage change 
                                 is calculated (e.g., 'hba1c_12m').

    Returns:
        None: This function prints a table with the mean percentage change and standard deviation for 
              both drug categories, without returning any value.
    """
    
    # Calculate percentages for each category
    sglt_percentage, sglt_std = get_perc(sglt_strata_actual[response_variable], sglt_strata_actual[baseline_val])
    dpp_percentage, dpp_std = get_perc(dpp_strata_actual[response_variable], dpp_strata_actual[baseline_val])

    # Data for the table
    data = [
        ["SGLT", f"{sglt_percentage:.2f}", f"{sglt_std:.2f}"],
        ["DPP", f"{dpp_percentage:.2f}", f"{dpp_std:.2f}"]
    ]

    # Print the table
    headers = ["Category", "Mean Percentage Change from Baseline (original dataset)", "standard deviation of the percentage change from Baseline (original dataset)"]
    print(tabulate(data, headers=headers))
    
def calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable, baseline_val):
    
    """
    Calculates and prints the mean change from the baseline and standard deviation for concordant 
    and discordant groups within SGLT and DPP drug classes, along with the treatment difference.

    This function evaluates how the response variable changes from its baseline value in different drug classes 
    (SGLT and DPP), splitting the data into concordant and discordant categories. It also computes the treatment 
    difference between concordant and discordant groups for both drug classes, displaying the results in a table format.

    Args:
        concordant_dpp (DataFrame): Data where DPP was correctly assigned (concordant group).
        discordant_dpp_sglt (DataFrame): Data where DPP was incorrectly assigned as SGLT (discordant group).
        concordant_sglt (DataFrame): Data where SGLT was correctly assigned (concordant group).
        discordant_sglt_dpp (DataFrame): Data where SGLT was incorrectly assigned as DPP (discordant group).
        response_variable (str): The name of the response variable for which the percentage change is calculated 
                                 (e.g., 'hba1c_12m').
        baseline_val (str): The name of the column representing the baseline value (e.g., 'baseline_hba1c').

    Returns:
        None: This function prints a table with the mean percentage change, standard deviation, and treatment 
              differences for the concordant and discordant groups.
    """

    concordant_sglt_percentage, concordant_sglt_std = get_perc(concordant_sglt[response_variable], concordant_sglt[baseline_val])
    discordant_sglt_dpp_percentage, discordant_sglt_dpp_std = get_perc(discordant_sglt_dpp[response_variable], discordant_sglt_dpp[baseline_val])
    concordant_dpp_percentage, concordant_dpp_std = get_perc(concordant_dpp[response_variable], concordant_dpp[baseline_val])
    discordant_dpp_sglt_percentage, discordant_dpp_sglt_std = get_perc(discordant_dpp_sglt[response_variable], discordant_dpp_sglt[baseline_val])

    sglt_diff = concordant_sglt_percentage - discordant_sglt_dpp_percentage
    dpp_diff = concordant_dpp_percentage - discordant_dpp_sglt_percentage
    # Data for the table
    data = [
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_percentage:.2f}",  f"{concordant_sglt_std:.2f}", f"{sglt_diff:.2f}"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_percentage:.2f}", f"{discordant_sglt_dpp_std:.2f}", ''],
        
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_percentage:.2f}",  f"{concordant_dpp_std:.2f}", f"{dpp_diff:.2f}"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_percentage:.2f}", f"{discordant_dpp_sglt_std:.2f}", ''],
    ]

    # Print the table
    headers = ["Category","Real value", "Predicted value", "Mean Change from Baseline", "std", 'treatment difference']
    print(tabulate(data, headers=headers))
    
    
def calculate_percentage_change_othre_responses(concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp, response_variable1, response_variable2, response_variable3,
            baseline_val1,baseline_val2, baseline_val3,
            label1, label2, label3):
    
    """
    Calculates and prints the mean change and standard deviation for other response variables in 
    concordant and discordant groups for SGLT and DPP drug classes.

    This function computes the mean change and standard deviation for other three different response variables 
    (if main response variable is hba1c, this method calculations are for lsl, hdl and bmi) in both concordant and 
    discordant groups within the SGLT and DPP drug classes. It displays the results in a table format organized by 
    the provided labels.

    Args:
        concordant_dpp (DataFrame): Data where DPP was correctly assigned (concordant group).
        discordant_dpp_sglt (DataFrame): Data where DPP was incorrectly assigned as SGLT (discordant group).
        concordant_sglt (DataFrame): Data where SGLT was correctly assigned (concordant group).
        discordant_sglt_dpp (DataFrame): Data where SGLT was incorrectly assigned as DPP (discordant group).
        response_variable1 (str): The first response variable for which the percentage change is calculated.
        response_variable2 (str): The second response variable for which the percentage change is calculated.
        response_variable3 (str): The third response variable for which the percentage change is calculated.
        baseline_val1 (str): The column representing the baseline value for the first response variable.
        baseline_val2 (str): The column representing the baseline value for the second response variable.
        baseline_val3 (str): The column representing the baseline value for the third response variable.
        label1 (str): The label associated with the first response variable.
        label2 (str): The label associated with the second response variable.
        label3 (str): The label associated with the third response variable.

    Returns:
        None: This function prints a table with the mean change and standard deviation for each response variable 
              across the concordant and discordant groups.
    """
    
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
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v1:.2f}",  f"{concordant_sglt_std_v1:.2f}"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v1:.2f}", f"{discordant_sglt_dpp_std_v1:.2f}"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v1:.2f}",  f"{concordant_dpp_std_v1:.2f}"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v1:.2f}", f"{discordant_dpp_sglt_std_v1:.2f}"],
        ['','','','',''],
        
        [label2,'','','',''],
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v2:.2f}",  f"{concordant_sglt_std_v2:.2f}"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v2:.2f}", f"{discordant_sglt_dpp_std_v2:.2f}"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v2:.2f}",  f"{concordant_dpp_std_v2:.2f}"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v2:.2f}", f"{discordant_dpp_sglt_std_v2:.2f}"],
        ['','','','',''],
        
        [label3,'','','',''],
        ["Concordant", "SGLT", "SGLT", f"{concordant_sglt_v3:.2f}",  f"{concordant_sglt_std_v3:.2f}"],
        ["Discordant", "DPP", "SGLT", f"{discordant_sglt_dpp_v3:.2f}", f"{discordant_sglt_dpp_std_v3:.2f}"],
        ['','','','',''],
        ["Concordant", "DPP", "DPP", f"{concordant_dpp_v3:.2f}",  f"{concordant_dpp_std_v3:.2f}"],
        ["Discordant", "SGLT", "DPP", f"{discordant_dpp_sglt_v3:.2f}", f"{discordant_dpp_sglt_std_v3:.2f}"],
        ['','','','',''],
    ]

    # Print the table
    headers = ["Category","Real value", "Predicted value", "Mean Change", "standard deviation of the change"]
    print(tabulate(data, headers=headers))
    
def calculate_count_diff(data, response_variable, baseline_val, predicted_change ):
    
    """
    Compares the real and predicted changes from baseline for a given response variable and returns count differences.

    This function computes and compares the actual change and predicted change from baseline for a given 
    response variable. It returns the count of instances where the actual change 
    is greater than or less than the predicted change, as well as the count of instances where both real and 
    predicted values are greater than the baseline.

    Args:
        data (DataFrame): The dataset containing the response, baseline, and predicted values.
        response_variable (str): The column name representing the actual response variable in the data.
        baseline_val (str): The column name representing the baseline value in the data.
        predicted_change (str): The column name representing the predicted change in the data.

    Returns:
        tuple: A tuple containing four values:
            - count_actual (int): The count of instances where the actual change from baseline is greater than the predicted change.
            - count_pred (int): The count of instances where the predicted change from baseline is greater than the actual change.
            - greater_than_bl_actual (int): The count of instances where the actual change from baseline is greater than 0.
            - greater_than_bl_pred (int): The count of instances where the predicted change from baseline is greater than 0.
    """
    
    real_change = (data[response_variable] - data[baseline_val])
    pred_change = (data[predicted_change] - data[baseline_val])
    
    count_actual = (real_change > pred_change).sum()
    count_pred = (real_change < pred_change).sum()
    
    greater_than_bl_actual = (real_change>0).sum()
    greater_than_bl_pred = (pred_change>0).sum()
    
    return count_actual, count_pred, greater_than_bl_actual, greater_than_bl_pred
    
def calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                          response_variable, baseline_val, predicted_change):
    
    """
    Compares real and predicted changes from baseline for both SGLT and DPP groups and prints the count differences.

    This function calculates the number of cases where the actual change from baseline is greater or lesser than the
    predicted change for concordant SGLT and DPP groups. It also counts and reports the number of cases where 
    the actual and predicted changes are greater than the baseline for each treatment group.

    Args:
        concordant_dpp (DataFrame): Subset of data where DPP is concordant between real and predicted values.
        discordant_dpp_sglt (DataFrame): Subset of data where the model predicted SGLT but the actual treatment was DPP.
        concordant_sglt (DataFrame): Subset of data where SGLT is concordant between real and predicted values.
        discordant_sglt_dpp (DataFrame): Subset of data where the model predicted DPP but the actual treatment was SGLT.
        response_variable (str): Column name representing the actual response variable (e.g., HbA1c change).
        baseline_val (str): Column name representing the baseline value.
        predicted_change (str): Column name representing the predicted change.

    Returns:
        None: This function prints the results in a table format showing the count of samples where the real 
              and predicted changes are lower than baseline, as well as the number of samples with changes greater 
              than the baseline in both real and predicted data.
    """
    
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
    
    """
    Identifies outliers in both actual and predicted data distributions using z-scores.

    This function computes z-scores for the actual response variable and the predicted changes, 
    and identifies outliers where the absolute z-score exceeds a threshold of 3. The indices 
    of these outliers are collected for both the actual and predicted datasets.

    Args:
        df (DataFrame): DataFrame containing predicted values for the response variable.
        df_act (DataFrame): DataFrame containing the actual observed values for the response variable.
        response_variable (str): Column name representing the actual response variable (e.g., HbA1c change).
        predicted_change (str): Column name representing the predicted change in the response variable.

    Returns:
        tuple: A tuple containing two lists:
            - outliers_act (list): Indices of outliers from the actual data where the z-score is greater than 3.
            - outliers_pred (list): Indices of outliers from the predicted data where the z-score is greater than 3.
    """
    
    z_scores_col1 = (df_act[response_variable] - np.mean(df_act[response_variable])) / np.std(df_act[response_variable])
    outliers_col1 = df_act[abs(z_scores_col1) > 3]
    outliers_act = outliers_col1.index.to_list()
    
    z_scores_col2 = (df[predicted_change] - np.mean(df[predicted_change])) / np.std(df[predicted_change])
    outliers_col2 = df[abs(z_scores_col2) > 3]
    outliers_pred = outliers_col2.index.to_list()

    return outliers_act, outliers_pred
    
def plot_scatter(df, df_act, df2, df_act2, baseline_val, predicted_change, response_variable):
    
    """
    Creates scatter and box plots to visualize observed and predicted values for two different drug classes (DPP and SGLT).

    This function sets up a 2x2 subplot layout:
    1. The first row contains box plots comparing observed and predicted values for DPP.
    2. The second row contains scatter plots of observed versus predicted values, along with fitted lines for 
    both drug classes.

    Args:
        df (pd.DataFrame): DataFrame containing predicted values and baseline information for DPP.
        df_act (pd.DataFrame): DataFrame containing actual values and baseline information for DPP.
        df2 (pd.DataFrame): DataFrame containing predicted values and baseline information for SGLT.
        df_act2 (pd.DataFrame): DataFrame containing actual values and baseline information for SGLT.
        baseline_val (str): Name of the column representing the baseline value.
        predicted_change (str): Name of the column representing the predicted change in response variable.
        response_variable (str): Name of the column representing the response variable.

    Save:
        The plot is displayed and save as an image file.
    """
    
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
    plt.savefig('scatter_plot_research_paper.png', dpi=300)

    # Show the plot
    plt.show()


def plot_scatter_with_CI(df, df_act, df2, df_act2, baseline_val, predicted_change, response_variable):
    
    """
    Plots scatter plots and box plots comparing observed and predicted values.

    This function creates a 2x2 subplot layout:
    1. The first row contains box plots comparing observed and predicted values for DPP and SGLT.
    2. The second row contains scatter plots with regression lines showing the relationship between the
    baseline value and the response variable for both DPP and SGLT (with confidence interval).

    Args:
        df (DataFrame): DataFrame containing predicted values and baseline information for DPP.
        df_act (DataFrame): DataFrame containing actual values and baseline information for DPP.
        df2 (DataFrame): DataFrame containing predicted values and baseline information for SGLT.
        df_act2 (DataFrame): DataFrame containing actual values and baseline information for SGLT.
        baseline_val (str): Name of the column representing the baseline value.
        predicted_change (str): Name of the column representing the predicted change in response variable.
        response_variable (str): Name of the column representing the response variable.

    Saves:
        An image file of the plot with a resolution of 300 DPI.
    """
    
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
    
    """
    Visualizes the drug class data by comparing predicted and actual response values.

    This function performs the following steps:
    1. Extracts relevant columns from the provided DataFrames for both predicted and actual drug classes.
    2. Identifies outliers in both the predicted and actual datasets using a distribution check.
    3. Drops the outlier entries from the datasets.
    4. Plots a scatter plot comparing the predicted and actual values of the response variable for both drug classes.

    Args:
        df (pd.DataFrame): DataFrame containing predicted values and drug class information.
        df_act (pd.DataFrame): DataFrame containing actual values and drug class information.
        df2 (pd.DataFrame): Second DataFrame for comparison, containing predicted values and drug class information.
        df_act2 (pd.DataFrame): Second DataFrame for comparison, containing actual values and drug class information.
        response_variable (str): Name of the column representing the response variable.
        predicted_change (str): Name of the column representing the predicted change in response variable.
        assigned_drug (str): Name of the column representing the assigned drug class.
        baseline_val (str): Name of the column representing the baseline value.

    Returns:
        pd.DataFrame: The modified DataFrame with outliers removed.
    """
    
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
    
    """
    Removes outliers based on the drug class from the predicted and actual datasets.

    This function identifies outliers in the predicted and actual datasets based on the specified response variable 
    and the predicted change. It removes these outliers from both the predicted and actual datasets, ensuring that 
    the final dataset excludes these extreme values.

    Args:
        df (DataFrame): The predicted dataset containing the response variable, predicted change, 
                        drug class, assigned drug, and baseline values.
        df_act (DataFrame): The actual dataset containing the same columns as df for comparison.
        response_variable (str): The name of the response variable to evaluate.
        predicted_change (str): The name of the column containing predicted changes.
        assigned_drug (str): The name of the column indicating the assigned drug class.
        baseline_val (str): The name of the column representing baseline values.

    Returns:
        DataFrame: A new DataFrame with outliers removed based on the drug class from the original predicted dataset.
    """
    
    df_ = df[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
    df_act_ = df_act[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]

    outliers_act, outliers_pred = check_distribution(df_, df_act_, response_variable, predicted_change)
    
    df_ = df_.drop(outliers_pred)
    df_act_ = df_act_.drop(outliers_act)
    df_new = df.drop(outliers_pred)
    
    return df_new

def save_data_for_ensemble(X_train_original, Y_train, X_test_original, Y_test, file_path):
    
    """
    Saves the training and testing datasets into a single CSV file for ensemble modeling.

    This function concatenates the original training features (X_train_original) with the corresponding 
    training labels (Y_train), and similarly for the testing data. The combined datasets are then saved 
    to a specified file path.

    Args:
        X_train_original (DataFrame): The original training feature set.
        Y_train (Series or DataFrame): The training labels corresponding to the training feature set.
        X_test_original (DataFrame): The original testing feature set.
        Y_test (Series or DataFrame): The testing labels corresponding to the testing feature set.
        file_path (str): The file path where the combined dataset will be saved as a CSV file.

    Returns:
        None: This function does not return any value; it directly saves the combined data to a file.
    """
    
    train_data = pd.concat([X_train_original, Y_train], axis=1)
    # Concatenate X_test and Y_test horizontally
    test_data = pd.concat([X_test_original, Y_test], axis=1)
    result = pd.concat([train_data, test_data], axis=0)
    result.to_csv(file_path)

def min_max_normalize(arr):
    
    """
    Perform min-max normalization

    Returns:
        normalized (arr): Normalized data
    """
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = (arr - min_val) / (max_val - min_val)
    return normalized

def get_feature_importance(model, X, file_path):
    
    """
    Extracts and averages feature importances from the base regressors of a model.

    This function retrieves model feature importances, normalizes them, and computes the average feature
    importances across all target variables. The results are then saved to a specified CSV file.

    Args:
        model (BaseEstimator): A trained model
        X (DataFrame): The input features used for training the model, used to reference feature names.
        file_path (str): The file path where the resulting feature importances DataFrame will be saved as a CSV.

    Returns:
        DataFrame: A DataFrame containing feature names and their averaged (across all 4 target variables) importance values, sorted 
                   in descending order.
    """
    
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
    
    """
    Extracts and averages feature importances from the base regressors of a Voting Regressor (for ensemble model calculations).

    This function iterates through the base regressors of a Voting Regressor, retrieves their feature importances,
    normalizes them, and computes the average feature importances across all base regressors. The results are 
    then saved to a specified CSV file.

    Args:
        model (VotingRegressor): A trained VotingRegressor model containing multiple base regressors.
        X (DataFrame): The input features used for training the model, used to reference feature names.
        file_path (str): The file path where the resulting feature importances DataFrame will be saved as a CSV.

    Returns:
        DataFrame or None: A DataFrame containing feature names and their averaged importance values, sorted 
                           in descending order. Returns None if no feature importances are found.
    """
    
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