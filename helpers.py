import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from skrebate import ReliefF
import statsmodels.regression.linear_model as sm
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from constants import SEED, DPP_VALUE, SGLT_VALUE, ORIGINAL_DPP_VALUE, ORIGINAL_SGLT_VALUE

def get_missing_val_percentage(df):
        """ function to read missing value percentage in the dataframe 

        Args:
            df : dataframe

        Returns:
            percentages: missing value percentages in each dataframe column
        """
        return (df.isnull().sum()* 100 / len(df))

def filter_by(df, condition_str):
    """function to filter by a condition

    Args:
        df: dataframe
        condition_str: condition to filter, send as a string

    Returns:
        filtered_df: filtered dataframe
    """
    if condition_str:
        condition = eval(condition_str)
        filtered_df = df[condition]
        return filtered_df
    
def countUsers(drug_id, df):
    df2 = df.apply(lambda x : True
                if x['drug_class'] == drug_id else False, axis = 1)
    number_of_rows = len(df2[df2 == True].index)
    return number_of_rows

def print_sample_count(df, X_train_, X_test_):

    print('==== sample count in preprocessed data =======')
    print(' number of dpp4 : ', countUsers(ORIGINAL_DPP_VALUE, df))
    print(' number of sglt2 : ', countUsers(ORIGINAL_SGLT_VALUE, df))

    print('==== sample count in training data =======')
    print(' number of dpp4 : ', countUsers(DPP_VALUE, X_train_))
    print(' number of sglt2 : ', countUsers(SGLT_VALUE, X_train_))

    print('==== sample count in testing data =======')
    print(' number of dpp4 : ', countUsers(DPP_VALUE, X_test_))
    print(' number of sglt2 : ', countUsers(SGLT_VALUE, X_test_))


def get_features_kbest(X_train, Y_train,i):
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

def get_features_ref(X_train, Y_train,i): 
    random.seed(SEED)
    model = RandomForestRegressor(random_state = 123)
    rfe = RFE(estimator=model, n_features_to_select=i)  
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


def print_val(name, pred_sglt, pred_dpp):
    print(name)
    print(pred_sglt)
    print(pred_dpp)
    
def find_lowest_respponse_value(pred_sglt, pred_dpp):
    values = [pred_sglt, pred_dpp]
    max_index = values.index(min(values))
    max_difference = [pred_sglt, pred_dpp][max_index]
    drug_class = [SGLT_VALUE, DPP_VALUE][max_index]
    return max_difference, drug_class

def find_highest_respponse_value(pred_sglt, pred_dpp):
    values = [pred_sglt, pred_dpp]
    max_index = values.index(max(values))
    max_difference = [pred_sglt, pred_dpp][max_index]
    drug_class = [SGLT_VALUE, DPP_VALUE][max_index]
    return max_difference, drug_class

def find_closest_to_42(pred_sglt, pred_dpp):
    values = [pred_sglt, pred_dpp]
    drug_classes = [SGLT_VALUE, DPP_VALUE]
    max_index = min(range(len(values)), key=lambda i: abs(values[i] - 42.0))
    closest_value = values[max_index]
    drug_class = drug_classes[max_index]
    return closest_value, drug_class