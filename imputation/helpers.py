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
    random.seed(SEED)
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
    random.seed(SEED)  # Fit the model before using RFE
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
    
    for col in Y_train.columns:
        model = sm.OLS(Y_train[col], X_train).fit()
        predictions_train.append(model.predict(X_train))
        models.append(model)

    # Make predictions for each output in Y_test
    predictions = np.column_stack([model.predict(X_test) for model in models])

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

def cross_val(train, X_train, Y_train, model, response_variable_list, n_splits=10):
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


def get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs, name = ''):
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