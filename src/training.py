import pandas as pd
import numpy as np
import random
import yaml
import os 
import sys

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, CompoundKernel
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import lightgbm as ltb
from tabulate import tabulate
from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import RegressorChain, MultiOutputRegressor

from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import COMMON_VARIABLE_PATH, SEED, TEST_PATH_WO_LDL_IMPUTATION,\
    TRAIN_PATH_WO_LDL_IMPUTATION, DPP_VALUE, SGLT_VALUE
from helper import read_data, preprocess, get_test_train_data, get_features_kbest, get_features_ref,\
    get_features_relieff, get_model_name, cross_val, get_scores, outlier_detect, pred_all,\
    find_lowest_respponse_value, find_highest_respponse_value, get_features_ref_multiout,\
    get_concordant_discordant, calculate_percentage_change, check_distribution

class Train:
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the full path to the CSV file
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)
        
        # Define the relative path to the CSV file from the script's directory
        relative_path_to_impute_train_data_wo_ldl = os.path.join("..", TRAIN_PATH_WO_LDL_IMPUTATION)
        relative_path_to_impute_test_data_wo_ldl = os.path.join("..", TEST_PATH_WO_LDL_IMPUTATION)
        
        # Get the absolute path to the CSV file
        self.file_path_train_data = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_train_data_wo_ldl))
        self.file_path_test_data = os.path.abspath(os.path.join(self.script_directory, relative_path_to_impute_test_data_wo_ldl))
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        self.correlated_variables = self.common_data['correlated_variables']
        self.items = ['drug_class']
        
    def check_distribution(self, df, df_act, response_variable, predicted_change):
        # Find outliers using z-score
        z_scores_col1 = (df_act[response_variable] - np.mean(df_act[response_variable])) / np.std(df_act[response_variable])
        outliers_col1 = df_act[abs(z_scores_col1) > 3]
        outliers_act = outliers_col1.index.to_list()

        z_scores_col2 = (df[predicted_change] - np.mean(df[predicted_change])) / np.std(df[predicted_change])
        outliers_col2 = df[abs(z_scores_col2) > 3]
        outliers_pred = outliers_col2.index.to_list()
        return outliers_act, outliers_pred
    
    
    def drug_class_visualization(self, df, df_act, response_variable, predicted_change, assigned_drug, baseline_val):
        df_ = df[[response_variable, predicted_change, 'drug_class', assigned_drug, baseline_val]]
        df_act_ = df_act[[response_variable
                        , predicted_change, 'drug_class', assigned_drug, baseline_val]]

        outliers_act, outliers_pred = check_distribution(df_, df_act_, response_variable, predicted_change)
        
        df_ = df_.drop(outliers_pred)
        df_act_ = df_act_.drop(outliers_act)
        return df_
    
    def predict_drug_classes(self, model, X_test, Y_train):
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
            
        
    def train_models(self, models, X_test, Y_test, X_train, Y_train, train,scaler,X_test_original):
        model_results = {}
        model_results_drugs = {}
        for model in models:
            if str(get_model_name(model)) == 'Sequential':
                model.compile(optimizer='adam', loss='mean_squared_error')
                random.seed(SEED)
                model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
                random.seed(SEED)
                model.fit(X_train, Y_train, epochs=250, batch_size=16, verbose=0)
            else:
                random.seed(SEED)
                model = cross_val(model, train, X_test, Y_test, X_train, Y_train, self.response_variable_list)
                random.seed(SEED)
                model.fit(X_train, Y_train)
            data_pred, model_results, model_results_drugs, score = get_scores(model, X_test, Y_test, X_train, Y_train, model_results, model_results_drugs)

            
        X_test_copy = self.predict_drug_classes(model, X_test, Y_train)
        
        denormalized_test_data = scaler.inverse_transform(X_test_original)
        denormalized_test_df = pd.DataFrame(denormalized_test_data, columns=X_test_original.columns)
        denormalized_test_df = denormalized_test_df.drop(['drug_class'], axis = 1)

        data = denormalized_test_df
        X_test_ = X_test_copy.copy()
        X_test_= X_test_.reset_index()
        Y_test_ = pd.DataFrame(Y_test)
        Y_test_ = Y_test_.reset_index()
        
        data[self.response_variable_list] = Y_test_[self.response_variable_list]
        data['assigned_drug_hba1c'] = X_test_['assigned_drug_hba1c']
        data['predicted_change_hba1c'] = X_test_['predicted_change_hba1c']
        data['assigned_drug_ldl'] = X_test_['assigned_drug_ldl']
        data['predicted_change_ldl'] = X_test_['predicted_change_ldl']
        data['assigned_drug_hdl'] = X_test_['assigned_drug_hdl']
        data['predicted_change_hdl'] = X_test_['predicted_change_hdl']
        data['assigned_drug_bmi'] = X_test_['assigned_drug_bmi']
        data['predicted_change_bmi'] = X_test_['predicted_change_bmi']
        data['drug_class'] = X_test_['drug_class']

        dpp_strata_hba1c = data[(data['assigned_drug_hba1c'] == DPP_VALUE)]
        sglt_strata_hba1c = data[(data['assigned_drug_hba1c'] == SGLT_VALUE)] 

        dpp_strata_ldl = data[(data['assigned_drug_ldl'] == DPP_VALUE)]
        sglt_strata_ldl = data[(data['assigned_drug_ldl'] == SGLT_VALUE)] 

        dpp_strata_hdl = data[(data['assigned_drug_hdl'] == DPP_VALUE)]
        sglt_strata_hdl = data[(data['assigned_drug_hdl'] == SGLT_VALUE)] 

        dpp_strata_bmi = data[(data['assigned_drug_bmi'] == DPP_VALUE)]
        sglt_strata_bmi = data[(data['assigned_drug_bmi'] == SGLT_VALUE)] 

        dpp_strata_actual = data[(data['drug_class'] == DPP_VALUE)]
        sglt_strata_actual = data[(data['drug_class'] == SGLT_VALUE)]  
        
        dpp_df_hba1c = self.drug_class_visualization(dpp_strata_hba1c, dpp_strata_actual, 'hba1c_12m','predicted_change_hba1c', 'assigned_drug_hba1c', 'hba1c_bl_6m')
        sglt_df_hba1c = self.drug_class_visualization(sglt_strata_hba1c, sglt_strata_actual,'hba1c_12m',
                                                'predicted_change_hba1c', 'assigned_drug_hba1c', 'hba1c_bl_6m')
        
        dpp_df_ldl = self.drug_class_visualization(dpp_strata_ldl, dpp_strata_actual, 'ldl_12m',
                                            'predicted_change_ldl', 'assigned_drug_ldl', 'ldl')
        sglt_df_ldl = self.drug_class_visualization(sglt_strata_ldl, sglt_strata_actual, 'ldl_12m',
                                            'predicted_change_ldl', 'assigned_drug_ldl', 'ldl')
        
        dpp_df_hdl = self.drug_class_visualization(dpp_strata_hdl, dpp_strata_actual, 'hdl_12m',
                                            'predicted_change_hdl', 'assigned_drug_hdl', 'hdl')
        sglt_df_hdl = self.drug_class_visualization(sglt_strata_hdl, sglt_strata_actual, 'hdl_12m',
                                            'predicted_change_hdl', 'assigned_drug_hdl', 'hdl')
        
        dpp_df_bmi = self.drug_class_visualization(dpp_strata_bmi, dpp_strata_actual, 'bmi_12m',
                                            'predicted_change_bmi', 'assigned_drug_bmi', 'bmi')
        sglt_df_bmi = self.drug_class_visualization(sglt_strata_bmi, sglt_strata_actual, 'bmi_12m',
                                            'predicted_change_bmi', 'assigned_drug_bmi', 'bmi')

        
        (concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
        concordant_sglt_hba1c, discordant_sglt_dpp_hba1c ) = get_concordant_discordant(dpp_df_hba1c,sglt_df_hba1c, data,
                                                                                    dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_hba1c')
        (concordant_dpp_ldl, discordant_dpp_sglt_ldl,
        concordant_sglt_ldl, discordant_sglt_dpp_ldl ) = get_concordant_discordant(dpp_df_ldl,sglt_df_ldl, data,
                                                                                    dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_ldl')
        (concordant_dpp_hdl, discordant_dpp_sglt_hdl,
        concordant_sglt_hdl, discordant_sglt_dpp_hdl ) = get_concordant_discordant(dpp_df_hdl,sglt_df_hdl, data,
                                                                                    dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_hdl')
        (concordant_dpp_bmi, discordant_dpp_sglt_bmi,
        concordant_sglt_bmi, discordant_sglt_dpp_bmi ) = get_concordant_discordant(dpp_df_bmi,sglt_df_bmi, data,
                                                                                    dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_bmi')


        print('\n -------- Percentage HBA1C  ---------')
        calculate_percentage_change(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, response_variable = 'hba1c_12m', baseline_val='hba1c_bl_6m')
        
        print('\n -------- Percentage LDL  ---------')
        calculate_percentage_change(concordant_dpp_ldl, discordant_dpp_sglt_ldl,
                    concordant_sglt_ldl, discordant_sglt_dpp_ldl, response_variable = 'ldl_12m', baseline_val='ldl')
        
        print('\n -------- Percentage HDL  ---------')
        calculate_percentage_change(concordant_dpp_hdl, discordant_dpp_sglt_hdl,
                    concordant_sglt_hdl, discordant_sglt_dpp_hdl, response_variable = 'hdl_12m', baseline_val='hdl')
        
        print('\n -------- Percentage BMI  ---------')
        calculate_percentage_change(concordant_dpp_bmi, discordant_dpp_sglt_bmi,
                    concordant_sglt_bmi, discordant_sglt_dpp_bmi, response_variable = 'bmi_12m', baseline_val='bmi')
        
        
        return model_results


    def run(self, algo, i, df_X_train, df_X_test, feats_hc = []):
        
        X_train_ = preprocess(df_X_train, self.response_variable_list)
        X_test_ = preprocess(df_X_test, self.response_variable_list)
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, X_test_before_scale = get_test_train_data(X_train_, X_test_, self.response_variable_list)

        X_test_original = X_test.copy()
        
        X_test_ = pd.DataFrame(X_test)
        X_train_ = pd.DataFrame(X_train)

        X_train = X_train.drop(['init_year'], axis = 1)
        X_test = X_test.drop(['init_year'], axis = 1)

        selected_features = []
        
        random.seed(42) 
        if algo == 'kbest':
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                feats = get_features_kbest(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'relieff':
            #selected_features.append(feats_hc)
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                feats = get_features_relieff(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'refMulti':
            #selected_list = feats_hc
            selected_list = get_features_ref_multiout(X_train, Y_train, i)
        elif algo=='ref':
            #selected_features.append(feats_hc)
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                feats = get_features_ref(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)

        if algo != 'refMulti':
            selected_list = sum(selected_features, [])
        
        for item in self.items:
            if item not in selected_list:
                selected_list.extend([item])

        # remove duplicate
        selected_list = np.unique(selected_list)
        number_of_features = len(selected_list)
        print('\n\n')
        print(selected_list.tolist())
        X_train_selected = X_train[selected_list]
        X_test_selected = X_test[selected_list]

        ################# OUTLIER CODE ################
        print('Shape of training data before removing outliers:', np.shape(X_train_selected))
        print('Shape of test data before removing outliers:', np.shape(X_test_selected))
        
        out_train, out_test = outlier_detect(X_train_selected, Y_train, X_test_selected, Y_test)
        
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

        X_test_before_scale = pd.DataFrame(X_test_before_scale.drop(out_test, axis = 0)) 
        X_test_original = pd.DataFrame(X_test_original.drop(out_test, axis = 0)) 

        ################
        
        train = X_train_selected.copy()
        train[self.response_variable_list] = Y_train.values

        xgb = XGBRegressor()

        lr = linear_model.LinearRegression(n_jobs = 10)

        kernel = DotProduct()# + WhiteKernel()

        gpr = GPR(kernel, alpha=1e-10, random_state=123)

        rfr = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=123)

        ridge = Ridge(alpha=0.001)

        mlpr = MLPRegressor(random_state=123, max_iter=2000, hidden_layer_sizes = (128), learning_rate= 'adaptive')

        gbr = GradientBoostingRegressor(random_state=0)

        catboost = CatBoostRegressor(iterations=40, learning_rate=0.1, depth=6, verbose = 0)

        ltbr = ltb.LGBMRegressor(max_depth = 6, learning_rate = 0.1, verbose = -1, verbose_eval = False)


        nn = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(number_of_features,)),  # Adjust input shape
        keras.layers.Dense(32, activation='relu'),  # Additional hidden layer
    #     keras.layers.Dense(64, activation='relu'),  # Another hidden layer
    #     keras.layers.Dense(32, activation='relu'),  # Yet another hidden layer
        keras.layers.Dense(4)  # Output layer with a single neuron for regression
    ])
        random.seed(SEED)
        
        #wrapper = MultiOutputRegressor(catboost)
        vr = VotingRegressor([ ('catboost', catboost), ('ltbr', ltbr), ('rfr', rfr)])
        
        # combinations for vr 
        # rfr + ltbr + cat + xgb
        # rfr + ltbr
        # rfr + cat
        # ltbr + cat
        # rfr + ltbr + cat

        wrapper = RegressorChain(vr, order=[0,1,2,3])

        models = [wrapper]
        random.seed(SEED) 
        model_results = self.train_models(models, X_test_selected, Y_test, X_train_selected, Y_train, train, scaler, X_test_original)
        return model_results

    
        
if __name__ == "__main__":
    print("Initialte model training...")
    train = Train()
    feature_size  = [2,3,4,5,6,7,8,9,10]
    feature_selection_list = ['kbest', 'relieff', 'ref', 'refMulti']
        
    df_X_train = read_data(train.file_path_train_data)
    df_X_test = read_data(train.file_path_test_data)
   
    for feature_method in feature_selection_list:
        print('\n\n\n===============  ', feature_method ,' feature selection method ','   ===================\n',)
        
        for i in feature_size:
            print('\n\n\n===============  ', i ,' features ','   ===================\n',)
            model_results = train.run(feature_method, i, df_X_train, df_X_test)
            table = []
            for model, score in model_results.items():
                table.append([model, score])

            table_str = tabulate(table, headers=['Model', 'Test R2 Score'], tablefmt='grid')
            print(table_str)
                
            print('\n\n\n')