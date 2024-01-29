import numpy as np
import pandas as pd
import os

from constants import X_TRAIN_PATH, X_TEST_PATH, BMI_PATH, HDL_PATH, LDL_PATH, HBA1C_PATH, \
    TESTING_DATA_PATH, IMPUTED_TRAINING_DATA_PATH
from preprocess import ImputationPreprocessing
from hdl import ImputationHDL
from ldl import ImputationLDL
from hba1c import ImputationHbA1c
from bmi import ImputationBMI

class Main():

    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        self.file_path_X_train = os.path.join(self.script_directory, X_TRAIN_PATH)
        self.file_path_X_test = os.path.join(self.script_directory, X_TEST_PATH)
        
        self.file_path_bmi = os.path.join(self.script_directory, BMI_PATH)
        self.file_path_hdl = os.path.join(self.script_directory, HDL_PATH)
        self.file_path_ldl = os.path.join(self.script_directory, LDL_PATH)
        self.file_path_hba1c = os.path.join(self.script_directory, HBA1C_PATH)
        
        self.file_path_imputed_train = os.path.join(self.script_directory, IMPUTED_TRAINING_DATA_PATH)
        self.file_path_imputed_test = os.path.join(self.script_directory, TESTING_DATA_PATH)
        
    def impute_data(self):

        imp = ImputationPreprocessing()
        df = imp.read_data()
        df, X_train, X_test, Y_train, Y_test, X, Y = imp.preprocess(df, 0.25)
        
        imputeBMI = ImputationBMI()
        df = imputeBMI.read_data()
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = imputeBMI.preprocess(df, 0.25)
        print('df_missing_val shape : ', df_missing_val.shape)
        X_train, X_test, selected_features = imputeBMI.feature_selection(df, X_train, Y_train, X_test)
        X_train, X_test, Y_train, Y_test = imputeBMI.remove_outliers(X_train, Y_train, X_test, Y_test)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeBMI.model_training(X_train, Y_train, X_test, Y_test)
        imputeBMI.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
        
        
        imputeHDL = ImputationHDL()
        df = imputeHDL.read_data()
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = imputeHDL.preprocess(df, 0.25)
        print('df_missing_val shape : ', df_missing_val.shape)
        X_train, X_test, selected_features = imputeHDL.feature_selection(df, X_train, Y_train, X_test)
        X_train, X_test, Y_train, Y_test = imputeHDL.remove_outliers(X_train, Y_train, X_test, Y_test)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHDL.model_training(X_train, Y_train, X_test, Y_test)
        imputeHDL.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
        
        imputeHba1c = ImputationHbA1c()
        df = imputeHba1c.read_data()
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = imputeHba1c.preprocess(df, 0.25)
        print('df_missing_val shape : ', df_missing_val.shape)
        X_train, X_test, selected_features = imputeHba1c.feature_selection(df, X_train, Y_train, X_test)
        X_train, X_test, Y_train, Y_test = imputeHba1c.remove_outliers(X_train, Y_train, X_test, Y_test)
        original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeHba1c.model_training(X_train, Y_train, X_test, Y_test)
        imputeHba1c.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
        
        # imputeLDL = ImputationLDL()
        # df = imputeLDL.read_data()
        # df, X_train, X_test, Y_train, Y_test, X, Y, scaler, df_missing_val, df_missing_val_original, df_original = imputeLDL.preprocess(df, 0.25)
        # print('df_missing_val shape : ', df_missing_val.shape)
        # X_train, X_test, selected_features = imputeLDL.feature_selection(df, X_train, Y_train, X_test)
        # X_train, X_test, Y_train, Y_test = imputeLDL.remove_outliers(X_train, Y_train, X_test, Y_test)
        # original_data_pred, model_results, model_results_drugs_ori, score_ori, model = imputeLDL.model_training(X_train, Y_train, X_test, Y_test)
        # imputeLDL.missing_value_prediction(model, df_missing_val, df_original, selected_features, df_missing_val_original)
        
    def delete_files(self):
        # Delete all the csv files
        file_paths_to_delete = [self.file_path_ldl, self.file_path_hba1c, self.file_path_hdl, self.file_path_bmi,
                                self.file_path_X_train, self.file_path_X_test]

        for file_path in file_paths_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully.")
            else:
                print(f"File {file_path} does not exist.")


    def get_dfs(self, df_orginal):
    
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

    def combine_data(self):
        df_X_train = pd.read_csv(self.file_path_X_train, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python',index_col=0)
        df_X_test = pd.read_csv(self.file_path_X_test, sep = ',',decimal = '.', encoding = 'utf-8', engine ='python',index_col=0)

        # List of file paths to read
        file_paths = [self.file_path_ldl, self.file_path_hba1c, self.file_path_hdl, self.file_path_bmi]

        # Initialize empty DataFrames
        df_ldl = pd.DataFrame()
        df_hba1c = pd.DataFrame()
        df_hdl = pd.DataFrame()
        df_bmi = pd.DataFrame()

        # Iterate over the list and read each file if it exists
        for file_name in file_paths:
            if os.path.exists(file_name):
                if 'ldl' in file_name:
                    df_ldl = pd.read_csv(file_name, sep=',', decimal='.', encoding='utf-8', engine='python', index_col=0)
                elif 'hba1c' in file_name:
                    df_hba1c = pd.read_csv(file_name, sep=',', decimal='.', encoding='utf-8', engine='python', index_col=0)
                elif 'hdl' in file_name:
                    df_hdl = pd.read_csv(file_name, sep=',', decimal='.', encoding='utf-8', engine='python', index_col=0)
                elif 'bmi' in file_name:
                    df_bmi = pd.read_csv(file_name, sep=',', decimal='.', encoding='utf-8', engine='python', index_col=0)
            else:
                print(f"File {file_name} does not exist.")
                
        print("original shape df_X_train: ", np.shape(df_X_train))
        print("original shape df_X_test: ", np.shape(df_X_test))

        df_X_train = self.get_dfs(df_X_train)
        df_X_test = self.get_dfs(df_X_test)

        print(df_bmi.shape)
        print(df_hba1c.shape)
        print(df_hdl.shape)
        print(df_ldl.shape)
        print(df_X_train.shape)
        print(df_X_test.shape)

        result_df =  df_X_train.copy()
        updates = {
            'ldl_12m': df_ldl,
            'bmi_12m': df_bmi,
            'hba1c_12m': df_hba1c,
            'hdl_12m': df_hdl
        }

        for col, df_update in updates.items():
            if not df_update.empty:
                result_df = result_df.drop([f'{col}'], axis=1)
                result_df[col] = df_update[col]

        print(result_df.shape)
        
        print(result_df[['id','hba1c_12m', 'ldl_12m','hdl_12m','bmi_12m']].isna().sum())
        print(df_X_test[['id','hba1c_12m', 'ldl_12m','hdl_12m','bmi_12m']].isna().sum())
        # Save combined data
        result_df.to_csv(self.file_path_imputed_train,index=True)
        df_X_test.to_csv(self.file_path_imputed_test,index=True)
    
if __name__ == "__main__":
    print("Initialte imputation...")
    main = Main()
    main.delete_files()
    main.impute_data()
    main.combine_data()