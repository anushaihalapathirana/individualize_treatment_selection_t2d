import pandas as pd
import numpy as np
import random
import yaml
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from constants import DATA_WITHOUT_DATES, DATA_ONLY_DATES, COMMON_VARIABLE_PATH, X_TEST_PATH, X_TRAIN_PATH, SEED
class ImputationPreprocessing:
    
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the full path to the CSV file
        self.file_path_without_dates = os.path.join(self.script_directory, DATA_WITHOUT_DATES)
        self.file_path_only_dates = os.path.join(self.script_directory, DATA_ONLY_DATES)
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)

        self.file_path_X_train = os.path.join(self.script_directory, X_TRAIN_PATH)
        self.file_path_X_test = os.path.join(self.script_directory, X_TEST_PATH)
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        self.correlated_variables = self.common_data['correlated_variables']


    def read_data(self):
        """Read data files and combine

        Returns:
            df: combined dataframe
        """
        df_new = pd.read_csv(self.file_path_without_dates, sep = ';',decimal = ',', encoding = 'utf-8', engine ='python')
        df_date = pd.read_csv(self.file_path_only_dates, sep = ';',decimal = ',', encoding = 'utf-8', engine ='python')

        # rename postinumero with id
        df = df_new.rename(columns={'potilasnumero': 'id'})
        df_date = df_date.rename(columns={'potilasnumero': 'id'})

        columns_to_add = self.common_data['columns_to_add']

        # Add selected columns from dfdate to df
        for col in columns_to_add:
            df[col] = df_date[col]
        
        return df

    def get_nan_count(self, df):
        """Print NaN count in selected columns

        Args:
            df : dataframe
        """
        selected_columns = df[['hba1c_12m', 'ldl_12m', 'hdl_12m', 'bmi_12m']].columns
        nan_counts = df[selected_columns].isna().sum()
        nan_info = pd.DataFrame({'Feature': selected_columns, 'NaN Count': nan_counts})
        print("\n NaN counts in resonse variables:")
        print(nan_info)
        
    
    def get_missing_val_percentage(self, df):
        """ function to read missing value percentage in the dataframe 

        Args:
            df : dataframe

        Returns:
            percentages: missing value percentages in each dataframe column
        """
        return (df.isnull().sum()* 100 / len(df))


    def preprocess(self, df, test_size):
        """Only focus on drug class SGLT and DPP
            2=GLP-1 analogues (A10BJ)
            3=DPP-4 inhibitors (A10BH)
            4=SGLT2 inhibitors (A10BK)

        Args:
            df : dataframe
            test_size (float): size of the test data. This use to split the data into training and test dataset.
        
        Returns: 
            df : Preprocessed dataframe
            X_train, X_test, Y_train, Y_test : After train and test split
            X, Y : X and Y before train test split

        """
        
        variables = df.columns
        thresh = self.common_data['thresh']
        keep = []
        rem = []
        print("original shape: ", np.shape(df))
        
        # remove all the records with drug class is not 2,3,or 4 

        df = df[(df['drug_class'] == 3) | (df['drug_class'] == 4) ]

        # replace ' ' as NaN
        df = df.replace(' ', np.NaN)
        print('Shape of data after removing other drug types:', np.shape(df))
        
        # Convert selected columns to float. 
        df['bmi'] = df['bmi'].astype(float)
        df['sp'] = df['sp'].astype(int)
        df['ika'] = df['ika'].astype(float)
        df['smoking'] = df['smoking'].astype(float)

        # print the nan counts
        self.get_nan_count(df)
        
        #delete columns with more than threshold NaN. get missing values < threshold feature name list
        missing_per = self.get_missing_val_percentage(df)
        
        for i in range(df.columns.shape[0]):
            if missing_per[i] <= thresh: #setting the threshold as 40%
                keep.append(variables[i])
            else :
                rem.append(variables[i])
        
        # Keep these columns, even if it has more than 40% missing values. 
        columns_to_keep = ['hba1c_prev_1y', 'date_hdl_12m', 'date_bmi_12m','date_ldl_12m','hba1c_12m',
                            'ldl_12m', 'hdl_12m', 'bmi_12m']
        
        for col in columns_to_keep:
            if col in rem:
                rem.remove(col)
        
        df = df.drop([x for x in rem if x in df.columns], axis=1)
        print('Shape of data after removing cols with less than %.2f percent values missing:' % (thresh), np.shape(df))

        #     remove correlated features 
        df = df.drop([x for x in self.correlated_variables if x in df.columns], axis=1)
        print('Shape of data after removing correlated features:', np.shape(df))
        
        # convert categorical to numeric
        cat_cols = []
        for i in cat_cols:
            labelencoder = LabelEncoder()
            df[i] = labelencoder.fit_transform(df[i])
            
        # calculate days from baseline to 12m response date.
        date_cols = ['date_hba_bl_6m','date_ldl_bl','date_bmi_bl','date_hdl_bl','date_12m', 'date_n1',
                    'date_ldl_12m', 'date_bmi_12m', 'date_hdl_12m']
        
        #convert dates into datetime format
        df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce', format='%m/%d/%Y')
        days_to_response_hba1c = df['date_12m'] - df['date_hba_bl_6m']
        days_to_response_bmi = df['date_bmi_12m'] - df['date_bmi_bl']
        days_to_response_hdl = df['date_hdl_12m'] - df['date_hdl_bl']
        days_to_response_ldl = df['date_ldl_12m'] - df['date_ldl_bl']
        
        df.loc[:,'days_hba1c'] = [x.days for x in days_to_response_hba1c]
        df.loc[:,'days_bmi'] = [x.days for x in days_to_response_bmi]
        df.loc[:,'days_hdl'] = [x.days for x in days_to_response_hdl]
        df.loc[:,'days_ldl'] = [x.days for x in days_to_response_ldl]
        
        print('Shape of full data with change + days', np.shape(df))

        #convert other "object" columns to numeric 
        convert = df.select_dtypes('object').columns
        df.loc[:, convert] = df[convert].apply(pd.to_numeric, downcast='float', errors='coerce')
        
        print('Shape of full data after selecting date range dates > 21 days', np.shape(df))
        
        # split data
        random.seed(SEED)
        # Save original data set
        original = df
        Y = df[self.response_variable_list]
        X = df.drop(self.response_variable_list, axis=1)
        random.seed(SEED)
        
        # Split into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=123)
        
        # save preprocessed df to csv
        result_df = pd.concat([X_train, Y_train], axis=1)
        test_df = pd.concat([X_test, Y_test], axis = 1)

        result_df.to_csv(self.file_path_X_train, index=True)
        test_df.to_csv(self.file_path_X_test, index=True)
        
        return df, X_train, X_test, Y_train, Y_test, X, Y

if __name__ == "__main__":
    imp = ImputationPreprocessing()
    df = imp.read_data()
    df, X_train, X_test, Y_train, Y_test, X, Y = imp.preprocess(df, 0.25)
