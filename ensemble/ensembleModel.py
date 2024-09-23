"""
This script contains calculations of combine all 4 output predictions based on the majority votes and importance-weight
In majority votes ensemble method, if the votes are tie, assigne that sample based on hba1c prediction value
"""

import sys
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from utils import calculate_accuracy, ensemble_based_on_majority, find_optimal_threshold, get_concordant_discordant, \
    calculate_change_diff
    
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

from constants import FEATURE_IMPORTANCE_DF_LOCATION, PREDICTED_DRUG_CLASS_FILE_LOCATION, DPP_VALUE, SGLT_VALUE
from helper import read_data, calculate_percentage_change

class EnsembleModel:
    
    def __init__(self):
        """
        Initializes an instance of the class by setting up file paths and variables.

        Attributes:
            file_path_feature_importance_df_file (str): Absolute path to the file containing feature importance data.
            
        Raises:
            FileNotFoundError: If the specified file paths cannot be found.
        """
        
        try:
            # Get the current script's directory
            self.script_directory = os.path.dirname(os.path.abspath(__file__))

            # Define the relative paths
            relative_paths = {
                'feature_importance': os.path.join("../", FEATURE_IMPORTANCE_DF_LOCATION),
                'predicted_drug': os.path.join("../", PREDICTED_DRUG_CLASS_FILE_LOCATION),  
            }

            # Get the absolute paths
            self.file_path_feature_importance_df_file = os.path.abspath(os.path.join(self.script_directory, relative_paths['feature_importance']))
            self.file_path_predicted_drug_file = os.path.abspath(os.path.join(self.script_directory, relative_paths['predicted_drug']))

        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise

    def get_relavant_data(self):
        """
        Reads data from specified file paths and returns relevant dataframes for drug predictions and feature importance.

        Returns:
            tuple: A tuple containing two dataframes:
                - df (DataFrame): A dataframe with selected columns related to assigned drugs and their features.
                - df_feature_importance (DataFrame): A dataframe containing the feature importance data.

        """

        df_predicted_drug_file = read_data(self.file_path_predicted_drug_file)
        df_feature_importance = read_data(self.file_path_feature_importance_df_file, index_col = None)
        
        df = df_predicted_drug_file[['assigned_drug_hba1c', 'assigned_drug_ldl', 'assigned_drug_hdl', 'assigned_drug_bmi', 'drug_class', 'hba1c_12m',
              'ldl_12m', 'hdl_12m', 'bmi_12m', 'hba1c_bl_6m', 'ldl', 'hdl', 'bmi', 'predicted_change_hba1c',
              'predicted_change_ldl', 'predicted_change_hdl', 'predicted_change_bmi']]
        return df, df_feature_importance

    def importance_weight_ensemple(self, df, feature_importance_df):
        
        """
        Computes the weighted sum of drug assignments based on feature importance values 
        and generates an ensemble decision on drug class assignment.

        Args:
            df (DataFrame): A DataFrame containing features related to drug assignment and the actual drug class.
            feature_importance_df (DataFrame): A DataFrame containing feature importance values with 'Feature' and 'Importance' columns.
            
        Returns:
            DataFrame: A copy of the input DataFrame with an additional column 'ensemble_drug', 
            which contains the binary classification (1 or 0) based on the weighted sum and the optimal threshold.
        """
    
        df_importance_weights = df.copy()

        # Create a dictionary to store the variables
        variables = {}
        # Iterate over the DataFrame and assign values to the dictionary
        for index, row in feature_importance_df.iterrows():
            variables[row['Feature']] = row['Importance']

        hba1c_cost = variables['hba1c_bl_6m']
        ldl_cost = variables['ldl']
        hdl_cost = variables['hdl']
        bmi_cost = variables['bmi']

        weighted_sum = (
            df_importance_weights['assigned_drug_hba1c'] * hba1c_cost +
            df_importance_weights['assigned_drug_ldl'] * ldl_cost +
            df_importance_weights['assigned_drug_hdl'] * hdl_cost +
            df_importance_weights['assigned_drug_bmi'] * bmi_cost
        )

        actual_values = df_importance_weights['drug_class']

        optimal_threshold = weighted_sum.mean()
        #optimal_threshold = weighted_sum.median()
        #optimal_threshold = find_optimal_threshold(actual_values, weighted_sum)

        # Create the new binary column based on the optimal threshold
        df_importance_weights['ensemble_drug'] = (weighted_sum >= optimal_threshold).astype(int)
        return df_importance_weights
        
    def ensemble_model_evaluation(self, df_):
        
        """
        Evaluates the performance of the ensemble model and assesses treatment selection outcomes.

        Args:
            df_ (DataFrame): A DataFrame containing the true drug class labels ('drug_class') 
                                and the ensemble model's predictions ('ensemble_drug').

        Returns:
            None: This method prints the evaluation metrics and treatment selection evaluation results.
        """
        # Ensemble model perfoemance evaluation
        precision = precision_score(df_['drug_class'], df_['ensemble_drug'])
        recall = recall_score(df_['drug_class'], df_['ensemble_drug'])

        print(f"Accuracy: {calculate_accuracy(df_, 'drug_class', 'ensemble_drug'):.2f}")
        print(f"F1 score: {f1_score(df_['drug_class'], df_['ensemble_drug'], average='weighted')}")
        print("Precision:", precision)
        print("Recall:", recall)

        cm = confusion_matrix(df_['drug_class'], df_['ensemble_drug'])
        print("Confusion Matrix:")
        print(cm)
        
        # Treatment selection evaluation 
        dpp_strata = df_[(df_['ensemble_drug'] == DPP_VALUE)]
        sglt_strata = df_[(df_['ensemble_drug'] == SGLT_VALUE)] 

        dpp_strata_actual = df_[(df_['drug_class'] == DPP_VALUE)]
        sglt_strata_actual = df_[(df_['drug_class'] == SGLT_VALUE)] 

        (concordant_dpp, discordant_dpp_sglt,
            concordant_sglt, discordant_sglt_dpp ) = get_concordant_discordant(dpp_strata,sglt_strata, df_,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'ensemble_drug')

        print('\n============= HBA1C ===================')    
        calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
                    concordant_sglt, discordant_sglt_dpp, response_variable = 'hba1c_12m', baseline_val='hba1c_bl_6m')
        
        print('\n============= LDL ===================')    

        calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
                    concordant_sglt, discordant_sglt_dpp, response_variable = 'ldl_12m', baseline_val='ldl')

        print('\n============= HDL ===================')    

        calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
                    concordant_sglt, discordant_sglt_dpp, response_variable = 'hdl_12m', baseline_val='hdl')

        print('\n============= BMI ===================')    

        calculate_percentage_change(concordant_dpp, discordant_dpp_sglt,
                    concordant_sglt, discordant_sglt_dpp, response_variable = 'bmi_12m', baseline_val='bmi')
        
        # Change calculated with respect to baseline - calculated for concordant only 
        calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                            'hba1c_12m', 'hba1c_bl_6m', 'predicted_change_hba1c', 'hba1c')

        calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                            'ldl_12m', 'ldl', 'predicted_change_ldl', 'ldl')

        calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                            'hdl_12m', 'hdl', 'predicted_change_hdl', 'hdl')

        calculate_change_diff(concordant_dpp, discordant_dpp_sglt, concordant_sglt, discordant_sglt_dpp,
                            'bmi_12m', 'bmi', 'predicted_change_bmi', 'bmi')


    def initialize(self):
        """
        Initializes and applying two different ensemble methods: 
        majority vote and importance-weighted regression, and evaluates their performance.

        Returns:
            None: The method prints the results of the ensemble model evaluations for both the majority vote 
                and importance-weight approaches.
        """
    
        df, df_feature_importance = self.get_relavant_data()
        
        print('\n ======= Majority vote model results =======')
        df_ = df.copy()
        df_.loc[:, 'ensemble_drug'] = ensemble_based_on_majority(df[['assigned_drug_hba1c', 'assigned_drug_ldl', 
                                                                     'assigned_drug_hdl', 'assigned_drug_bmi']], 
                                                                    'assigned_drug_hba1c', 'ensemble_drug')
        self.ensemble_model_evaluation(df_)
        
        print('\n ======= Model based on regression model weights =======')
        df_importance_weights = self.importance_weight_ensemple(df, df_feature_importance)
        self.ensemble_model_evaluation(df_importance_weights)
        
        
if __name__ == "__main__":
    print("Initialte optimal model training...")
    ensembleModel = EnsembleModel()
    # Call the initialize method to execute the workflow
    ensembleModel.initialize()