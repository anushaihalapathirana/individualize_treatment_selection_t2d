import numpy as np
import pandas as pd
import yaml

from tabulate import tabulate
from constants import SGLT_VALUE, DPP_VALUE
from helpers import print_val, find_lowest_respponse_value, find_highest_respponse_value, print_strata_data,\
    check_aggreement

class Treatment:
    def __init__(self, X_test, Y_train, model, scaler, X_test_original):
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        
        self.X = X_test.copy()
        self.X_test_copy = X_test.copy()
        self.X_test_copy['assigned_drug_hba1c'] = np.nan
        self.X_test_copy['predicted_change_hba1c'] = np.nan
        self.X_test_copy['assigned_drug_ldl'] = np.nan
        self.X_test_copy['predicted_change_ldl'] = np.nan
        self.X_test_copy['assigned_drug_hdl'] = np.nan
        self.X_test_copy['predicted_change_hdl'] = np.nan
        self.X_test_copy['assigned_drug_bmi'] = np.nan
        self.X_test_copy['predicted_change_bmi'] = np.nan
        
        self.assigned_drug_class_0 = np.nan
        self.assigned_drug_class_1 = np.nan
        self.assigned_drug_class_2 = np.nan
        self.assigned_drug_class_3 = np.nan
        self.max_change_0 = np.nan
        self.max_change_1 = np.nan
        self.max_change_2 = np.nan
        self.max_change_3 = np.nan
        
        self.model = model
        self.scaler = scaler
        self.Y_train = Y_train
        self.X_test_original = X_test_original
    
    def pred_all(self, model, row, drug_class):
        if drug_class == SGLT_VALUE:
            pred_sglt_ = model.predict(row.values[None])[0]
            row['drug_class'] = DPP_VALUE
            pred_dpp_ = model.predict(row.values[None])[0]
    #         print_val('SGLT', pred_sglt, pred_dpp)
            
        elif drug_class == DPP_VALUE:
            pred_dpp_ = model.predict(row.values[None])[0]
            row['drug_class'] = SGLT_VALUE
            pred_sglt_ = model.predict(row.values[None])[0]
    #         print_val('DPP', pred_sglt, pred_dpp)
            
        else:
            print('Worng drug class')
        return pred_sglt_, pred_dpp_
    
    def assign_drug(self):
        for index, row in self.X.iterrows():
            drug_class = row['drug_class']    
            pred_original = self.model.predict(row.values[None])[0]
            pred_sglt, pred_dpp = self.pred_all(self.model, row, drug_class) 
            
            for j in range(self.Y_train.shape[1]):
                variable_change_name = f"max_change_{j}"
                variable_drug_name = f"assigned_drug_class_{j}"
                
                if (self.Y_train.iloc[:,j].name == 'hdl_12m'):
                    temp_max_change, temp_assigned_drug_class = find_highest_respponse_value(pred_sglt[j], pred_dpp[j])
                else:
                    temp_max_change, temp_assigned_drug_class = find_lowest_respponse_value(pred_sglt[j], pred_dpp[j])
                # Update the original variables
                globals()[variable_change_name] = temp_max_change
                globals()[variable_drug_name] = temp_assigned_drug_class
            
        #     print('actual: ', drug_class, 'assigned: ', assigned_drug_class_0)
            self.X_test_copy.at[index, 'assigned_drug_hba1c'] = self.assigned_drug_class_0
            self.X_test_copy.at[index, 'predicted_change_hba1c'] = self.max_change_0
            
            self.X_test_copy.at[index, 'assigned_drug_ldl'] = self.assigned_drug_class_1
            self.X_test_copy.at[index, 'predicted_change_ldl'] = self.max_change_1
            
            self.X_test_copy.at[index, 'assigned_drug_hdl'] = self.assigned_drug_class_2
            self.X_test_copy.at[index, 'predicted_change_hdl'] = self.max_change_2
            
            self.X_test_copy.at[index, 'assigned_drug_bmi'] = self.assigned_drug_class_3
            self.X_test_copy.at[index, 'predicted_change_bmi'] = self.max_change_3
            
    
    def assign_starta(self):
        denormalized_test_data = self.scaler.inverse_transform(self.X_test_original)
        denormalized_test_df = pd.DataFrame(denormalized_test_data, columns=self.X_test_original.columns)
        denormalized_test_df = denormalized_test_df.drop(['drug_class'], axis = 1)

        data = denormalized_test_df
        X_test_ = self.X_test_copy.copy()
        X_test_= X_test_.reset_index()
        Y_test = pd.DataFrame(self.Y_test)
        Y_test = Y_test.reset_index()

        data[self.response_variable_list] = Y_test[self.response_variable_list]

        data['assigned_drug_hba1c'] = X_test_['assigned_drug_hba1c']
        data['predicted_change_hba1c'] = X_test_['predicted_change_hba1c']
        data['assigned_drug_ldl'] = X_test_['assigned_drug_ldl']
        data['predicted_change_ldl'] = X_test_['predicted_change_ldl']
        data['assigned_drug_hdl'] = X_test_['assigned_drug_hdl']
        data['predicted_change_hdl'] = X_test_['predicted_change_hdl']
        data['assigned_drug_bmi'] = X_test_['assigned_drug_bmi']
        data['predicted_change_bmi'] = X_test_['predicted_change_bmi']
        data['drug_class'] = X_test_['drug_class']

        # drug classes after normalization
        # 0 = GLP-1 
        # 0.5 = DPP-4 
        # 1 = SGLT2

        # 2=GLP-1 analogues (A10BJ)
        # 3=DPP-4 inhibitors (A10BH)
        # 4=SGLT2 inhibitors (A10BK)

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
        
        strata_dictionary = {
            'dpp_strata_hba1c': dpp_strata_hba1c,
            'sglt_strata_hba1c': sglt_strata_hba1c,
            'dpp_strata_ldl': dpp_strata_ldl,
            'sglt_strata_ldl': sglt_strata_ldl,
            'dpp_strata_hdl': dpp_strata_hdl,
            'sglt_strata_hdl': sglt_strata_hdl,
            'dpp_strata_bmi': dpp_strata_bmi,
            'sglt_strata_bmi': sglt_strata_bmi,
            'dpp_strata_actual': dpp_strata_actual,
            'sglt_strata_actual': sglt_strata_actual
        }
        
        print_strata_data(strata_dictionary)
        return strata_dictionary
        
    def get_concordant_discordant(self, dpp_strata, sglt_strata, data, dpp_strata_actual, sglt_strata_actual, variable_name):

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
            ["Concordant", "DPP", "DPP", concordant_dpp_count, f"{concordant_dpp_percentage:.2f}%"],
            ["Discordant", "SGLT", "DPP", discordant_sglt_dpp_count, f"{discordant_sglt_dpp_percentage:.2f}%"],
            
            ["Concordant", "SGLT","SGLT", concordant_sglt_count, f"{concordant_sglt_percentage:.2f}%"],
            ["Discordant", "DPP", "SGLT", discordant_dpp_sglt_count, f"{discordant_dpp_sglt_percentage:.2f}%"],
        ]

        # Print the table
        print(tabulate(data, headers=["Category","Real value", "Predicted value",  "Count", "Percentage of Predicted cases"]))
        print('\n')
        
        return ( concordant_dpp, discordant_dpp_sglt,
                concordant_sglt, discordant_sglt_dpp)
        
    def print_concordant_discordant(self, strata_dictionary, data):
        print('HBA1C')
        (concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
            concordant_sglt_hba1c, discordant_sglt_dpp_hba1c ) = self.get_concordant_discordant(strata_dictionary['dpp_strata_hba1c'], strata_dictionary['sglt_strata_hba1c'],\
                                                                                                data, strata_dictionary['dpp_strata_actual'], strata_dictionary['sglt_strata_actual'],\
                                                                                                variable_name = 'assigned_drug_hba1c')
        print('LDL')
        (concordant_dpp_ldl, discordant_dpp_sglt_ldl,
            concordant_sglt_ldl, discordant_sglt_dpp_ldl ) = self.get_concordant_discordant(strata_dictionary['dpp_strata_ldl'], strata_dictionary['sglt_strata_ldl'],\
                                                            data, strata_dictionary['dpp_strata_actual'], strata_dictionary['sglt_strata_actual'],\
                                                            variable_name = 'assigned_drug_ldl')
        print('HDL')
        (concordant_dpp_hdl, discordant_dpp_sglt_hdl,
            concordant_sglt_hdl, discordant_sglt_dpp_hdl ) = self.get_concordant_discordant(strata_dictionary['dpp_strata_hdl'],strata_dictionary['sglt_strata_hdl'],\
                                                            data, strata_dictionary['dpp_strata_actual'], strata_dictionary['sglt_strata_actual'],\
                                                            variable_name = 'assigned_drug_hdl')
        print('BMI') 
        (concordant_dpp_bmi, discordant_dpp_sglt_bmi,
            concordant_sglt_bmi, discordant_sglt_dpp_bmi ) = self.get_concordant_discordant(strata_dictionary['dpp_strata_bmi'], strata_dictionary['sglt_strata_bmi'],\
                                                            data, strata_dictionary['dpp_strata_actual'], strata_dictionary['sglt_strata_actual'],\
                                                            variable_name = 'assigned_drug_bmi')
        con_dis_groups = {
            'concordant_dpp_hba1c': concordant_dpp_hba1c,
            'discordant_dpp_sglt_hba1c': discordant_dpp_sglt_hba1c,
            'concordant_sglt_hba1c': concordant_sglt_hba1c,
            'discordant_sglt_dpp_hba1c': discordant_sglt_dpp_hba1c,
            'concordant_dpp_ldl': concordant_dpp_ldl,
            'discordant_dpp_sglt_ldl': discordant_dpp_sglt_ldl,
            'concordant_sglt_ldl': concordant_sglt_ldl,
            'discordant_sglt_dpp_ldl': discordant_sglt_dpp_ldl,
            'concordant_dpp_hdl': concordant_dpp_hdl,
            'discordant_dpp_sglt_hdl': discordant_dpp_sglt_hdl,
            'concordant_sglt_hdl': concordant_sglt_hdl,
            'discordant_sglt_dpp_hdl': discordant_sglt_dpp_hdl,
            'concordant_dpp_bmi': concordant_dpp_bmi,
            'discordant_dpp_sglt_bmi': discordant_dpp_sglt_bmi,
            'concordant_sglt_bmi': concordant_sglt_bmi,
            'discordant_sglt_dpp_bmi': discordant_sglt_dpp_bmi,
        }

        return con_dis_groups
            

                