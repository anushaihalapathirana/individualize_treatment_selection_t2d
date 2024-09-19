import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import shap
import ipywidgets as widgets
import warnings
import os 
import sys

from tabulate import tabulate
from sklearn.multioutput import RegressorChain

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from helper import read_data, preprocess, get_test_train_data, get_features_kbest, get_features_ref, get_features_ref_multiout,\
    get_features_relieff, outlier_detect, train_models, get_concordant_discordant, print_change_mean, percentage_change_original_data,\
    calculate_percentage_change, calculate_percentage_change_othre_responses, calculate_change_diff, drug_class_visualization,\
    drug_class_outlier_remove, save_data_for_ensemble, get_feature_importance, predict_drug_classes, print_strata_stats,\
    get_feature_importance_for_voting_regressor, get_strata

from constants import COMMON_VARIABLE_PATH, SEED, TEST_PATH_WO_LDL_IMPUTATION, TRAIN_PATH_WO_LDL_IMPUTATION, DPP_VALUE,\
    SGLT_VALUE, SCATTER_PLOT_ACTUAL_VS_PRED, SHAP_SUMMARY_PLOT_HBA1C, SHAP_SUMMARY_PLOT_LDL, SHAP_SUMMARY_PLOT_HDL,\
    SHAP_SUMMARY_PLOT_BMI, IMAGE_FONT_SIZE_24, IMAGE_LABEL_SIZE_24, IMAGE_DPI_300, IMAGE_FONT_SIZE_18, IMAGE_FONT_SIZE_14,\
    PREDICTED_DRUG_CLASS_FILE_LOCATION, PREPROCESSED_DATA_FILE_LOCATION, FEATURE_IMPORTANCE_DF_LOCATION

warnings.filterwarnings('ignore')

class BaseModel:
    """
    This class represents the base model
    """

    def __init__(self, feature_list, model, isVRModel = False):
        """
        Initializes an instance of the class by setting up file paths and loading common variables.

        Parameters:
            feature_list (list): A list of features to be used by the model.
            model (object): The model instance that will be used for training and prediction.
            isVRModel (bool, optional): A flag indicating whether the model is a VotingRegresson model. Defaults to False.

        Attributes:
            script_directory (str): The directory of the current script.
            file_path_train_data (str): Absolute path to the training data CSV file.
            file_path_test_data (str): Absolute path to the test data CSV file.
            file_path_common_variables (str): Absolute path to the YAML file containing common variables.
            file_path_predicted_drug_file (str): Absolute path to the CSV file for predicted drug classes.
            file_path_preprocessed_data_file (str): Absolute path to the preprocessed data file.
            file_path_feature_importance_df_file (str): Absolute path to the file containing feature importance data.
            path_to_scatter_plot_actual_vs_pred (str): Absolute path to the scatter plot image of actual vs predicted values.
            path_to_shap_sp_hba1c (str): Absolute path to the SHAP summary plot image for HbA1c.
            path_to_shap_sp_ldl (str): Absolute path to the SHAP summary plot image for LDL.
            path_to_shap_sp_hdl (str): Absolute path to the SHAP summary plot image for HDL.
            path_to_shap_sp_bmi (str): Absolute path to the SHAP summary plot image for BMI.
            common_data (dict): Data loaded from the YAML file, including response variables and correlated variables.
            response_variable_list (list): List of response variables extracted from common_data.
            correlated_variables (list): List of correlated variables extracted from common_data.
            items (list): A list containing the items to include in selected features.
            isshap (bool): A flag indicating whether SHAP (SHapley Additive exPlanations) values are to be calculated. 
                          Defaults to False, since it takes time to run.

        Raises:
            FileNotFoundError: If the specified file paths cannot be found.
            yaml.YAMLError: If there is an error reading the YAML file.
        """
        
        try:
            # Get the current script's directory
            self.script_directory = os.path.dirname(os.path.abspath(__file__))

            # Define the relative paths
            relative_paths = {
                'train_data': os.path.join("../../", TRAIN_PATH_WO_LDL_IMPUTATION),
                'test_data': os.path.join("../../", TEST_PATH_WO_LDL_IMPUTATION),
                'common_variables': os.path.join("../", COMMON_VARIABLE_PATH),
                'predicted_drug': os.path.join("../../", PREDICTED_DRUG_CLASS_FILE_LOCATION),
                'preprocessed_data': os.path.join("../../", PREPROCESSED_DATA_FILE_LOCATION),
                'feature_importance': os.path.join("../../", FEATURE_IMPORTANCE_DF_LOCATION),
                'scatter_plot': os.path.join("../../", SCATTER_PLOT_ACTUAL_VS_PRED),
                'shap_hba1c': os.path.join("../../", SHAP_SUMMARY_PLOT_HBA1C),
                'shap_ldl': os.path.join("../../", SHAP_SUMMARY_PLOT_LDL),
                'shap_hdl': os.path.join("../../", SHAP_SUMMARY_PLOT_HDL),
                'shap_bmi': os.path.join("../../", SHAP_SUMMARY_PLOT_BMI),
            }

            # Get the absolute paths
            self.file_path_train_data = os.path.abspath(os.path.join(self.script_directory, relative_paths['train_data']))
            self.file_path_test_data = os.path.abspath(os.path.join(self.script_directory, relative_paths['test_data']))
            self.file_path_common_variables = os.path.abspath(os.path.join(self.script_directory, relative_paths['common_variables']))
            self.file_path_predicted_drug_file = os.path.abspath(os.path.join(self.script_directory, relative_paths['predicted_drug']))
            self.file_path_preprocessed_data_file = os.path.abspath(os.path.join(self.script_directory, relative_paths['preprocessed_data']))
            self.file_path_feature_importance_df_file = os.path.abspath(os.path.join(self.script_directory, relative_paths['feature_importance']))
            self.path_to_scatter_plot_actual_vs_pred = os.path.abspath(os.path.join(self.script_directory, relative_paths['scatter_plot']))
            self.path_to_shap_sp_hba1c = os.path.abspath(os.path.join(self.script_directory, relative_paths['shap_hba1c']))
            self.path_to_shap_sp_ldl = os.path.abspath(os.path.join(self.script_directory, relative_paths['shap_ldl']))
            self.path_to_shap_sp_hdl = os.path.abspath(os.path.join(self.script_directory, relative_paths['shap_hdl']))
            self.path_to_shap_sp_bmi = os.path.abspath(os.path.join(self.script_directory, relative_paths['shap_bmi']))

            # Read common variables from a YAML file
            try:
                with open(self.file_path_common_variables, 'r') as file:
                    self.common_data = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Common variables file not found: {self.file_path_common_variables}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error reading YAML file: {e}")

            # Initialize other attributes
            self.response_variable_list = self.common_data['response_variable_list']
            self.correlated_variables = self.common_data['correlated_variables']
            self.items = ['drug_class']
            self.isshap = False
            self.feature_list = feature_list
            self.model = model
            self.isVRModel = isVRModel

        except Exception as e:
            print(f"An error occurred during initialization: {e}")
            raise
        
    def run(self, algo, i, df_X_train, df_X_test):
        
        """
        Runs the model training and evaluation process with feature selection (not in this file), outlier removal, and model training.

        This method preprocesses the training and testing data, performs feature selection based on the specified algorithm,
        removes outliers, and trains the model using the selected features. It then returns the results of the model training.

        Parameters:
            algo (str): The feature selection algorithm to use. Options are:
                - 'kbest': Select features using the k-best method.
                - 'relieff': Select features using the ReliefF method.
                - 'refMulti': Select features using the REF Multi-output method.
                - 'ref': Select features using the REF method.
                - Any other value: Uses the predefined feature list. - Default option in this file
            i (int): The number of features to select or the threshold for feature selection algorithms - Not use in this file
            df_X_train (pd.DataFrame): The training data.
            df_X_test (pd.DataFrame): The testing data.

        Returns:
            tuple: A tuple containing:
                - model_results (dict): A dictionary of model results including performance metrics.
                - model (object): The trained model.
                - X_test_selected (DataFrame): The testing data features after feature selection and outlier removal.
                - Y_test (DataFrame): The testing data target variables after outlier removal.
                - X_train_selected (DataFrame): The training data features after feature selection and outlier removal.
                - Y_train (DataFrame): The training data target variables after outlier removal.
                - train (DataFrame): The training data with selected features and target variables after outlier removal.
                - scaler (object): The scaler used for data normalization.
                - X_test_original (DataFrame): The original testing data features after outlier removal.
                - selected_list (list): The list of selected features.
                - X_test_before_scale (DataFrame): The testing data features before scaling and after outlier removal.
                - X_train_original (DataFrame): The original training data features after outlier removal.
        """
    
        X_train_ = preprocess(df_X_train, self.response_variable_list)
        X_test_ = preprocess(df_X_test, self.response_variable_list)
        df, X_train, X_test, Y_train, Y_test, X, Y, scaler, X_test_before_scale = get_test_train_data(X_train_, X_test_, self.response_variable_list)

        X_test_original = X_test.copy()
        X_train_original = X_train.copy()

        X_test_ = pd.DataFrame(X_test)
        X_train_ = pd.DataFrame(X_train)

        X_train = X_train.drop(['init_year'], axis = 1)
        X_test = X_test.drop(['init_year'], axis = 1)

        selected_features = []
        
        random.seed(SEED) 
        if algo == 'kbest':
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                random.seed(SEED)
                feats = get_features_kbest(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'relieff':
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                random.seed(SEED)
                feats = get_features_relieff(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        elif algo == 'refMulti':
            random.seed(SEED)
            selected_list = get_features_ref_multiout(X_train, Y_train, i)
        elif algo=='ref':
            random.seed(SEED)
            for j in range(Y_train.shape[1]):  # Assuming Y.shape[1] is the number of target features
                feats = get_features_ref(X_train, Y_train.iloc[:, j],i)
                selected_features.append(feats)
        else:
            # During training, this model already selected relevant features.
            selected_list = self.feature_list
            
        if algo == 'kbest' or algo == 'relieff' or algo == 'ref' :
            selected_list = sum(selected_features, [])
        
        for item in self.items:
            if item not in selected_list:
                selected_list.extend([item])

        # remove duplicate
        selected_list = np.unique(selected_list)
        number_of_features = len(selected_list)
        print('\n\n')
        print(selected_list)
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

        # remove test outliers from X_test_before_scale and X_test_original
        X_test_before_scale = pd.DataFrame(X_test_before_scale.drop(out_test, axis = 0)) 
        X_test_original = pd.DataFrame(X_test_original.drop(out_test, axis = 0)) 
        X_train_original = pd.DataFrame(X_train_original.drop(out_train, axis = 0)) 
        
        ################
        
        train = X_train_selected.copy()
        train[self.response_variable_list] = Y_train.values
        
        random.seed(SEED)
        wrapper = RegressorChain(self.model, order=[0,1,2,3])
        
        models = wrapper 
        random.seed(SEED) 
        model_results, model = train_models(models, X_test_selected, Y_test, X_train_selected, Y_train, train, scaler, X_test_original, self.response_variable_list)
        return model_results, model, X_test_selected, Y_test, X_train_selected, Y_train, train, scaler, X_test_original, selected_list, X_test_before_scale, X_train_original

    def execute_model(self, df_X_train, df_X_test):
        
        """
        Executes the model training and evaluation process, prints the model performance results, and returns various data components.

        This method calls the `run` method to perform model training and evaluation with a default ('defaukt') and number of features.
        It then formats and prints the results in a tabular format and returns the results and relevant data components for further analysis.

        Parameters:
            df_X_train (DataFrame): The training data.
            df_X_test (DataFrame): The testing data.

        Returns:
            tuple: A tuple containing:
                - model_results (dict): A dictionary of model results including performance metrics for each model.
                - model (object): The trained model.
                - X_test (DataFrame): The testing data features after feature selection and outlier removal.
                - Y_test (DataFrame): The testing data target variables after outlier removal.
                - X_train (DataFrame): The training data features after feature selection and outlier removal.
                - Y_train (DataFrame): The training data target variables after outlier removal.
                - train (DataFrame): The training data with selected features and target variables after outlier removal.
                - scaler (object): The scaler used for data normalization.
                - X_test_original (DataFrame): The original testing data features after outlier removal.
                - selected_list (list): The list of selected features.
                - X_test_before_scale (DataFrame): The testing data features before scaling and after outlier removal.
                - X_train_original (DataFrame): The original training data features after outlier removal.

        Prints:
            A formatted table displaying the model performance results, including the model names and their corresponding test R2 scores.
        """
    
        model_results, model, X_test, Y_test, X_train, Y_train, train, scaler, X_test_original, \
        selected_list, X_test_before_scale, X_train_original = self.run('default',8, df_X_train, df_X_test)
        table = []
        for model_, score in model_results.items():
            table.append([model_, score])

        table_str = tabulate(table, headers=['Model', 'Test R2 Score'], tablefmt='grid')
        print(table_str)   
        print('\n\n\n')
        return model_results, model, X_test, Y_test, X_train, Y_train, train, scaler, X_test_original, \
        selected_list, X_test_before_scale, X_train_original


    def plot_actual_pred(self, df, pred_, drug_class):

        """
        Plots observed vs. predicted values for multiple target features, with separate scatter plots for each target feature.
        
        This method creates subplots for each target feature and plots the observed values against the predicted values.
        It also includes a regression line for each subplot to visualize the relationship between observed and predicted values.
        Different colors are used to distinguish between different drug classes.

        Parameters:
            df (DataFrame): A DataFrame containing the observed values for each target feature. 
                            Each column represents a different target feature.
            pred_ (ndarray): A NumPy array containing the predicted values for each target feature.
            drug_class (ndarray): A NumPy array indicating the drug class for each observation.
                                    It is used to color-code the scatter plots.

        Returns:
            None

        Notes:
            - It uses a color scheme with 'blue' and 'maroon' for different drug classes and plots a regression line in black.
            - The resulting plot is saved to the file path specified by `self.path_to_scatter_plot_actual_vs_pred` with high resolution.
        """
    
        # Number of target features
        num_targets = df.shape[1]
        # rearrange this names list if any changes to df response column order
        names = ['HbA1c Outcome', 'LDL Outcome', 'HDL Outcome', 'BMI Outcome']
        # Create subplots for each target feature
        fig, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(15, 5))

        # Set a common title
        #fig.suptitle('Observed vs. Predicted Values', fontsize= 14)
        colors = ['blue', 'maroon'] 
        scatter_plots = []
        # Plot each target feature separately
        for i, val in enumerate(names):
            ax = axes[i] if num_targets > 1 else axes  # Handle the case when there's only one target

            for idx, drug in enumerate(np.unique(drug_class)):
                if idx==0:
                    drug_name = 'DPP-4'
                else: drug_name = 'SGLT2'
                indices = np.where(drug_class == drug)[0]
                scatter = ax.scatter(df.iloc[indices, i], pred_[indices, i], marker='o', c=colors[idx], s=10, cmap='coolwarm', label=f'{drug_name}')
                scatter_plots.append(scatter)  # Store the scatter plot object
            
            #scatter = ax.scatter(df.iloc[:, i], pred_[:, i], marker='o', s=10, c=drug_class ,cmap='coolwarm', label=f'Target {i + 1} Data Points')
            regression_line = ax.plot(df.iloc[:, i], np.poly1d(np.polyfit(df.iloc[:, i], pred_[:, i], 1))(df.iloc[:, i]), color='black', linestyle='--', label=f'Regression Line')
            #sns.regplot(x=df.iloc[:, i], y=pred_[:, i], ax=ax, scatter=False, color='black', label='Regression Line', ci=95)

            #ax.set_xlabel(f'Observed Values', fontsize=IMAGE_FONT_SIZE_18)
            #ax.set_ylabel(f'Predicted Values',fontsize=IMAGE_FONT_SIZE_18)
            
            # Set the subplot title
            ax.set_title(f'{val}', fontsize=IMAGE_FONT_SIZE_18)
            
            ax.grid(True)
            
        # Set common X and Y axis labels
        fig.supxlabel('Observed Values', fontsize=IMAGE_FONT_SIZE_18, y=0.08)
        fig.supylabel('Predicted Values', fontsize=IMAGE_FONT_SIZE_18, x=0.01)

        handles, labels = ax.get_legend_handles_labels()

        # Create legend outside the loop
        fig.legend(handles, labels, loc=(0.85, 0.14), fontsize=IMAGE_FONT_SIZE_14)
        # Adjust layout for better appearance
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure with higher resolution
        plt.savefig(self.path_to_scatter_plot_actual_vs_pred, dpi=IMAGE_DPI_300, bbox_inches='tight')
        plt.show()
        
    def get_shap_values(self, model, X_train, X_test, selected_list):
        """
        Computes SHAP values for the given model to interpret its predictions.

        This method uses the SHAP KernelExplainer to compute SHAP values for the specified model, 
        providing insights into feature importance and how each feature contributes to the model's predictions.

        Parameters:
            model (object): The trained model for which SHAP values are to be computed. It should have a `predict` method that can be used by the SHAP explainer.
            X_train (DataFrame): The training data used to initialize the SHAP explainer. This data is used to compute the SHAP values for the test data.
            X_test (DataFrame): The testing data for which SHAP values will be computed. The `selected_list` specifies the subset of features to include in the SHAP calculation.
            selected_list (list): A list of feature names that specifies which features to include when computing the SHAP values for `X_test`.

        Returns:
            tuple: A tuple containing:
                - explainer (shap.KernelExplainer): The SHAP explainer instance used to compute the SHAP values.
                - shap_values (list of np.ndarray): The computed SHAP values for the test data. 
                            The list contains one array for each output of the model, representing the contribution of each feature to the model's prediction.
        """
        explainer = shap.KernelExplainer(model=model.predict, data=X_train, link="identity")
        shap_values = explainer.shap_values(X=X_test[selected_list])
        return explainer, shap_values
        
    def get_shap_label(self, Y_test, val):
        """
        Generates a dropdown widget to select a label from the columns of Y_test.

        This method creates a list of column labels from `Y_test` DataFrame and pairs 
        each label with its index in a tuple. A dropdown widget is then created with these tuples 
        to allow for selecting a specific label by index. The selected label index can be accessed 
        using `current_label.value`.

        Args:
            Y_test (DataFrame): The DataFrame containing the target labels (columns) from which to select.
            val (int): The default index value for the dropdown selection.

        Returns:
            tuple: A tuple containing:
                - list_of_labels (list): List of all label names (columns) from Y_test.
                - current_label (widgets.Dropdown): A dropdown widget to select a label.
        """
        # Create the list of all labels for the drop down list
        list_of_labels = Y_test.columns.to_list()

        # Create a list of tuples so that the index of the label is what is returned
        tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

        # Create a widget for the labels and then display the widget
        current_label = widgets.Dropdown(
            options=tuple_of_labels, value=val, description="Select Label:"
        )

        # Display the dropdown list (Note: access index value with 'current_label.value')
        return list_of_labels, current_label

    def rename_features(self, df):
        """
        Renames the columns of the provided DataFrame (X_test) to more descriptive feature names.

        Args:
            df (DataFrame): The input DataFrame containing the original feature names.

        Returns:
            DataFrame: A new DataFrame with the columns renamed to more descriptive feature names.
        """

        X_test_features = df.rename(columns={
            'P_Krea': 'Creatinine',
            'bmi': 'Baseline BMI',
            'drug_class':'Drug class',
            'eGFR':'Glomerular filtration rate',
            'gluk':'Glucose',
            'hba1c_bl_18m':'HbA1c 6-18 months before',
            'hba1c_bl_6m':'Baseline HbA1c',
            'hdl':'Baseline HDL',
            'ika':'Age',
            'ldl':'Baseline LDL',
            'obese':'Obese',
            't2d_dur_y':'T2D Duration (years)',
            'trigly':'Triglycerides'
            }, inplace=False)
        return X_test_features
    
    def save_shap_summary_plot(self, shap_values, list_of_labels, current_label, X_test_features):
        """
        Generates and saves a SHAP summary plot for the current model feature importance analysis.

        The SHAP summary plot visualizes the impact of individual features on the model's output. 
        High positive SHAP values indicate that increasing the feature's value will increase the predicted output, 
        while low or negative SHAP values have the opposite effect.

        Args:
            shap_values (list): A list of SHAP values for the different output labels/features.
            list_of_labels (list): A list of labels corresponding to the output features of the model.
            current_label (int): The index of the current label to be displayed in the SHAP summary plot.
            X_test_features (DataFrame): The input test data with feature names to be used in the SHAP summary plot.

        Returns:
            None: The function saves the SHAP summary plot as an image file and displays it.
        """
        
        print(f"Current Label Shown: {list_of_labels[current_label.value]}\n")

        shap.summary_plot(
            shap_values=shap_values[current_label.value], features=X_test_features, show=False, plot_size=[15,15]
        )

        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(labelsize=IMAGE_LABEL_SIZE_24)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=IMAGE_FONT_SIZE_24)

        cb_ax = fig.axes[1]
        cb_ax.tick_params(labelsize=IMAGE_LABEL_SIZE_24)
        cb_ax.set_ylabel("Feature value", fontsize=IMAGE_FONT_SIZE_24)

        plt.savefig(self.path_to_shap_sp_bmi ,bbox_inches='tight', dpi=IMAGE_DPI_300)
        plt.show()

    def assign_sample_to_stratas(self, scaler, X_test_original, X_test_copy, Y_test):
        """
        Assigns samples to different strata (groups) based on drug assignments and predicted outcomes,
        denormalizes the test data, and saves the results to a CSV file.

        This method:
        - Denormalizes the test data using the provided scaler.
        - Segments the data into various strata (DPP and SGLT groups) for different biomarkers and drug classes.
        - Saves the final dataframe with assigned drugs and predicted changes to a CSV file.

        Args:
            scaler (object): The scaler used to normalize the original test data. It is used to revert the normalization.
            X_test_original (DataFrame): The original test dataset, which will be denormalized.
            X_test_copy (DataFrame): A copy of the test dataset that contains assigned drug and predicted changes.
            Y_test (DataFrame): The actual test target values for the response variables.

        Returns:
            tuple: 
            - data (DataFrame): The final dataframe containing denormalized test data and assigned drug strata.
            - dpp_strata_hba1c (DataFrame): Subset of data where the assigned drug for HbA1c is DPP.
            - sglt_strata_hba1c (DataFrame): Subset of data where the assigned drug for HbA1c is SGLT.
            - dpp_strata_ldl (DataFrame): Subset of data where the assigned drug for LDL is DPP.
            - sglt_strata_ldl (DataFrame): Subset of data where the assigned drug for LDL is SGLT.
            - dpp_strata_hdl (DataFrame): Subset of data where the assigned drug for HDL is DPP.
            - sglt_strata_hdl (DataFrame): Subset of data where the assigned drug for HDL is SGLT.
            - dpp_strata_bmi (DataFrame): Subset of data where the assigned drug for BMI is DPP.
            - sglt_strata_bmi (DataFrame): Subset of data where the assigned drug for BMI is SGLT.
            - dpp_strata_actual (DataFrame): Subset of data where the actual drug class is DPP.
            - sglt_strata_actual (DataFrame): Subset of data where the actual drug class is SGLT.
        """
    
        denormalized_test_data = scaler.inverse_transform(X_test_original)
        denormalized_test_df = pd.DataFrame(denormalized_test_data, columns=X_test_original.columns)
        denormalized_test_df = denormalized_test_df.drop(['drug_class'], axis = 1)

        data = denormalized_test_df
        X_test_ = X_test_copy.copy()
        X_test_= X_test_.reset_index()
        Y_test = pd.DataFrame(Y_test)
        Y_test = Y_test.reset_index()
        
        data[self.response_variable_list] = Y_test[self.response_variable_list]

        # Add drug assignments and predictions
        columns_to_add = ['assigned_drug_hba1c', 'predicted_change_hba1c', 
                        'assigned_drug_ldl', 'predicted_change_ldl',
                        'assigned_drug_hdl', 'predicted_change_hdl', 
                        'assigned_drug_bmi', 'predicted_change_bmi', 'drug_class']
        data[columns_to_add] = X_test_[columns_to_add]

        # save data to csv
        data.to_csv(self.file_path_predicted_drug_file)
        
        # Define the list of biomarkers and drug columns
        biomarkers = ['hba1c', 'ldl', 'hdl', 'bmi']
        strata = {}
        
        # Iterate through each biomarker to extract DPP and SGLT groups
        for biomarker in biomarkers:
            strata[f'dpp_strata_{biomarker}'] = get_strata(data, f'assigned_drug_{biomarker}', DPP_VALUE)
            strata[f'sglt_strata_{biomarker}'] = get_strata(data, f'assigned_drug_{biomarker}', SGLT_VALUE)

        # Extract actual drug class strata
        strata['dpp_strata_actual'] = get_strata(data, 'drug_class', DPP_VALUE)
        strata['sglt_strata_actual'] = get_strata(data, 'drug_class', SGLT_VALUE)

        return (data, 
                strata['dpp_strata_hba1c'], strata['sglt_strata_hba1c'], 
                strata['dpp_strata_ldl'], strata['sglt_strata_ldl'], 
                strata['dpp_strata_hdl'], strata['sglt_strata_hdl'], 
                strata['dpp_strata_bmi'], strata['sglt_strata_bmi'], 
                strata['dpp_strata_actual'], strata['sglt_strata_actual'])

    def initialize(self):
        
        """
        Initializes the workflow for model execution, data analysis, and visualization of results.

        This method performs the following tasks:
        1. Reads training and test datasets from the specified file paths.
        2. Executes the machine learning model, printing model performance and results.
        3. Prepares data for ensemble models and calculates feature importances.
        4. Generates actual vs. predicted plots for the model's predictions.
        5. Computes SHAP values for model interpretation if SHAP analysis is enabled.
        6. Predicts drug classes based on the trained model and assigns samples to strata for further analysis.
        7. Computes concordant and discordant treatment effects for various clinical responses before and after outlier removal.
        8. Visualizes observed vs. predicted drug classes for various clinical responses.
        9. Removes outliers from predicted values for more accurate analysis.
        10. Calculates and prints the percentage change in clinical responses and evaluates differences between observed and predicted values.

        Args:
            None

        Returns:
            None
        """
        df_X_train = read_data(self.file_path_train_data)
        df_X_test = read_data(self.file_path_test_data)
        
        # print model performance
        model_results, model, X_test, Y_test, X_train, Y_train, train, scaler, X_test_original, \
            selected_list, X_test_before_scale, X_train_original = self.execute_model(df_X_train, df_X_test)

        # calculations for ensemble model
        save_data_for_ensemble(X_train_original, Y_train, X_test_original, Y_test, self.file_path_preprocessed_data_file)
        if(self.isVRModel):
            average_feature_importances = get_feature_importance_for_voting_regressor(model, X_test, self.file_path_feature_importance_df_file)
        else:
            average_feature_importances = get_feature_importance(model, X_test, self.file_path_feature_importance_df_file)
        
        # plot actual vs predicted plot
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        self.plot_actual_pred(Y_test, y_pred, X_test['drug_class'])
        
        # shap calculations
        if (self.isshap):
            shap.initjs()
            explainer, shap_values = self.get_shap_values(model, X_train, X_test, selected_list)
            list_of_labels, current_label = self.get_shap_label(Y_test, 3)
            X_test_features = self.rename_features(X_test)
            self.save_shap_summary_plot(shap_values, list_of_labels, current_label, X_test_features)
            
        # Predict drug classes
        X_test_copy = predict_drug_classes(model, X_test, Y_train)
        data, dpp_strata_hba1c, sglt_strata_hba1c, dpp_strata_ldl, sglt_strata_ldl, dpp_strata_hdl, sglt_strata_hdl, dpp_strata_bmi,\
                sglt_strata_bmi, dpp_strata_actual, sglt_strata_actual = self.assign_sample_to_stratas(scaler, X_test_original, X_test_copy, Y_test)
        
        # print strata stats
        print_strata_stats(dpp_strata_actual, sglt_strata_actual, dpp_strata_hba1c, sglt_strata_hba1c, dpp_strata_ldl, sglt_strata_ldl,
                        dpp_strata_hdl, sglt_strata_hdl, dpp_strata_bmi, sglt_strata_bmi)
        
        # concordant and discordant treatment effect: before outlier removal
        print('HBA1C')
        (concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
            concordant_sglt_hba1c, discordant_sglt_dpp_hba1c ) = get_concordant_discordant(dpp_strata_hba1c,sglt_strata_hba1c, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_hba1c')
        print('LDL')
        (concordant_dpp_ldl, discordant_dpp_sglt_ldl,
            concordant_sglt_ldl, discordant_sglt_dpp_ldl ) = get_concordant_discordant(dpp_strata_ldl,sglt_strata_ldl, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                    variable_name = 'assigned_drug_ldl')
        print('HDL')
        (concordant_dpp_hdl, discordant_dpp_sglt_hdl,
            concordant_sglt_hdl, discordant_sglt_dpp_hdl ) = get_concordant_discordant(dpp_strata_hdl,sglt_strata_hdl, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_hdl')
        print('BMI') 
        (concordant_dpp_bmi, discordant_dpp_sglt_bmi,
            concordant_sglt_bmi, discordant_sglt_dpp_bmi ) = get_concordant_discordant(dpp_strata_bmi,sglt_strata_bmi, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_bmi')

        # visualize observed vs predicted drug classes
        df_ = drug_class_visualization(dpp_strata_hba1c, dpp_strata_actual, sglt_strata_hba1c,sglt_strata_actual,
                                            response_variable='hba1c_12m', 
                                            predicted_change='predicted_change_hba1c',
                                            assigned_drug='assigned_drug_hba1c',
                                            baseline_val='hba1c_bl_6m')
        df_ldl_ = drug_class_visualization(dpp_strata_ldl, dpp_strata_actual, sglt_strata_ldl,sglt_strata_actual,
                                        response_variable='ldl_12m', 
                                        predicted_change='predicted_change_ldl',
                                        assigned_drug='assigned_drug_ldl',
                                        baseline_val='ldl')
        df_hdl_ = drug_class_visualization(dpp_strata_hdl, dpp_strata_actual, sglt_strata_hdl,sglt_strata_actual,
                                        response_variable='hdl_12m', 
                                        predicted_change='predicted_change_hdl',
                                        assigned_drug='assigned_drug_hdl',
                                        baseline_val='hdl')
        df_bmi_ = drug_class_visualization(dpp_strata_bmi, dpp_strata_actual, sglt_strata_bmi,sglt_strata_actual,
                                        response_variable='bmi_12m', 
                                        predicted_change='predicted_change_bmi',
                                        assigned_drug='assigned_drug_bmi',
                                        baseline_val='bmi')

        # remove outliers in predicted values
        dpp_df_hba1c = drug_class_outlier_remove(dpp_strata_hba1c, dpp_strata_actual, 'hba1c_12m','predicted_change_hba1c',
                                                'assigned_drug_hba1c', 'hba1c_bl_6m')
        sglt_df_hba1c = drug_class_outlier_remove(sglt_strata_hba1c, sglt_strata_actual,'hba1c_12m',
                                                'predicted_change_hba1c', 'assigned_drug_hba1c', 'hba1c_bl_6m')
            
        dpp_df_ldl = drug_class_outlier_remove(dpp_strata_ldl, dpp_strata_actual, 'ldl_12m',
                                                'predicted_change_ldl', 'assigned_drug_ldl', 'ldl')
        sglt_df_ldl = drug_class_outlier_remove(sglt_strata_ldl, sglt_strata_actual, 'ldl_12m',
                                                'predicted_change_ldl', 'assigned_drug_ldl', 'ldl')
            
        dpp_df_hdl = drug_class_outlier_remove(dpp_strata_hdl, dpp_strata_actual, 'hdl_12m',
                                                'predicted_change_hdl', 'assigned_drug_hdl', 'hdl')
        sglt_df_hdl = drug_class_outlier_remove(sglt_strata_hdl, sglt_strata_actual, 'hdl_12m',
                                                'predicted_change_hdl', 'assigned_drug_hdl', 'hdl')
            
        dpp_df_bmi = drug_class_outlier_remove(dpp_strata_bmi, dpp_strata_actual, 'bmi_12m',
                                                'predicted_change_bmi', 'assigned_drug_bmi', 'bmi')
        sglt_df_bmi = drug_class_outlier_remove(sglt_strata_bmi, sglt_strata_actual, 'bmi_12m',
                                                'predicted_change_bmi', 'assigned_drug_bmi', 'bmi')
        
        
        print('HBA1C')
        (concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
            concordant_sglt_hba1c, discordant_sglt_dpp_hba1c ) = get_concordant_discordant(dpp_df_hba1c,sglt_df_hba1c, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_hba1c')

        print('\n============= HBA1C ===================')    
        print_change_mean(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, response_variable = 'hba1c_12m')

        print('\n\n====== Average change of other 3 responses =========')
        calculate_percentage_change_othre_responses(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, 
                    response_variable1='ldl_12m', response_variable2='hdl_12m', response_variable3='bmi_12m',
                    baseline_val1='ldl',baseline_val2='hdl', baseline_val3='bmi',
                    label1='LDL', label2='HDL', label3='BMI')
            
        print('\n\n====== Percentage in Original data =========')
        percentage_change_original_data(dpp_strata_actual, sglt_strata_actual,baseline_val='hba1c_bl_6m', response_variable = 'hba1c_12m')


        calculate_change_diff(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c, concordant_sglt_hba1c, discordant_sglt_dpp_hba1c,
                            'hba1c_12m', 'hba1c_bl_6m', 'predicted_change_hba1c')
        
        
        
        print('LDL')
        (concordant_dpp_ldl, discordant_dpp_sglt_ldl,
            concordant_sglt_ldl, discordant_sglt_dpp_ldl ) = get_concordant_discordant(dpp_df_ldl,sglt_df_ldl, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_ldl')

        print('\n============= LDL ===================')    
        print_change_mean(concordant_dpp_ldl, discordant_dpp_sglt_ldl,
                    concordant_sglt_ldl, discordant_sglt_dpp_ldl, response_variable = 'ldl_12m')


        print('\n\n====== Average change of other 3 responses =========')
        calculate_percentage_change_othre_responses(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, 
                    response_variable1='hba1c_12m', response_variable2='hdl_12m', response_variable3='bmi_12m',
                    baseline_val1='hba1c_bl_6m',baseline_val2='hdl', baseline_val3='bmi',
                    label1='HBA1C', label2='HDL', label3='BMI')

        print('\n\n====== Percentage in Original data =========')
        percentage_change_original_data(dpp_strata_actual, sglt_strata_actual, baseline_val='ldl', response_variable = 'ldl_12m')


        calculate_change_diff(concordant_dpp_ldl, discordant_dpp_sglt_ldl, concordant_sglt_ldl, discordant_sglt_dpp_ldl,
                            'ldl_12m', 'ldl', 'predicted_change_ldl')


        
        print('HDL')
        (concordant_dpp_hdl, discordant_dpp_sglt_hdl,
            concordant_sglt_hdl, discordant_sglt_dpp_hdl ) = get_concordant_discordant(dpp_df_hdl,sglt_df_hdl, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_hdl')
        print('\n============= HDL ===================')    
        print_change_mean(concordant_dpp_hdl, discordant_dpp_sglt_hdl,
                    concordant_sglt_hdl, discordant_sglt_dpp_hdl, response_variable = 'hdl_12m')


        print('\n\n====== Average change of other 3 responses =========')
        calculate_percentage_change_othre_responses(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, 
                    response_variable1='hba1c_12m', response_variable2='ldl_12m', response_variable3='bmi_12m',
                    baseline_val1='hba1c_bl_6m',baseline_val2='ldl', baseline_val3='bmi',
                    label1='HBA1C', label2='LDL', label3='BMI')

        print('\n\n====== Percentage in Original data =========')
        percentage_change_original_data(dpp_strata_actual, sglt_strata_actual, baseline_val='hdl', response_variable = 'hdl_12m')


        calculate_change_diff(concordant_dpp_hdl, discordant_dpp_sglt_hdl, concordant_sglt_hdl, discordant_sglt_dpp_hdl,
                            'hdl_12m', 'hdl', 'predicted_change_hdl')



        
        print('BMI')
        (concordant_dpp_bmi, discordant_dpp_sglt_bmi,
            concordant_sglt_bmi, discordant_sglt_dpp_bmi ) = get_concordant_discordant(dpp_df_bmi,sglt_df_bmi, data,
                                                                                        dpp_strata_actual, sglt_strata_actual,
                                                                                        variable_name = 'assigned_drug_bmi')

        print('\n============= BMI ===================')    
        print_change_mean(concordant_dpp_bmi, discordant_dpp_sglt_bmi,
                    concordant_sglt_bmi, discordant_sglt_dpp_bmi, response_variable = 'bmi_12m')

        print('\n\n====== Average change of other 3 responses =========')
        calculate_percentage_change_othre_responses(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, 
                    response_variable1='hba1c_12m', response_variable2='ldl_12m', response_variable3='hdl_12m',
                    baseline_val1='hba1c_bl_6m',baseline_val2='ldl', baseline_val3='hdl',
                    label1='HBA1C', label2='LDL', label3='HDL')

        print('\n\n====== Percentage in Original data =========')
        percentage_change_original_data(dpp_strata_actual, sglt_strata_actual,baseline_val='bmi',response_variable = 'bmi_12m')


        calculate_change_diff(concordant_dpp_bmi, discordant_dpp_sglt_bmi, concordant_sglt_bmi, discordant_sglt_dpp_bmi,
                            'bmi_12m', 'bmi', 'predicted_change_bmi')


        print('\n\n====== Percentage =========')
        calculate_percentage_change(concordant_dpp_hba1c, discordant_dpp_sglt_hba1c,
                    concordant_sglt_hba1c, discordant_sglt_dpp_hba1c, response_variable = 'hba1c_12m', baseline_val='hba1c_bl_6m')

        print('\n\n====== Percentage =========')
        calculate_percentage_change(concordant_dpp_ldl, discordant_dpp_sglt_ldl,
                    concordant_sglt_ldl, discordant_sglt_dpp_ldl, response_variable = 'ldl_12m', baseline_val='ldl')

        print('\n\n====== Percentage =========')
        calculate_percentage_change(concordant_dpp_hdl, discordant_dpp_sglt_hdl,
                    concordant_sglt_hdl, discordant_sglt_dpp_hdl, response_variable = 'hdl_12m', baseline_val='hdl')

        print('\n\n====== Percentage =========')
        calculate_percentage_change(concordant_dpp_bmi, discordant_dpp_sglt_bmi,
                    concordant_sglt_bmi, discordant_sglt_dpp_bmi,  response_variable = 'bmi_12m', baseline_val='bmi' )
    
if __name__ == "__main__":
    print("Initialte optimal model training...")
    baseModel = BaseModel()
    # Call the initialize method to execute the workflow
    baseModel.initialize() 