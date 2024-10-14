# Paths to original data
DATA_WITHOUT_DATES = 'resources/data/HTx_ind_treat_res_new_data_update_without_dates_08032024.csv'
DATA_ONLY_DATES = 'resources/data/HTx_ind_treat_res_new_data_update_only_dates_08032024.csv'

# Paths to the training and test data before the imputation of response variables
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'

# Paths to training and test data after the imputation of response variables
TRAIN_PATH_IMPUTED = 'resources/data/train_data.csv'
TEST_PATH_IMPUTED = 'resources/data/test_data.csv'

# Paths to training and test data after the imputation of response variables (without imputation of LDL response variable)
TRAIN_PATH_WO_LDL_IMPUTATION = 'resources/data/train_data_wo_LDL_imputation.csv'
TEST_PATH_WO_LDL_IMPUTATION = 'resources/data/test_data_wo_LDL_imputation.csv'

# Paths to data files that used for single-treatment selection methods
PREDICTED_DRUG_CLASS_FILE_LOCATION = 'resources/output/pred_drug_classes.csv'
PREPROCESSED_DATA_FILE_LOCATION = 'resources/output/preprocessed_data.csv'
FEATURE_IMPORTANCE_DF_LOCATION = 'resources/output/feature_importance_dataframe.csv'

# Path to data stats file
DATA_STATS_FILE_LOCATION = 'resources/output/data_stats.csv'

# Paths to the visualization plots
SCATTER_PLOT_ACTUAL_VS_PRED = 'resources/img/multi_output_scatter_plot.jpeg'
SCATTER_BOX_PLOT_HBA1C = 'resources/img/scatter_and_box_plot_hba1c.jpeg'
SCATTER_BOX_PLOT_LDL = 'resources/img/scatter_and_box_plot_ldl.jpeg'
SCATTER_BOX_PLOT_HDL = 'resources/img/scatter_and_box_plot_hdl.jpeg'
SCATTER_BOX_PLOT_BMI = 'resources/img/scatter_and_box_plot_bmi.jpeg'
SHAP_SUMMARY_PLOT_HBA1C = 'resources/img/shap_summary_plot_hba1c.jpeg'
SHAP_SUMMARY_PLOT_LDL = 'resources/img/shap_summary_plot_ldl.jpeg'
SHAP_SUMMARY_PLOT_HDL = 'resources/img/shap_summary_plot_hdl.jpeg'
SHAP_SUMMARY_PLOT_BMI = 'resources/img/shap_summary_plot_bmi.jpeg'

# Paths to the directories, notebooks and scripts
BEST_MODELS_DIRECTORY = 'src/models/best_models'
BEST_MODELS_NOTEBOOKS_DIRECTORY = 'src/models/notebooks'
ENSEMBLE_PYTHON_FILE_PATH = 'ensemble/ensembleModel.py'
ENSEMBLE_PYTHON_NOTEBOOK_PATH = 'ensemble_models.ipynb'

# Paths to the data after imputation of response variables. Files for each response variable
BMI_PATH = 'data/bmi.csv'
HDL_PATH = 'data/hdl.csv'
LDL_PATH = 'data/ldl.csv'
HBA1C_PATH = 'data/hba1c.csv'

# Path to the yaml file containing common variables
COMMON_VARIABLE_PATH = '../common_variables.yaml'

# Codes for drug classes
SGLT_VALUE = 1
DPP_VALUE = 0
ORIGINAL_DPP_VALUE = 3
ORIGINAL_SGLT_VALUE = 4

SEED = 42

# Constants for image generation
IMAGE_FONT_SIZE_24 = 24
IMAGE_LABEL_SIZE_24 = 24
IMAGE_FONT_SIZE_18 = 18
IMAGE_FONT_SIZE_14 = 14
IMAGE_DPI_300 = 300



