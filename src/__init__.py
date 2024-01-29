import pandas as pd
import numpy as np
import random
from matplotlib.pyplot import pie, axis, show
import seaborn as sns
import missingno as msno
from scipy import stats
import matplotlib.pyplot as plt
import yaml
import os 
import sys

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import statsmodels.regression.linear_model as sm
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, CompoundKernel
import sklearn_relief as sr
from skrebate import ReliefF
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as ltb
from sklearn.svm import SVR
from scipy.stats import ks_2samp
from tabulate import tabulate

from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE


from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.exceptions import DataConversionWarning

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from constants import COMMON_VARIABLE_PATH, X_TEST_PATH, X_TRAIN_PATH

class Exp:
    def __init__(self):
        # Get the current script's directory
        self.script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the full path to the CSV file
        self.file_path_common_variables = os.path.join(self.script_directory, COMMON_VARIABLE_PATH)

        self.file_path_X_train = os.path.join(self.script_directory, X_TRAIN_PATH)
        self.file_path_X_test = os.path.join(self.script_directory, X_TEST_PATH)
        
        # Read common variables from a YAML file
        with open(self.file_path_common_variables, 'r') as file:
            self.common_data = yaml.safe_load(file)

        self.response_variable_list = self.common_data['response_variable_list']
        self.correlated_variables = self.common_data['correlated_variables']
        print('done')