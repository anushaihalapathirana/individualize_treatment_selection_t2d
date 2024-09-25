# Multi-Output Explainable AI based algorithm to Individualize Treatment Selection between SGLT2 and DPP-4 Inhibitor Therapies in Type 2 Diabetes Patients

This repository contains the code for developing a multi-output machine learning (ML) model, including training, validation, and testing phases. The model is designed to predict key health outcomes for patients with Type 2 diabetes, 12 months after drug initiation.

## Prediction Targets
The primary efficacy outcomes (prediction targets) include:

- HbA1c (Hemoglobin A1c)
- LDL cholesterol (Low-Density Lipoprotein)
- HDL cholesterol (High-Density Lipoprotein)
- BMI (Body Mass Index)

## Treatment Selection Strategy
The treatment selection process is conducted in two steps:

1. Multi-Treatment Strategy
In this method, each health parameter (HbA1c, LDL, HDL, BMI) is evaluated separately. Based on the predicted outcome for each parameter, a patient is assigned one of two possible treatments.

2. Single-Treatment Strategy
This strategy aggregates the results from the multi-treatment method and assigns a single treatment to each patient. It individualizes treatment selection by considering multiple health outcomes. We experimented with two aggregation methods:

Majority Voting: The treatment that is selected most frequently across health parameters.
Importance-Weighted Aggregation: Each treatment is weighted based on the importance of the predicted outcomes.

All the experiments carriedout with Type 2 diabetes dataset.

## Directory Structure

- `ensemble/`: Contains code for developing single treatment strategies based on aggregation methods.
- `imputation/`: Includes code used to impute missing values in response variables.
- `Other models (Appendix)/`: Contains high-performing models of various machine learning types that were developed during the experiments.
- `resources/`: 
    - `data/`: Contains the dataset used in the study. (Note: This folder is currently empty due to privacy agreements regarding data usage.)
    - `img`: Contains figures and visualizations generated during the analysis.
    - `output`: Includes the data generated throughout the analysis and modeling process. (Note: This folder is currently empty due to privacy agreements regarding data usage.)
- `src/`: Contains code for training, evaluating, and analyzing the performance of machine learning models and SHAP explanations.
- `notebooks/`: This directory contains Jupyter notebook versions of the Python scripts. The notebooks in this directory replicate the functionality of the corresponding `.py` files, except for the notebooks located in the `Other models/` directory. The `Other models/` directory contains notebooks that do not have corresponding `.py` files.
- `test/`: These folders dedicated for testing helper functions. The tests are implemented using pytest. You can run the tests to verify the correctness of the helper functions and their integration into the overall project.

## Usage

1. **Clone the Repository**: Clone this repository to your local machine using 
```bash 
git clone https://github.com/anushaihalapathirana/individualize_treatment_selection_t2d.git
``````
2. **Install Dependencies**: Install the necessary dependencies using 
```bash
pip install -r requirements.txt
```
3. **Data Preparation**: Ensure dataset files are in the `resources/data/` directory.
4. **Imputate response variables**: To perform initial preprocessing, and impute all four response variables, execute the __init__.py script. This script handles:

    - Initiating the preprocessing steps.
    - Training the four imputation models.
    - Imputing missing values for the four response variables.

5. **Model Training and Evaluation**: Run `training.py` scripts in the `src/models/` directory to train and evaluate ML models.
6. **Multi-Treatment Selection**: To run the optimal model for multi-treatment selection, execute the `optimalModel.py` script located in the `src/models/` directory. 
    
    Additional other high-performance model scripts can be found in the `src/models/best_models/` directory.
Other high performance model scripts are in the `src/models/best_models` directory.
7. **Single-Treatment Selection**: To perform single-treatment selection, follow these steps:
    - Run the Optimal Model for Multi-Treatment Selection: This will saves the necessary files in the `resources/output/` directory
    - Run the Ensemble Model: After the optimal model has been run, execute the `ensemble/ensembleModel.py` script. This script will apply both the majority voting and weight-importance aggregation methods for treatment selection.

        To run and evaluate all the best models for the single treatment strategy, execute the script `runAllEnsembleModels.py` located at `ensemble/` directory. 
        
        This script will execute all .py file in the `src/models/best_models/` directory and displays the reuslts using both aggregation methods. By running this script, you can easily analyze the performance of all the models in one go.
8. **Review Results**: Explore output figures in the `resources/img/` directory.
9. **Run Test Cases**: Ensure pytest is installed. To run the test files, navigate to the directory containing the test files. For example:
    ```bash 
    cd imputation/test
    ``````
    Execute the tests using the following command
    ```bash 
    pytest
    ``````
    After running the tests, you will see a summary of the test results in your terminal, indicating which tests passed or failed.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

