import lightgbm as ltb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from catboost import CatBoostRegressor

from baseModel import BaseModel

class Model3(BaseModel):
    def __init__(self): 
        feature_list = ['C10A', 'MD_RCT_mmol_mol', 'P_Krea', 'bmi', 'comb_comp_enn' ,'concordant_dis' ,'drug_class' ,'eGFR', 'gluk',
                        'hba1c_bl_18m' ,'hba1c_bl_6m', 'hdl', 'ika', 'insulin', 'ldl' ,'met_oad0', 'n_of_dis', 'renal_insuf' ,'smoking',
                        'sum_diab_drugs' ,'t2d_dur_y' ,'trigly']

        rfr = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=123)
        ltbr = ltb.LGBMRegressor(max_depth = 6, learning_rate = 0.1, verbose = -1, verbose_eval = False)
        catboost = CatBoostRegressor(iterations=40,
                            learning_rate=0.1,
                            depth=6, verbose = 0)
        model = VotingRegressor([('ltbr', ltbr), ('rfr', rfr), ('catboost', catboost)])

 
        # Initialize the BaseModel (parent class)
        super().__init__(feature_list, model, isVRModel=True)
        
    def start(self):
        # call the parent class's initialize method
        super().initialize()  
        
if __name__ == "__main__":
    print("Initialte optimal model training...")
    model3 = Model3()
    model3.start()
