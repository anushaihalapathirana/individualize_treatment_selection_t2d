import lightgbm as ltb

from baseModel import BaseModel

class OptimalModel(BaseModel):
    def __init__(self): 
        feature_list = ['P_Krea', 'bmi', 'drug_class', 'eGFR', 'gluk', 'hba1c_bl_18m', 'hba1c_bl_6m', 'hdl', 'ika', 'ldl', 'obese', 't2d_dur_y', 'trigly']
        model = ltb.LGBMRegressor(max_depth = 6, learning_rate = 0.1, verbose = -1, verbose_eval = False)
 
        # Initialize the BaseModel (parent class)
        super().__init__(feature_list, model)
        
    def start(self):
        # call the parent class's initialize method
        super().initialize()  
        
if __name__ == "__main__":
    print("Initialte optimal model training...")
    optModel = OptimalModel()
    optModel.start()