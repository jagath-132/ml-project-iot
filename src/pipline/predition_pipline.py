import os, sys
import pandas as pd
from src.logger import  *
from src.utils.mail_utils import load_object







# class PredictionPipeline:

#     def __init__(self):

#         pass

#     def predict(self, features):

#         try:
#             model_path_cl=os.path.join("final_model", "model_classification.pkl")
#             model_path_rg=os.path.join("final_model", "model_regression.pkl")
#             preprocessor_path=os.path.join("final_model", "preprocessor.pkl")
#             model=load_object(model_path)
#             preprocessor=load_object(preprocessor_path)
#             scaled_data = preprocessor.transform(features)
#             preds = model.predict(scaled_data)
#             return preds
#         except Exception as e:
#             logger.error(f"An Error Occured In : {e}")
#             raise e


class CustomData:
    def __init__(  self,
                 Machine_Type : str,
                 Temperature_C : float,
                 Vibration_mms : float,
                 Sound_dB : float,
                 Oil_Level_pct : float,
                 Coolant_Level_pct : float,
                 Power_Consumption_kW : float,
                 Maintenance_History_Count : float,
                 Failure_History_Count : float,
                 AI_Supervision : str,
                 Error_Codes_Last_30_Days: float
         ):

        self.Machine_Type = Machine_Type
        self.Temperature_C = Temperature_C
        self.Vibration_mms = Vibration_mms
        self.Sound_dB = Sound_dB
        self.Oil_Level_pct = Oil_Level_pct
        self.Coolant_Level_pct = Coolant_Level_pct
        self.Power_Consumption_kW = Power_Consumption_kW
        self.Maintenance_History_Count = Maintenance_History_Count
        self.Failure_History_Count = Failure_History_Count
        self.AI_Supervision = AI_Supervision
        self.Error_Codes_Last_30_Days = Error_Codes_Last_30_Days

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Machine_Type": [self.Machine_Type],
                "Temperature_C": [self.Temperature_C],
                "Vibration_mms": [self.Vibration_mms],
                "Sound_dB": [self.Sound_dB],
                "Oil_Level_pct": [self.Oil_Level_pct],
                "Coolant_Level_pct": [self.Coolant_Level_pct],
                "Power_Consumption_kW": [self.Power_Consumption_kW],
                "Maintenance_History_Count": [self.Maintenance_History_Count],
                "Failure_History_Count": [self.Failure_History_Count],
                "AI_Supervision": [self.AI_Supervision],
                "Error_Codes_Last_30_Days": [self.Error_Codes_Last_30_Days]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e
