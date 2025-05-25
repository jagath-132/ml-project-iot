import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.pipline.predition_pipline import CustomData
from src.utils.mail_utils import load_object

# Title
st.title("Machine Failure Prediction")
st.markdown("Enter the machine parameters below to predict potential failure or regression output.")

# Input form
with st.form("prediction_form"):
    st.subheader("Machine Parameters")
    
    
    
    Machine_Type = st.selectbox("Machine Type", options=['Mixer', 'Industrial_Chiller', 'Pick_and_Place', 'Vision_System',
       'Shuttle_System', 'Labeler', 'Automated_Screwdriver',
       'Shrink_Wrapper', 'Laser_Cutter', 'CMM', 'CNC_Lathe', 'Dryer',
       'Valve_Controller', 'Furnace', 'Carton_Former', 'Hydraulic_Press',
       'Compressor', 'AGV', 'Robot_Arm', 'Conveyor_Belt',
       'Forklift_Electric', 'Press_Brake', 'Boiler', 'Vacuum_Packer',
       'XRay_Inspector', 'Crane', '3D_Printer', 'Palletizer', 'Grinder',
       'CNC_Mill', 'Injection_Molder', 'Heat_Exchanger', 'Pump'])
    Temperature_C = st.number_input("Temperature (°C)", min_value=0, step=1)
    Vibration_mms = st.number_input("Vibration (mm/s)", min_value=0, step=1)
    Sound_dB = st.number_input("Sound Level (dB)", min_value=0, step=1)
    Oil_Level_pct = st.number_input("Oil Level (%)", min_value=0, step=1)
    Coolant_Level_pct = st.number_input("Coolant Level (%)", min_value=0, step=1)
    Power_Consumption_kW = st.number_input("Power Consumption (kW)", min_value=0, step=1)
    Maintenance_History_Count = st.number_input("Maintenance History Count", min_value=0, step=1)
    Failure_History_Count = st.number_input("Failure History Count", min_value=0, step=1)
    AI_Supervision = st.selectbox("AI Supervision", options=["True", "False"])
    Error_Codes_Last_30_Days = st.number_input("Error Codes (last 30 days)", min_value=0, step=1)
       
    submit = st.form_submit_button("Predict")

# Predict and show result
if submit:
    data = CustomData(Machine_Type= Machine_Type,
                      Temperature_C= Temperature_C,
                      Vibration_mms= Vibration_mms,
                      Sound_dB= Sound_dB,
                      Oil_Level_pct= Oil_Level_pct,
                      Coolant_Level_pct= Coolant_Level_pct,
                      Power_Consumption_kW= Power_Consumption_kW,
                      Maintenance_History_Count= Maintenance_History_Count,
                      Failure_History_Count= Failure_History_Count,
                      AI_Supervision= AI_Supervision,
                      Error_Codes_Last_30_Days= Error_Codes_Last_30_Days)
    
    final_data = data.get_data_as_data_frame()
    
    preprocessor= load_object("final_model/preprocessor.pkl")
    
    classification_model=load_object("final_model/model_classification.pkl")
    
    regression_model =load_object("final_model/model_regression.pkl")
    
    pre_data = preprocessor.transform(final_data)

    classification_pred = classification_model.predict(pre_data)


    risk_score_pred = regression_model.predict(pre_data)[0]
    
    result_classification = "Ture" if classification_pred[0] == 0 else "False"
    
    result_risk_score = f"{risk_score_pred:.2f}"
    
    
    st.write(result_classification)
    
    st.write(result_risk_score)
    