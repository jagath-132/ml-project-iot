import os, sys
import json 
import certifi
import pandas as pd
import numpy as np
import pymongo 
from src.logger import *
from dotenv import load_dotenv, find_dotenv
load_dotenv()
mongo_url = os.getenv("mongoDB_url")


class ExtractTransformLoadMongoDB:
    def __init__(self):
        pass
    
    def transform_to_json(self, filepath : str):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)
            data = data.sample(n=80000, random_state=42)  # deterministic sampling
            
            records = list(json.loads(data.T.to_json()).values())
            
            return records
        
        except Exception as e:
            logger.error(f"error founded in {e}")
            
    def insert_data_mongoDB(self, records, data_base, collection):
        try:
            self.data_base = data_base
            self.records = records
            self.collection = collection
            self.mongoclient = pymongo.MongoClient(mongo_url, tlsCAFile=certifi.where())
            self.data_base = self.mongoclient[self.data_base]
            self.collection = self.data_base[self.collection]
            self.collection.insert_many(self.records)
            
            return (list(self.records))
        except Exception as e:
            logger.error(f"error founded in {e}")
            
            raise
            
            











if __name__ == "__main__":
    
    file_path = r"data\factory_sensor_simulator_2040.csv"
    
    Data_Base_Name = "IOTMLPROJECT"
    
    Data_collection_name = "IOTDATA"
    
    extract = ExtractTransformLoadMongoDB()
    records = extract.transform_to_json(file_path)
    print(records)
    
    numofrecords = extract.insert_data_mongoDB(records, Data_Base_Name, Data_collection_name)
    print(numofrecords)