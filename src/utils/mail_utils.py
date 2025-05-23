import yaml
import os
import sys
import numpy as np
import pickle


def read_yaml_file(file_path : str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise e
    
    
    


def write_yaml(file_path : str, content : object, replace : bool = False) -> None:
    
    try:
        if replace :
            
            if os.path.exists(file_path):
                
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, "w") as yaml_file:
            return yaml.dump(content, yaml_file)
        
    except Exception as e:
        raise e
    
    
    
def save_numpy_array_data(file_path : str, array : np.array):
    
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_object:
            np.save(file_object, array)
            
    except Exception as e:
        raise e
    
    
    
    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
    except Exception as e:
        raise e
    
    

def load_object(file_path: str ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    
    except Exception as e:
        raise e
    
    
    

def load_numpy_array_data(file_path: str) -> np.array:

    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise  e
    
    
    
    
