import os 
import sys
import numpy as np



CLASSIFICATION_TARGET_COLUMN = "Failure_Within_7_Days"

REGRESSION_TARGET_COLUMN = "emaining_Useful_Life_days"

PIPPLELINE_NAME = ""

ARTIFACT_DIR = "artifact"

FILE_NAME = "raw.csv"

TRAIN_FILE_NAME = "train.csv"

TEST_FILE_NAME = "test.csv"

TRAIN_REGRESSION_FILE = "train_regression.csv"

TEST_REGRESSION_FILE = "test_regression.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

SAVED_MODEL_DIR =os.path.join("saved_models")

CLASSIFICATION_MODEL_FILE_NAME = "model_classification.pkl"

REGRESSION_MODEL_FILE_NAME = "model_regression.pkl"





"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "IOTDATA"
DATA_INGESTION_DATABASE_NAME: str = "IOTMLPROJECT"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2