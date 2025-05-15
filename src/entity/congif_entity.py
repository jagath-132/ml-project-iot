import os
import sys
from datetime import datetime
from dataclasses import dataclass
from src.constant import *



class TrainingPipelineConfig:
    def __init__(self, timestamp:str=datetime.now()):

        timestamp=timestamp.strftime("%m%d%Y%H%M_%S")
        self.artifact_name = ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir = os.path.join("final_model")
        self.timestamp: str = timestamp





class DataIngestionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME
            )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = DATA_INGESTION_DATABASE_NAME
