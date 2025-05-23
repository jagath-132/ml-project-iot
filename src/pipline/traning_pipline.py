import os
import sys
from src.entity.congif_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig,
    DataValidationConfig
)
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config= self.training_pipeline_config)

            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()

            return data_ingestion_artifact

        except Exception as e:
            raise e
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config= self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)

            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        
        except Exception as e:
            raise e
        
    def run_pipline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            print(data_ingestion_artifact)
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            print(data_validation_artifact)
            return data_validation_artifact
        except Exception as e:
            raise e
        