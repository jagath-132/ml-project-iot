import os
import sys
from src.entity.congif_entity import (
    TrainingPipelineConfig, 
    DataIngestionConfig,
)
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion


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
        
    def run_pipline(self):
        try: 
            data_ingestion_ar = self.start_data_ingestion()
            return data_ingestion_ar
        except Exception as e:
            raise e