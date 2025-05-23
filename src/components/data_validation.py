import os
import pandas as pd
import numpy as np
from src.constant import *
from src.entity.artifact_entity import *
from src.entity.congif_entity import *
from src.logger import *
from src.utils.mail_utils import *
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact, 
                 data_validation_config:DataValidationConfig):
        
        """
        Initialize the DataValidation class with ingestion artifacts and validation configurations.
        """

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error
            raise e
        
    @staticmethod    
    def read_csv(file_path)->pd.DataFrame:

        """
        Read a CSV file and return it as a pandas DataFrame.
        """
         
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise e

    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:

        """
        Validate if the DataFrame contains the expected number of columns based on the schema.
        """

        try:
            number_of_columns=len(self._schema_config['columns'])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise e

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:

        """
        Detect dataset drift by comparing the distributions of each column between two datasets.
        """

        try:
            status = True
            report ={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml(file_path=drift_report_file_path,content=report)
        except Exception as e:
            raise e
        
    def initiate_data_validation(self)->DataIngestionArtifact:

        """
        Execute the data validation process:
        1. Load training and testing datasets.
        2. Validate the number of columns in both datasets.
        3. Detect dataset drift between training and testing datasets.
        4. Save validated datasets and generate a validation artifact.
        """

        try:
            train_path = self.data_ingestion_artifact.trained_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_csv(train_path)
            test_dataframe = DataValidation.read_csv(test_path)

            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"

            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise e