import os
import sys
from src.mlproject.exception import CustomeException
from src.mlproject.logger import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str =  os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv(os.path.join('notebook/data','raw.csv'))

            logging.info("Reading from MySQL database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            train_set , test_test = train_test_split(df, test_size=0.2,random_state=57)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_test.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Data Ingestion is complated")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                

        except Exception as e:
            raise CustomeException(e,sys)
        


    