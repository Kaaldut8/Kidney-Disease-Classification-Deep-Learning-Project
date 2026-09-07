import kagglehub
import os
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config



    def download_file(self):
        kagglehub.dataset_download(self.config.source_URL, output_dir=self.config.local_data_file)
        logger.info(f"Dataset downloaded at path: {self.config.local_data_file}")