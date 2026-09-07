import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, DataTransformationConfig, PrepareBaseModelConfig, PrepareCallbacksConfig, TrainingConfig, FineTuningConfig, ModelEvaluationConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)



    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config



    def get_data_transformation_config(self) -> DataTransformationConfig:
            config = self.config.data_transformation
    
            create_directories([config.root_dir])
    
            data_transformation_config = DataTransformationConfig(
                root_dir=config.root_dir,
                source_dir=config.source_dir
            )
    
            return data_transformation_config



    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model
            params = self.params
    
            create_directories([config.root_dir])
    
            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=config.root_dir,
                model_path=config.model_path,
                params_image_size=params.IMAGE_SIZE,
                params_learning_rate=params.LEARNING_RATE,
                params_include_top=params.INCLUDE_TOP,
                params_weights=params.WEIGHTS,
                params_classes=params.CLASSES
            )
    
            return prepare_base_model_config



    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
            config = self.config.prepare_callbacks
    
            model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
    
            create_directories([config.root_dir, config.tensorboard_root_log_dir, model_ckpt_dir])
    
            prepare_callbacks_config = PrepareCallbacksConfig(
                root_dir=config.root_dir,
                tensorboard_root_log_dir=config.tensorboard_root_log_dir,
                checkpoint_model_filepath=config.checkpoint_model_filepath
            )
    
            return prepare_callbacks_config



    def get_training_config(self) -> TrainingConfig:
            training = self.config.training
            get_model = self.config.prepare_base_model
            params = self.params
            training_data = self.config.data_transformation
    
            create_directories([training.root_dir])
    
            training_config = TrainingConfig(
                root_dir=training.root_dir,
                trained_model_path=training.trained_model_path,
                updated_base_model_path=get_model.model_path,
                training_data=training_data.root_dir,
                params_epochs=params.EPOCHS,
                params_batch_size=params.BATCH_SIZE,
                params_is_augmentation=params.AUGMENTATION,
                params_image_size=params.IMAGE_SIZE
            )
    
            return training_config



    def get_fine_tuning_config(self) -> FineTuningConfig:
            training = self.config.training
            training_data = self.config.data_transformation
            params = self.params
    
            fine_tuning_config = FineTuningConfig(
                trained_model_path=training.trained_model_path,
                fine_tuned_model_path=training.fine_tuned_model_path,
                training_data=training_data.root_dir,
                params_epochs=params.FINE_TUNE_EPOCHS,
                params_batch_size=params.BATCH_SIZE,
                params_is_augmentation=params.AUGMENTATION,
                params_image_size=params.IMAGE_SIZE,
                fine_tune_layers=params.FINE_TUNE_LAYERS,
                fine_tune_learning_rate=params.FINE_TUNE_LEARNING_RATE
            )
    
            return fine_tuning_config



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
            config = self.config.model_evaluation
            training = self.config.training
            training_data = self.config.data_transformation
            params = self.params
    
            create_directories([config.root_dir])
    
            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config.root_dir,
                model_path=training.fine_tuned_model_path,
                training_data=training_data.root_dir,
                all_params=params,
                params_image_size=params.IMAGE_SIZE,
                params_batch_size=params.BATCH_SIZE
            )
    
            return model_evaluation_config