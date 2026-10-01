from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier.components.training import Training
from cnnClassifier.components.fine_tuning import FineTuning
from cnnClassifier import logger



STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass



    def main(self):
        config = ConfigurationManager()

        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callbacks_list = prepare_callbacks.get_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.train(callbacks_list=callbacks_list)

        fine_tuning_config = config.get_fine_tuning_config()
        fine_tuning = FineTuning(config=fine_tuning_config)
        fine_tuning.train(callbacks_list=callbacks_list)



if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e