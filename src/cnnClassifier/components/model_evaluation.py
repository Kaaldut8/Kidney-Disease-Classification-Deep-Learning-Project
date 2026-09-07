import keras
from cnnClassifier.config.configuration import ModelEvaluationConfig
import os
from pathlib import Path
from cnnClassifier.utils.common import save_json



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config



    def _test_generator(self):
        datagen = keras.src.legacy.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input
        )

        self.test_generator = datagen.flow_from_directory(
            os.path.join(self.config.training_data, "test"),
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=False
        )



    @staticmethod
    def load_model(path: Path) -> keras.Model:
        return keras.models.load_model(path)



    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)



    def save_score(self):
        scores = {"loss": self.score[0], "accuracy":self.score[1]}
        save_json(path=os.path.join(self.config.root_dir, "scores.json"), data=scores)