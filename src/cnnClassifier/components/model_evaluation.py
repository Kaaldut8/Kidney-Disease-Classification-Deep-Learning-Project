import json
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



    def evaluation_trained(self):
        self.model = self.load_model(self.config.trained_model_path)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)



    def evaluation_fine_tuned(self):
        self.model = self.load_model(self.config.fine_tuned_model_path)
        self._test_generator()
        self.score = self.model.evaluate(self.test_generator)



    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}

        score_path = Path(os.path.join(self.config.root_dir, "scores.json"))

        if score_path.exists():
            with open(score_path, "r") as f:
                all_scores = json.load(f)
        else:
            all_scores = []

        all_scores.append(scores)

        save_json(path=score_path, data=all_scores)