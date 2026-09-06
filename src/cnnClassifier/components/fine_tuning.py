import keras
import os
from cnnClassifier.entity.config_entity import FineTuningConfig
from cnnClassifier.components.training import Training



class FineTuning:
    def __init__(self, config: FineTuningConfig):
        self.config = config



    def get_model(self):
        model = keras.models.load_model(self.config.trained_model_path)
        return model



    @staticmethod
    def _freeze_unfreeze(self, model):
        model.trainable = False

        base_model = model.get_layer("vgg16")

        for layer in base_model.layers[-self.config.fine_tune_layers:]:
            layer.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.fine_tuned_learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"]
        )

        return model



    def train(self, callbacks_list: list):
        model = self.get_model()

        model = self._freeze_unfreeze(model)

        Training.train_valid_test_generator()

        model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks_list
        )

        model.save(self.config.fine_tuned_model_path)