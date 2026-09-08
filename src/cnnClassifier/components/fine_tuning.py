import keras
import os
from cnnClassifier.entity.config_entity import FineTuningConfig
from cnnClassifier.components.training import Training



class FineTuning(Training):
    def __init__(self, config: FineTuningConfig):
        self.config = config



    def get_model(self):
        model = keras.models.load_model(self.config.trained_model_path)
        return model



    def _freeze_unfreeze(self, model):
        conv_layer = [layer for layer in model.layers if layer.name.startswith("block")]

        for layer in conv_layer[-self.config.fine_tune_layers:]:
            layer.trainable = True

        for layer in model.layers:
            print(layer.name, layer.trainable)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.fine_tune_learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"]
        )

        model.summary()
        return model



    def train(self, callbacks_list: list):
        model = self.get_model()

        model = self._freeze_unfreeze(model)

        self.train_valid_generator()

        model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks_list
        )

        model.save(self.config.fine_tuned_model_path)