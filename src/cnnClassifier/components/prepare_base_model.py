from cnnClassifier import logger
import keras
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config



    def get_base_model(self):
        base_model = keras.applications.vgg16.VGG16(
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
            input_shape=self.config.params_image_size,
            name="vgg16"
        )
        
        return base_model



    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten)

        full_model = keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model



    def update_base_model(self, freeze_all=True, freeze_till=None):
        self.full_model = self._prepare_full_model(
            model=self.get_base_model(),
            classes=self.config.params_classes,
            freeze_all=freeze_all,
            freeze_till=freeze_till,
            learning_rate=self.config.params_learning_rate
        )

        self.full_model.save(self.config.model_path)