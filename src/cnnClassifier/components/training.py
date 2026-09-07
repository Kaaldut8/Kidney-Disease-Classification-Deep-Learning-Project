import keras
import os
from cnnClassifier.entity.config_entity import TrainingConfig



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config



    def get_model(self):
        model = keras.models.load_model(self.config.updated_base_model_path)
        return model


    def train_valid_generator(self):
        datagen = keras.src.legacy.preprocessing.image.ImageDataGenerator(
            preprocessing_function=keras.applications.vgg16.preprocess_input
        )

        if self.config.params_is_augmentation:
            train_datagen = keras.src.legacy.preprocessing.image.ImageDataGenerator(
                preprocessing_function=keras.applications.vgg16.preprocess_input,
                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
            )
        else:
            train_datagen = datagen

        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.config.training_data, "train"),
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=True
        )

        self.valid_generator = datagen.flow_from_directory(
            os.path.join(self.config.training_data, "val"),
            target_size=self.config.params_image_size[:2],
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=False
        )



    def train(self, callbacks_list: list):
        model = self.get_model()

        self.train_valid_generator()

        model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callbacks_list
        )

        model.save(self.config.trained_model_path)