import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import keras



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename



    def predict(self):
        model = load_model("artifacts/training/stage2_model.keras")

        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        result = np.argmax(model.predict(test_image), axis=1)

        print(result)

        if result[0] == 0:
            prediction = "Cyst"
        elif result[0] == 1:
            prediction = "Normal"
        elif result[0] == 2:
            prediction = "Stone"
        elif result[0] == 3:
            prediction = "Tumor"
        
        return [{"image": prediction}]