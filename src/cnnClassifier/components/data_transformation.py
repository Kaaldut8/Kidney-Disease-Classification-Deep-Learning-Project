import os
import shutil
from cnnClassifier import logger
from sklearn.model_selection import train_test_split
from cnnClassifier.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config



    def transform_data(self):
        classes = os.listdir(self.config.source_dir)

        for class_name in classes:
            class_path = os.path.join(self.config.source_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            images = os.listdir(class_path)
            train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
            for split, split_images in [("train", train_images),("val", val_images),("test", test_images)]:
                destination = os.path.join(self.config.root_dir, split, class_name)
                os.makedirs(destination, exist_ok=True)
                for image in split_images:
                    if not os.path.exists(os.path.join(destination, image)):
                        shutil.copy2(
                            os.path.join(class_path, image),
                            os.path.join(destination, image)
                        )