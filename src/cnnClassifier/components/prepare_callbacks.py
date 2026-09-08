import keras
import time
import os
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig



class PrepareCallbacks:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config



    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        return keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)



    @property
    def _create_ckpt_callbacks(self):
        return keras.callbacks.ModelCheckpoint(
            self.config.checkpoint_model_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )



    @property
    def _create_early_stopping(self):
        return keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            mode="max",
            restore_best_weights=True,
            verbose=1
        )



    @property
    def _create_reduce_lr(self):
        return keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=2,
            mode="max",
            min_lr=1e-7,
            verbose=1
        )



    def get_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_early_stopping,
            self._create_reduce_lr
        ]