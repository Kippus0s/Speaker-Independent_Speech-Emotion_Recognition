import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, Callback
import time 

class EpochTimer(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1} took {epoch_duration:.2f} seconds.")

class Earlystop_1(EarlyStopping):
    def __init__(self): 
        super().__init__(monitor='val_accuracy', 
                        patience=3,restore_best_weights=True)          
                               

class Lr_callback_1(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self):
        super().__init__(
                        monitor='val_loss',
                        factor=0.1,
                        patience=3,
                        verbose=0,
                        mode='auto',
                        min_delta=0.0001,
                        cooldown=0,
                        min_lr=1e-6)


