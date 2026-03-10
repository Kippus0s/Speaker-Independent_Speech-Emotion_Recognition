# train_model.py 
import os
import time
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import argparse
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, AirAbsorption
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utilities.class_weight import class_weight_maker
from prepare_tf_datasets import *
from models import model_specs, callbacks

#Determinism via seeding and enforcing deterministic gpu operations - ensuring identical results between runs
tf.keras.backend.clear_session()
tf.config.experimental.enable_op_determinism()
seed = 69
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Argument handling
def parse_args():
    parser = argparse.ArgumentParser(description="Train SER model")

    parser.add_argument(
        "DATASET",
        type=str,
        choices=["emodb", "iemocap", "ravdess", "savee"],
        help="Dataset name"
    )

    parser.add_argument(
        "DATATYPE",
        type=str,
        choices=["wav", "mel", "mfcc"],
        help="Input data representation"
    )

    parser.add_argument(
        "SAMPLE_RATE",
        type=int,
        help="Audio sample rate"
    )

    parser.add_argument(
        "SAMPLE_DURATION",
        type=int,
        help="Audio duration in seconds"
    )

    parser.add_argument(
        "BATCH_SIZE",
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "PREPROCESSED_ROOT_DIR",
        type=str,
        help="Preprocessed dataset subdirectory, this is the same as output directory used in dataset_preprocess.py"

    )

    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the model configuration to train, in the models subdirectory"

    )
  

    args = parser.parse_args()
    return args


args = parse_args()

def main(args):

    DATASET = args.DATASET
    DATATYPE = args.DATATYPE
    SAMPLE_RATE = args.SAMPLE_RATE
    SAMPLE_DURATION = args.SAMPLE_DURATION
    BATCH_SIZE = args.BATCH_SIZE
    PREPROCESSED_ROOT_DIR = args.PREPROCESSED_ROOT_DIR    
    model_name = args.model_name
    

    #Running utility functions to get the dataset path and shape of the data (duration, sample size and data representation all factors)
    DATASET_PATH = get_dataset_path(DATATYPE, PREPROCESSED_ROOT_DIR, DATASET)
    root_path = os.path.join(os.getcwd(), DATASET_MAP[DATASET.lower()])
    df_train = pd.read_csv(os.path.join(root_path, "train.csv"))
    INPUT_SHAPE = get_input_shape(DATASET, DATATYPE, DATASET_PATH, df_train, SAMPLE_RATE, SAMPLE_DURATION)
  
    #Creating the tensorflow dataset objects, and building the model, passing the training data to normalise
    train_ds, val_ds, test_ds, INPUT_SHAPE = create_tf_datasets(DATASET, DATATYPE, SAMPLE_RATE, SAMPLE_DURATION, BATCH_SIZE, PREPROCESSED_ROOT_DIR)


    # Data integrity check 
    for spect, label in train_ds.unbatch().take(3):
        print("Data shape:", spect.shape)
        print("Label:", label.numpy())
        print("Snippet:", spect.numpy())
    print("datasets created succesfully")

    # turn the string into a callable
    model_name = getattr(model_specs, model_name)
    model, model_callbacks, epoch_count = model_name(INPUT_SHAPE, train_ds) 
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer = opt,loss='sparse_categorical_crossentropy',
                metrics=['accuracy']) 
    print("Model compiled successfully")
    #Quick test
    cw = class_weight_maker(train_ds)
    history = model.fit(train_ds, 
                 epochs=epoch_count,
                 validation_data=val_ds,callbacks=[model_callbacks], class_weight = cw)

    print("quick test complete: ",model.evaluate(test_ds))

    #Saving weights and training history 
    filename = str(DATASET) + "_" + str(DATATYPE)
    os.makedirs("weights", exist_ok=True)
    model.save(os.path.join(os.getcwd(), "weights", filename + "_weights.keras"))

    os.makedirs("training_history", exist_ok=True)
    with open((os.path.join(os.getcwd(), "training_history", filename + "_history")), 'w') as f:
        json.dump(history.history, f)




    #Generate normalised confusion matrices
    os.makedirs("confusion_matrices", exist_ok=True)
    y_true = []
    y_pred = []
    for x, y in test_ds:
        preds = model.predict(x,verbose=0)  
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    label_map = LABEL_MAP[DATASET]
    cm = confusion_matrix(y_true, y_pred, labels= range(len(label_map)))



    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=label_map)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("SAVEE MFCC Optimised model 2", fontsize=16)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    disp.plot(cmap="Blues",ax=ax)
    plt.savefig((os.path.join(os.getcwd(), "confusion_matrices", filename + "_cm.png", )), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main(args)

