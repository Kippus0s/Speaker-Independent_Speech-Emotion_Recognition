import opensmile
import audb
import audformat
import os
import numpy as np
import pandas as pd
import sys 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataset_preprocess import DATASET_SPEAKER_DEFAULTS, get_dataset_paths

"""
This script exctracts emobase features using opensmile from the raw sound files, and trains a logistic regression classifier using them

Useage:
python train_baselines.py <dataset> <sample_rate> <sample_duration>
Arguments

dataset (sys.argv[1])
The name of the dataset you want to process.
Must correspond to a dataset handled in get_dataset_paths() and DATASET_SPEAKER_DEFAULTS.
Valid datasets: emodb, savee, iemocap, ravdess 

sample_rate (sys.argv[2])
The audio sample rate in Hz to use for feature extraction.
Passed directly to opensmile and audformat.Media.
Example: 16000

sample_duration (sys.argv[3])
The length of audio (in seconds) to process from each file.
Should be a string that will concatenate "s" in smile.process_files.
Example: "4" → extracts 4 seconds of audio
"""



dataset = sys.argv[1]
dataset_name, data_path, dataset_path, csv_path = get_dataset_paths(dataset)
dataset_path = os.path.normpath(dataset_path)
df = pd.read_csv(csv_path)
SAMPLE_RATE = sys.argv[2]
SAMPLE_DURATION = sys.argv[3]
#Loading some files from EmoDB note the path doesnt need EmoDB/archive I guess because "emodb" is the first parameter

def extract_features(dataset_name, df, SAMPLE_RATE):
    db = audformat.Database(name=dataset, usage=audformat.define.Usage.UNRESTRICTED)
    # Populate audb dataset, columns become audb database schemes, with proper indexes
    file_index = pd.Index(df['file'], name="file")
    for col in df.columns:
        if col == "file":
            continue

        db[col] = audformat.Table(index=file_index)  # <-- same index as files
        db.schemes[col] = audformat.Scheme()
        db[col][col] = audformat.Column(scheme_id=col)
        db[col][col].set(df[col].values)

    # Initialise files table with file paths as index
    file_index = pd.Index(df['file'], name="file")
    db["files"] = audformat.Table(index=file_index)
    db.media["recording"] = audformat.Media(
        format="wav",
        sampling_rate=SAMPLE_RATE,
        channels=1
    )

    
    # Initialising smile class for feature extraction with files and parameters (feature set etc)
    files = db.files
    print("len files", len(files))
    if dataset == "savee":
        file_paths = [os.path.join(dataset_path, f + ".wav") for f in db.files]
    else: 
        file_paths = [os.path.normpath(os.path.join(dataset_path, f)) for f in db.files]
    print("len file_paths", len(file_paths))
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers = 5
    )
    # Extract features from each file
    # After extracting features
    features = smile.process_files(file_paths, ends=[SAMPLE_DURATION+"s"] * len(file_paths),
                                root=db.root)
    print("len features", len(features))

    # Make the file paths a column instead of index
    features = features.reset_index()  # keeps the index as a column called 'file'

    # Now merge
    dfemotion = df[['file', 'emotion']].copy()    
    dfspeaker = df[['file', 'speaker']].copy()
    
    if dataset == "savee":
        dfemotion['file'] = dfemotion['file'].apply(lambda f: os.path.join(dataset_path, f+".wav"))
        dfspeaker['file'] = dfspeaker['file'].apply(lambda f: os.path.join(dataset_path, f+".wav"))        
    else:
        dfemotion['file'] = dfemotion['file'].apply(lambda f: os.path.join(dataset_path, f))
        dfspeaker['file'] = dfspeaker['file'].apply(lambda f: os.path.join(dataset_path, f))    
    
    combined_df = pd.merge(features, dfemotion, on='file')
    combined_df = pd.merge(combined_df, dfspeaker, on='file')
    return combined_df

def data_split(dataset, features_df):
    
    test_speaker = DATASET_SPEAKER_DEFAULTS[dataset]['test_speaker']
    print(test_speaker)
    print(len(features_df['speaker']))

    # Split based on speaker column
    df_test = features_df[features_df['speaker'].isin(test_speaker)].reset_index(drop=True)
    df_train = features_df[~features_df['speaker'].isin(test_speaker)].reset_index(drop=True)

    # Print basic stats
    print(f"Train samples: {len(df_train)}")
    print(f"Test samples (speaker {test_speaker}): {len(df_test)}")

    print("Emotion distribution:")
    print("Full dataset:", features_df['emotion'].value_counts())
    print("Train:", df_train['emotion'].value_counts())
    print("Test:", df_test['emotion'].value_counts())

    return df_train, df_test

def train_logreg(df_train, df_test):    
    # Extract features (drop 'emotion' and 'speaker' etc)
    X_train = df_train.iloc[:, 3:-2].values
    X_test = df_test.iloc[:, 3:-2].values

    # Extract labels
    Y_train = df_train['emotion'].values
    Y_test = df_test['emotion'].values

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    #pca = PCA(n_components=n_components)
    #X_train = pca.fit_transform(X_train)
    #X_test = pca.transform(X_test)

    
    classifier = LogisticRegression(C=1, max_iter=10000)
    classifier.fit(X_train, Y_train)    
    score = classifier.score(X_test, Y_test)
    return score

features_df = extract_features(dataset, df, SAMPLE_RATE)
df_train, df_test = data_split(dataset, features_df)
score = train_logreg(df_train, df_test)    
print(score)


