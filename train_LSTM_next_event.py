#!/usr/bin/env python

"""
This script takes an event log in csv format and trains an LSTM model for predicting the next event in a case.

Author: Irene Teinemaa
"""

from keras.models import Sequential
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint


# path to the input dataset
input_filename = "event_log.csv"

# path where the LSTM models (checkpoints) are saved
checkpoint_filepath = "model_weights.{epoch:02d}-{val_loss:.2f}.hdf5"

# LSTM parameters
lstmsize = 48
dropout = 0.2
optim = 'rmsprop'
activation = 'softmax'
loss = 'categorical_crossentropy'
nb_epoch = 10
batch_size = 64
validation_ratio = 0.2

# relevant column names
activity_col = "Activity"
case_id_col = "Case ID"
timestamp_col = "Complete Timestamp"


# read the dataset
data = pd.read_csv(input_filename, sep=";")
data[timestamp_col] = pd.to_datetime(data[timestamp_col])

# one-hot encode the activity
cat_data = pd.get_dummies(data[[activity_col]])
dt_final = pd.concat([data[[case_id_col, timestamp_col]], cat_data], axis=1).fillna(0)

# add dummy columns for case start and case end
dt_final["START"] = 0
dt_final["END"] = 0

# assign model dimensions
grouped = dt_final.groupby(case_id_col)
max_events = grouped.size().max()  # maximum case length
data_dim = dt_final.shape[1] - 2  # our input dataset will contain columns for each activity type, including the dummy start and end activities. We are excluding timestamp and case_id, therefore -2
time_dim = max_events + 1  # +1 comes from adding the artificial start points. We are not considering the end points here, because the LSTM input is one less than the case length (the last training sample for a case predicts the end event)

# generate one-hot vectors representing the dummy endpoints
start = np.zeros(data_dim, dtype=int)
start[-2] = 1
end = np.zeros(data_dim, dtype=int)
end[-1] = 1

print('Constructing LSTM input data...')
X = np.zeros((len(dt_final)+len(grouped), time_dim, data_dim))
y = np.zeros((len(dt_final)+len(grouped), data_dim))
case_idx = 0
for name, group in grouped:
    group = group.sort_values(timestamp_col, ascending=True, kind="mergesort").as_matrix()[:,2:]
    # adding the artificial start and end-points to the case
    group = np.vstack([start, group, end])
    # generate training samples for each prefix of the case, where LSTM input is the prefix and prediction target is the next event
    for i in range(1, len(group)):
        X[case_idx] = pad_sequences(group[np.newaxis,:i,:], maxlen=time_dim)
        y[case_idx] = group[i,:]
        case_idx += 1
        
print('Building model...')
model = Sequential()
model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim)))
model.add(Dropout(dropout))
model.add(Dense(data_dim, activation=activation))
        
print('Compiling model...')
model.compile(loss=loss, optimizer=optim)

print("Training...")
checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True, save_weights_only=True)
model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_split=validation_ratio, callbacks=[checkpointer])
print("Done.")
