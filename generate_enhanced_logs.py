#!/usr/bin/env python

"""
This script takes an LSTM model for predicting the next event in a case and outputs enhanced event logs by generating additional traces using the LSTM model.

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
from datetime import datetime, timedelta
from collections import defaultdict


input_filename = "event_log.csv"
lstm_weights_file = "model_weights.01-0.76.hdf5"
enhanced_log_template = "event_log_enhanced%s.csv"

added_traces_ratios = [0, 1, 2] # Each ratio generates a new log file containing both the existing and generated traces.
# 0 - only existing traces
# 1 - equal proportion of existing and generated traces (proportion of existing and generated traces is 1:1)
# 2 - proportion of existing and generated traces is 1:2 

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
timestamp_format = '%Y/%m/%d %H:%M:%S.%f'
start_event = "START"
end_event = "END"

# LSTM parameters
lstmsize = 48
dropout = 0.2
optim = 'rmsprop'
activation = 'softmax'
loss = 'categorical_crossentropy'
nb_epoch = 10
batch_size = 64
validation_ratio = 0.2


def get_event_as_onehot(event_idx):
    event = np.zeros(data_dim)
    event[event_idx] = 1
    return event

def generate_trace():
    event_idx = start_idx
    events = get_event_as_onehot(event_idx)[np.newaxis,:]
    trace = []
    while col_idxs[event_idx] != end_event:# and len(trace) < max_events:
        event_idx = np.random.choice(len(col_idxs), 1, p=model.predict(pad_sequences(events[np.newaxis,:,:], maxlen=time_dim))[0])[0]
        event = get_event_as_onehot(event_idx)
        events = np.vstack([events, get_event_as_onehot(event_idx)])
        trace.append(col_idxs[event_idx])
    return tuple(trace[:-1])


# read original log
data = pd.read_csv(input_filename, sep=";")

# which traces exist in the original log
existing_traces = set()
existing_trace_lengths = defaultdict(int)
grouped = data.groupby(case_id_col)
for name, group in grouped:
    group = group.sort_values(timestamp_col)
    existing_traces.add(tuple(group[activity_col]))
    existing_trace_lengths[len(group)] += 1

# prepare data
cat_data = pd.get_dummies(data[[activity_col]])
dt_final = pd.concat([data[[case_id_col, timestamp_col]], cat_data], axis=1).fillna(0)
dt_final[start_event] = 0
dt_final[end_event] = 0
grouped = dt_final.groupby(case_id_col)
n_existing_traces = len(grouped)

# generate dict of activity idxs
col_idxs = {idx:col.replace("%s_"%activity_col, "") for idx, col in enumerate(cat_data.columns)}
col_idxs[len(col_idxs)] = start_event
col_idxs[len(col_idxs)] = end_event
start_idx = col_idxs.keys()[col_idxs.values().index(start_event)]


# load LSTM model
max_events = grouped.size().max()
data_dim = dt_final.shape[1] - 2
time_dim = max_events + 1

model = Sequential()
model.add(LSTM(lstmsize, input_shape=(time_dim, data_dim)))
model.add(Dropout(dropout))
model.add(Dense(data_dim, activation=activation))
model.compile(loss=loss, optimizer=optim)

model.load_weights(lstm_weights_file)

# generate enhanced logs
for added_trace_ratio in added_traces_ratios:

    n_added_traces = int(n_existing_traces * added_trace_ratio)
    
    with open(enhanced_log_template%added_trace_ratio, "w") as fout:
        fout.write("%s,%s,%s\n"%(case_id_col, activity_col, timestamp_col))
        
        # write existing traces to file
        for row_idx, row in data.iterrows():
            fout.write("%s,%s,%s\n"%(row[case_id_col], row[activity_col], row[timestamp_col]))
        
        # generate new traces
        n_existing = 0
        np.random.seed(22)
        for i in range(n_added_traces):
            trace = generate_trace()
            start_time = datetime.now()
            if trace in existing_traces:
                n_existing += 1
            for event in trace:
                # add a timestamp for the event (not part of the LSTM)
                timestamp = datetime.strftime(start_time + timedelta(days=1), timestamp_format)
                fout.write("%s,%s,%s\n"%("new%s"%(i+1), event, timestamp))
    print("Total added: %s, # generated traces of existing variants (present in original event log): %s, # generated traces of new variants: %s"%(n_added_traces, n_existing, n_added_traces - n_existing))
