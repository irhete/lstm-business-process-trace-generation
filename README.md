# lstm-business-process-trace-generation
This repository contains scripts for:
* training an LSTM for predicting the next event in a trace: train_LSTM_next_event.py
* generating new traces with this model: generate_enhanced_logs.py

The code is written with Python 2.7 in mind.

Required libraries are:
* pandas
* numpy
* keras

It is recommended to train the LSTM using a GPU, as it will significantly speed up the calculations.
