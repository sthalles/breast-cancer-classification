# breast-cancer-classification

This document accompanies all the codebase developed in the project. Here is an overview of the codebase structure. You can find the complete code at: 


- The file ''Create-dataset.ipynb'', a jupyter notebook, reads, normalizes and implements the bagging training protocol for training the ConvNets~\cite{lecun1999object} using Tensorflow 2.0. \cite{tensorflow20} This script generates tfrecords files required by Tensorflow.
- The files ''fine-tune-config.yaml'' and ''from-scratch-config.yaml'' define the hyperparameters for training the deep learning models. These configurations state which ConvNet should be trained, which dataset should be used, the input format, and some hyperparameters.
- The python script ''train.py'', trains each neural network based on the config files.
- The python script: ''ensemble-eval.py'' read each trained model and evaluates each separately and as an ensemble. It displays all the metrics reported in this document.
- files ''utils.py'' and ''model.py'' contains some helper function to load and preprocess the data.
