# Stochastic Push-Forward Model Repository

## Overview

This repository implements a **Stochastic Push-Forward Model** framework for time-series prediction tasks. It integrates a convolutional autoencoder (CAE) with an LSTM-based sequence-to-sequence (Seq2Seq) model and provides tools for efficient data handling, training, and evaluation.

---

## Repository Structure

### **1. `train.py`**
This file contains the core pipeline, encapsulated in the `StochasticPushForward` class, which handles:

- Loading training and test datasets.
- Training an LSTM model with early stopping.
- Recursive transformation of time-series data.
- Evaluation using metrics like accumulated errors and SSIM.

Key methods:
- `load_data(path, batch_size)`
- `load_test_data(path)`
- `train_lstm_with_stop_condition(...)`
- `loop_prediction(...)`
- `process_and_save_transformed_data(depth)`

---

### **2. `dataset.py`**
Contains dataset classes for managing time-series data:

- **`SingleSourceDataset`**:
  - Extracts input-output pairs from a single time-series dataset.
  - Configurable lookback and lookahead parameters.

- **`MultiSourceDataset`**:
  - Handles two datasets with configurable sampling probabilities.
  - Alternates sampling between datasets for balanced training.

---

### **3. `models.py`**
Defines the neural network models used in the framework:

- **`Encoder`**: Compresses input data using convolutional layers and a dense layer.
- **`Decoder`**: Reconstructs input data from the encoded latent representation.
- **`CAE`**: Combines the encoder and decoder for unsupervised learning of compact representations.
- **`Seq2Seq`**: Implements an LSTM-based sequence prediction model.

---

## Contact

Hao Zhou - haozhou0713@outlook.com / h40.zhou@hdr.qut.edu.au

Sibo Cheng - sibo.cheng@imperial.ac.uk