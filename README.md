# Human-to-Machine Date Translation Using Attention-Based Deep Learning

## Overview
This project implements a Seq2Seq model with attention in PyTorch to convert human-readable dates (e.g., "23 January 01") to machine-readable format (e.g., "2001-01-23"). The model achieved **96% accuracy** on the validation set.

## Key Features
- **Encoder**: LSTM-based encoder processes input sequences.
- **Attention**: Helps decoder focus on relevant parts of input.
- **Decoder**: LSTM-based decoder generates machine-readable dates.
- **Teacher Forcing**: Used during training for better performance.

## Dataset & Preprocessing
- Converts human-readable dates to machine-readable formats.
- Tokenizes and pads input/output sequences.
  
## Model Hyperparameters
- `BATCH_SIZE`: 64
- `EPOCHS`: 6
- `HIDDEN_DIM`: 128
- `EMBEDDING_DIM`: 64
- `ATTENTION_DIM`: 64
- `LEARNING_RATE`: 0.001

## Results
The trained model achieved **96% accuracy**.
