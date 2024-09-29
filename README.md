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

## Usage
1. **Training**: Run `python date_converter.py` to train and evaluate the model.
2. **Inference**: Predicts new date formats from input and saves results in `answer_rohit.xlsx`.

## Results
The trained model is saved as `seq2seq_date_converter_rohit.pth`, with **96% accuracy**.
