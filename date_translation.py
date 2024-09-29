import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from babel.dates import format_date
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

# Set random seeds for reproducibility
SEED = 15
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define Constants
Tx = 30  # Length of input sequences
Ty = 12  # Length of output sequences (including <sos> and <eos>)
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 128
DECODER_HIDDEN_DIM = 128
ATTENTION_DIM = 64
BIDIRECTIONAL = False
DROPOUT = 0.5

# Define Special Tokens
SOS_TOKEN = '$'
EOS_TOKEN = '^'
UNK_TOKEN = '*'
PAD_TOKEN = '='

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Vocabulary Placeholder (to be initialized later)
machine_vocab = []

# =========================
# Data Processing Functions
# =========================

def read_dataset(file_path):
    dataset = []
    human_vocab = set()
    machine_vocab = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace("'", "")
            human_readable, machine_readable = line.strip().split(', ')
            human_readable = human_readable.strip("'\"").lower()
            machine_readable = f"{SOS_TOKEN}{machine_readable.strip('\"')}{EOS_TOKEN}".lower()
            dataset.append((human_readable, machine_readable))
            human_vocab.update(human_readable)
            machine_vocab.update(machine_readable)
    
    # Add special tokens and sort
    human_vocab = sorted(set([UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(human_vocab)))
    machine_vocab = sorted(set([SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN] + list(machine_vocab)))
    
    # Create dictionaries
    human_vocab = {char: idx for idx, char in enumerate(human_vocab)}
    machine_vocab = {char: idx for idx, char in enumerate(machine_vocab)}
    inv_machine_vocab = {idx: char for char, idx in machine_vocab.items()}
    
    return dataset, human_vocab, machine_vocab, inv_machine_vocab

def read_infer_data(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            human_readable = line.strip().strip("'\"").lower()
            dataset.append(human_readable)
    return dataset

def string_to_int(string, length, vocab):
    string = string.replace(',', '').lower()
    string = string[:length].ljust(length, PAD_TOKEN)
    return [vocab.get(char, vocab[UNK_TOKEN]) for char in string]

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty, infer=False):
    if infer:
        return torch.tensor([string_to_int(seq, Tx, human_vocab) for seq in dataset], dtype=torch.long)
    X, Y = zip(*dataset)
    X_int = [string_to_int(seq, Tx, human_vocab) for seq in X]
    Y_int = [string_to_int(seq, Ty, machine_vocab) for seq in Y]
    return torch.tensor(X_int, dtype=torch.long), torch.tensor(Y_int, dtype=torch.long)

# =====================
# PyTorch Dataset Class
# =====================

class DateDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X[idx], self.Y[idx]
        return self.X[idx]

# ===================
# Model Definitions
# ===================

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers=1, bidirectional=True, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=machine_vocab[PAD_TOKEN])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))  # [batch_size, Tx, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: [batch_size, Tx, hidden_dim * num_directions]
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # [batch_size, hidden_dim * 2]
            cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)      # [batch_size, hidden_dim * 2]
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim, attention_dim=64):
        super(Attention, self).__init__()
        self.attn = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        batch_size, Tx, encoder_hidden_dim = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, Tx, 1)  # [batch_size, Tx, decoder_hidden_dim]
        concat = torch.cat((hidden, encoder_outputs), dim=2)  # [batch_size, Tx, decoder_hidden_dim + encoder_hidden_dim]
        energy = torch.tanh(self.attn(concat))  # [batch_size, Tx, attention_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, Tx]
        attention_weights = self.softmax(attention)  # [batch_size, Tx]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, encoder_hidden_dim]
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, decoder_hidden_dim, encoder_hidden_dim, attention, n_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=machine_vocab[PAD_TOKEN])
        self.lstm = nn.LSTM(embed_dim + encoder_hidden_dim, decoder_hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(decoder_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_dim]
        context, attention_weights = self.attention(hidden, encoder_outputs)  # [batch_size, encoder_hidden_dim]
        context = context.unsqueeze(1)  # [batch_size, 1, encoder_hidden_dim]
        lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embed_dim + encoder_hidden_dim]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        output = output.squeeze(1)  # [batch_size, decoder_hidden_dim]
        prediction = self.fc_out(output)  # [batch_size, output_dim]
        return prediction, hidden.squeeze(0), cell.squeeze(0), attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, machine_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.machine_vocab_size = machine_vocab_size
        self.fc_hidden = nn.Linear(encoder.hidden_dim * (2 if BIDIRECTIONAL else 1), decoder.decoder_hidden_dim)
        self.fc_cell = nn.Linear(encoder.hidden_dim * (2 if BIDIRECTIONAL else 1), decoder.decoder_hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.machine_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        hidden = self.fc_hidden(hidden)  # [batch_size, decoder_hidden_dim]
        cell = self.fc_cell(cell)        # [batch_size, decoder_hidden_dim]

        input = trg[:, 0]  # <sos> token

        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1

        return outputs

# ===========================
# Training and Evaluation
# ===========================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=0.5)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
    print("Training Completed!")

def evaluate_model(model, val_loader, device, inv_machine_vocab, machine_vocab):
    model.eval()
    correct_predictions = 0
    total_predictions = len(val_loader)

    with torch.no_grad():
        for src, trg in tqdm(val_loader, desc="Evaluating"):
            src, trg = src.to(device), trg.to(device)
            encoder_outputs, hidden, cell = model.encoder(src)
            hidden = model.fc_hidden(hidden)
            cell = model.fc_cell(cell)

            input_token = trg[:, 0]
            decoded_tokens = []

            for _ in range(Ty - 1):
                output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
                top1 = output.argmax(1)
                decoded_tokens.append(top1)
                input_token = top1

            predicted_output = ''.join([inv_machine_vocab.get(token.item(), '') for token in decoded_tokens])
            expected_output = ''.join([inv_machine_vocab.get(token.item(), '') for token in trg[0][1:]])
            if predicted_output == expected_output:
                correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Validation Exact Match Accuracy: {accuracy:.2f}%")

# =====================
# Inference Functions
# =====================

def infer(model, src_sentence, human_vocab, machine_vocab, inv_machine_vocab):
    model.eval()
    with torch.no_grad():
        src_int = string_to_int(src_sentence, Tx, human_vocab)
        src_tensor = torch.tensor([src_int], dtype=torch.long).to(DEVICE)

        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        hidden = model.fc_hidden(hidden)
        cell = model.fc_cell(cell)

        input_token = torch.tensor([machine_vocab[SOS_TOKEN]], dtype=torch.long).to(DEVICE)
        decoded_tokens = []
        attention_weights_all = []

        for _ in range(Ty - 1):
            output, hidden, cell, attention_weights = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            if top1.item() == machine_vocab[EOS_TOKEN]:
                break
            decoded_tokens.append(top1.item())
            attention_weights_all.append(attention_weights.cpu().numpy())
            input_token = top1

        decoded_string = ''.join([inv_machine_vocab.get(token, '') for token in decoded_tokens])
    return decoded_string, np.array(attention_weights_all)

def plot_attention_heatmap(src_sentence, tgt_sentence, attention_weights, file_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(attention_weights[:, :len(src_sentence)], cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(src_sentence)))
    ax.set_yticks(np.arange(len(tgt_sentence)))

    ax.set_xticklabels(src_sentence, rotation=90)
    ax.set_yticklabels(tgt_sentence)

    ax.set_xlabel('Source Sentence')
    ax.set_ylabel('Target Sentence')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def evaluate_and_show_attention(model, input_sentence, inv_machine_vocab):
    output, attentions = infer(model, input_sentence, human_vocab, machine_vocab, inv_machine_vocab)
    print(f"Input: {input_sentence}\nPredicted Output: {output}")
    attentions = attentions.reshape(-1, Tx)[:len(output)]
    plot_attention_heatmap(input_sentence, output, attentions, 'AttentionMap.png')

# =====================
# Main Function
# =====================

def main():
    # Paths to the dataset files (Update these paths as necessary)
    TRAIN_FILE = '/data2/home2/pvishal/DL_NLP/NewStart/Assignment2_train.txt'
    VALIDATION_FILE = '/data2/home2/pvishal/DL_NLP/NewStart/Assignment2_validation.txt'
    INFERENCE_FILE = '/home2/pvishal/DL_NLP/experiment/Assignment2_Test.txt'
    MODEL_PATH = '/home2/pvishal/DL_NLP/NewStart/seq2seq_date_converter_rohit.pth'

    # Read and preprocess training and validation data
    train_dataset, human_vocab, machine_vocab_dict, inv_machine_vocab = read_dataset(TRAIN_FILE)
    X_train, Y_train = preprocess_data(train_dataset, human_vocab, machine_vocab_dict, Tx, Ty)
    train_data = DateDataset(X_train, Y_train)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataset, _, _, _ = read_dataset(VALIDATION_FILE)
    X_val, Y_val = preprocess_data(validation_dataset, human_vocab, machine_vocab_dict, Tx, Ty)
    val_data = DateDataset(X_val, Y_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    # Update global machine_vocab
    global machine_vocab
    machine_vocab = machine_vocab_dict

    # Model instantiation
    input_dim = len(human_vocab)
    output_dim = len(machine_vocab)
    embed_dim = 64
    encoder_hidden_dim = HIDDEN_DIM * (2 if BIDIRECTIONAL else 1)

    encoder = Encoder(input_dim, embed_dim, HIDDEN_DIM, bidirectional=BIDIRECTIONAL, dropout=DROPOUT).to(DEVICE)
    attention = Attention(DECODER_HIDDEN_DIM, encoder_hidden_dim, ATTENTION_DIM).to(DEVICE)
    decoder = Decoder(output_dim, embed_dim, DECODER_HIDDEN_DIM, encoder_hidden_dim, attention, n_layers=1, dropout=DROPOUT).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE, output_dim).to(DEVICE)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=machine_vocab[PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training
    train_model(model, train_loader, optimizer, criterion, EPOCHS, DEVICE)

    # Validation
    evaluate_model(model, val_loader, DEVICE, inv_machine_vocab, machine_vocab)

    # Save the Model
    torch.save(model.state_dict(), 'seq2seq_date_converter_rohit.pth')

    # Inference Example
    example_input = "23 January 01"
    predicted_date, _ = infer(model, example_input, human_vocab, machine_vocab, inv_machine_vocab)
    print(f"Input: {example_input}\nPredicted Output: {predicted_date}")

    # Plot Attention for Example
    evaluate_and_show_attention(model, example_input, inv_machine_vocab)

    # Inference on New Data
    inference_dataset = read_infer_data(INFERENCE_FILE)
    X_infer = preprocess_data(inference_dataset, human_vocab, machine_vocab, Tx, Ty, infer=True)
    infer_data = DateDataset(X_infer)
    infer_loader = DataLoader(infer_data, batch_size=1, shuffle=False)
    
    def perform_inference(model, infer_loader, inv_machine_vocab):
        model.eval()
        answers = []
        with torch.no_grad():
            for src in tqdm(infer_loader, desc="Performing Inference"):
                src = src.to(DEVICE)
                encoder_outputs, hidden, cell = model.encoder(src)
                hidden = model.fc_hidden(hidden)
                cell = model.fc_cell(cell)

                input_token = torch.tensor([machine_vocab[SOS_TOKEN]], dtype=torch.long).to(DEVICE)
                decoded_tokens = []

                for _ in range(Ty - 1):
                    output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
                    top1 = output.argmax(1)
                    if top1.item() == machine_vocab[EOS_TOKEN]:
                        break
                    decoded_tokens.append(top1.item())
                    input_token = top1

                predicted_output = ''.join([inv_machine_vocab.get(token, '') for token in decoded_tokens])
                answers.append(predicted_output.rstrip(EOS_TOKEN))
        
        df = pd.DataFrame(answers)
        df.to_excel('answer_rohit.xlsx', header=None, index=False)
        print("Inference completed and results saved to 'answer_rohit.xlsx'.")

    perform_inference(model, infer_loader, inv_machine_vocab)

if __name__ == '__main__':
    main()
