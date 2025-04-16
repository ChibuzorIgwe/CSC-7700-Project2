# Language Model Training & Evaluation with Validation Loss (RNN, LSTM, Transformer)

import torch
from tokenizer import load_jsonl, BPEWrapper, TextDataset
from train_and_evaluation import train_model, evaluate_bleu, compute_perplexity, plot_loss_curve_with_validation, evaluate_prompt_response
from lstm_module import LSTMLanguageModel
from rnn_module import RNNLanguageModel
from Transformer_module import TransformerLanguageModel

# === CONFIGURATION ===
train_file = "/scratch/cigweo1/project2_chibuzor/train.jsonl"
test_file = "/scratch/cigweo1/project2_chibuzor/test.jsonl"
tokenizer_model = "/scratch/cigweo1/project2_chibuzor/bpe_tokenizer.model"

# Hyperparameters
vocab_size = 10000
embed_dim = 256
hidden_dim = 512
num_layers = 2
num_heads = 8
max_len = 512
epochs = 30
batch_size = 128
lr = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === LOAD DATA ===
tokenizer = BPEWrapper(tokenizer_model)
train_data = load_jsonl(train_file)
test_data = load_jsonl(test_file)
train_dataset = TextDataset(train_data, tokenizer, max_len=max_len)
test_dataset = TextDataset(test_data, tokenizer, max_len=max_len)

# List of models to train sequentially
model_types = ['rnn', 'lstm', 'transformer']

# === SEQUENTIAL TRAINING ===
for model_type in model_types:
    print(f"\nTraining {model_type.upper()} model...")

    # === MODEL SETUP ===
    if model_type == 'rnn':
        model = RNNLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers)
    elif model_type == 'lstm':
        model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers)
    elif model_type == 'transformer':
        model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, num_layers)

    # === TRAINING WITH VALIDATION ===
    print(f"Training {model_type.upper()} model...")
    loss_history, val_loss_history = train_model(model, train_dataset, val_dataset=test_dataset,
                                                 epochs=epochs, batch_size=batch_size, lr=lr, device=device, model_type=model_type)

    plot_loss_curve_with_validation(loss_history, val_loss_history, model_type)

    # === EVALUATION ===
    ppl = compute_perplexity(model, test_dataset, batch_size, device=device, model_type=model_type)
    bleu = evaluate_bleu(model, test_dataset, batch_size, device=device, model_type=model_type)

    print(f"\n Model Evaluation Metrics:")
    print(f"{'Model':<15} {'Perplexity':<15} {'BLEU Score':<15}")
    print(f"{model_type:<15} {ppl:<15.2f} {bleu:<15.4f}")

    # === PROMPT RESPONSES ===
    prompt1 = "Which do you prefer? Dogs or cats?"
    custom_prompt = "Tell me a story about a lonely robot"
    response1 = evaluate_prompt_response(model, tokenizer, prompt1, device=device)
    response2 = evaluate_prompt_response(model, tokenizer, custom_prompt, device=device)

    print(f"\nPrompt 1: {prompt1}\nResponse: {response1}")
    print(f"\nPrompt 2: {custom_prompt}\nResponse: {response2}")
