import os
import socket
import torch
import pandas as pd
from torch import nn, optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import send_tensor, recv_tensor

# === Config ===
MODEL_ID = "distilgpt2"
MODEL_DIR = "./distilgpt2_local"
CSV_PATH = "wikitext_small.csv"
SAVE_DIR = "./distilgpt2_finetuned"
NUM_SHARDS = 5
EPOCHS = 10
BATCH_SIZE = 4
MAX_SEQ_LEN = 64

# === Load tokenizer and model ===
if os.path.exists(MODEL_DIR):
    print("Loading distilgpt2 from local cache...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
else:
    print("Downloading distilgpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

tokenizer.pad_token = tokenizer.eos_token

# === Load dataset from CSV ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
all_texts = df["text"].tolist()

# === Extract embedding and final layers ===
wte = model.transformer.wte
wpe = model.transformer.wpe
drop = model.transformer.drop
ln_f = model.transformer.ln_f
lm_head = model.lm_head

# === Optimizer and loss ===
trainable_params = list(ln_f.parameters()) + list(lm_head.parameters())
optimizer = optim.Adam(trainable_params, lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

# === TCP Shard Communication ===
def run_shard(port, tensor):
    try:
        print(f"[INFO] Connecting to shard on port {port}...")
        with socket.create_connection(("localhost", port), timeout=10) as sock:
            print(f"[INFO] Connected to port {port}. Sending tensor with shape {list(tensor.shape)}")
            send_tensor(sock, tensor.cpu())
            result = recv_tensor(sock)
            print(f"[INFO] Received tensor from port {port} with shape {list(result.shape)}")
            return result
    except Exception as e:
        print(f"[ERROR] Failed to communicate with shard {port}: {e}")
        exit(1)

# === Training loop ===
model.train()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    total_loss = 0.0

    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = inputs["input_ids"]
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1)

        hidden_states = wte(input_ids) + wpe(position_ids)
        hidden_states = drop(hidden_states)

        for shard_id in range(NUM_SHARDS):
            port = 9000 + shard_id
            hidden_states = run_shard(port, hidden_states)

        hidden_states = ln_f(hidden_states)
        logits = lm_head(hidden_states[:, :-1, :])
        labels = input_ids[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\n Batch {i // BATCH_SIZE + 1}: Loss = {loss.item():.4f} {'-' * 20}")


        total_loss += loss.item()

    avg_loss = total_loss / (len(all_texts) // BATCH_SIZE)
    print(f"\n Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f} {'='*20}")

# === Save fine-tuned model ===
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\n Model saved to {SAVE_DIR}")
