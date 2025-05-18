import os
import socket
import time
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

EPOCHS = 20
BATCH_SIZE = 4
MAX_SEQ_LEN = 64
NUM_SHARDS = 5

# === Shard IPs and Ports ===
SHARD_ADDRS = {
    0: ("192.168.1.45", 9000),
    1: ("192.168.1.45", 9001),
    2: ("192.168.1.47", 9002),
    3: ("192.168.1.45", 9003),
    4: ("192.168.1.45", 9004),
}

# === Load tokenizer and model ===
if os.path.exists(MODEL_DIR):
    print("Loading distilgpt2 from local cache...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
else:
    print("Downloading distilgpt2...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

tokenizer.pad_token = tokenizer.eos_token

# === Load dataset ===
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
all_texts = df["text"].tolist()

# === Extract model parts ===
wte = model.transformer.wte
wpe = model.transformer.wpe
drop = model.transformer.drop
ln_f = model.transformer.ln_f
lm_head = model.lm_head

# === Optimizer and loss ===
trainable_params = list(ln_f.parameters()) + list(lm_head.parameters())
optimizer = optim.Adam(trainable_params, lr=5e-4)
loss_fn = nn.CrossEntropyLoss()

# === Log training for CSV ===
training_log = []

# === Function to send/receive from shard ===
def run_shard(shard_id, tensor):
    ip, port = SHARD_ADDRS[shard_id]
    try:
        print(f"[shard {shard_id}] Connecting to shard {shard_id} at {ip}:{port}...")
        with socket.create_connection((ip, port), timeout=10) as sock:
            print(f"[shard {shard_id}] Connected. Sending tensor with shape {list(tensor.shape)}")
            send_tensor(sock, tensor.cpu())
            result = recv_tensor(sock)
            print(f"[shard {shard_id}] Received tensor from shard {shard_id} with shape {list(result.shape)}")
            return result
    except Exception as e:
        print(f"[shard {shard_id}] Failed to communicate with shard {shard_id} at {ip}:{port}: {e}")
        exit(1)

# === Training loop ===
model.train()
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    start_time = time.time()
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
            hidden_states = run_shard(shard_id, hidden_states)

        hidden_states = ln_f(hidden_states)
        logits = lm_head(hidden_states[:, :-1, :])
        labels = input_ids[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"\n Batch {i // BATCH_SIZE + 1}: Loss = {loss.item():.4f} {'-' * 20}")
        total_loss += loss.item()

    duration = time.time() - start_time
    avg_loss = total_loss / (len(all_texts) // BATCH_SIZE)
    print(f"Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}, Time: {duration:.2f} sec")

    training_log.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "duration_sec": round(duration, 2)
    })

    # Save CSV after each epoch
    log_df = pd.DataFrame(training_log)
    log_path = os.path.join(SAVE_DIR, "training_log.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_df.to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")

# === Save fine-tuned model ===
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")
