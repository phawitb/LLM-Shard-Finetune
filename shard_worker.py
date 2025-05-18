# shard_worker.py
import sys
import socket
import torch
from transformers import GPT2LMHeadModel
from utils import split_model, send_tensor, recv_tensor

# === Config ===
SHARD_INDEX = int(sys.argv[1])
PORT = 9000 + SHARD_INDEX
DEVICE = "cpu"
MODEL_PATH = "./distilgpt2_local"

# === Helper: get local IP address ===
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# === Load model blocks for this shard ===
print(f"[SHARD {SHARD_INDEX}] Loading model from {MODEL_PATH}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
blocks = split_model(model, f"device{SHARD_INDEX}", num_shards=5)

# === TCP Server Setup ===
ip_address = get_local_ip()
print(f"[SHARD {SHARD_INDEX}] Ready on {ip_address}:{PORT}")
print(f"[SHARD {SHARD_INDEX}] Listening for tensor input...")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((ip_address, PORT))
server.listen(1)

while True:
    conn, addr = server.accept()
    print(f"[SHARD {SHARD_INDEX}] Connection from {addr}")
    with conn:
        try:
            hidden = recv_tensor(conn).to(DEVICE)
            print(f"[SHARD {SHARD_INDEX}] Received tensor with shape: {list(hidden.shape)}")

            for block in blocks:
                hidden = block(hidden)[0]

            send_tensor(conn, hidden.cpu())
            print(f"[SHARD {SHARD_INDEX}] Sent processed tensor back")
        except Exception as e:
            print(f"[SHARD {SHARD_INDEX}] ERROR: {e}")
