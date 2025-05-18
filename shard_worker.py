# shard_worker.py
import sys
import torch
import socket
from transformers import GPT2LMHeadModel
from utils import split_model, send_tensor, recv_tensor

SHARD_INDEX = int(sys.argv[1])
PORT = 9000 + SHARD_INDEX
DEVICE = "cpu"

print(f"🟢 Starting shard {SHARD_INDEX} on port {PORT}")

model = GPT2LMHeadModel.from_pretrained("./distilgpt2_local")
blocks = split_model(model, f"device{SHARD_INDEX}", num_shards=5)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", PORT))
server.listen(1)

while True:
    conn, _ = server.accept()
    with conn:
        try:
            hidden = recv_tensor(conn).to(DEVICE)
            for block in blocks:
                hidden = block(hidden)[0]
            send_tensor(conn, hidden.cpu())
        except Exception as e:
            print(f"❌ [Shard {SHARD_INDEX}] Error: {e}")
