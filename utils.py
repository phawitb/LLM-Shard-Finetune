# utils.py
import pickle
import struct

# === TCP tensor send/receive ===

def send_tensor(sock, tensor):
    data = pickle.dumps(tensor)
    length = struct.pack('>I', len(data))  # 4-byte length header
    sock.sendall(length + data)

def recv_tensor(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    data_len = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, data_len)
    return pickle.loads(data)

def recvall(sock, size):
    buf = b''
    while len(buf) < size:
        part = sock.recv(size - len(buf))
        if not part:
            return None
        buf += part
    return buf

# === Model block splitter ===

def split_model(model, device_id, num_shards=5):
    index = int(device_id.replace("device", ""))
    total_blocks = len(model.transformer.h)
    blocks_per_shard = total_blocks // num_shards
    start = index * blocks_per_shard
    end = (index + 1) * blocks_per_shard
    return model.transformer.h[start:end]
