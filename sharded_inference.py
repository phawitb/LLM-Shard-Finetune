
import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from shard_utils import split_model
import math

# === Config ===
MODEL_PATH = "./distilgpt2_finetuned"
NUM_SHARDS = 5
DEVICE = "cpu"
MAX_GEN = 30
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.8

# === Local implementation of top_k_top_p_filtering ===
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, filter_value), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for b in range(logits.size(0)):
            indices = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices] = filter_value
    return logits

# === Load tokenizer and model ===
print("📦 Loading fine-tuned model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# === Shard the model ===
wte, wpe, drop, blocks_device0 = split_model(model, "device0", num_shards=NUM_SHARDS)
blocks_device1 = split_model(model, "device1", num_shards=NUM_SHARDS)
blocks_device2 = split_model(model, "device2", num_shards=NUM_SHARDS)
blocks_device3 = split_model(model, "device3", num_shards=NUM_SHARDS)
blocks_device4, ln_f, lm_head = split_model(model, "device4", num_shards=NUM_SHARDS)

# === Generate function (sampling + perplexity) ===
def generate_sample(prompt, max_new_tokens=MAX_GEN):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()
    loss_total = 0.0
    token_count = 0

    for _ in range(max_new_tokens):
        position_ids = torch.arange(0, generated.size(1)).unsqueeze(0)

        with torch.no_grad():
            hidden_states = wte(generated) + wpe(position_ids)
            hidden_states = drop(hidden_states)
            for block in blocks_device0:
                hidden_states = block(hidden_states)[0]
            for block in blocks_device1:
                hidden_states = block(hidden_states)[0]
            for block in blocks_device2:
                hidden_states = block(hidden_states)[0]
            for block in blocks_device3:
                hidden_states = block(hidden_states)[0]
            for block in blocks_device4:
                hidden_states = block(hidden_states)[0]
            hidden_states = ln_f(hidden_states)

            logits = lm_head(hidden_states[:, -1, :] / TEMPERATURE)

            # Sampling
            filtered_logits = top_k_top_p_filtering(logits, top_k=TOP_K, top_p=TOP_P)
            next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            # Calculate loss for perplexity
            label = next_token_id.view(-1)
            log_prob = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_prob, label, reduction='mean')
            loss_total += loss.item()
            token_count += 1

            generated = torch.cat([generated, next_token_id], dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    avg_loss = loss_total / token_count if token_count > 0 else float("inf")
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return text, avg_loss, perplexity

# === Sample prompts ===
sample_prompts = [
    "The future of AI is",
    "Edge devices can",
    "Federated learning helps",
    "Machine learning on Raspberry Pi"
]

# === Run inference ===
print("🧠 Sharded Inference with Sampling & Perplexity (Fixed)")
for prompt in sample_prompts:
    output, loss, ppl = generate_sample(prompt)
    print(f"📝 Prompt: {prompt}")
    print(f"➡️ Output: {output}")
    print(f"📉 Per-token Loss: {loss:.4f} | 🤖 Perplexity: {ppl:.2f}\n")
