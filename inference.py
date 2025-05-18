# inference_builtin.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# === Config ===
MODEL_DIR = "./distilgpt2_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 100

# === Load model & tokenizer ===
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(DEVICE)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# === Inference Function ===
def generate_from_prompts(prompts, max_length=MAX_LENGTH):
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_k=40,
                top_p=0.92,
                temperature=0.7,
                repetition_penalty=1.1,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(generated_text)
    return results

# === Example Prompts ===
if __name__ == "__main__":
    test_prompts = [
        "Artificial intelligence is transforming industries",
        "The future of education will be",
        "Climate change is",
        "Self-driving cars rely on",
        "Space exploration enables",
        "Healthcare is being revolutionized by",
        "Social media has",
        "Blockchain technology is used for",
        "Quantum computing may solve",
        "Ethical concerns in AI involve"
    ]

    outputs = generate_from_prompts(test_prompts)

    for i, (inp, out) in enumerate(zip(test_prompts, outputs), 1):
        print("=" * 60)
        print(f"[{i}] INPUT : {inp}")
        print(f"[{i}] OUTPUT: {out}")
