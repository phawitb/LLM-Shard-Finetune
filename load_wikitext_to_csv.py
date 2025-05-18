import os
import pandas as pd
from datasets import load_dataset

# === Configuration ===
CSV_PATH = "wikitext_small.csv"
NUM_SAMPLES = 100  

# === Load dataset ===
if os.path.exists(CSV_PATH):
    print(f"Loading dataset from local CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
else:
    print("Downloading dataset from Hugging Face (wikitext-2-raw-v1)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = [text.strip() for text in dataset["text"] if len(text.strip()) > 0][:NUM_SAMPLES]

    df = pd.DataFrame({"text": texts})
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} samples to {CSV_PATH}")

# === Preview ===
print("Dataset preview:")
print(df.head())
