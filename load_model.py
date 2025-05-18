from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

print("Downloading and caching distilgpt2 model...")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

print("Saving to ./distilgpt2_local ...")
model.save_pretrained("./distilgpt2_local")
tokenizer.save_pretrained("./distilgpt2_local")

print("Download complete. Files saved to ./distilgpt2_local")
