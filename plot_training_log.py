import pandas as pd
import matplotlib.pyplot as plt
import os

# === Path to training log CSV ===
LOG_PATH = "./distilgpt2_finetuned/training_log.csv"
SAVE_DIR = os.path.dirname(LOG_PATH)

# === Check file exists ===
if not os.path.exists(LOG_PATH):
    raise FileNotFoundError(f"Log file not found: {LOG_PATH}")

# === Load data ===
df = pd.read_csv(LOG_PATH)

# === Plot Loss Curve ===
plt.figure(figsize=(10, 6))  # ใหญ่ขึ้น
plt.plot(df["epoch"], df["avg_loss"], marker='o', linewidth=2, label="Average Loss")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss per Epoch", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
loss_path = os.path.join(SAVE_DIR, "loss_curve.png")
plt.savefig(loss_path, dpi=300)  # บันทึกแบบชัด
print(f"Saved loss curve to {loss_path}")
plt.show()

# === Plot Duration Curve (if available) ===
if "duration_sec" in df.columns:
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["duration_sec"], marker='s', color='orange', linewidth=2, label="Time per Epoch (sec)")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Duration (sec)", fontsize=12)
    plt.title("Training Time per Epoch", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    time_path = os.path.join(SAVE_DIR, "duration_curve.png")
    plt.savefig(time_path, dpi=300)
    print(f"Saved duration curve to {time_path}")
    plt.show()
