import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------
# Paths
# ---------------------------
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/kalam_brain/kalam_finetune_model"
OUTPUT_PATH = "models/kalam_brain/merged_model"

# ---------------------------
# Load base + adapter
# ---------------------------
print("🧠 Loading base TinyLlama model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("🔗 Loading Kalam LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# ---------------------------
# Merge LoRA → Base
# ---------------------------
print("🧬 Merging LoRA weights into base model...")
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ Merged model saved at: {OUTPUT_PATH}")
