import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["HF_HOME"] = "/tmp/hf_cache"


import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from src.model import TinyQwenVL  # wherever you defined your class
from preprocess import collate_fn
import argparse
# Parse arguments
parser = argparse.ArgumentParser(description="Inference script for TinyQwenVL")
parser.add_argument("--best_model_path", type=str, required=True, help="Path to the best model checkpoint")
args = parser.parse_args()

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu" # forcing cpu device
# paths or model IDs
SIGLIP_MODEL = "google/siglip-so400m-patch14-224"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"

# load processor & tokenizer
processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
# add special tokens extended
tokenizer.add_special_tokens({
    'pad_token': '<|endoftext|>',
    # Notes: Adding unknown special tokens is not supported
    # <|extra_0|> as begin image token
    # <|extra_1|> as end image token
    'additional_special_tokens': ['<|extra_0|>', '<|extra_1|>']
})
tokenizer.padding_side='right'


# load model
model = TinyQwenVL(torch_dtype=torch.float32).to(device)
# load checkpoint

best_model_path = args.best_model_path
model.text_decoder.resize_token_embeddings(len(tokenizer))

state_dict = torch.load(best_model_path, map_location=device)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
print("adapter._precomputed_rope: ", model.adapter._precomputed_rope)
if model.adapter._precomputed_rope == True:
    model.adapter._precomputed_rope = False
model.eval()
print(model.get_model_stats())
# 2. Prepare inputs
batch = [{
    "question": "Please carefully observe the image and come up with a caption for the image.",
    "answer": [],
    "image": Image.open("data/images/meowselfie.jpg")
}]

data = collate_fn(batch=batch, 
            processor=processor,
            tokenizer=tokenizer)
input_ids = data["input_ids"].to(device)
pixel_values = data["pixel_values"].to(device)
attention_mask = data["attention_mask"].to(device)

# # remove eos token
# input_ids = input_ids[:,:-1]
# attention_mask = attention_mask[:,:-1]
print(input_ids.shape)
print(pixel_values.shape)
print(attention_mask.shape)
print(f"pixel_values:\n{pixel_values}")
# Greedy generation loop
#    — we’ll append one token at a time

num_return_sequences = 4
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=64,
        num_beams=4,
        do_sample=True,
        top_p=0.95,
        temperature=1.5,
        top_k=40,
        min_p=0,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    print(outputs.shape)

print(f"Prompt:{tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
print(f"Generate {num_return_sequences} caption(s):\n")
for i, seq_i in enumerate(outputs):
    generated_text = tokenizer.decode(seq_i, skip_special_tokens=True)
    print(f"Caption {i+1}:\n{generated_text}")
