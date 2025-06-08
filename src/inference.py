import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from src.model import TinyQwenVL  # wherever you defined your class

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# paths or model IDs
SIGLIP_MODEL = "google/siglip-so400m-patch14-224"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"

# load model
model = TinyQwenVL(
    vision_dim=1152,
    lang_dim=896,
    num_queries=256,
    freeze_vision=False,
    freeze_text=False,
    torch_dtype=torch.float32,
).to(device)

# load checkpoint

model.eval()

# load processor & tokenizer
processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)

# 2. Prepare inputs
# — image
image = Image.open("../data/images/untitled.jpg").convert("RGB")
vision_inputs = processor(images=image, return_tensors="pt")
pixel_values = vision_inputs.pixel_values.to(device)       # [1, 3, H, W]

# — text prompt
prompt = "Describe the scene in the image:" 
text_inputs = tokenizer(prompt, return_tensors="pt")
input_ids     = text_inputs.input_ids.to(device)           # [1, seq_len]
attention_mask= text_inputs.attention_mask.to(device)      # [1, seq_len]

# 4. Greedy generation loop
#    — we’ll append one token at a time
generated_ids = input_ids
for step in range(50):
    # rebuild attention mask: ones for existing tokens
    cur_attn = torch.ones(1, model.num_queries + 2 + generated_ids.shape[1], device=device)
    with torch.no_grad():
        outs = model(
            pixel_values=pixel_values,
            input_ids=generated_ids,
            attention_mask=cur_attn
        )
    next_logits = outs[:, -1, :]                     # [1, vocab_size]
    next_token  = next_logits.argmax(dim=-1).unsqueeze(-1)  # [1,1]
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    if next_token.item() == tokenizer.eos_token_id:
        break

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("→", generated_text)
