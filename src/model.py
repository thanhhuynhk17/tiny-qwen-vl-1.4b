import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM
from torch.nn.init import trunc_normal_

from src.utils.rope2d import get_rope_2d_angles, apply_2d_rope

SIGLIP_MODEL = "google/siglip-so400m-patch14-224"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"

class PositionAwareAdapter(nn.Module):
    """_summary_
    Query was randomly initialized and learned during training.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, vision_dim=1152, lang_dim=896, num_queries=256, num_heads=16, patch_size=14):
        super().__init__()
        self.vision_dim = vision_dim
        self.lang_dim = lang_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.patch_size = patch_size

        # trainable queries
        self.queries = nn.Parameter(torch.zeros(num_queries, vision_dim))
        trunc_normal_(self.queries, std=0.02)

        self.proj_keys = nn.Linear(vision_dim, vision_dim, bias=False)
        self.proj_values = nn.Linear(vision_dim, vision_dim, bias=False)

        self.attention = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)

        self.proj_out = nn.Linear(vision_dim, lang_dim, bias=False)

        # # Moved to buffer registration in forward to handle potential device issues
        # self.register_buffer("pos_embed_cos", None)
        # self.register_buffer("pos_embed_sin", None)
        # self.register_buffer("key_pos_embed_cos", None)
        # self.register_buffer("key_pos_embed_sin", None)
        self._precomputed_rope = False # Flag to check if RoPE is computed

        self.ln_q = nn.LayerNorm(vision_dim)
        self.ln_kv = nn.LayerNorm(vision_dim)
        
    def forward(self, vision_features):
        device = vision_features.device
        # [B, N, D] B=batch size, N=vision tokens (like 256), D=vision dim (e.g., 1152)
        batch_size, vision_tokens, vision_dim = vision_features.shape

        # Compute RoPE angles only once and register as buffers
        if not self._precomputed_rope:
            with torch.no_grad():
                # Compute pos_embed_cos and pos_embed_sin for queries
                pos_embed_cos, pos_embed_sin = get_rope_2d_angles(vision_dim, int(self.num_queries**0.5))
                # Move to device before registering as buffer
                self.register_buffer("pos_embed_cos", pos_embed_cos.to(device))
                self.register_buffer("pos_embed_sin", pos_embed_sin.to(device))

                # Compute key_pos_embed_cos and key_pos_embed_sin for keys
                if vision_tokens != self.num_queries:
                    key_pos_embed_cos, key_pos_embed_sin = get_rope_2d_angles(vision_dim, int(vision_tokens**0.5))
                    # Move to device before registering as buffer
                    self.register_buffer("key_pos_embed_cos", key_pos_embed_cos.to(device))
                    self.register_buffer("key_pos_embed_sin", key_pos_embed_sin.to(device))
                else:
                    # If vision tokens == num_queries, keys use the same RoPE as queries
                    self.register_buffer("key_pos_embed_cos", self.pos_embed_cos.clone().to(device))
                    self.register_buffer("key_pos_embed_sin", self.pos_embed_sin.clone().to(device))
            self._precomputed_rope = True

        # Project vision features to keys and values
        keys = self.proj_keys(vision_features)
        keys = self.ln_kv(keys)
        keys_rope2d = apply_2d_rope(keys,
                                    self.key_pos_embed_cos.to(dtype=vision_features.dtype),
                                    self.key_pos_embed_sin.to(dtype=vision_features.dtype))

        # repeat query [batch_size] times
        queries = self.ln_q(self.queries)
        queries = queries.unsqueeze(0).repeat(batch_size, 1, 1) # [B, 256, D] # Compressed to 256 vision tokens
        queries_rope2d = apply_2d_rope(
                            queries,
                            self.pos_embed_cos.to(dtype=queries.dtype),
                            self.pos_embed_sin.to(dtype=queries.dtype))

        # Project vision features to values
        values = self.proj_values(vision_features)
        # attention
        attn_out, _ = self.attention(queries_rope2d, keys_rope2d, values)
        out_llm = self.proj_out(attn_out)
        return out_llm

class TinyQwenVL(nn.Module):
    def __init__(self,
            vision_dim=1152,
            lang_dim=896,
            num_queries=256,
            freeze_vision: bool = False,
            freeze_text: bool = False,
            torch_dtype=torch.float32
        ):
        super().__init__()
        self.vision_dim = vision_dim
        self.lang_dim = lang_dim
        self.num_queries = num_queries
        self.torch_dtype = torch_dtype

        # vision encoder
        self.vision_encoder = AutoModel.from_pretrained(
            SIGLIP_MODEL,
        ).to(dtype=torch_dtype)

        # position aware adapter
        self.adapter = PositionAwareAdapter(
            vision_dim=self.vision_dim,
            lang_dim=self.lang_dim,
            num_queries=self.num_queries
        ).to(dtype=torch_dtype)  # force vision part float32 to avoid NaN issue

        # text decoder
        self.text_decoder = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL,
            trust_remote_code=True,
        ).to(dtype=torch_dtype)

        # freeze
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if freeze_text:
            for param in self.text_decoder.parameters():
                param.requires_grad = False

    def forward(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor = None):

        # Step 1: Get vision features
        # image size: 224 x 224, patch: 14
        # -> vision features: (224/14)^2 = 256 tokens
        vision_output = self.vision_encoder.vision_model(pixel_values=pixel_values,
                                                        output_hidden_states=True)
        vision_features = vision_output.last_hidden_state  # (B, 256, vision_dim)

        # Step 2: Adapt vision features
        adapted_features = self.adapter(vision_features)  # (B, 256, lang_dim)

        # Step 3: Insert vision adapted embeddings with text embeddings (if input_ids provided)
        # Get text‐token embeddings: (B, 1 + N_adapter + 1 + seq_len, D)
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        inserted_embeds = self.insert_embed(
            adapted_features,
            text_embeds
        )   # (B, 256 + 2 + seq_len, lang_dim)

        # Step 4: Pass through language model
        outputs = self.text_decoder(inputs_embeds=inserted_embeds,
                                    attention_mask=attention_mask)

        # Return only logits during inference
        return outputs.logits # [B, 256 + 2 + seq_len, vocab_size]

    def insert_embed(
        self,
        adapted_features: torch.Tensor,
        text_embeds: torch.Tensor):
        """
        Replace all padding tokens for image with actually `adapted_features`
        (from the PositionAwareAdapter) on the text embeddings.

        Args:
            adapted_features: (batch_size, N_adapter, D)              — tensor from adapter
            text_embeds:      (batch_size, N_adapter + 2 + seq_len)   — tensor from text decoder

        - seq_len: text tokens (eos_token included)

        Returns:
            inserted_embeddings:    (batch_size, N_adapter + 2 + seq_len, D)
        """
        batch_size, N_adapter, D = adapted_features.shape

        # Insert to text_embeds
        text_embeds[:,1:N_adapter+1,:] = adapted_features
        return text_embeds
    
    def generate(self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        max_new_tokens=32,
        num_beams=5,
        num_return_sequences=1,
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=1.0,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs):
        """
        Generate text based on visual and textual inputs.

        Args:
            input_ids (torch.Tensor): The tokenized input text.
            attention_mask (torch.Tensor, optional): The attention mask for the input text.
            pixel_values (torch.Tensor, optional): The pixel values of the input image.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 32.
            num_beams (int, optional): Number of beams for beam search. Defaults to 5.
            num_return_sequences (int, optional): The number of independently computed returned sequences for each element in the batch. Defaults to 1.
            do_sample (bool, optional): Whether to use sampling; use greedy decoding otherwise. Defaults to False.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to None.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to None.
            temperature (float, optional): The value used to modulate the next token probabilities. Defaults to 1.0.
            pad_token_id (int, optional): The id of the padding token.
            eos_token_id (int, optional): The id of the end-of-sequence token.
            **kwargs: Additional keyword arguments passed to the text_decoder.generate method.

        Returns:
            torch.Tensor: The generated token ids.
        """
        # Step 1: Get vision features
        vision_output = self.vision_encoder.vision_model(pixel_values=pixel_values,
                                                        output_hidden_states=True)
        vision_features = vision_output.last_hidden_state  # (B, 256, vision_dim)

        # Step 2: Adapt vision features
        adapted_features = self.adapter(vision_features)  # (B, 256, lang_dim)

        # Step 3: Insert vision adapted embeddings with text embeddings
        text_embeds = self.text_decoder.get_input_embeddings()(input_ids)
        inserted_embeds = self.insert_embed(adapted_features, text_embeds)  # (B, 256 + 2 + seq_len, lang_dim)

        # Step 4: Prepare inputs for generation
        model_inputs = {
            "inputs_embeds": inserted_embeds,
            "attention_mask": attention_mask
        }

        # Step 5: Call the text decoder's generate method with all parameters
        return self.text_decoder.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
