# Tiny-Qwen-VL Model (v1.4B)

- Total parameters: 1,380,450,994
- Model size: 2633.00 MB (2.57 GB)

This repository and its contents are provided for academic and research purposes only.

## Backbone Information

This repository hosts the checkpoint for the Tiny-Qwen-VL model, version 1.4B. The model architecture consists of the following components:

- **Vision Encoder**: based on the `google/siglip-so400m-patch14-224` architecture with vision embeddings of size 1152.

- **Position-Aware Adapter**: Utilizes learnable queries with a fixed length of 256, combined with a 2D Rotary Positional Embedding (RoPE) implementation to effectively encode positional information.

- **Text Decoder**: Utilizes the Qwen/Qwen2.5-0.5B architecture with text embeddings dimension of 896.

These components work in synergy to enable efficient multimodal learning and inference.

## Checkpoint

To download the pre-trained checkpoint, ensure you have `gdown` installed. You can install it using pip:

```bash
pip install gdown
```

Then, use the following command:

```bash
gdown --id 1_8HpI6WMNU1wr8ABSlsdFMrEwzvgvvcr -O best_tinyqwenvl_1.4B.pth
```

## Important Notice

The `tinyqwenvl-1-4b.ipynb` notebook currently supports only training stage 1. Training stage 2 will be added in a future update. Please check back for the latest updates.

## Upcoming Features

The next release will introduce `model.generate`, enabling text generation directly from the Tiny-Qwen-VL model. Stay tuned for further details and implementation updates.

