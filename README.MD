# Tiny-Qwen-VL Model (v1.4B)

- Total parameters: 1,380,450,994
- Model size: 2633.00 MB (2.57 GB)

This repository and its contents are provided for academic and research purposes only.

## Project Directory Structure

Below is the directory structure of the project:

<pre>
data/
    └── images/
        ├── dog_and_girl.jpeg
        ├── frame1.jpg
        ├── meowselfie.jpg
        └── untitled.jpg
notebooks/
    ├── rope2d_vs_abs_pos_embed.ipynb
    └── tinyqwenvl-1-4b.ipynb
src/
    ├── utils/
        ├── __init__.py
        ├── abs_pos_embed.py
        ├── rope2d.py
        └── vision_utils.py
    ├── __init__.py
    ├── inference.py
    ├── model.py
    ├── preprocess.py
    └── train.py
.gitignore
README.MD
requirements.txt
</pre>

## Backbone Information

This repository hosts the checkpoint for the Tiny-Qwen-VL model, version 1.4B. The model architecture consists of the following components:

- **Vision Encoder**: based on the `google/siglip-so400m-patch14-224` architecture with vision embeddings of size 1152.

- **Position-Aware Adapter**: Utilizes learnable queries with a fixed length of 256, combined with a 2D Rotary Positional Embedding (RoPE) implementation to effectively encode positional information.

- **Text Decoder**: Utilizes the Qwen/Qwen2.5-0.5B architecture with text embeddings dimension of 896.

These components work in synergy to enable efficient multimodal learning and inference.

## Checkpoint

To download the pre-trained checkpoint, use the provided `hf_download_ckpt.py` script.

```bash
python hf_download_ckpt.py
```

## Important Notice

The `tinyqwenvl-1-4b.ipynb` notebook currently supports only training stage 1. Training stage 2 will be added in a future update. Please check back for the latest updates.

## Inference

```bash
python src/inference.py
```

Output:

```bash

```
