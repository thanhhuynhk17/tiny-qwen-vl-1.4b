import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" # Can sometimes rate limit, so set to 0 to disable
from huggingface_hub import snapshot_download

def main():
    snapshot_download(
        repo_id = "thanhhuynhk17/tiny-qwen-vl-1.4b",
        local_dir = "./models/checkpoints",
        allow_patterns = ["checkpoints/*.pth"],
    )

if __name__ == "__main__":
    main()