import os
from huggingface_hub import snapshot_download, hf_hub_download

#TODO support more models with argparse

# # original model weights
# snapshot_download(
#     repo_id="meta-llama/Llama-2-7b",
#     token=TOKEN)

# converted GGUF
snapshot_download(
    repo_id="TheBloke/Llama-2-70B-GGUF",
    local_dir="/root/llama_models/Llama-2-70B-GGUF",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=32,
)
