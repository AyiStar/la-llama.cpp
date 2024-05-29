import os
from huggingface_hub import snapshot_download, hf_hub_download

# original model weights, need token
if 'HF_TOKEN' in os.environ:
    snapshot_download(
        repo_id="meta-llama/Llama-2-7b",
        local_dir="./Meta-Llama-2-7b",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ['HF_TOKEN'],
    )
    
    snapshot_download(
        repo_id="meta-llama/Llama-2-13b",
        local_dir="./Meta-Llama-2-13b",
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ['HF_TOKEN'],
    )
else:
    raise ValueError('Meta LLaMA requires HuggingFace Tokens')