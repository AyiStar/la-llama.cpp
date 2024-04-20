import os
from huggingface_hub import snapshot_download, hf_hub_download

#TODO support more models with argparse

# TOKEN = os.environ['HF_TOKEN']

# original model weights
# snapshot_download(
#     repo_id="meta-llama/Llama-2-7b",
#     local_dir="/home/loongson/llama_models/Llama-2-7B-GGUF",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     token=TOKEN,
# )


# converted GGUF
# snapshot_download(
#     repo_id="TheBloke/Llama-2-7B-GGUF",
#     local_dir="/home/loongson/llama_models/Llama-2-7B-GGUF",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     max_workers=1,
# )

for quant_type in ['Q4_0', 'Q4_K_S', 'Q4_K_M']:
    hf_hub_download(
        repo_id="TheBloke/Llama-2-7B-GGUF",
        filename=f"llama-2-7b.{quant_type}.gguf",
        local_dir="/home/loongson/llama_models/Llama-2-7B-GGUF",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
