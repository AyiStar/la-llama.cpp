import os
import pathlib
from huggingface_hub import snapshot_download, hf_hub_download

base_path = pathlib.Path(__file__).parent.resolve()

repo_id_list = [
    "TinyLlama/TinyLlama_v1.1",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-2-13b",
    "meta-llama/Meta-Llama-3-8B",
    "sydonayrex/Blackjack-Llama3-21B",
    "TheBloke/LLaMA-30b-GGUF"
]

for repo_id in repo_id_list:
    model_name = repo_id.split('/')[1]
    local_dir = base_path / model_name
    if os.path.exists(local_dir):
        print(f'There has been existing model weights for {model_name}, skip')
        continue
    snapshot_download(
        repo_id=repo_id,
        local_dir=base_path / model_name,
        local_dir_use_symlinks=False,
        resume_download=True,
        # token=os.environ['HF_TOKEN'],
    )