from huggingface_hub import snapshot_download

local_path = snapshot_download(repo_id="Qwen/Qwen3-4B")
print(local_path)