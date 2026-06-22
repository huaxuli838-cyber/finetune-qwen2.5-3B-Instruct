from modelscope import snapshot_download

model_id = "qwen/Qwen2.5-3B-Instruct"
cache_dir = "."
print(f"Starting download of {model_id} to {cache_dir}")
path = snapshot_download(model_id, cache_dir=cache_dir)
print(f"Download completed: {path}")
