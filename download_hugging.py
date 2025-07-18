import os
from huggingface_hub import snapshot_download
# 设置HTTP代理
os.environ['http_proxy'] = 'http://10.249.36.23:8243'

# 设置HTTPS代理
os.environ['https_proxy'] = 'http://10.249.36.23:8243'
snapshot_download(
    repo_id="GD-ML/FLUX-Text",  # 模型名称
    local_dir="weights",  # 本地保存路径
)