import torch, platform

print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device:", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
