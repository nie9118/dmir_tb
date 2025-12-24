import torch
print("PyTorch版本:", torch.__version__)
print("是否支持ROCm:", torch.cuda.is_available())  # 关键：AMD GPU上应返回True
print("设备数量:", torch.cuda.device_count())
print("当前设备:", torch.cuda.get_device_name(0))