import torch
print(torch.cuda.is_available())  # 如果输出 True，说明成功认出显卡！
print(torch.version.cuda)         # 这会显示你安装的 PyTorch 自带的 CUDA 版本（比如 12.1）
