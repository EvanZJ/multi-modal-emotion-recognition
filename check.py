import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
x = torch.randn(3, 3).cuda()
y = torch.matmul(x, x)  # This triggers cuBLAS
print("✅ Success!")