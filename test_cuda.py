import torch

print("\n")
print("-" * 50)
print(f"torch version: {torch.__version__}")
print(f"Is AI models using GPU?:  {torch.cuda.is_available()}")
print("-" * 50)
print("\n")