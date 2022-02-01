import torch
from py3nvml.py3nvml import *
import gc


def get_cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = c - a  # free inside cache
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f"\ntotal    : {info.total/1000000} * 10^6")
    print(f"free     : {info.free/1000000} * 10^6")
    print(f"used     : {info.used/1000000} * 10^6")


print(get_cuda_memory_info())
gc.collect()
torch.cuda.empty_cache()
print(get_cuda_memory_info())


print("\n")
print("-" * 50)
print(f"torch version: {torch.__version__}")
print(f"Is AI models using GPU?:  {torch.cuda.is_available()}")
print("-" * 50)
print("\n")
