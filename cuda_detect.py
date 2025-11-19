import torch

# Ignore this file, it's just for testing CUDA with PyTorch
def check_cuda_pytorch():
    """
    检查 PyTorch 是否识别并能使用 CUDA。
    """
    print("--- PyTorch CUDA 状态检查 ---")

    # 1. 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print("CUDA is available! PyTorch can use your GPU.")
        
        # 2. 获取 CUDA 设备的数量
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        # 3. 遍历并打印每个 GPU 的信息
        for i in range(num_gpus):
            print(f"--- GPU {i} Information ---")
            # 获取当前 GPU 的名称
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  Device Name: {gpu_name}")
            
            # 获取当前 GPU 的能力 (Compute Capability)
            # 例如 (8,6) 代表 Sm_86
            capability = torch.cuda.get_device_capability(i)
            print(f"  Compute Capability: {capability[0]}.{capability[1]}")
            
            # 获取当前 GPU 的总内存
            total_memory_bytes = torch.cuda.get_device_properties(i).total_memory
            total_memory_gb = total_memory_bytes / (1024**3) # 转换为 GB
            print(f"  Total Memory: {total_memory_gb:.2f} GB")

            # 示例：尝试将一个张量移动到 GPU 上
            try:
                tensor_on_gpu = torch.rand(3, 3).cuda(i)
                print(f"  Successfully moved a tensor to GPU {i}.")
                print(f"  Tensor device: {tensor_on_gpu.device}")
            except Exception as e:
                print(f"  Failed to move a tensor to GPU {i}: {e}")
                
    else:
        print("CUDA is NOT available. PyTorch will use your CPU.")
        print("  Possible reasons: CUDA not installed, NVIDIA drivers not properly configured, or PyTorch not built with CUDA support.")

if __name__ == "__main__":
    check_cuda_pytorch()
