import torch
import time
from typing import List

def occupy_amd_gpu_with_utilization(
    target_mem_percent: int = 80,
    target_util_percent: int = 50,
    gpu_ids: List[int] = None,
    interval: float = 1.0  
):
    if not torch.cuda.is_available():
        return

    
    available_gpus = list(range(torch.cuda.device_count()))
    if not available_gpus:
        return

    gpu_ids = gpu_ids or available_gpus
    gpu_ids = [g for g in gpu_ids if g in available_gpus]
    if not gpu_ids:
        return

    memory_blocks = []
    compute_tensors = []

    for gpu_id in gpu_ids:
        torch.cuda.set_device(gpu_id)
        props = torch.cuda.get_device_properties(gpu_id)
        total_mem = props.total_memory
        total_mem_gb = total_mem / (1024 **3)

        target_mem = total_mem * (target_mem_percent / 100) * 0.9 
        element_size = 4  
        num_elements = int(target_mem / element_size)
        mem_block = torch.empty(
            (num_elements,),
            dtype=torch.float32,
            device=f'cuda:{gpu_id}',
            requires_grad=False
        )
        mem_block.uniform_(-1, 1)  
        memory_blocks.append(mem_block)

        compute_size = 1024 * 10  
        compute_tensor = torch.randn(
            (compute_size, compute_size),
            dtype=torch.float32,
            device=f'cuda:{gpu_id}',
            requires_grad=False
        )
        compute_tensors.append((gpu_id, compute_tensor))


    compute_time = interval * (target_util_percent / 100)
    sleep_time = interval - compute_time

    try:
        while True:
            start = time.time()
            for gpu_id, tensor in compute_tensors:
                torch.cuda.set_device(gpu_id)
                repeats = 10  
                for _ in range(repeats):
                    result = torch.matmul(tensor, tensor)
                torch.cuda.synchronize(gpu_id)
            
            actual_compute = time.time() - start
            adjust_sleep = max(0, sleep_time - (actual_compute - compute_time))
            time.sleep(adjust_sleep)

    except KeyboardInterrupt:
        print("\n程序退出，释放GPU资源")

if __name__ == "__main__":
    # 示例：占用0号和1号AMD GPU，显存80%，利用率50%
    occupy_amd_gpu_with_utilization(
        target_mem_percent=75,
        target_util_percent=50,
        gpu_ids=[4,5,6,7],
        interval=1.0  
    )