import os
import torch
import time
from multiprocessing import Process

# 指定需要测试的显卡编号（4、5、6、7）
GPU_IDS = [4, 5, 6, 7]


def gpu_stress_test(gpu_id):
    """对单张显卡进行压力测试，动态调整显存分配"""
    # 强制启用显存扩展段分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    # 隔离目标显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        if not torch.cuda.is_available():
            print(f"显卡 {gpu_id} 不可用")
            return

        device = torch.device("cuda:0")
        print(f"开始测试显卡 {gpu_id}，设备：{device}")

        # 逐步分配显存，每次分配10%，直到占用80%
        total_mem = torch.cuda.get_device_properties(device).total_memory
        target_mem = int(total_mem * 0.8)
        allocated = 0
        tensors = []
        chunk = 1024 ** 2 * 256  # 初始 chunk 为 1GB（可根据需要调整）

        while allocated < target_mem:
            try:
                # 每次分配一个小张量，避免一次性溢出
                tensors.append(torch.randn(chunk // 4, device=device))  # float32 占4字节
                torch.cuda.synchronize()
                allocated += chunk
                print(f"显卡 {gpu_id} 已分配 {allocated / (1024 ** 3):.2f} GB 显存")
            except RuntimeError as e:
                # 若当前 chunk 过大，缩小后重试
                chunk = chunk // 2
                if chunk < 1024 ** 2 * 16:  # 最小 chunk 为 16MB
                    raise RuntimeError(f"显卡 {gpu_id} 显存分配失败：{e}")
                print(f"调整 chunk 大小为 {chunk / (1024 ** 2):.2f} MB，继续分配...")

        # 执行轻量化计算任务（矩阵加法），维持高利用率
        x = torch.randn(2048, 2048, device=device)
        y = torch.randn(2048, 2048, device=device)
        while True:
            z = x + y
            torch.cuda.synchronize()
            time.sleep(0.001)  # 极短延迟，保证计算连续性

    except Exception as e:
        print(f"显卡 {gpu_id} 测试出错：{e}")
    finally:
        print(f"显卡 {gpu_id} 测试终止")


if __name__ == "__main__":
    processes = []
    for gpu in GPU_IDS:
        p = Process(target=gpu_stress_test, args=(gpu,))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n用户终止测试，正在停止所有进程...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("所有进程已停止")