import os
import torch
import time
from multiprocessing import Process

# 指定需要测试的显卡编号（4、5、6、7）
GPU_IDS = [4, 5, 6, 7]


def gpu_stress_test(gpu_id):
    """对单张显卡进行压力测试"""
    # 设置当前进程仅可见目标显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        # 检查显卡是否可用
        if not torch.cuda.is_available():
            print(f"显卡 {gpu_id} 不可用")
            return

        device = torch.device(f"cuda:{0}")  # 由于已指定可见显卡，此处索引为0
        print(f"显卡 {gpu_id}，设备：{device}")

        # 获取显卡总显存（GB），用于分配接近满的显存
        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        # 预留少量显存（避免完全占满导致程序崩溃），分配95%的显存
        alloc_mem = int(total_mem * 0.8)

        # 创建大张量占用显存
        x = torch.randn(alloc_mem * 1024 ** 3 // 4, device=device)  # float32占4字节

        # 循环执行计算任务（矩阵乘法），维持高利用率
        while True:
            # 矩阵乘法（计算密集型任务）
            x = torch.matmul(x.reshape(-1, 1024), torch.randn(1024, 1024, device=device)).flatten()
            time.sleep(0.01)  # 轻微延迟，避免过度占用CPU

    except Exception as e:
        print(f"显卡 {gpu_id} 出错：{e}")
    finally:
        print(f"显卡 {gpu_id} 终止")


if __name__ == "__main__":
    # 为每张显卡启动一个独立进程
    processes = []
    for gpu in GPU_IDS:
        p = Process(target=gpu_stress_test, args=(gpu,))
        p.start()
        processes.append(p)

    # 等待所有进程（可通过Ctrl+C终止）
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n正在停止所有进程...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("进程已停止")