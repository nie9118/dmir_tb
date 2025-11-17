#!/bin/bash

# GPU压力测试脚本 - 占用4、5、6、7号GPU
# 作者：AI助手
# 注意：请谨慎使用，这会导致GPU温度升高

# 设置要使用的GPU设备
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# 检查CUDA是否可用
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        exit 1
    fi

    if ! python3 -c "import torch" &> /dev/null; then
        echo "错误: 未找到PyTorch，请先安装PyTorch"
        exit 1
    fi
}

# 显示GPU状态
show_gpu_status() {
    echo "当前GPU状态:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv -i 4,5,6,7
    echo "----------------------------------------"
}

# 主循环函数
main_loop() {
    local script_dir=$(dirname "$0")
    local python_script="$script_dir/main.py"

    # 创建Python压力测试脚本
    cat > "$python_script" << 'EOF'
import torch
import time
import sys

def gpu_stress_test():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU")

        # 为每个GPU创建张量
        tensors = []
        matrix_sizes = [8000, 9000, 10000, 11000]

        for i in range(num_gpus):
            try:
                torch.cuda.set_device(i)
                size = matrix_sizes[i % len(matrix_sizes)]

                a = torch.randn(size, size, device=f'cuda:{i}')
                b = torch.randn(size, size, device=f'cuda:{i}')
                tensors.append((a, b, i))

            except RuntimeError as e:
                print(f"GPU {i} 显存分配失败: {e}")
                continue

        iteration = 0

        while True:
            try:
                for a, b, gpu_id in tensors:
                    torch.cuda.set_device(gpu_id)

                    c = torch.matmul(a, b)
                    d = torch.relu(c)
                    e = torch.sigmoid(d)
                    torch.cuda.synchronize(gpu_id)


                time.sleep(0.01)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"执行过程中出现错误: {e}")
                time.sleep(1)

    else:

if __name__ == "__main__":
    gpu_stress_test()
EOF

    # 运行Python压力测试脚本
    python3 "$python_script"
}

# 清理函数
cleanup() {
    echo "正在清理..."
    # 杀死可能遗留的Python进程
    pkill -f "gpu_stress.py"
    # 清理GPU内存
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
"
}

# 设置信号处理
trap cleanup EXIT INT TERM


# 检查环境
check_cuda

# 显示初始状态
show_gpu_status

# 启动监控线程（可选）
(
    while true; do
        sleep 30
        show_gpu_status
    done
) &

# 运行主循环
main_loop
show_gpu_status