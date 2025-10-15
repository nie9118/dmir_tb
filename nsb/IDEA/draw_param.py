import argparse
import os
import numpy as np
from matplotlib import pyplot as plt


def draw_yq(model, input, target, path):
    # 确保输入是一维
    input = input.squeeze()
    target = target.squeeze()

    # 检查维度
    assert input.ndim == 1, f"input 必须是 1 维，实际维度：{input.ndim}"
    assert target.ndim == 1, f"target 必须是 1 维，实际维度：{target.ndim}"

    # 绘图
    plt.plot(range(len(input)), input, label='Input', linewidth=2)
    plt.plot(range(len(target)), target, label='Ground Truth', linewidth=5, color='#C45C69')
    plt.legend()

    # 保存图像
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='NSTS', required=False)
parser.add_argument('-size', type=int, default=1)
parser.add_argument('-datasets', type=str, default='custom', nargs='+')
parser.add_argument('-models', type=str, nargs='+')
parser.add_argument('-device', default=0, type=int)
args = parser.parse_args()

for model in args.models:
    for dataset in args.datasets:
        for pred_len in [48]:
            # 加载数据
            inputs = np.load(f'./draw_data/{model}/{dataset}/{pred_len}/u.npy', allow_pickle=True)
            targets = np.load(f'./draw_data/{model}/{dataset}/{pred_len}/true.npy', allow_pickle=True)

            # 调试数据形状
            print(f"inputs.shape = {inputs.shape}")
            print(f"targets.shape = {targets.shape}")

            # 创建保存目录
            root_path = f"./IDEA_draw_pictures/{dataset}/{pred_len}/{model}"
            os.makedirs(root_path, exist_ok=True)

            # 遍历每个样本、通道和特征
            for i in range(targets.shape[0]):
                for j in range(32):
                    for k in range(7):
                        path = os.path.join(root_path, f'{i}_{k}.png')
                        input_slice = inputs[0][j, :]
                        target_slice = targets[0][j, :, k]
                        draw_yq(model, input_slice, target_slice, path)