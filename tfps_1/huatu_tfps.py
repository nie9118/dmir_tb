import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")


# 移除全局的 context 设置，以便更好地单独控制字体大小
# sns.set_context("notebook", font_scale=1.2)

def plot_time_series_modified(input_file, pred_file, truth_file, feature_idx=-1):
    # 1. 读取数据
    df_input = pd.read_csv(input_file, header=None)
    df_pred = pd.read_csv(pred_file, header=None)
    df_truth = pd.read_csv(truth_file, header=None)

    # 2. 准备数据
    history_data = df_input.iloc[:, feature_idx].values
    time_history = np.arange(len(history_data))

    start_idx = len(history_data)
    pred_len = len(df_pred)

    future_data_pred = df_pred.iloc[:, feature_idx].values
    future_data_truth = df_truth.iloc[:, feature_idx].values
    time_future = np.arange(start_idx, start_idx + pred_len)

    # 3. 绘图
    plt.figure(figsize=(15, 6))

    # 绘制历史 (线段更粗: linewidth=4)
    plt.plot(time_history, history_data, label='History (Input)', color='#1f77b4', linewidth=4)

    # 绘制连接线 (可选，为了视觉连贯)
    plt.plot([time_history[-1], time_future[0]], [history_data[-1], future_data_truth[0]],
             color='grey', linestyle=':', alpha=0.5)

    # 绘制真实值 (线段更粗: linewidth=4)
    plt.plot(time_future, future_data_truth, label='Ground Truth', color='green', linestyle='-', alpha=0.5, linewidth=4)

    # 绘制预测值 (线段更粗: linewidth=4)
    plt.plot(time_future, future_data_pred, label='Prediction', color='#d62728', linestyle='-', linewidth=4)

    # 分割线
    plt.axvline(x=start_idx, color='k', linestyle='--', linewidth=2, alpha=0.8)  # 分割线也稍微加粗

    # 获取当前 y 轴范围以放置文本
    y_min, y_max = plt.ylim()
    plt.text(start_idx, y_max * 0.98, '  Prediction Start', color='k', verticalalignment='top', fontsize=12)

    col_idx_display = df_input.shape[1] + feature_idx if feature_idx < 0 else feature_idx
    # 标题变小 (fontsize=12) 并且靠左 (loc='left')
    plt.title(f'Time Series Forecasting (Feature index: {col_idx_display})', fontsize=12, loc='left')

    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig('forecast_plot_modified.png', dpi=300)
    # plt.show() # 在notebook中显示


if __name__ == "__main__":
    plot_time_series_modified('sample_0_input_x.csv', 'sample_0_pred_out.csv', 'sample_0_truth_y.csv')