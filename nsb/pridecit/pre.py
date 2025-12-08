import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from timefeatures import time_features


def process_input_data(x, scaler=None, timeenc=0, freq='h'):
    """
    按照Dataset_ETT_hour类的方式处理输入数据

    参数:
        x: 输入的原始数据，形状为(96, 8)
        scaler: 标准化器，如果为None则会新创建并拟合
        timeenc: 时间编码方式，0或1
        freq: 时间频率，默认为'h'（小时）

    返回:
        processed_x: 处理后的输入数据
        x_mark: 时间特征标记
        scaler: 使用的标准化器
    """
    # 转换为DataFrame以便处理
    df_data = pd.DataFrame(x)

    # 标准化处理
    if scaler is None:
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(df_data.values)
    else:
        processed_data = scaler.transform(df_data.values)

    # 生成时间戳（这里假设数据是连续的小时数据，从当前时间往前推95小时）
    # 实际应用中可能需要根据实际时间调整
    end_time = pd.Timestamp.now()
    start_time = end_time - pd.Timedelta(hours=len(x) - 1)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    df_stamp = pd.DataFrame({'date': timestamps})

    # 处理时间特征
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
        x_mark = df_stamp.drop(['date'], axis=1).values
    elif timeenc == 1:
        x_mark = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        x_mark = x_mark.transpose(1, 0)

    # 转换为张量并增加批次维度
    processed_x = torch.FloatTensor(processed_data).unsqueeze(0)
    x_mark = torch.FloatTensor(x_mark).unsqueeze(0)

    return processed_x, x_mark, scaler


def predict_and_save(model_path, input_data, output_csv, pred_len=96, timeenc=0, freq='h'):
    """
    使用保存的模型进行预测并保存结果

    参数:
        model_path: 模型.pth文件路径
        input_data: 输入数据，形状为(96, 8)
        output_csv: 输出CSV文件路径
        pred_len: 预测长度
        timeenc: 时间编码方式
        freq: 时间频率
    """
    # 检查输入数据形状
    if input_data.shape != (96, 8):
        raise ValueError("输入数据必须是形状为(96, 8)的数组")

    # 加载模型
    model = torch.load(model_path)
    model.eval()

    # 处理输入数据
    processed_x, x_mark, scaler = process_input_data(input_data, timeenc=timeenc, freq=freq)

    # 进行预测
    with torch.no_grad():
        # 假设模型输入为(seq_x, seq_x_mark)，输出为预测结果
        # 根据实际模型结构可能需要调整
        pred = model(processed_x, x_mark)
        # 取预测部分（根据模型输出格式调整）
        pred = pred[:, -pred_len:, :].numpy()

    # 反标准化
    pred_original = scaler.inverse_transform(pred.squeeze(0))

    # 准备保存数据
    # 输入数据的原始尺度
    input_original = scaler.inverse_transform(processed_x.squeeze(0).numpy())

    # 创建DataFrame保存结果
    # 输入数据
    input_df = pd.DataFrame(input_original, columns=[f'input_feat_{i}' for i in range(8)])
    input_df['type'] = 'input'
    input_df['index'] = range(len(input_df))

    # 预测数据
    pred_df = pd.DataFrame(pred_original, columns=[f'pred_feat_{i}' for i in range(8)])
    pred_df['type'] = 'prediction'
    pred_df['index'] = range(len(input_df), len(input_df) + len(pred_df))

    # 合并并保存
    result_df = pd.concat([input_df, pred_df], ignore_index=True)
    result_df.to_csv(output_csv, index=False)

    print(f"结果已保存到 {output_csv}")
    return result_df


# 使用示例
if __name__ == "__main__":
    # 生成示例输入数据 (96, 8)
    sample_input = np.random.randn(96, 8)  # 实际应用中替换为真实数据

    # 模型路径
    model_path = "ckp/etth1.checkpoint.pth"  # 替换为实际模型路径

    # 输出CSV路径
    output_csv = "prediction_results.csv"

    # 执行预测并保存
    result = predict_and_save(model_path, sample_input, output_csv)