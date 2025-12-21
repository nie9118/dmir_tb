import re
from typing import List, Dict, Tuple
import argparse


def parse_result_pair(model_line: str, metric_line: str) -> Dict[str, any]:
    """
    解析一组结果（模型名称行 + 指标行）
    关键修正：第二个"mse"实际是"mae"，进行正确映射
    """
    # 清理空格
    model_name = model_line.strip()
    metric_str = metric_line.strip()

    if not model_name or not metric_str:
        return None

    # 解析指标
    metrics = {}
    # 匹配所有指标（按顺序提取）
    metric_pattern = r'(\w+):([\d\.]+)'
    matches = re.findall(metric_pattern, metric_str)

    # 处理指标映射（修正名称错误）
    # 指标顺序：mse(正确) → mse(实际是mae) → rmse → mape → mspe
    metric_keys = ['mse', 'mae', 'rmse', 'mape', 'mspe']
    for idx, (key, value) in enumerate(matches):
        try:
            float_value = float(value)
            # 根据位置映射正确的指标名称
            if idx < len(metric_keys):
                correct_key = metric_keys[idx]
                metrics[correct_key] = float_value
            else:
                # 额外的指标按原名称保留
                metrics[key] = float_value
        except ValueError:
            continue

    # 验证关键指标是否齐全
    required_metrics = ['mse', 'mae', 'rmse', 'mape', 'mspe']
    missing_metrics = [req for req in required_metrics if req not in metrics]
    if missing_metrics:
        print(f"警告：该组结果缺少指标{missing_metrics}，已跳过")
        print(f"  模型名称：{model_name}")
        print(f"  指标行：{metric_str}")
        return None

    # 从模型名称中解析超参数
    hyper_params = parse_hyper_parameters(model_name)

    return {
        'model_name': model_name,
        'metrics': metrics,
        'hyper_params': hyper_params
    }


def parse_hyper_parameters(model_name: str) -> Dict[str, any]:
    """
    从模型名称中提取超参数
    支持的超参数：bs, ft, sl, ll, pl, dm, nh, ial, pdl, cal, df, ebtime, Exp
    """
    hyper_params = {}

    # 定义超参数模式（键值对，如bs16 -> bs:16）
    param_patterns = [
        (r'bs(\d+)', 'batch_size'),  # 批次大小
        (r'ft([A-Za-z]+)', 'feature_type'),  # 特征类型
        (r'sl(\d+)', 'seq_len'),  # 序列长度
        (r'll(\d+)', 'lookback_len'),  # 回溯长度
        (r'pl(\d+)', 'predict_len'),  # 预测长度
        (r'dm(\d+)', 'd_model'),  # 模型维度
        (r'nh(\d+)', 'n_heads'),  # 注意力头数
        (r'ial(\d+)', 'inner_attention_layers'),  # 内部注意力层数
        (r'pdl(\d+)', 'proj_dropout_layer'),  # 投影dropout层
        (r'cal(\d+)', 'cross_attention_layers'),  # 交叉注意力层数
        (r'df(\d+)', 'd_ff'),  # 前馈网络维度
        (r'ebtime([A-Za-z]+)', 'time_embedding'),  # 时间嵌入类型
        (r'Exp_(\d+)', 'experiment_id'),  # 实验ID
    ]

    for pattern, param_name in param_patterns:
        match = re.search(pattern, model_name)
        if match:
            value = match.group(1)
            # 尝试转换为整数
            if value.isdigit():
                hyper_params[param_name] = int(value)
            else:
                hyper_params[param_name] = value

    return hyper_params


def load_results(file_path: str) -> List[Dict[str, any]]:
    """
    加载并解析结果文件（每个结果占2行：模型名称行 + 指标行）
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]  # 读取所有非空行
        total_lines = len(lines)

        # 按两行一组解析（模型行 + 指标行）
        for i in range(0, total_lines, 2):
            if i + 1 >= total_lines:
                print(f"警告：第{i + 1}行缺少对应的指标行，已跳过")
                continue

            model_line = lines[i]
            metric_line = lines[i + 1]

            # 验证模型行和指标行的格式（简单校验）
            if 'TimeBridge' not in model_line:
                print(f"警告：第{i + 1}行不是有效的模型名称行，已跳过")
                continue
            if 'mse:' not in metric_line:
                print(f"警告：第{i + 2}行不是有效的指标行，已跳过")
                continue

            # 解析这一组结果
            result = parse_result_pair(model_line, metric_line)
            if result:
                results.append(result)

    return results


def get_top_n_results(results: List[Dict[str, any]],
                      metric_name: str = 'mse',
                      top_n: int = 5,
                      ascending: bool = True) -> List[Dict[str, any]]:
    """
    根据指定指标获取前n个结果（默认找最小值）
    """
    # 过滤掉没有该指标的结果
    valid_results = [r for r in results if metric_name in r['metrics']]

    if not valid_results:
        print(f"警告：没有找到包含{metric_name}指标的有效结果")
        return []

    # 按指标排序
    sorted_results = sorted(valid_results,
                            key=lambda x: x['metrics'][metric_name],
                            reverse=not ascending)

    # 返回前n个
    return sorted_results[:top_n]


def print_results_summary(results: List[Dict[str, any]], metric_name: str):
    """
    打印结果摘要，包括超参数和关键指标
    """
    if not results:
        return

    print(f"\n=== 按{metric_name}排序的前{len(results)}个最优结果 ===")
    print("=" * 180)
    print(f"{'排名':<4} {'实验ID':<8} {'核心超参数':<80} "
          f"{metric_name:<10} {'mse':<10} {'mae':<10} {'rmse':<10} {'mape':<10} {'mspe':<10}")
    print("=" * 180)

    for idx, result in enumerate(results, 1):
        hp = result['hyper_params']
        metrics = result['metrics']

        # 构建核心超参数字符串
        hp_items = [
            f"bs:{hp.get('batch_size', 'N/A')}",
            f"sl:{hp.get('seq_len', 'N/A')}",
            f"pl:{hp.get('predict_len', 'N/A')}",
            f"dm:{hp.get('d_model', 'N/A')}",
            f"nh:{hp.get('n_heads', 'N/A')}",
            f"ial:{hp.get('inner_attention_layers', 'N/A')}",
            f"df:{hp.get('d_ff', 'N/A')}",
            f"ebtime:{hp.get('time_embedding', 'N/A')}"
        ]
        hp_str = ", ".join(hp_items)

        # 打印行（保留6位小数，MSPE保留整数）
        print(f"{idx:<4} {hp.get('experiment_id', 'N/A'):<8} {hp_str:<80} "
              f"{metrics[metric_name]:<10.6f} "
              f"{metrics['mse']:<10.6f} "
              f"{metrics['mae']:<10.6f} "
              f"{metrics['rmse']:<10.6f} "
              f"{metrics['mape']:<10.6f} "
              f"{metrics['mspe']:<10.0f}")

    print("=" * 180)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='提取模型结果中MSE/MAE最小的前n个结果，并显示超参数',
        epilog='注意：1. 文档格式要求：每个结果占2行（模型名称行+指标行）；2. 已自动修正指标名称错误（第二个"mse"→"mae"）'
    )
    parser.add_argument('--file', type=str, required=True, help='结果文件路径（如result_long_term_forecast.txt）')
    parser.add_argument('--top_n', type=int, default=5, help='要显示的前n个最优结果，默认5个')
    parser.add_argument('--metric', type=str, default='mse',
                        choices=['mse', 'mae', 'rmse', 'mape', 'mspe'],
                        help='排序指标（默认mse）：mse(均方误差)、mae(平均绝对误差)、rmse(均方根误差)、mape(平均绝对百分比误差)、mspe(平均平方百分比误差)')
    parser.add_argument('--detail', action='store_true', help='显示详细信息（完整模型名称+全部超参数）')

    args = parser.parse_args()

    # 加载并解析结果
    print(f"正在加载结果文件: {args.file}")
    print(f"注意：文件格式要求每个结果占2行（模型名称行+指标行），正在解析...")
    results = load_results(args.file)
    print(f"\n成功加载 {len(results)} 条有效结果（已自动修正指标名称错误：第二个'mse'→'mae'）")

    # 获取前n个最优结果
    top_results = get_top_n_results(
        results,
        metric_name=args.metric,
        top_n=args.top_n,
        ascending=True  # 指标越小越好，固定为升序
    )

    # 打印结果摘要
    print_results_summary(top_results, args.metric)

    # 显示详细信息（如果需要）
    if args.detail and top_results:
        print("\n=== 详细信息展示 ===")
        for idx, result in enumerate(top_results, 1):
            print(f"\n【第{idx}名】")
            print(f"模型全名: {result['model_name']}")
            print(f"完整超参数: ")
            for param, value in sorted(result['hyper_params'].items()):
                print(f"  - {param}: {value}")
            print(f"完整指标值: ")
            for metric, value in sorted(result['metrics'].items()):
                print(f"  - {metric}: {value:.6f}")


if __name__ == "__main__":
    # 使用示例（复制到命令行运行）：
    # 1. 查找MSE最小的前5个结果
    # python find_search.py --file result_long_term_forecast.txt --metric mse

    # 2. 查找MAE最小的前10个结果
    # python find_search.py --file result_long_term_forecast.txt --top_n 10 --metric mae

    # 3. 查找RMSE最小的前3个结果，并显示详细信息
    # python find_search.py --file result_long_term_forecast.txt --top_n 3 --metric rmse --detail
    main()