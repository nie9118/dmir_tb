import os
import re
import json
import time
import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Union, Set, Any
import numpy as np
import pandas as pd
import pynvml
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import KBinsDiscretizer
from tabicl.sklearn.classifier import TabICLClassifier
import torch

# 常量定义
CLASSIFICATION_TASKS = {'binclass', 'multiclass'}
SKIP_REGRESSION = True
COERCE_NUMERIC = True
FIXED_GPUS = 8  # 固定使用8卡并行
DEFAULT_DATA_ROOT = '/vast/users/guangyi.chen/causal_group/zijian.li/LDM/datasets'
DEFAULT_MODEL_PATH = "//vast/users/guangyi.chen/causal_group/zijian.li/LDM/tabicl_latent3/stage1/checkpoint/dir"
DEFAULT_OUTDIR = 'talent_log'


# 用于显存监控
def convert_features(X: np.ndarray, enabled: bool) -> np.ndarray:
    """
    可选地将特征矩阵强制转换为数值类型。
    启用时，无法解析为数值的列会被序数编码——每个不同的字符串会被分配一个稳定的整数ID（0, 1, 2, ...）。
    """
    X = np.asarray(X)
    if not enabled:
        return X

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    df = pd.DataFrame(X)
    encoded = pd.DataFrame(index=df.index)

    for col in df.columns:
        series = df.iloc[:, col]
        numeric_series = pd.to_numeric(series, errors='coerce')

        if series.isna().equals(numeric_series.isna()):
            encoded[col] = numeric_series
        else:
            string_series = series.astype("string")
            codes, uniques = pd.factorize(string_series, sort=True)
            codes = codes.astype(np.int32)
            if (codes == -1).any():
                codes[codes == -1] = len(uniques)
            encoded[col] = codes

    return encoded.fillna(0).values.astype(np.float32)


def count_missing(values: np.ndarray) -> int:
    """统计数组中的缺失值数量"""
    if values is None:
        return 0
    arr = np.asarray(values)
    if arr.dtype.kind in {"f", "c"}:
        return int(np.isnan(arr).sum())
    mask = pd.isna(pd.DataFrame(arr))
    return int(mask.values.sum())


def load_array(file_path: Path) -> np.ndarray:
    """
    从文件加载数组
    支持格式：.npy/.npz, .parquet, .csv/.tsv
    """
    suffix = file_path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(file_path, allow_pickle=False)
        except ValueError:
            arr = np.load(file_path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        return np.asarray(arr)
    if suffix == '.parquet':
        return pd.read_parquet(file_path).values
    sep = '\t' if suffix == '.tsv' else None
    return pd.read_csv(file_path, sep=sep, header=None).values


def find_data_files(dataset_dir: Path):
    """
    在数据集目录中查找训练/验证/测试集文件
    返回格式：(训练集文件元组), (验证集文件元组), (测试集文件元组)
    元组结构：(数值特征文件, 类别特征文件, 标签文件)
    """
    files = [p for p in dataset_dir.iterdir() if p.is_file()]
    lower = {p.name.lower(): p for p in files}

    def by_suffix(key: str):
        """根据后缀查找文件"""
        for name, p in lower.items():
            if name.endswith(key):
                return p
        return None

    n_train = by_suffix('n_train.npy')
    c_train = by_suffix('c_train.npy')
    y_train = by_suffix('y_train.npy')
    n_val = by_suffix('n_val.npy')
    c_val = by_suffix('c_val.npy')
    y_val = by_suffix('y_val.npy')
    n_test = by_suffix('n_test.npy')
    c_test = by_suffix('c_test.npy')
    y_test = by_suffix('y_test.npy')

    if y_train and y_test and (n_train or c_train) and (n_test or c_test):
        val_pair = None
        if y_val and (n_val or c_val):
            val_pair = (n_val, c_val, y_val)
        return (n_train, c_train, y_train), val_pair, (n_test, c_test, y_test)

    # 查找单表文件候选
    table_candidates = [p for p in files if p.suffix.lower() in {'.npy', '.npz', '.csv', '.tsv', '.parquet'}]
    if len(table_candidates) == 1:
        return table_candidates[0], None, None
    return None, None, None


def load_dataset_info(dataset_dir: Path) -> Optional[dict]:
    """加载数据集的info.json文件"""
    p = dataset_dir / 'info.json'
    if not p.exists():
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"读取 {p} 失败: {exc}")
        return None


def load_pair(X_path: Path, y_path: Path, context: str = "", coerce_numeric: bool = False,
              dataset_id: str | None = None, missing_registry: set[str] | None = None):
    """
    加载特征-标签对文件
    :param X_path: 特征文件路径
    :param y_path: 标签文件路径
    :param context: 日志上下文
    :param coerce_numeric: 是否强制转换为数值特征
    :param dataset_id: 数据集ID（用于缺失值登记）
    :param missing_registry: 缺失值数据集登记集合
    :return: (特征数组, 标签数组)
    """
    X = load_array(X_path)
    y = load_array(y_path)
    log_nan_presence(f"{context or X_path.stem}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{context or X_path.stem}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=context or X_path.stem)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_split(num_path: Optional[Path], cat_path: Optional[Path], y_path: Path,
               context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None):
    """
    加载拆分的特征文件（数值+类别）和标签文件
    :param num_path: 数值特征文件路径
    :param cat_path: 类别特征文件路径
    :param y_path: 标签文件路径
    :param context: 日志上下文
    :param coerce_numeric: 是否强制转换为数值特征
    :param dataset_id: 数据集ID（用于缺失值登记）
    :param missing_registry: 缺失值数据集登记集合
    :return: (合并后的特征数组, 标签数组)
    """
    feats = []
    base = context or (num_path.stem if num_path else (cat_path.stem if cat_path else y_path.stem))
    if num_path:
        Xn = np.asarray(load_array(num_path))
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        log_nan_presence(f"{base}-num_raw", Xn, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xn)
    if cat_path:
        Xc = np.asarray(load_array(cat_path))
        if Xc.ndim == 1:
            Xc = Xc.reshape(-1, 1)
        log_nan_presence(f"{base}-cat_raw", Xc, dataset_id=dataset_id, missing_registry=missing_registry)
        feats.append(Xc)
    if not feats:
        raise ValueError("缺少数值/类别特征文件")
    n = feats[0].shape[0]
    for i, f in enumerate(feats):
        if f.shape[0] != n:
            raise ValueError(f"特征数量不一致: #{i} 有 {f.shape[0]} vs {n}")
    X = feats[0] if len(feats) == 1 else np.concatenate(feats, axis=1)
    log_nan_presence(f"{base}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    y = np.asarray(load_array(y_path))
    log_nan_presence(f"{base}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.squeeze(1)
        elif y.shape[0] == 1:
            y = y.squeeze(0)
    X, y = handle_missing_entries(X, y, context=base)
    X = convert_features(X, coerce_numeric)
    return X, y


def load_table(file_path: Union[Path, Tuple], context: str = "", coerce_numeric: bool = False,
               dataset_id: str | None = None, missing_registry: set[str] | None = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    通用加载函数，支持多种输入格式
    :param file_path: 单个文件路径 或 (特征,标签)元组 或 (数值特征,类别特征,标签)元组
    :param context: 日志上下文
    :param coerce_numeric: 是否强制转换为数值特征
    :param dataset_id: 数据集ID（用于缺失值登记）
    :param missing_registry: 缺失值数据集登记集合
    :return: (特征数组, 标签数组)
    """
    if isinstance(file_path, (tuple, list)):
        if len(file_path) == 2:
            Xp, yp = Path(file_path[0]), Path(file_path[1])
            return load_pair(Xp, yp, context=context, coerce_numeric=coerce_numeric,
                             dataset_id=dataset_id, missing_registry=missing_registry)
        if len(file_path) == 3:
            num_path, cat_path, y_path = file_path
            return load_split(Path(num_path) if num_path else None,
                              Path(cat_path) if cat_path else None,
                              Path(y_path),
                              context=context, coerce_numeric=coerce_numeric,
                              dataset_id=dataset_id, missing_registry=missing_registry)
        raise ValueError(f"不支持的load_table输入元组格式: {file_path}")

    path: Path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {'.npy', '.npz'}:
        try:
            arr = np.load(path, allow_pickle=False)
        except ValueError:
            arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[list(arr.files)[0]]
        data = np.asarray(arr)
    elif suffix == '.parquet':
        data = pd.read_parquet(path).values
    else:
        sep = '\t' if suffix == '.tsv' else None
        data = pd.read_csv(path, sep=sep, header=None).values

    if data.ndim == 1:
        raise ValueError(f"{path} 中包含不支持的一维数据")

    log_target = context or str(path)
    log_nan_presence(f"{log_target}-raw", data, dataset_id=dataset_id, missing_registry=missing_registry)

    col0 = data[:, 0]
    try:
        uniques0 = np.unique(col0)
    except Exception:
        uniques0 = np.array([])

    # 启发式拆分标签列：第一列唯一值少则选第一列，否则选最后一列
    if 0 < uniques0.size < max(2, data.shape[0] // 2):
        y = data[:, 0]
        X = data[:, 1:]
        which = 'first'
    else:
        y = data[:, -1]
        X = data[:, :-1]
        which = 'last'

    log_nan_presence(f"{log_target}-X_raw", X, dataset_id=dataset_id, missing_registry=missing_registry)
    log_nan_presence(f"{log_target}-y_raw", y, dataset_id=dataset_id, missing_registry=missing_registry)

    X = np.asarray(pd.DataFrame(X).values)
    y = pd.Series(y).values
    X, y = handle_missing_entries(X, y, context=log_target)
    X = convert_features(X, coerce_numeric)
    logging.info(f"{log_target}: 使用单文件启发式拆分标签 (取 {which} 列)")
    return X, y


def get_gpu_memory_mib(device_id: int = 0) -> Optional[float]:
    """
    获取指定GPU设备的当前已用显存（单位：MiB）
    :param device_id: GPU设备索引
    :return: 已用显存值（MiB），失败时返回None
    """
    # 静态变量，确保错误只记录一次
    if not hasattr(get_gpu_memory_mib, "has_logged_error"):
        setattr(get_gpu_memory_mib, "has_logged_error", False)

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)  # 字节转换为MiB
    except pynvml.NVMLError as e:
        if not getattr(get_gpu_memory_mib, "has_logged_error", False):
            logging.warning(f"无法查询 GPU 显存 (device {device_id}): {e}. 后续将不再显示显存信息。")
            setattr(get_gpu_memory_mib, "has_logged_error", True)
        return None
    except Exception as e:
        if not getattr(get_gpu_memory_mib, "has_logged_error", False):
            logging.warning(f"查询 GPU 显存时发生未知错误: {e}. 后续将不再显示显存信息。")
            setattr(get_gpu_memory_mib, "has_logged_error", True)
        return None


def split_train_test(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=42):
    """
    拆分训练集和测试集
    分类任务自动分层抽样，回归任务随机抽样
    """
    stratify = y if (len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y))) else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def summarize_task_types(dirs: List[Path]) -> None:
    """统计数据集目录列表中的任务类型分布"""
    counts = {'regression': 0, 'binclass': 0, 'multiclass': 0, 'unknown': 0}
    for d in dirs:
        info = load_dataset_info(d)
        t = (str(info.get('task_type', '')).lower() if info else '')
        if not t:
            counts['unknown'] += 1
        elif t in counts:
            counts[t] += 1
        else:
            counts['unknown'] += 1
    logging.info("任务统计: regression=%d, binclass=%d, multiclass=%d, unknown=%d, 总计=%d",
                 counts['regression'], counts['binclass'], counts['multiclass'], counts['unknown'], len(dirs))


def log_nan_presence(name: str, data: np.ndarray, dataset_id: str | None = None,
                     missing_registry: set[str] | None = None) -> None:
    """
    记录数据中NaN的存在情况
    :param name: 数据标识名称
    :param data: 待检查数组
    :param dataset_id: 数据集ID
    :param missing_registry: 缺失值数据集登记集合
    """
    if missing_registry is None:
        return
    if count_missing(data) > 0 and dataset_id:
        missing_registry.add(dataset_id)


def handle_missing_entries(X: np.ndarray, y: np.ndarray, context: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    处理特征和标签中的缺失值
    （示例实现，实际逻辑需根据业务需求调整）
    :param X: 特征数组
    :param y: 标签数组
    :param context: 日志上下文
    :return: 处理后的特征和标签数组
    """
    return X, y


def get_ckpt_metadata(ckpt_path: str) -> dict:
    """提取ckpt中的重要元数据（超参数、训练信息等）"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metadata = {
            'keys': list(ckpt.get('state_dict', ckpt).keys()) if isinstance(ckpt, dict) else [],
            'epoch': ckpt.get('epoch', 'unknown'),
            'step': ckpt.get('step', 'unknown'),
            'hyperparameters': ckpt.get('hyper_parameters', {}),
            'feature_dim': None,
            'output_dim': None
        }

        # 尝试提取输入特征维度（根据常见参数名推断）
        state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        for key in state_dict:
            if 'encoder' in key and 'weight' in key and len(state_dict[key].shape) >= 2:
                metadata['feature_dim'] = state_dict[key].shape[1]
            if 'classifier' in key and 'weight' in key and len(state_dict[key].shape) >= 2:
                metadata['output_dim'] = state_dict[key].shape[0]
        return metadata
    except Exception as e:
        logging.warning(f"提取ckpt元数据失败: {e}")
        return {'keys': [], 'epoch': 'unknown', 'step': 'unknown', 'hyperparameters': {}}


def log_ckpt_metadata(ckpt_path: str, outdir: Path) -> None:
    """记录ckpt元数据到日志文件"""
    metadata = get_ckpt_metadata(ckpt_path)
    with open(outdir / "ckpt_metadata.txt", "w") as f:
        f.write(f"Checkpoint Path: {ckpt_path}\n")
        f.write(f"Epoch: {metadata['epoch']}\n")
        f.write(f"Step: {metadata['step']}\n")
        f.write(f"Feature Dimension: {metadata['feature_dim']}\n")
        f.write(f"Output Dimension: {metadata['output_dim']}\n")
        f.write("\nHyperparameters:\n")
        for k, v in metadata['hyperparameters'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nParameter Keys:\n")
        for key in metadata['keys']:
            f.write(f"  {key}\n")


class AdaptableTabICLClassifier(TabICLClassifier):
    """可适配不同参数checkpoint的分类器"""

    def __init__(self, *args, **kwargs):
        self.ckpt_metadata = kwargs.pop('ckpt_metadata', None)
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """重写模型加载方法，实现参数兼容"""
        try:
            # 加载原始checkpoint
            state_dict = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in state_dict:  # 处理包含state_dict键的情况
                state_dict = state_dict['state_dict']

            # 获取当前模型的参数名
            model_state = self.model.state_dict()

            # 过滤并适配参数
            filtered_state = {}
            mismatched = []
            for name, param in state_dict.items():
                # 处理参数名前缀差异（如'model.'前缀）
                cleaned_name = name.replace('model.', '').replace('module.', '')

                if cleaned_name in model_state:
                    # 处理维度不匹配但可兼容的情况（如偏置项）
                    if model_state[cleaned_name].shape == param.shape:
                        filtered_state[cleaned_name] = param
                    else:
                        # 尝试截断或扩展参数（仅建议用于偏置等简单参数）
                        if len(model_state[cleaned_name].shape) == 1:
                            min_len = min(len(model_state[cleaned_name]), len(param))
                            filtered_state[cleaned_name] = torch.nn.Parameter(
                                torch.cat([
                                    param[:min_len],
                                    model_state[cleaned_name][min_len:] if min_len < len(
                                        model_state[cleaned_name]) else torch.tensor([])
                                ])
                            )
                            logging.warning(
                                f"参数维度不匹配，已适配: {cleaned_name} {param.shape} -> {model_state[cleaned_name].shape}")
                        else:
                            mismatched.append(f"{cleaned_name} ({param.shape} vs {model_state[cleaned_name].shape})")
                else:
                    mismatched.append(f"{name} (不存在于当前模型)")

            # 加载过滤后的参数
            self.model.load_state_dict(filtered_state, strict=False)

            # 记录不匹配的参数
            if mismatched:
                with open(Path(self.model_path).parent / "mismatched_params.log", "a") as f:
                    f.write(f"Model: {self.model_path}\n")
                    for msg in mismatched:
                        f.write(f"  {msg}\n")
                logging.warning(f"发现{len(mismatched)}个不匹配参数，已记录到日志")

        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise


def run_on_gpu(model_path: str, dirs: List[Path], gpu_physical_id: int, results_list, merge_val: bool,
               coerce_numeric: bool, skip_regression: bool):
    """
    在单个GPU上运行数据集评测任务
    :param model_path: 模型文件路径
    :param dirs: 分配给该GPU的数据集目录列表
    :param gpu_physical_id: GPU物理设备ID
    :param results_list: 多进程共享的结果列表
    :param merge_val: 是否将验证集合并到训练集
    :param coerce_numeric: 是否强制转换为数值特征
    :param skip_regression: 是否跳过回归任务
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_physical_id)
    try:
        import torch
        torch.cuda.set_device(0)
    except Exception:
        pass

    logging.info(f"[GPU {gpu_physical_id}] 启动，分配到 {len(dirs)} 个数据集")

    # 延迟导入以避免初始化问题
    from tabicl.sklearn.classifier import TabICLClassifier  # 保留原始导入
    # 读取ckpt元数据
    ckpt_metadata = get_ckpt_metadata(model_path)
    # 使用适配性分类器
    clf = AdaptableTabICLClassifier(
        verbose=False,
        model_path=model_path,
        ckpt_metadata=ckpt_metadata  # 传入元数据
    )

    missing_datasets: set[str] = set()

    for d in dirs:
        try:
            info = load_dataset_info(d)
            ttype = (str(info.get('task_type', '')).lower() if info else None)
            if skip_regression and ttype == 'regression':
                logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 回归任务")
                continue

            train_path, val_path, test_path = find_data_files(d)
            if train_path is None and test_path is None:
                logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 未识别数据文件")
                continue

            if train_path and test_path:
                X_train, y_train = load_table(train_path, context=f"{d.name}-train",
                                              coerce_numeric=coerce_numeric, dataset_id=d.name,
                                              missing_registry=missing_datasets)

                X_test, y_test = load_table(test_path, context=f"{d.name}-test",
                                            coerce_numeric=coerce_numeric, dataset_id=d.name,
                                            missing_registry=missing_datasets)
            else:
                logging.info(f"[GPU {gpu_physical_id}] {d.name}: 只有单文件，当前策略跳过")
                continue

            # 合并验证集到训练集
            if merge_val and val_path:
                X_val, y_val = load_table(val_path, context=f"{d.name}-val", coerce_numeric=coerce_numeric,
                                          dataset_id=d.name, missing_registry=missing_datasets)
                if X_val.ndim == 3 and X_val.shape[1] == 1:
                    X_val = X_val.squeeze(1)
                if X_val.ndim == 1:
                    X_val = X_val.reshape(-1, 1)
                y_val = np.asarray(y_val)
                if y_val.ndim > 1 and y_val.shape[-1] == 1:
                    y_val = y_val.reshape(-1)
                X_train = np.concatenate([X_train, X_val], axis=0)
                y_train = np.concatenate([y_train, y_val], axis=0)
                logging.info(
                    f"[GPU {gpu_physical_id}] {d.name}: 已将validation split合并进训练，总计 {X_train.shape[0]} 条训练样本")

            # 确保输入形状正确
            if X_train.ndim == 3 and X_train.shape[1] == 1:
                X_train = X_train.squeeze(1)
            if X_test.ndim == 3 and X_test.shape[1] == 1:
                X_test = X_test.squeeze(1)
            X_train = X_train.astype(np.float32, copy=False)
            X_test = X_test.astype(np.float32, copy=False)

            # 处理连续标签
            tgt_type = None
            try:
                tgt_type = type_of_target(y_train)
            except Exception:
                tgt_type = None

            if ttype is None and tgt_type is not None and tgt_type.startswith('continuous'):
                if skip_regression:
                    logging.info(f"[GPU {gpu_physical_id}] 跳过 {d.name}: 连续标签 (可能为回归任务)")
                    continue

            # 开始评测并记录显存
            t0 = time.time()

            clf.fit(X_train, y_train)
            mem_after_fit = get_gpu_memory_mib(0)

            y_pred = clf.predict(X_test)
            mem_after_predict = get_gpu_memory_mib(0)

            acc = float(np.mean(y_pred == y_test))
            dt = time.time() - t0

            # 计算峰值显存
            valid_mems = []
            if mem_after_fit is not None:
            if mem_after_fit is not None:
                valid_mems.append(mem_after_fit)
            if mem_after_predict is not None:
                valid_mems.append(mem_after_predict)
            peak_mem_mib = max(valid_mems) if valid_mems else None

            logging.info(f"[GPU {gpu_physical_id}] {d.name}: acc={acc:.4f}, time={dt:.2f}s, "
                         f"peak_vram={peak_mem_mib:.1f}MiB" if peak_mem_mib else "")

            results_list.append((d.name, acc, dt, peak_mem_mib))

        except Exception as e:
            logging.exception(f"[GPU {gpu_physical_id}] 评测失败 {d.name}: {e}")


def evaluate_model(model_path: str, data_root: Path, outdir_root: Path, merge_val: bool,
                   coerce_numeric: bool, skip_regression: bool) -> Tuple[str, int, float, float, float, float, float]:
    """
    评测单个模型并返回汇总结果
    :param model_path: 模型文件路径
    :param data_root: 数据集根目录
    :param outdir_root: 输出目录根路径
    :param merge_val: 是否将验证集合并到训练集
    :param coerce_numeric: 是否强制转换为数值特征
    :param skip_regression: 是否跳过回归任务
    :return: (模型标签, 成功评测数, 平均准确率, 总耗时, 平均耗时, 平均峰值显存, 最大峰值显存)
    """
    model_tag = Path(model_path).stem
    outdir = outdir_root / model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    log_ckpt_metadata(model_path, outdir)

    # 准备日志
    file_handler = logging.FileHandler(outdir / 'bench_talent.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)

    dirs = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    summarize_task_types(dirs)

    # 确定可用GPU数量
    try:
        available_gpus = int(os.environ.get("NUM_GPUS", "0"))
    except Exception:
        available_gpus = 0

    num_gpus = FIXED_GPUS
    if available_gpus > 0:
        num_gpus = min(FIXED_GPUS, available_gpus)
    if num_gpus < FIXED_GPUS:
        logging.info(f"检测到 {num_gpus} 张GPU（少于固定{FIXED_GPUS}张），将按{num_gpus}张并行。")

    # 初始化pynvml
    try:
        pynvml.nvmlInit()
        logging.info("pynvml初始化成功，将监控GPU显存。")
    except Exception as e:
        logging.warning(f"pynvml初始化失败: {e}. 无法监控GPU显存。")

    # 数据集分片到不同GPU
    shards: List[List[Path]] = [[] for _ in range(num_gpus)]
    for i, d in enumerate(dirs):
        shards[i % num_gpus].append(d)

    # 使用多进程在多个GPU上并行评测
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        results_list = manager.list()

        procs = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=run_on_gpu,
                args=(model_path, shards[gpu_id], gpu_id, results_list, merge_val, coerce_numeric, skip_regression),
                daemon=False,
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # 关闭pynvml
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        results = list(results_list)
        results.sort(key=lambda x: x[0])

        # 保存详细结果
        detailed_path = outdir / "talent_detailed.txt"
        if results:
            with open(detailed_path, "w") as f:
                f.write("dataset\taccuracy\ttime_s\tpeak_vram_mib\n")
                for name, acc, dur, vram in results:
                    vram_str = f"{vram:.2f}" if vram is not None else "N/A"
                    f.write(f"{name}\t{acc:.6f}\t{dur:.3f}\t{vram_str}\n")

            # 计算汇总统计
            total_time = sum(dur for _, _, dur, _ in results)
            avg_time = total_time / len(results)
            avg_acc = sum(acc for _, acc, _, _ in results) / len(results)

            valid_vram = [vram for _, _, _, vram in results if vram is not None and vram > 0]
            avg_vram = sum(valid_vram) / len(valid_vram) if valid_vram else 0.0
            max_vram = max(valid_vram) if valid_vram else 0.0

            # 保存汇总结果
            summary_path = outdir / "talent_summary.txt"
            with open(summary_path, "w") as f:
                f.write(f"Model: {model_tag}\n")
                f.write(f"Total datasets: {len(results)}\n")
                f.write(f"Average accuracy: {avg_acc:.6f}\n")
                f.write(f"Total time s: {total_time:.3f}\n")
                f.write(f"Average time s: {avg_time:.3f}\n")
                f.write(f"Average Peak VRAM (MiB): {avg_vram:.2f}\n")
                f.write(f"Overall Max Peak VRAM (MiB): {max_vram:.2f}\n")

            logging.info(f"[{model_tag}] 汇总完成：{detailed_path} / {summary_path}")
            logging.getLogger().removeHandler(file_handler)
            return model_tag, len(results), avg_acc, total_time, avg_time, avg_vram, max_vram
        else:
            logging.info(f"[{model_tag}] 没有成功的评测结果。")
            logging.getLogger().removeHandler(file_handler)
            return model_tag, 0, float("nan"), 0.0, float("nan"), 0.0, 0.0


def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=None,
                    help="单个模型ckpt路径（与--models_dir互斥）")
    ap.add_argument("--models_dir", type=str, default=DEFAULT_MODEL_PATH,
                    help="包含多个*.ckpt的目录；将按文件名排序依次评测")
    ap.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    ap.add_argument("--max-datasets", type=int, default=None, help="限制评测的数据集数量")
    ap.add_argument("--verbose", action="store_true", help="启用详细日志")
    ap.add_argument("--merge-val", action="store_true", help="将验证集合并到训练集")
    ap.add_argument("--no-coerce-numeric", dest="coerce_numeric", action="store_false",
                    help="禁用非数值特征自动转换为数值编码")
    ap.add_argument("--include-regression", dest="skip_regression", action="store_false",
                    help="包含回归任务（默认跳过）")
    ap.set_defaults(coerce_numeric=True, skip_regression=True)
    return ap.parse_args()


def main():
    """主函数：解析参数并执行模型评测"""
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()

    data_root = Path(args.data_root)
    outdir_root = Path(args.outdir)
    outdir_root.mkdir(parents=True, exist_ok=True)

    # 确定待评测模型列表
    model_paths: List[str] = []
    if args.models_dir:
        md = Path(args.models_dir)
        files = [p for p in md.iterdir() if p.is_file() and p.suffix.lower() in {".ckpt", ".pt", ".pth"}]
        if not files:
            logging.warning(f"在 {md} 未找到*.ckpt/*.pt/*.pth")

        # 排序函数：数字优先（常为step），否则按修改时间
        def sort_key(p: Path):
            nums = re.findall(r"\d+", p.stem)
            if nums:
                return (0, int(nums[-1]), p.stem)
            return (1, int(p.stat().st_mtime), p.stem)

        ordered = sorted(files, key=sort_key)

        # 过滤逻辑：仅保留每50 step的模型
        filtered = []
        for p in ordered:
            nums = re.findall(r"\d+", p.stem)
            if nums:
                step = int(nums[-1])
                if step % 50 == 0:  # 每50 step
                    filtered.append(p)
            else:
                filtered.append(p)

        model_paths.extend([str(p) for p in filtered])
        logging.info("将按顺序评测：%s", " -> ".join(Path(p).stem for p in model_paths))
    elif args.model_path:
        model_paths.append(args.model_path)
    else:
        model_paths.append(DEFAULT_MODEL_PATH)

    # 总表：追加写入
    master_path = outdir_root / "all_models_summary.tsv"
    if not master_path.exists():
        with open(master_path, "w") as f:
            f.write(
                "model_name\ttotal_datasets\taverage_accuracy\ttotal_time_s\taverage_time_s\tavg_peak_vram_mib\tmax_peak_vram_mib\n")

    t0_all = time.perf_counter()
    for mpth in model_paths:
        t0 = time.perf_counter()
        result = evaluate_model(
            mpth,
            data_root,
            outdir_root,
            merge_val=args.merge_val,
            coerce_numeric=args.coerce_numeric,
            skip_regression=args.skip_regression
        )
        model_tag, total, avg_acc, total_t, avg_t, avg_vram, max_vram = result

        with open(master_path, "a") as f:
            avg_acc_str = f"{avg_acc:.6f}" if not np.isnan(avg_acc) else "nan"
            avg_t_str = f"{avg_t:.3f}" if not np.isnan(avg_t) else "nan"
            avg_vram_str = f"{avg_vram:.2f}" if avg_vram > 0 else "N/A"
            max_vram_str = f"{max_vram:.2f}" if max_vram > 0 else "N/A"
            f.write(
                f"{model_tag}\t{total}\t{avg_acc_str}\t{total_t:.3f}\t{avg_t_str}\t{avg_vram_str}\t{max_vram_str}\n")

        logging.info(f"[{model_tag}] 完成，耗时 {time.perf_counter() - t0:.2f}s")

    logging.info(f"全部模型完成，总耗时 {time.perf_counter() - t0_all:.2f}s")
    print("\n汇总总表：", master_path)


if __name__ == "__main__":
    main()