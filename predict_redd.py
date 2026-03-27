"""
使用实验权重对 REDD 数据进行预测，并保存为可复用训练数据。

默认行为：
1) 使用 --exp 对应权重
2) 预测 config 中 train_buildings
3) 保存到 data/，文件名：数据集-算法-训练房间.h5
4) 文件内仅写入 NILMTK 风格键

示例:
    python predict_redd.py --exp 32 --weight-name best.pth
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
import torch

from inference import Seq2PointInference, Seq2PointParallelInference
from utils.config_utils import load_config


DEFAULT_REDD_FALLBACK_PATHS = [
    'data/redd_simple.h5',
    'data/redd.h5',
]


def _cleanup_nilmtk_temp_handles() -> None:
    """关闭 nilmtk 读取过程中遗留的 /tmp/nilmtk-*.h5 句柄。"""
    try:
        open_handlers = list(getattr(tables.file._open_files, 'handlers', []))
    except Exception:
        return

    for handler in open_handlers:
        filename = getattr(handler, 'filename', '') or ''
        if '/tmp/nilmtk-' in filename:
            try:
                handler.close()
            except Exception:
                continue


def resolve_exp_name(exp: Optional[str]) -> Optional[str]:
    """兼容 --exp 33 与 --exp exp33 两种写法。"""
    if exp is None:
        return None

    candidates = [exp]
    if not exp.startswith('exp'):
        candidates.insert(0, f"exp{exp}")

    for name in candidates:
        if os.path.isdir(os.path.join('runs', name)):
            return name

    return exp


def resolve_checkpoint_and_config(
    checkpoint: Optional[str],
    exp: Optional[str],
    weight_name: str,
    config_path: Optional[str]
) -> Tuple[str, Optional[str]]:
    """解析 checkpoint 和 config 路径。"""
    resolved_exp = resolve_exp_name(exp)

    checkpoint_path = checkpoint
    if checkpoint_path is None:
        if resolved_exp is None:
            raise ValueError("Please provide either --checkpoint or --exp")
        checkpoint_path = os.path.join('runs', resolved_exp, 'weights', weight_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config_path is not None:
        return checkpoint_path, config_path

    if resolved_exp is not None:
        exp_config = os.path.join('runs', resolved_exp, 'config.yaml')
        if os.path.exists(exp_config):
            return checkpoint_path, exp_config

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) == 'weights':
        inferred_cfg = os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml')
        if os.path.exists(inferred_cfg):
            return checkpoint_path, inferred_cfg

    return checkpoint_path, None


def _load_mains_from_simple_h5(data_path: str, building: int) -> np.ndarray:
    """从简化格式读取 mains: building_<id>/mains。"""
    with h5py.File(data_path, 'r') as f:
        group_key = f'building_{building}'
        if group_key not in f:
            raise ValueError(f"{group_key} not found in {data_path}")
        if 'mains' not in f[group_key]:
            raise ValueError(f"mains not found in {group_key}")
        return f[group_key]['mains'][:].astype(np.float32)


def _load_appliance_from_simple_h5(data_path: str, building: int, appliance: str) -> np.ndarray:
    """从简化格式读取电器功率: building_<id>/<appliance>。"""
    with h5py.File(data_path, 'r') as f:
        group_key = f'building_{building}'
        if group_key not in f:
            raise ValueError(f"{group_key} not found in {data_path}")
        if appliance not in f[group_key]:
            raise ValueError(f"{appliance} not found in {group_key}")
        return f[group_key][appliance][:].astype(np.float32)


def _load_mains_from_nilmtk_h5(data_path: str, building: int) -> np.ndarray:
    """从 NILMTK 格式读取 mains: building<id>/elec/meter*（通过 nilmtk API）。"""
    try:
        from nilmtk import DataSet
    except Exception as exc:
        raise ImportError(
            "Reading NILMTK-format REDD data requires nilmtk. "
            "Please install nilmtk or use a simplified HDF5 file."
        ) from exc

    dataset = DataSet(data_path)
    try:
        if building not in dataset.buildings:
            raise ValueError(f"building{building} not found in {data_path}")
        mains = dataset.buildings[building].elec.mains().power_series_all_data()
        if hasattr(mains, 'dropna'):
            mains = mains.dropna()
        if hasattr(mains, 'to_numpy'):
            mains = mains.to_numpy()
        mains = np.asarray(mains, dtype=np.float32).reshape(-1)
        return mains
    finally:
        if hasattr(dataset, 'store') and dataset.store is not None:
            dataset.store.close()
        _cleanup_nilmtk_temp_handles()


def _load_appliance_from_nilmtk_h5(data_path: str, building: int, appliance: str) -> np.ndarray:
    """从 NILMTK 格式读取电器功率: building<id>/elec/meter*（通过 nilmtk API）。"""
    try:
        from nilmtk import DataSet
    except Exception as exc:
        raise ImportError(
            "Reading NILMTK-format REDD data requires nilmtk. "
            "Please install nilmtk or use a simplified HDF5 file."
        ) from exc

    dataset = DataSet(data_path)
    try:
        if building not in dataset.buildings:
            raise ValueError(f"building{building} not found in {data_path}")

        elec = dataset.buildings[building].elec
        try:
            series = elec[appliance].power_series_all_data()
        except Exception as exc:
            raise ValueError(f"{appliance} not found in building{building}") from exc

        if hasattr(series, 'dropna'):
            series = series.dropna()
        if hasattr(series, 'to_numpy'):
            series = series.to_numpy()
        return np.asarray(series, dtype=np.float32).reshape(-1)
    finally:
        if hasattr(dataset, 'store') and dataset.store is not None:
            dataset.store.close()
        _cleanup_nilmtk_temp_handles()


def load_redd_mains(
    data_path: str,
    building: int,
    start: int = 0,
    length: Optional[int] = None
) -> np.ndarray:
    """读取 REDD 指定 building 的 mains 波形并裁剪。"""
    with h5py.File(data_path, 'r') as f:
        root_keys = list(f.keys())

    if any(k.startswith('building_') for k in root_keys):
        mains = _load_mains_from_simple_h5(data_path, building)
    elif any(k.startswith('building') for k in root_keys):
        mains = _load_mains_from_nilmtk_h5(data_path, building)
    else:
        raise ValueError(f"Unsupported HDF5 structure: {data_path}")

    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= len(mains):
        raise ValueError(f"--start {start} out of range. Data length={len(mains)}")

    if length is None:
        return mains[start:]
    if length <= 0:
        raise ValueError("--length must be > 0")

    end = min(start + length, len(mains))
    return mains[start:end]


def load_redd_appliance(
    data_path: str,
    building: int,
    appliance: str,
    start: int = 0,
    length: Optional[int] = None
) -> np.ndarray:
    """读取 REDD 指定 building 的原始电器波形并裁剪。"""
    with h5py.File(data_path, 'r') as f:
        root_keys = list(f.keys())

    if any(k.startswith('building_') for k in root_keys):
        values = _load_appliance_from_simple_h5(data_path, building, appliance)
    elif any(k.startswith('building') for k in root_keys):
        values = _load_appliance_from_nilmtk_h5(data_path, building, appliance)
    else:
        raise ValueError(f"Unsupported HDF5 structure: {data_path}")

    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= len(values):
        raise ValueError(f"--start {start} out of range. Data length={len(values)}")

    if length is None:
        return values[start:]
    if length <= 0:
        raise ValueError("--length must be > 0")

    end = min(start + length, len(values))
    return values[start:end]


def _unique_existing_paths(paths: List[str]) -> List[str]:
    """返回去重后的存在路径，保持原始顺序。"""
    seen = set()
    valid_paths: List[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            valid_paths.append(path)
    return valid_paths


def _load_with_fallback(
    loader,
    candidate_paths: List[str],
    building: int,
    what: str,
    appliance: Optional[str] = None,
    start: int = 0,
    length: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """按候选路径依次尝试读取 building 数据，返回数据与实际命中的路径。"""
    errors: List[str] = []

    for path in _unique_existing_paths(candidate_paths):
        try:
            if appliance is None:
                values = loader(path, building, start=start, length=length)
            else:
                values = loader(path, building, appliance, start=start, length=length)
            return values, path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    error_text = '; '.join(errors) if errors else 'no existing candidate file'
    raise ValueError(
        f"Failed to load {what} for building {building}. Tried paths: {candidate_paths}. Details: {error_text}"
    )


def _power_df_to_array(df: pd.DataFrame) -> np.ndarray:
    """从 NILMTK 风格表中提取功率列为 1D float32。"""
    if isinstance(df, pd.Series):
        values = df.to_numpy()
    else:
        values = df.iloc[:, 0].to_numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _to_1d_series(data) -> pd.Series:
    """Convert NILMTK meter output to a 1D pandas Series."""
    if isinstance(data, pd.Series):
        series = data
    elif isinstance(data, pd.DataFrame):
        series = data.iloc[:, 0]
    else:
        series = pd.Series(data)

    series = series.dropna()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series


def _align_nilmtk_series(mains: pd.Series, appliance: pd.Series) -> pd.DataFrame:
    """Align mains and appliance by timestamp; fallback to minute resampling when needed."""
    mains = mains.sort_index()
    appliance = appliance.sort_index()

    aligned = pd.concat(
        [mains.rename('mains'), appliance.rename('appliance')],
        axis=1,
        join='inner'
    ).dropna()

    if not aligned.empty:
        return aligned

    if isinstance(mains.index, pd.DatetimeIndex) and isinstance(appliance.index, pd.DatetimeIndex):
        mains_resampled = mains.resample('1T').mean().fillna(0)
        appliance_resampled = appliance.resample('1T').mean().fillna(0)
        return pd.concat(
            [mains_resampled.rename('mains'), appliance_resampled.rename('appliance')],
            axis=1
        ).fillna(0)

    min_len = min(len(mains), len(appliance))
    return pd.DataFrame({
        'mains': mains.to_numpy()[:min_len],
        'appliance': appliance.to_numpy()[:min_len],
    })


def _series_to_power_df(series: pd.Series) -> pd.DataFrame:
    """Convert series to NILMTK-like table format (power/active)."""
    values = np.asarray(series.to_numpy(), dtype=np.float32).reshape(-1)
    columns = pd.MultiIndex.from_tuples([('power', 'active')])
    return pd.DataFrame(values.reshape(-1, 1), index=series.index, columns=columns)


def _slice_aligned_df_by_pos(aligned: pd.DataFrame, start: int = 0, length: Optional[int] = None) -> pd.DataFrame:
    """Slice aligned dataframe by positional start/length."""
    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= len(aligned):
        raise ValueError(f"--start {start} out of range. Data length={len(aligned)}")

    if length is None:
        return aligned.iloc[start:]
    if length <= 0:
        raise ValueError("--length must be > 0")

    end = min(start + length, len(aligned))
    return aligned.iloc[start:end]


def _load_aligned_nilmtk_pair_df(
    data_path: str,
    building: int,
    appliance: str,
    start: int = 0,
    length: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load mains/appliance aligned pairs from NILMTK source using same semantics as training loader."""
    try:
        from nilmtk import DataSet
    except Exception as exc:
        raise ImportError(
            "Reading NILMTK-format REDD data requires nilmtk. "
            "Please install nilmtk or use a simplified HDF5 file."
        ) from exc

    try:
        dataset = DataSet(data_path)
    except Exception:
        _cleanup_nilmtk_temp_handles()
        raise
    try:
        if building not in dataset.buildings:
            raise ValueError(f"building{building} not found in {data_path}")

        elec = dataset.buildings[building].elec
        mains_series = _to_1d_series(elec.mains().power_series_all_data())

        appliance_series = None
        try:
            appliance_series = _to_1d_series(elec[appliance].power_series_all_data())
        except Exception:
            for meter in elec.submeters().meters:
                found = False
                for app in getattr(meter, 'appliances', []):
                    app_type = getattr(app, 'type', None)
                    app_name = app_type.get('type') if isinstance(app_type, dict) else app_type
                    if app_name == appliance:
                        appliance_series = _to_1d_series(meter.power_series_all_data())
                        found = True
                        break
                if found:
                    break

        if appliance_series is None:
            raise ValueError(f"{appliance} not found in building{building}")

        aligned = _align_nilmtk_series(mains_series, appliance_series)
        aligned = _slice_aligned_df_by_pos(aligned, start=start, length=length)

        mains_df = _series_to_power_df(aligned['mains'])
        target_df = _series_to_power_df(aligned['appliance'])
        return mains_df, target_df
    finally:
        if hasattr(dataset, 'store') and dataset.store is not None:
            dataset.store.close()
        _cleanup_nilmtk_temp_handles()


def _load_nilmtk_meter1_df(
    data_path: str,
    building: int,
    start: int = 0,
    length: Optional[int] = None,
) -> pd.DataFrame:
    """Load NILMTK meter1 table for a building and preserve original index."""
    key = f'/building{building}/elec/meter1'
    with pd.HDFStore(data_path, mode='r') as store:
        if key not in store.keys():
            raise ValueError(f"{key} not found in {data_path}")
        df = store.get(key)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if start < 0:
        raise ValueError("--start must be >= 0")
    if start >= len(df):
        raise ValueError(f"--start {start} out of range. Data length={len(df)}")

    if length is None:
        return df.iloc[start:]
    if length <= 0:
        raise ValueError("--length must be > 0")
    end = min(start + length, len(df))
    return df.iloc[start:end]


def auto_device() -> str:
    """自动选择可用设备。"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def resolve_algorithm_name(config: dict, checkpoint: dict) -> str:
    """推断算法名称用于文件命名。"""
    is_parallel = (
        config.get('training', {}).get('mode', 'single') == 'parallel'
        or config.get('model', {}).get('name') == 'parallel'
        or 'seq2point_state_dict' in checkpoint
    )
    if is_parallel:
        return 'parallel'
    return config.get('model', {}).get('name', 'seq2point')


def resolve_target_buildings(args, config: dict) -> List[int]:
    """解析要预测的 building 列表。"""
    if args.rooms:
        return [int(b) for b in args.rooms]

    building = args.room if args.room is not None else args.building
    if building is not None:
        return [int(building)]

    train_buildings = config.get('data', {}).get('train_buildings', [])
    if train_buildings:
        return [int(b) for b in train_buildings]

    if 'building' in config.get('data', {}):
        return [int(config['data']['building'])]

    test_buildings = config.get('data', {}).get('test_buildings', [])
    if test_buildings:
        return [int(test_buildings[0])]

    raise ValueError('Please provide --room/--building/--rooms, or set train_buildings in config')


def _to_power_df(values: np.ndarray) -> pd.DataFrame:
    """构造 NILMTK 常见的 power/active 列结构。"""
    time_index = pd.date_range('2011-01-01', periods=len(values), freq='s')
    columns = pd.MultiIndex.from_tuples([('power', 'active')])
    return pd.DataFrame(values.reshape(-1, 1), index=time_index, columns=columns)


def save_predictions_as_h5(
    output_path: str,
    appliance_name: str,
    dataset_name: str,
    algorithm_name: str,
    train_buildings: List[int],
    test_buildings: List[int],
    by_building_data: Dict[int, Dict[str, Any]],
) -> None:
    """
    保存预测结果为 HDF5（严格 NILMTK 风格）：
    /buildingX/elec/meter1, /buildingX/elec/meter2
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with pd.HDFStore(output_path, mode='w') as store:
        for building_id, payload in by_building_data.items():
            mains_df = payload.get('mains_df')
            app_df = payload.get('target_df')

            if mains_df is None:
                aggregate = payload['aggregate'].astype(np.float32)
                mains_df = _to_power_df(aggregate)

            mains_key = f'/building{building_id}/elec/meter1'
            app_key = f'/building{building_id}/elec/meter2'

            store.put(mains_key, mains_df, format='table')
            if app_df is not None:
                store.put(app_key, app_df, format='table')

    with h5py.File(output_path, 'a') as f:
        f.attrs['dataset'] = dataset_name
        f.attrs['appliance'] = appliance_name
        f.attrs['algorithm'] = algorithm_name
        f.attrs['train_buildings'] = np.array(train_buildings, dtype=np.int32)
        f.attrs['test_buildings'] = np.array(test_buildings, dtype=np.int32)


def build_default_output_path(config: dict, checkpoint: dict) -> str:
    """构建默认输出文件名：数据集-电器-算法-训练房间.h5（保存到 data/）。"""
    dataset_name = str(config.get('data', {}).get('dataset', 'dataset'))
    appliance_name = str(config.get('data', {}).get('appliance', 'appliance'))
    algorithm_name = resolve_algorithm_name(config, checkpoint)
    train_buildings = [str(b) for b in config.get('data', {}).get('train_buildings', [])]
    train_room_tag = '_'.join(train_buildings) if train_buildings else 'none'
    filename = f'{dataset_name}-{appliance_name}-{algorithm_name}-{train_room_tag}.h5'
    return os.path.join('data', filename)


def plot_predictions(
    aggregate: np.ndarray,
    appliance_name: str,
    pred_single: Optional[np.ndarray] = None,
    pred_seq2point: Optional[np.ndarray] = None,
    pred_transformer: Optional[np.ndarray] = None,
    pred_sum: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """绘制预测波形。"""
    if pred_single is not None:
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        x = np.arange(len(aggregate))

        axes[0].plot(x, aggregate, color='tab:blue', alpha=0.7, label='Aggregate')
        axes[0].set_title('Aggregate Power')
        axes[0].set_ylabel('Power (W)')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(x, pred_single, color='tab:red', alpha=0.7, label=f'{appliance_name} Predicted')
        axes[1].set_title(f'{appliance_name} Prediction')
        axes[1].set_xlabel('Time (samples)')
        axes[1].set_ylabel('Power (W)')
        axes[1].grid(True)
        axes[1].legend()

    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        x = np.arange(len(aggregate))

        axes[0, 0].plot(x, aggregate, color='tab:blue', alpha=0.7)
        axes[0, 0].set_title('Aggregate Power')
        axes[0, 0].grid(True)

        axes[0, 1].plot(x, pred_seq2point, color='tab:green', alpha=0.7)
        axes[0, 1].set_title('Seq2Point Prediction')
        axes[0, 1].grid(True)

        axes[1, 0].plot(x, pred_transformer, color='tab:orange', alpha=0.7)
        axes[1, 0].set_title('Transformer Prediction')
        axes[1, 0].grid(True)

        axes[1, 1].plot(x, pred_sum, color='tab:purple', alpha=0.7)
        axes[1, 1].set_title('Sum Prediction')
        axes[1, 1].grid(True)

        for ax in axes.flat:
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Power (W)')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict REDD waveform and export reusable HDF5 dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (.pth)')
    parser.add_argument('--exp', type=str, default=None, help='Experiment under runs/ (e.g. exp33 or 33)')
    parser.add_argument(
        '--weight-name',
        type=str,
        default='best.pth',
        choices=['best.pth', 'best_metrics.pth', 'last.pth'],
        help='Weight filename used with --exp'
    )
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--data-path', type=str, default=None, help='REDD HDF5 path (default from config)')
    parser.add_argument('--rooms', type=int, nargs='+', default=None, help='Batch predict multiple buildings')
    parser.add_argument('--room', type=int, default=None, help='REDD 房间编号（在本项目中映射为 building id）')
    parser.add_argument('--building', type=int, default=None, help='与 --room 等价，优先级低于 --room')
    parser.add_argument('--start', type=int, default=0, help='起始样本索引')
    parser.add_argument('--length', type=int, default=None, help='读取样本长度，不填表示读取到末尾')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='Inference device')
    parser.add_argument('--infer-batch-size', type=int, default=None, help='Batch size for window inference (default: read from config training.batch_size)')
    parser.add_argument('--output', type=str, default=None, help='Output h5 path')
    parser.add_argument('--plot', action='store_true', help='Show prediction waveform plot')
    parser.add_argument('--plot-save', type=str, default=None, help='Save plot to png path')

    args = parser.parse_args()

    checkpoint_path, config_path = resolve_checkpoint_and_config(
        checkpoint=args.checkpoint,
        exp=args.exp,
        weight_name=args.weight_name,
        config_path=args.config
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if config_path is not None:
        config = load_config(config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError('No config found. Please provide --config')

    target_buildings = resolve_target_buildings(args, config)
    train_buildings = [int(b) for b in config.get('data', {}).get('train_buildings', [])]
    test_buildings = [int(b) for b in config.get('data', {}).get('test_buildings', [])]
    infer_batch_size = args.infer_batch_size
    if infer_batch_size is None:
        infer_batch_size = int(config.get('training', {}).get('batch_size', 4096))
    if infer_batch_size <= 0:
        raise ValueError('--infer-batch-size must be > 0')

    data_path = args.data_path or config['data']['data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found: {data_path}')

    candidate_data_paths = [data_path] + DEFAULT_REDD_FALLBACK_PATHS
    primary_data_path = _unique_existing_paths(candidate_data_paths)[0]

    device = args.device or auto_device()

    is_parallel = (
        config.get('training', {}).get('mode', 'single') == 'parallel'
        or config.get('model', {}).get('name') == 'parallel'
        or 'seq2point_state_dict' in checkpoint
    )

    if is_parallel:
        inferencer = Seq2PointParallelInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )
    else:
        inferencer = Seq2PointInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )

    by_building_data: Dict[int, Dict[str, Any]] = {}
    for building in target_buildings:
        try:
            mains_df, target_raw_df = _load_aligned_nilmtk_pair_df(
                data_path=primary_data_path,
                building=building,
                appliance=config['data']['appliance'],
                start=args.start,
                length=args.length,
            )
            used_path = primary_data_path

        except ValueError as exc:
            # 若原始数据不存在目标电器，保持缺失：仅保留 mains，不生成预测电器曲线。
            if 'not found in building' in str(exc):
                mains_only_df = _load_nilmtk_meter1_df(
                    data_path=primary_data_path,
                    building=building,
                    start=args.start,
                    length=args.length,
                )
                by_building_data[building] = {
                    'mains_df': mains_only_df,
                    'aggregate': _power_df_to_array(mains_only_df),
                }
                print(
                    f'Skipped appliance replacement for building {building}: '
                    f'{config["data"]["appliance"]} missing in source (source: {primary_data_path})'
                )
                continue
            raise

        aggregate = _power_df_to_array(mains_df)
        if is_parallel:
            _, _, pred_sum = inferencer.predict_sequence(
                aggregate,
                batch_size=infer_batch_size,
                show_progress=True
            )
            target_pred = pred_sum
        else:
            target_pred = inferencer.predict_sequence(
                aggregate,
                batch_size=infer_batch_size,
                show_progress=True
            )

        replace_len = min(len(target_raw_df), len(target_pred))
        target_df = target_raw_df.iloc[:replace_len].copy()
        target_df.iloc[:, 0] = target_pred[:replace_len].astype(np.float32)

        mains_df_trim = mains_df.iloc[:replace_len]
        by_building_data[building] = {
            'mains_df': mains_df_trim,
            'target_df': target_df,
            'aggregate': _power_df_to_array(mains_df_trim),
            'target': _power_df_to_array(target_df),
        }
        print(
            f'Replaced appliance for building {building}: rows={replace_len} '
            f'(aligned source: {used_path})'
        )

    # 保持测试集原始标签不变：将 test_buildings 写入输出，并使用原始 appliance 数据。
    for building in test_buildings:
        if building in by_building_data:
            continue

        try:
            mains_df, target_df = _load_aligned_nilmtk_pair_df(
                data_path=primary_data_path,
                building=building,
                appliance=config['data']['appliance'],
                start=args.start,
                length=args.length,
            )
            mains_path = primary_data_path

            aligned_len = min(len(mains_df), len(target_df))
            mains_df = mains_df.iloc[:aligned_len]
            target_df = target_df.iloc[:aligned_len]
            by_building_data[building] = {
                'mains_df': mains_df,
                'target_df': target_df,
                'aggregate': _power_df_to_array(mains_df),
                'target': _power_df_to_array(target_df),
            }
            print(
                f'Kept raw test building {building}: rows={aligned_len} '
                f'(aligned source: {mains_path})'
            )
        except ValueError as exc:
            if 'not found in building' in str(exc):
                mains_only_df = _load_nilmtk_meter1_df(
                    data_path=primary_data_path,
                    building=building,
                    start=args.start,
                    length=args.length,
                )
                by_building_data[building] = {
                    'mains_df': mains_only_df,
                    'aggregate': _power_df_to_array(mains_only_df),
                }
                print(
                    f'Kept test building {building} without appliance meter: '
                    f'{config["data"]["appliance"]} missing in source (source: {primary_data_path})'
                )
                continue
            raise

        except Exception:
            # 无法按 NILMTK 表复制时，回退到原有数组读取路径。
            aggregate, mains_path = _load_with_fallback(
                loader=load_redd_mains,
                candidate_paths=candidate_data_paths,
                building=building,
                what='test aggregate',
                start=args.start,
                length=args.length
            )
            target_raw, target_path = _load_with_fallback(
                loader=load_redd_appliance,
                candidate_paths=[mains_path] + candidate_data_paths,
                building=building,
                what='test raw appliance',
                appliance=config['data']['appliance'],
                start=args.start,
                length=args.length
            )

            min_len = min(len(aggregate), len(target_raw))
            by_building_data[building] = {
                'aggregate': aggregate[:min_len],
                'target': target_raw[:min_len]
            }
            print(
                f'Kept raw test building {building}: {min_len} samples '
                f'(mains source: {mains_path}, appliance source: {target_path})'
            )

    algorithm_name = resolve_algorithm_name(config, checkpoint)
    output_path = args.output or build_default_output_path(config, checkpoint)
    save_predictions_as_h5(
        output_path=output_path,
        appliance_name=config['data']['appliance'],
        dataset_name=str(config['data'].get('dataset', 'dataset')),
        algorithm_name=algorithm_name,
        train_buildings=train_buildings,
        test_buildings=test_buildings,
        by_building_data=by_building_data,
    )

    print('Prediction complete')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Data path: {data_path}')
    print(f'Target buildings: {target_buildings}')
    print(f'Inference batch size: {infer_batch_size}')
    print(f'Output (HDF5): {output_path}')

    if args.plot or args.plot_save:
        first_building = target_buildings[0]
        payload = by_building_data[first_building]
        plot_predictions(
            aggregate=payload['aggregate'],
            appliance_name=config['data']['appliance'],
            pred_single=payload['target'] if not is_parallel else None,
            pred_seq2point=None,
            pred_transformer=None,
            pred_sum=payload['target'] if is_parallel else None,
            save_path=args.plot_save,
        )


if __name__ == '__main__':
    main()
