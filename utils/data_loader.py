"""
NILM Seq2Point 的数据加载与预处理。
"""

import numpy as np
import pandas as pd
import h5py
from typing import Tuple, List, Optional
import tables
import torch
from torch.utils.data import Dataset, DataLoader


class NILMDataset(Dataset):
    """NILM 数据的 PyTorch Dataset。"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform=None
    ):
        """
        Args:
            X: 输入序列，形状 (N, window_size)
            y: 目标值，形状 (N,)
            transform: 可选的数据变换
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y


class NILMDataLoader:
    """NILM 数据集加载器。"""
    
    def __init__(
        self,
        data_path: str,
        appliance: str,
        window_size: int = 599,
        window_stride: int = 1,
        max_power: float = None
    ):
        """
        初始化数据加载器。
        
        Args:
            data_path: HDF5 数据文件路径
            appliance: 目标电器名称
            window_size: 输入窗口长度
            window_stride: 滑窗步长
            max_power: 归一化最大功率
        """
        self.data_path = data_path
        self.appliance = appliance
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_power = max_power
        
        self.mean_aggregate = 0.0
        self.std_aggregate = 1.0

    @staticmethod
    def _cleanup_nilmtk_temp_handles(data_path: Optional[str] = None) -> None:
        """关闭 nilmtk 读取过程中遗留的 PyTables 句柄。"""
        try:
            open_handlers = list(getattr(tables.file._open_files, 'handlers', []))
        except Exception:
            return

        for handler in open_handlers:
            filename = getattr(handler, 'filename', '') or ''
            if '/tmp/nilmtk-' in filename or (data_path and data_path in filename):
                try:
                    handler.close()
                except Exception:
                    continue
        
    def load_data(
        self,
        buildings: List[int],
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 HDF5 文件加载数据。
        
        Args:
            buildings: 建筑 ID 列表
            normalize: 是否归一化
            
        Returns:
            (X, y)，X 为聚合功率窗口，y 为电器功率
        """
        with h5py.File(self.data_path, 'r') as f:
            root_keys = list(f.keys())

        if any(k.startswith('building_') for k in root_keys):
            X_list, y_list = self._load_simple_h5_data(buildings)
        elif any(k.startswith('building') for k in root_keys):
            X_list, y_list = self._load_nilmtk_h5_data(buildings)
        else:
            raise ValueError(
                f"Unsupported HDF5 structure in {self.data_path}. "
                "Expected simplified 'building_*' groups or NILMTK 'building*/elec/meter*' groups."
            )
        
        if not X_list:
            raise ValueError(f"No data loaded for buildings {buildings}")
            
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        
        # 归一化处理
        if normalize:
            self.mean_aggregate = np.mean(X)
            self.std_aggregate = np.std(X)
            X = (X - self.mean_aggregate) / (self.std_aggregate + 1e-8)
            
            if self.max_power:
                y = y / self.max_power
        
        return X, y

    def _load_simple_h5_data(self, buildings: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load data from simplified format: building_<id>/{mains,<appliance>}."""
        X_list = []
        y_list = []

        with h5py.File(self.data_path, 'r') as f:
            for building_id in buildings:
                building_key = f'building_{building_id}'

                if building_key not in f:
                    print(f"Warning: {building_key} not found in dataset")
                    continue

                mains = f[building_key]['mains'][:]

                if self.appliance not in f[building_key]:
                    print(f"Warning: {self.appliance} not found in {building_key}")
                    continue

                appliance = f[building_key][self.appliance][:]
                min_len = min(len(mains), len(appliance))
                mains = mains[:min_len]
                appliance = appliance[:min_len]

                if min_len < self.window_size:
                    print(
                        f"Warning: {building_key} has only {min_len} aligned samples, "
                        f"which is smaller than window_size={self.window_size}"
                    )
                    continue

                X_building, y_building = self._create_windows(mains, appliance)
                X_list.append(X_building)
                y_list.append(y_building)

        return X_list, y_list

    def _load_nilmtk_h5_data(self, buildings: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load data directly from NILMTK format: building<id>/elec/meter*."""
        X_list = []
        y_list = []

        try:
            from nilmtk import DataSet
        except Exception as exc:
            raise ImportError(
                "Reading NILMTK-format HDF5 requires nilmtk to be installed in the current environment."
            ) from exc

        try:
            dataset = DataSet(self.data_path)
        except AttributeError as exc:
            # 兼容仅包含 /buildingX/elec/meter* 表结构但缺少 NILMTK 根 metadata 的文件。
            if "Attribute 'metadata' does not exist" in str(exc):
                print(
                    "Warning: NILMTK root metadata not found. "
                    "Falling back to table-based reader for /buildingX/elec/meter*."
                )
                self._cleanup_nilmtk_temp_handles(self.data_path)
                return self._load_nilmtk_table_h5_data(buildings)
            self._cleanup_nilmtk_temp_handles(self.data_path)
            raise

        try:
            available_buildings = set(dataset.buildings.keys())
            for building_id in buildings:
                if building_id not in available_buildings:
                    print(f"Warning: building{building_id} not found in dataset")
                    continue

                building = dataset.buildings[building_id]
                elec = building.elec

                mains_series = self._to_1d_series(elec.mains().power_series_all_data())
                appliance_series = self._get_appliance_series_from_nilmtk(elec)

                if appliance_series is None:
                    print(f"Warning: {self.appliance} not found in building{building_id}")
                    continue

                aligned = self._align_nilmtk_series(mains_series, appliance_series)
                if len(aligned) < self.window_size:
                    print(
                        f"Warning: building{building_id} has only {len(aligned)} aligned samples, "
                        f"which is smaller than window_size={self.window_size}"
                    )
                    continue

                mains = aligned['mains'].to_numpy(dtype=np.float32)
                appliance = aligned['appliance'].to_numpy(dtype=np.float32)
                X_building, y_building = self._create_windows(mains, appliance)
                X_list.append(X_building)
                y_list.append(y_building)
        finally:
            if hasattr(dataset, 'store') and dataset.store is not None:
                dataset.store.close()
            self._cleanup_nilmtk_temp_handles(self.data_path)

        return X_list, y_list

    def _load_nilmtk_table_h5_data(self, buildings: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load NILMTK-like table data without requiring NILMTK metadata."""
        X_list = []
        y_list = []

        with pd.HDFStore(self.data_path, mode='r') as store:
            available_keys = set(store.keys())

            for building_id in buildings:
                mains_key = f'/building{building_id}/elec/meter1'
                app_key = f'/building{building_id}/elec/meter2'

                if mains_key not in available_keys:
                    print(f"Warning: {mains_key} not found in dataset")
                    continue

                if app_key not in available_keys:
                    print(
                        f"Warning: {app_key} not found in dataset. "
                        "For metadata-free NILMTK-like files, meter2 is expected as target appliance."
                    )
                    continue

                mains_series = self._to_1d_series(store.get(mains_key))
                appliance_series = self._to_1d_series(store.get(app_key))

                aligned = self._align_nilmtk_series(mains_series, appliance_series)
                if len(aligned) < self.window_size:
                    print(
                        f"Warning: building{building_id} has only {len(aligned)} aligned samples, "
                        f"which is smaller than window_size={self.window_size}"
                    )
                    continue

                mains = aligned['mains'].to_numpy(dtype=np.float32)
                appliance = aligned['appliance'].to_numpy(dtype=np.float32)
                X_building, y_building = self._create_windows(mains, appliance)
                X_list.append(X_building)
                y_list.append(y_building)

        self._cleanup_nilmtk_temp_handles(self.data_path)
        return X_list, y_list

    @staticmethod
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

    def _get_appliance_series_from_nilmtk(self, elec) -> Optional[pd.Series]:
        """Try to resolve target appliance series from NILMTK meter group."""
        try:
            appliance_meter = elec[self.appliance]
            return self._to_1d_series(appliance_meter.power_series_all_data())
        except Exception:
            pass

        try:
            for meter in elec.submeters().meters:
                for app in getattr(meter, 'appliances', []):
                    app_type = getattr(app, 'type', None)
                    app_name = app_type.get('type') if isinstance(app_type, dict) else app_type
                    if app_name == self.appliance:
                        return self._to_1d_series(meter.power_series_all_data())
        except Exception:
            return None

        return None

    @staticmethod
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
    
    def _create_windows(
        self,
        aggregate: np.ndarray,
        appliance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从序列数据生成滑动窗口。
        
        Args:
            aggregate: 聚合功率
            appliance: 电器功率
            
        Returns:
            窗口化后的 (X, y)
        """
        X = []
        y = []
        
        # 目标为窗口中点
        offset = self.window_size // 2
        
        for i in range(0, len(aggregate) - self.window_size + 1, self.window_stride):
            window = aggregate[i:i + self.window_size]
            target = appliance[i + offset]
            
            X.append(window)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def denormalize_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        将预测值反归一化回原始尺度。
        
        Args:
            predictions: 归一化后的预测值
            
        Returns:
            反归一化后的预测值
        """
        if self.max_power:
            return predictions * self.max_power
        return predictions
    
    def denormalize_aggregate(
        self,
        aggregate: np.ndarray
    ) -> np.ndarray:
        """
        将聚合功率反归一化回原始尺度。
        
        Args:
            aggregate: 归一化后的聚合功率
            
        Returns:
            反归一化后的聚合功率
        """
        return aggregate * self.std_aggregate + self.mean_aggregate


def create_dataloaders(
    data_loader: NILMDataLoader,
    train_buildings: List[int],
    test_buildings: List[int],
    batch_size: int = 256,
    val_split: float = 0.1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = None
) -> dict:
    """
    创建训练、验证和测试数据加载器。
    
    Args:
        data_loader: NILMDataLoader 实例
        train_buildings: 训练建筑列表
        test_buildings: 测试建筑列表
        batch_size: 批大小
        val_split: 验证集比例
        num_workers: 数据加载线程数
        shuffle: 是否打乱训练数据
        pin_memory: 是否开启 pin_memory，默认自动判断 (torch.cuda.is_available())
        
    Returns:
        包含 'train'/'val'/'test' 的字典
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # 加载训练数据
    X_train, y_train = data_loader.load_data(train_buildings, normalize=True)
    
    # 切分训练/验证集
    val_size = int(len(X_train) * val_split)
    
    if shuffle:
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    # 加载测试数据
    X_test, y_test = data_loader.load_data(test_buildings, normalize=True)
    
    # 构建数据集
    train_dataset = NILMDataset(X_train, y_train)
    val_dataset = NILMDataset(X_val, y_val)
    test_dataset = NILMDataset(X_test, y_test)
    
    # 构建数据加载器
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return dataloaders
