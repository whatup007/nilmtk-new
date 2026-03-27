"""
将 NILMTK 格式 HDF5 转换为 Seq2Point 使用的简化格式。
该脚本从 NILMTK 数据中抽取并生成更简洁的结构。
"""

import h5py
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm


# NILMTK 仪表编号到电器名称映射（REDD 数据集）
REDD_APPLIANCE_MAP = {
    1: {  # Building 1
        5: 'fridge',
        6: 'dishwasher',
        8: 'microwave',
        11: 'washer_dryer',
        20: 'lighting',
    },
    2: {  # Building 2
        5: 'lighting',
        6: 'washer_dryer',
        7: 'microwave',
        8: 'fridge',
        9: 'dishwasher',
    },
    3: {  # Building 3
        5: 'electric_heat',
        6: 'washer_dryer',
        7: 'fridge',
        8: 'dishwasher',
        9: 'disposal',
    },
    4: {  # Building 4
        5: 'lighting',
        6: 'furnace',
        7: 'kitchen',
        8: 'washer_dryer',
    },
    5: {  # Building 5
        5: 'lighting',
        6: 'bathroom_gfi',
        7: 'electric_heat',
        8: 'stove',
        9: 'fridge',
    },
    6: {  # Building 6
        5: 'kitchen_outlets',
        6: 'washer_dryer',
        7: 'electric_heat',
        8: 'stove',
        9: 'kitchen_outlets_2',
    }
}


def read_nilmtk_dataset(file_path: str, building: int, meter: int):
    """
    从 NILMTK 格式 HDF5 读取指定电表数据。
    
    Args:
        file_path: HDF5 文件路径
        building: 建筑编号
        meter: 电表编号
        
    Returns:
        功率数据数组
    """
    try:
        # 使用 pandas 读取 PyTables 格式
        meter_path = f'building{building}/elec/meter{meter}'
        
        # 通过 pandas 读取
        df = pd.read_hdf(file_path, meter_path)
        
        # 提取功率列（通常在第一列）
        if len(df.columns) > 0:
            power = df.iloc[:, 0].values
            # 将 NaN 替换为 0
            power = np.nan_to_num(power, nan=0.0)
            return power
        else:
            return None
            
    except (KeyError, FileNotFoundError, ValueError) as e:
        return None
    except Exception as e:
        print(f"    Error reading meter {meter}: {e}")
        return None


def convert_nilmtk_to_simple(
    input_file: str,
    output_file: str,
    buildings: list = None,
    appliance_map: dict = None
):
    """
    将 NILMTK 格式转换为简化格式。
    
    Args:
        input_file: 输入 NILMTK HDF5 文件
        output_file: 输出简化 HDF5 文件
        buildings: 需要处理的建筑列表（默认全部）
        appliance_map: 自定义电器映射
    """
    if appliance_map is None:
        appliance_map = REDD_APPLIANCE_MAP
    
    if buildings is None:
        buildings = list(appliance_map.keys())
    
    print(f"Converting {input_file} to {output_file}")
    print(f"Buildings to process: {buildings}\n")
    
    with h5py.File(output_file, 'w') as f_out:
        for building_id in buildings:
            print(f"Processing building {building_id}...")
            
            # 创建建筑分组
            building_grp = f_out.create_group(f'building_{building_id}')
            
            # 读取聚合功率（meter 1/2 通常为 mains）
            print("  Reading mains...")
            mains1 = read_nilmtk_dataset(input_file, building_id, 1)
            mains2 = read_nilmtk_dataset(input_file, building_id, 2)
            
            if mains1 is None and mains2 is None:
                print(f"  Warning: No mains data found for building {building_id}")
                continue
            
            # 合并两路 mains
            if mains1 is not None and mains2 is not None:
                mains = mains1 + mains2
            elif mains1 is not None:
                mains = mains1
            else:
                mains = mains2
            
            # 保存 mains
            building_grp.create_dataset('mains', data=mains, compression='gzip')
            print(f"  Mains: {len(mains):,} samples, Mean: {np.mean(mains):.2f}W")
            
            # 读取电器功率
            if building_id in appliance_map:
                for meter_num, appliance_name in appliance_map[building_id].items():
                    print(f"  Reading {appliance_name} (meter {meter_num})...")
                    appliance_data = read_nilmtk_dataset(input_file, building_id, meter_num)
                    
                    if appliance_data is not None:
                        # 对齐与 mains 长度
                        min_len = min(len(mains), len(appliance_data))
                        appliance_data = appliance_data[:min_len]
                        
                        # 保存电器数据
                        building_grp.create_dataset(
                            appliance_name,
                            data=appliance_data,
                            compression='gzip'
                        )
                        print(f"    └─ {appliance_name}: {len(appliance_data):,} samples, "
                              f"Mean: {np.mean(appliance_data):.2f}W, "
                              f"Max: {np.max(appliance_data):.2f}W")
                    else:
                        print(f"    └─ {appliance_name}: Not found")
            
            print()
    
    print(f"✅ Conversion complete! Saved to {output_file}")


def inspect_converted_file(file_path: str):
    """查看转换后的文件内容。"""
    print(f"\n{'='*70}")
    print(f"Inspecting converted file: {file_path}")
    print(f"{'='*70}\n")
    
    with h5py.File(file_path, 'r') as f:
        buildings = [k for k in f.keys() if k.startswith('building_')]
        
        print(f"Found {len(buildings)} building(s):\n")
        
        for building_key in sorted(buildings):
            building = f[building_key]
            print(f"{building_key}:")
            
            for dataset_name in building.keys():
                data = building[dataset_name][:]
                print(f"  - {dataset_name}: {len(data):,} samples, "
                      f"Mean={np.mean(data):.2f}W, Max={np.max(data):.2f}W")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Convert NILMTK format HDF5 to simplified format'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/redd.h5',
        help='Input NILMTK HDF5 file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/redd_simple.h5',
        help='Output simplified HDF5 file'
    )
    parser.add_argument(
        '--buildings',
        type=int,
        nargs='+',
        default=None,
        help='Buildings to convert (default: all)'
    )
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Inspect the output file after conversion'
    )
    
    args = parser.parse_args()
    
    # 执行转换
    convert_nilmtk_to_simple(
        args.input,
        args.output,
        args.buildings
    )
    
    # 按需检查输出文件
    if args.inspect:
        inspect_converted_file(args.output)


if __name__ == '__main__':
    main()
