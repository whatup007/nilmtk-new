"""
数据预处理脚本，用于生成训练所需的数据集。
"""

import os
import sys
import argparse
import h5py
import numpy as np
from nilmtk import DataSet
from nilmtk.disaggregate import CombinatorialOptimisation

# 将项目根目录加入 sys.path，便于直接运行脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_nilmtk_to_h5(
    input_path: str,
    output_path: str,
    appliances: list = None,
    buildings: list = None
):
    """
    将 NILMTK 数据集转换为 Seq2Point 需要的简化 HDF5 格式。
    
    Args:
        input_path: NILMTK 数据集路径
        output_path: 输出 HDF5 文件路径
        appliances: 提取的电器列表
        buildings: 处理的建筑列表
    """
    print(f"Loading dataset from {input_path}...")
    dataset = DataSet(input_path)
    
    if buildings is None:
        buildings = dataset.buildings.keys()
    
    if appliances is None:
        # 常见电器列表
        appliances = ['fridge', 'microwave', 'dishwasher', 'washing_machine', 'kettle']
    
    print(f"Processing buildings: {buildings}")
    print(f"Extracting appliances: {appliances}")
    
    with h5py.File(output_path, 'w') as f_out:
        for building_id in buildings:
            print(f"\nProcessing building {building_id}...")
            
            building = dataset.buildings[building_id]
            building_grp = f_out.create_group(f'building_{building_id}')
            
            # 提取聚合功率
            try:
                elec = building.elec
                mains = elec.mains().power_series_all_data()
                
                if not mains.empty:
                    # 重采样到 1 分钟
                    mains = mains.resample('1T').mean().fillna(0)
                    mains_data = mains.values.flatten()
                    
                    building_grp.create_dataset('mains', data=mains_data, compression='gzip')
                    print(f"  Mains: {len(mains_data)} samples")
                else:
                    print(f"  Warning: No mains data for building {building_id}")
                    continue
                
            except Exception as e:
                print(f"  Error extracting mains: {e}")
                continue
            
            # 提取电器功率
            for appliance_name in appliances:
                try:
                    appliance = elec[appliance_name]
                    appliance_data = appliance.power_series_all_data()
                    
                    if not appliance_data.empty:
                        # 重采样到 1 分钟
                        appliance_data = appliance_data.resample('1T').mean().fillna(0)
                        appliance_array = appliance_data.values.flatten()
                        
                        # 对齐与聚合功率长度
                        min_len = min(len(mains_data), len(appliance_array))
                        appliance_array = appliance_array[:min_len]
                        
                        building_grp.create_dataset(
                            appliance_name,
                            data=appliance_array,
                            compression='gzip'
                        )
                        print(f"  {appliance_name}: {len(appliance_array)} samples")
                    else:
                        print(f"  Warning: No data for {appliance_name}")
                        
                except KeyError:
                    print(f"  {appliance_name}: Not found")
                except Exception as e:
                    print(f"  Error extracting {appliance_name}: {e}")
    
    print(f"\nDataset saved to {output_path}")


def inspect_h5_file(file_path: str):
    """
    查看 HDF5 文件结构与数据概况。
    
    Args:
        file_path: HDF5 文件路径
    """
    print(f"\nInspecting {file_path}...")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                print(f"    Mean: {np.mean(obj[:]):.2f}")
                print(f"    Max: {np.max(obj[:]):.2f}")
                print(f"    Min: {np.min(obj[:]):.2f}")
        
        f.visititems(print_structure)
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Preprocess NILM datasets')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input NILMTK dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output HDF5 file'
    )
    parser.add_argument(
        '--appliances',
        type=str,
        nargs='+',
        default=None,
        help='List of appliances to extract'
    )
    parser.add_argument(
        '--buildings',
        type=int,
        nargs='+',
        default=None,
        help='List of buildings to process'
    )
    parser.add_argument(
        '--inspect',
        type=str,
        default=None,
        help='Inspect an HDF5 file'
    )
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_h5_file(args.inspect)
    else:
        convert_nilmtk_to_h5(
            args.input,
            args.output,
            args.appliances,
            args.buildings
        )


if __name__ == '__main__':
    main()
