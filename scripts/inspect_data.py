"""
检查 HDF5 数据文件结构与内容概况。
"""

import sys
import h5py
import numpy as np
import argparse


def inspect_hdf5(file_path: str, detailed: bool = False):
    """
    查看 HDF5 文件结构与内容。
    
    Args:
        file_path: HDF5 文件路径
        detailed: 是否显示详细统计信息
    """
    print(f"\n{'=' * 70}")
    print(f"Inspecting: {file_path}")
    print("=" * 70)
    
    try:
        with h5py.File(file_path, 'r') as f:
            
            def print_attrs(name, obj):
                """打印 HDF5 对象属性。"""
                indent = "  " * name.count('/')
                
                if isinstance(obj, h5py.Group):
                    print(f"\n{indent}📁 Group: {name}/")
                    if detailed and len(obj.attrs) > 0:
                        print(f"{indent}   Attributes:")
                        for key, val in obj.attrs.items():
                            print(f"{indent}     {key}: {val}")
                
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}📊 Dataset: {name}")
                    print(f"{indent}   Shape: {obj.shape}")
                    print(f"{indent}   Dtype: {obj.dtype}")
                    print(f"{indent}   Size: {obj.size:,} elements")
                    
                    if detailed and obj.size > 0:
                        data = obj[:]
                        if np.issubdtype(data.dtype, np.number):
                            print(f"{indent}   Statistics:")
                            print(f"{indent}     Mean:   {np.mean(data):.4f}")
                            print(f"{indent}     Std:    {np.std(data):.4f}")
                            print(f"{indent}     Min:    {np.min(data):.4f}")
                            print(f"{indent}     Max:    {np.max(data):.4f}")
                            print(f"{indent}     Median: {np.median(data):.4f}")
                            
                            # Count non-zero
                            non_zero = np.count_nonzero(data)
                            print(f"{indent}     Non-zero: {non_zero:,} ({non_zero/obj.size*100:.1f}%)")
            
            # 输出文件结构
            print("\nFile Structure:")
            print("-" * 70)
            f.visititems(print_attrs)
            
            # 汇总信息
            print("\n" + "=" * 70)
            print("Summary:")
            print("-" * 70)
            
            # 统计组与数据集数量
            num_groups = 0
            num_datasets = 0
            
            def count_items(name, obj):
                nonlocal num_groups, num_datasets
                if isinstance(obj, h5py.Group):
                    num_groups += 1
                elif isinstance(obj, h5py.Dataset):
                    num_datasets += 1
            
            f.visititems(count_items)
            
            print(f"Total Groups: {num_groups}")
            print(f"Total Datasets: {num_datasets}")
            
            # 列出建筑与电器
            buildings = [key for key in f.keys() if key.startswith('building')]
            if buildings:
                print(f"\nBuildings found: {len(buildings)}")
                for building in sorted(buildings):
                    appliances = list(f[building].keys())
                    print(f"  {building}: {', '.join(appliances)}")
            
            print("=" * 70)
            
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")


def compare_datasets(file1: str, file2: str):
    """
    对比两个 HDF5 文件的建筑信息。
    
    Args:
        file1: 第一个 HDF5 文件路径
        file2: 第二个 HDF5 文件路径
    """
    print(f"\n{'=' * 70}")
    print(f"Comparing Datasets")
    print("=" * 70)
    
    try:
        with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
            
            buildings1 = set(k for k in f1.keys() if k.startswith('building'))
            buildings2 = set(k for k in f2.keys() if k.startswith('building'))
            
            print(f"\nFile 1: {file1}")
            print(f"  Buildings: {len(buildings1)}")
            print(f"  {sorted(buildings1)}")
            
            print(f"\nFile 2: {file2}")
            print(f"  Buildings: {len(buildings2)}")
            print(f"  {sorted(buildings2)}")
            
            common = buildings1 & buildings2
            only1 = buildings1 - buildings2
            only2 = buildings2 - buildings1
            
            print(f"\nCommon buildings: {len(common)}")
            if common:
                print(f"  {sorted(common)}")
            
            print(f"\nOnly in file 1: {len(only1)}")
            if only1:
                print(f"  {sorted(only1)}")
            
            print(f"\nOnly in file 2: {len(only2)}")
            if only2:
                print(f"  {sorted(only2)}")
            
            print("=" * 70)
            
    except Exception as e:
        print(f"Error: {e}")


def export_sample(file_path: str, building: int, output_file: str):
    """
    导出指定建筑的样本数据为 numpy 文件。
    
    Args:
        file_path: HDF5 文件路径
        building: 建筑 ID
        output_file: 输出文件路径
    """
    try:
        with h5py.File(file_path, 'r') as f:
            building_key = f'building_{building}'
            
            if building_key not in f:
                print(f"Error: {building_key} not found in dataset")
                return
            
            if 'mains' not in f[building_key]:
                print(f"Error: No mains data in {building_key}")
                return
            
            mains = f[building_key]['mains'][:]
            
            # 截取前 10000 个样本
            sample = mains[:10000]
            
            np.save(output_file, sample)
            print(f"Exported {len(sample)} samples to {output_file}")
            print(f"Shape: {sample.shape}")
            print(f"Range: [{sample.min():.2f}, {sample.max():.2f}]")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Inspect HDF5 files')
    parser.add_argument(
        'file',
        type=str,
        help='Path to HDF5 file'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--compare',
        type=str,
        help='Compare with another HDF5 file'
    )
    parser.add_argument(
        '--export',
        type=int,
        metavar='BUILDING',
        help='Export sample from building'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sample.npy',
        help='Output file for export'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_datasets(args.file, args.compare)
    elif args.export:
        export_sample(args.file, args.export, args.output)
    else:
        inspect_hdf5(args.file, args.detailed)


if __name__ == '__main__':
    main()
