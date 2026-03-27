"""
检查数据的实际采样周期。
"""

import h5py
import numpy as np


def check_sampling_info(file_path: str):
    """检查数据采样信息。"""
    print(f"\n{'='*70}")
    print(f"数据采样信息检查: {file_path}")
    print(f"{'='*70}\n")
    
    with h5py.File(file_path, 'r') as f:
        buildings = [k for k in f.keys() if k.startswith('building_')]
        
        if not buildings:
            print("未找到建筑数据")
            return
        
        # Check first building
        building_key = buildings[0]
        building = f[building_key]
        
        print(f"检查 {building_key}:\n")
        
        if 'mains' in building:
            mains = building['mains'][:]
            
            print(f"聚合功率（Mains）:")
            print(f"  总样本数: {len(mains):,}")
            print(f"  平均功率: {np.mean(mains):.2f} W")
            print(f"  功率范围: [{np.min(mains):.2f}, {np.max(mains):.2f}] W")
            
            # 假设数据是连续的，估算总时长
            # REDD数据集：聚合功率通常每3秒一个样本
            # 从NILMTK文档得知，REDD的mains是每1秒或3秒采样
            
            print(f"\n采样周期估算:")
            print(f"  如果数据连续采样，可能的采样周期:")
            for period_sec in [1, 3, 6]:
                total_duration_days = (len(mains) * period_sec) / 86400
                print(f"    - 每 {period_sec} 秒: 约 {total_duration_days:.1f} 天")
            
        # Check appliances
        appliances = [k for k in building.keys() if k != 'mains']
        if appliances:
            print(f"\n电器数据:")
            for app_name in appliances[:3]:  # Show first 3
                app_data = building[app_name][:]
                print(f"  {app_name}:")
                print(f"    样本数: {len(app_data):,}")
                print(f"    平均功率: {np.mean(app_data):.2f} W")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'data/redd_simple.h5'
    
    check_sampling_info(file_path)
    
    print("\n" + "="*70)
    print("说明:")
    print("="*70)
    print("""
REDD 数据集的标准采样周期：
  - 聚合功率 (mains): 每 1 秒采样（某些建筑为 3 秒）
  - 单个电器: 每 3 秒采样

由于数据已经预处理，当前模型接收的是原始采样频率的数据。
如果需要调整采样周期，可以在数据转换时添加重采样功能。
    """)
    print("="*70 + "\n")
