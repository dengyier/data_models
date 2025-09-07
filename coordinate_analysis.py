import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_coordinate_system():
    """分析坐标系统的设计逻辑"""
    print("=" * 60)
    print("坐标系统分析")
    print("=" * 60)
    
    # 关键位置
    fake_target = np.array([0, 0, 0])           # 假目标（原点）
    real_target = np.array([0, 193, 0])         # 真目标视线点
    M1_init = np.array([20000, 0, 2000])        # 导弹M1初始位置
    FY1_init = np.array([17800, 0, 1800])       # 无人机FY1初始位置
    
    print("关键位置坐标:")
    print(f"假目标（原点）:     {fake_target}")
    print(f"真目标视线点:       {real_target}")
    print(f"导弹M1初始位置:     {M1_init}")
    print(f"无人机FY1初始位置:  {FY1_init}")
    
    print(f"\n坐标轴含义分析:")
    print(f"X轴: 主要作战方向轴")
    print(f"  - 导弹从x=20000m处向原点(x=0)飞行")
    print(f"  - 无人机从x=17800m处向原点方向飞行")
    print(f"  - 这是一个20公里的作战距离")
    
    print(f"Y轴: 横向偏移轴")
    print(f"  - 大部分初始位置y=0（在主轴线上）")
    print(f"  - 真目标稍微偏移到y=193m")
    print(f"  - 这是一个相对较小的横向距离")
    
    print(f"Z轴: 高度轴")
    print(f"  - 导弹初始高度: 2000m")
    print(f"  - 无人机初始高度: 1800m")
    print(f"  - 这是典型的飞行高度")
    
    # 计算距离
    missile_to_target = np.linalg.norm(M1_init - fake_target)
    drone_to_target = np.linalg.norm(FY1_init - fake_target)
    
    print(f"\n距离分析:")
    print(f"导弹到假目标距离: {missile_to_target:.1f}m ≈ {missile_to_target/1000:.1f}km")
    print(f"无人机到假目标距离: {drone_to_target:.1f}m ≈ {drone_to_target/1000:.1f}km")
    
    print(f"\n为什么X坐标这么大？")
    print(f"1. 现实作战场景: 导弹通常从数十公里外发射")
    print(f"2. 反应时间需求: 需要足够距离来部署烟幕干扰")
    print(f"3. 坐标系设计: 以假目标为原点，导弹从远处接近")
    print(f"4. 比例关系: X轴是主要作战轴，Y/Z轴是辅助调整轴")
    
    # 最新结果分析
    print(f"\n最新优化结果分析:")
    result_positions = [
        [17792.4, 0.0, 1800.0],  # 投放点1
        [17708.8, 0.0, 1800.0],  # 投放点2
        [17610.0, 0.0, 1800.0],  # 投放点3
        [17724.0, 0.0, 1796.0],  # 起爆点1
        [17625.2, 0.0, 1794.1],  # 起爆点2
        [17511.2, 0.0, 1791.7],  # 起爆点3
    ]
    
    print("投放点和起爆点坐标:")
    labels = ["投放点1", "投放点2", "投放点3", "起爆点1", "起爆点2", "起爆点3"]
    for i, (pos, label) in enumerate(zip(result_positions, labels)):
        print(f"{label}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
    
    print(f"\n坐标特点:")
    print(f"- X坐标范围: 17511-17792m (约17.5-17.8km)")
    print(f"- Y坐标: 基本为0 (在主轴线上)")
    print(f"- Z坐标范围: 1792-1800m (高空飞行)")
    
    print(f"\n这说明:")
    print(f"1. 作战发生在17-18公里距离上")
    print(f"2. 烟幕弹在导弹接近过程中的关键位置起爆")
    print(f"3. Y坐标小是因为都在主攻击轴线上")
    print(f"4. Z坐标相对稳定，保持在飞行高度")

def create_coordinate_visualization():
    """创建坐标系统可视化"""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D图
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 关键位置
    fake_target = np.array([0, 0, 0])
    real_target = np.array([0, 193, 0])
    M1_init = np.array([20000, 0, 2000])
    FY1_init = np.array([17800, 0, 1800])
    
    # 最新结果位置
    drop_points = np.array([
        [17792.4, 0.0, 1800.0],
        [17708.8, 0.0, 1800.0],
        [17610.0, 0.0, 1800.0]
    ])
    
    explosion_points = np.array([
        [17724.0, 0.0, 1796.0],
        [17625.2, 0.0, 1794.1],
        [17511.2, 0.0, 1791.7]
    ])
    
    # 绘制关键点
    ax1.scatter(*fake_target, color='black', s=100, label='Fake Target (Origin)')
    ax1.scatter(*real_target, color='green', s=100, label='Real Target')
    ax1.scatter(*M1_init, color='red', s=100, label='Missile M1 Start')
    ax1.scatter(*FY1_init, color='blue', s=100, label='Drone FY1 Start')
    
    # 绘制投放点和起爆点
    ax1.scatter(drop_points[:, 0], drop_points[:, 1], drop_points[:, 2], 
                color='cyan', s=80, label='Drop Points')
    ax1.scatter(explosion_points[:, 0], explosion_points[:, 1], explosion_points[:, 2], 
                color='orange', s=80, label='Explosion Points')
    
    # 绘制导弹轨迹
    t_array = np.linspace(0, 60, 100)
    missile_traj = np.array([M1_init * (1 - 300 * t / np.linalg.norm(M1_init)) for t in t_array])
    missile_traj = missile_traj[missile_traj[:, 0] >= 0]  # 只显示到原点
    ax1.plot(missile_traj[:, 0], missile_traj[:, 1], missile_traj[:, 2], 'r--', alpha=0.7, label='Missile Trajectory')
    
    ax1.set_xlabel('X (m) - Main Combat Axis')
    ax1.set_ylabel('Y (m) - Lateral Offset')
    ax1.set_zlabel('Z (m) - Altitude')
    ax1.legend()
    ax1.set_title('3D Combat Scenario Overview')
    
    # X-Z平面图（主要作战平面）
    ax2 = fig.add_subplot(222)
    ax2.scatter(fake_target[0], fake_target[2], color='black', s=100, label='Fake Target')
    ax2.scatter(real_target[0], real_target[2], color='green', s=100, label='Real Target')
    ax2.scatter(M1_init[0], M1_init[2], color='red', s=100, label='Missile M1')
    ax2.scatter(FY1_init[0], FY1_init[2], color='blue', s=100, label='Drone FY1')
    ax2.scatter(drop_points[:, 0], drop_points[:, 2], color='cyan', s=80, label='Drop Points')
    ax2.scatter(explosion_points[:, 0], explosion_points[:, 2], color='orange', s=80, label='Explosion Points')
    ax2.plot(missile_traj[:, 0], missile_traj[:, 2], 'r--', alpha=0.7, label='Missile Trajectory')
    
    ax2.set_xlabel('X (m) - Distance from Target')
    ax2.set_ylabel('Z (m) - Altitude')
    ax2.legend()
    ax2.set_title('X-Z Plane: Main Combat View')
    ax2.grid(True, alpha=0.3)
    
    # 距离分析图
    ax3 = fig.add_subplot(223)
    distances = [
        np.linalg.norm(M1_init),
        np.linalg.norm(FY1_init),
        np.linalg.norm(drop_points[0]),
        np.linalg.norm(explosion_points[0])
    ]
    labels = ['Missile Start', 'Drone Start', 'Drop Point', 'Explosion Point']
    colors = ['red', 'blue', 'cyan', 'orange']
    
    bars = ax3.bar(labels, distances, color=colors, alpha=0.7)
    ax3.set_ylabel('Distance from Origin (m)')
    ax3.set_title('Distance Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, dist in zip(bars, distances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 100,
                f'{dist:.0f}m\n({dist/1000:.1f}km)', ha='center', va='bottom')
    
    # 坐标范围对比
    ax4 = fig.add_subplot(224)
    
    all_positions = np.vstack([
        M1_init.reshape(1, -1),
        FY1_init.reshape(1, -1),
        drop_points,
        explosion_points
    ])
    
    x_range = [all_positions[:, 0].min(), all_positions[:, 0].max()]
    y_range = [all_positions[:, 1].min(), all_positions[:, 1].max()]
    z_range = [all_positions[:, 2].min(), all_positions[:, 2].max()]
    
    ranges = [x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]]
    axis_labels = ['X Range', 'Y Range', 'Z Range']
    
    bars = ax4.bar(axis_labels, ranges, color=['red', 'green', 'blue'], alpha=0.7)
    ax4.set_ylabel('Coordinate Range (m)')
    ax4.set_title('Coordinate Range Comparison')
    ax4.grid(True, alpha=0.3)
    
    for bar, range_val in zip(bars, ranges):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{range_val:.0f}m', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('coordinate_system_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_coordinate_system()
    create_coordinate_visualization()