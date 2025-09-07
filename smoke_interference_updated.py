import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import math

class SmokeInterferenceModel:
    def __init__(self):
        # 物理常数
        self.g = 9.8  # 重力加速度
        self.smoke_sink_speed = 3.0  # 烟幕下沉速度 m/s
        self.effective_radius = 10.0  # 有效遮蔽半径 m
        self.effective_time = 20.0  # 有效遮蔽时间 s
        self.missile_speed = 300.0  # 导弹速度 m/s
        
        # 目标信息
        self.fake_target = np.array([0, 0, 0])  # 假目标位置
        self.real_target_center = np.array([0, 200, 0])  # 真目标底面圆心
        self.real_target_radius = 7.0  # 真目标半径
        self.real_target_height = 10.0  # 真目标高度
        
        # 更新的视线点（根据新逻辑）
        self.target_bottom_view = np.array([0, 193, 0])  # 真目标底面视线点
        self.target_top_view = np.array([0, 193, 10])    # 真目标顶面视线点
        
        # 导弹初始位置
        self.M1_init = np.array([20000, 0, 2000])
        self.M2_init = np.array([19000, 600, 2100])
        self.M3_init = np.array([18000, -600, 1900])
        
        # 无人机初始位置
        self.FY1_init = np.array([17800, 0, 1800])
        self.FY2_init = np.array([12000, 1400, 1400])
        self.FY3_init = np.array([6000, -3000, 700])
        self.FY4_init = np.array([11000, 2000, 1800])
        self.FY5_init = np.array([13000, -2000, 1300])
    
    def missile_position(self, t, missile_init):
        """计算导弹在时刻t的位置"""
        # 导弹直指假目标
        direction = self.fake_target - missile_init
        direction_norm = np.linalg.norm(direction)
        unit_direction = direction / direction_norm
        
        return missile_init + self.missile_speed * t * unit_direction
    
    def drone_position(self, t, drone_init, speed, direction_angle):
        """计算无人机在时刻t的位置"""
        x = drone_init[0] + speed * t * np.cos(direction_angle)
        y = drone_init[1] + speed * t * np.sin(direction_angle)
        z = drone_init[2]  # 等高度飞行
        return np.array([x, y, z])
    
    def smoke_drop_position(self, drop_time, drone_init, speed, direction_angle):
        """计算烟幕弹投放点位置"""
        return self.drone_position(drop_time, drone_init, speed, direction_angle)
    
    def smoke_explosion_position(self, drop_time, explosion_delay, drone_init, speed, direction_angle):
        """计算烟幕弹起爆点位置"""
        drop_pos = self.smoke_drop_position(drop_time, drone_init, speed, direction_angle)
        
        # 烟幕弹脱离无人机后的运动
        # 水平方向保持无人机速度，垂直方向自由落体
        horizontal_speed_x = speed * np.cos(direction_angle)
        horizontal_speed_y = speed * np.sin(direction_angle)
        
        x = drop_pos[0] + horizontal_speed_x * explosion_delay
        y = drop_pos[1] + horizontal_speed_y * explosion_delay
        z = drop_pos[2] - 0.5 * self.g * explosion_delay**2
        
        return np.array([x, y, z])
    
    def smoke_cloud_center(self, t, explosion_time, explosion_pos):
        """计算烟幕云团中心在时刻t的位置"""
        if t < explosion_time:
            return None  # 还未起爆
        
        time_since_explosion = t - explosion_time
        if time_since_explosion > self.effective_time:
            return None  # 已失效
        
        # 云团以3m/s速度下沉
        z_offset = -self.smoke_sink_speed * time_since_explosion
        return explosion_pos + np.array([0, 0, z_offset])
    
    def point_to_line_segment_distance(self, point, line_start, line_end):
        """计算点到直线段的距离（更精确的实现）"""
        # 直线段向量
        line_vec = line_end - line_start
        # 点到直线段起点的向量
        point_vec = point - line_start
        
        # 直线段长度
        line_length_sq = np.dot(line_vec, line_vec)
        
        if line_length_sq == 0:
            # 直线段退化为点
            return np.linalg.norm(point_vec)
        
        # 计算投影参数t
        t = np.dot(point_vec, line_vec) / line_length_sq
        
        if t < 0:
            # 投影点在线段起点之前
            return np.linalg.norm(point_vec)
        elif t > 1:
            # 投影点在线段终点之后
            return np.linalg.norm(point - line_end)
        else:
            # 投影点在线段上
            projection = line_start + t * line_vec
            return np.linalg.norm(point - projection)
    
    def is_effective_shielding(self, t, missile_init, explosion_time, explosion_pos):
        """判断在时刻t是否形成有效遮蔽"""
        # 1. 时间有效性检查
        if t < explosion_time or t > explosion_time + self.effective_time:
            return False
        
        cloud_center = self.smoke_cloud_center(t, explosion_time, explosion_pos)
        if cloud_center is None:
            return False
        
        missile_pos = self.missile_position(t, missile_init)
        
        # 2. 位置有效性检查：云团需要位于导弹与真目标之间
        # 检查x坐标是否在导弹和目标之间
        if not (missile_pos[0] >= cloud_center[0] >= self.target_bottom_view[0]):
            return False
        
        # 3. 空间有效性检查：计算云团中心到视线段的距离
        # 检查两条视线：导弹到真目标底面和顶面
        dist_to_bottom_line = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_bottom_view
        )
        dist_to_top_line = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_top_view
        )
        
        # 如果任一视线与烟幕球体相交（距离≤半径），则形成有效遮蔽
        return (dist_to_bottom_line <= self.effective_radius or 
                dist_to_top_line <= self.effective_radius)
    
    def calculate_shielding_time(self, missile_init, explosion_time, explosion_pos, 
                               time_start=0, time_end=100, dt=0.01):
        """计算有效遮蔽时间"""
        total_time = 0
        t = max(time_start, explosion_time)  # 从起爆时间开始
        end_time = min(time_end, explosion_time + self.effective_time)  # 到失效时间结束
        
        while t <= end_time:
            if self.is_effective_shielding(t, missile_init, explosion_time, explosion_pos):
                total_time += dt
            t += dt
        
        return total_time
    
    def check_constraints(self, missile_init, explosion_time, explosion_pos):
        """检查物理约束条件"""
        # 1. 云团需要在导弹和目标之间的x方向
        missile_at_explosion = self.missile_position(explosion_time, missile_init)
        if not (missile_at_explosion[0] >= explosion_pos[0] >= self.target_bottom_view[0]):
            return False
        
        # 2. 云团高度合理性检查
        if explosion_pos[2] < 0 or explosion_pos[2] > 3000:
            return False
        
        # 3. 云团y坐标不能偏离太远
        if abs(explosion_pos[1]) > 500:  # 允许一定偏差
            return False
        
        return True

# 问题1的实现
def solve_problem1():
    """问题1：FY1以120m/s朝假目标飞行，1.5s后投放，3.6s后起爆"""
    model = SmokeInterferenceModel()
    
    # 参数设置
    drone_speed = 120.0  # m/s
    # 计算朝向假目标的方向角
    direction_vector = model.fake_target - model.FY1_init
    direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
    
    drop_time = 1.5  # 投放时间
    explosion_delay = 3.6  # 起爆延迟
    
    # 计算起爆点
    explosion_pos = model.smoke_explosion_position(
        drop_time, explosion_delay, model.FY1_init, drone_speed, direction_angle
    )
    explosion_time = drop_time + explosion_delay
    
    print("问题1结果（更新逻辑）：")
    print(f"无人机飞行方向角: {direction_angle:.4f} rad ({np.degrees(direction_angle):.2f}°)")
    print(f"烟幕弹投放点: {model.smoke_drop_position(drop_time, model.FY1_init, drone_speed, direction_angle)}")
    print(f"烟幕弹起爆点: {explosion_pos}")
    print(f"起爆时间: {explosion_time}s")
    
    # 检查约束条件
    constraints_ok = model.check_constraints(model.M1_init, explosion_time, explosion_pos)
    print(f"约束条件满足: {constraints_ok}")
    
    # 计算有效遮蔽时间
    shielding_time = model.calculate_shielding_time(
        model.M1_init, explosion_time, explosion_pos
    )
    
    print(f"有效遮蔽时长: {shielding_time:.2f}s")
    
    # 详细分析遮蔽过程
    print("\n遮蔽过程分析:")
    for t in np.arange(explosion_time, explosion_time + 10, 1.0):
        is_shielding = model.is_effective_shielding(t, model.M1_init, explosion_time, explosion_pos)
        cloud_center = model.smoke_cloud_center(t, explosion_time, explosion_pos)
        missile_pos = model.missile_position(t, model.M1_init)
        if cloud_center is not None:
            # 计算距离用于调试
            dist_bottom = model.point_to_line_segment_distance(
                cloud_center, missile_pos, model.target_bottom_view
            )
            dist_top = model.point_to_line_segment_distance(
                cloud_center, missile_pos, model.target_top_view
            )
            print(f"t={t:.1f}s: 遮蔽={is_shielding}, 云团中心=({cloud_center[0]:.1f},{cloud_center[1]:.1f},{cloud_center[2]:.1f})")
            print(f"       导弹位置=({missile_pos[0]:.1f},{missile_pos[1]:.1f},{missile_pos[2]:.1f})")
            print(f"       距离底面视线={dist_bottom:.2f}m, 距离顶面视线={dist_top:.2f}m")
    
    return shielding_time

# 问题2的实现
def solve_problem2():
    """问题2：优化FY1的飞行参数，使遮蔽时间最长"""
    model = SmokeInterferenceModel()
    
    def objective_function(params):
        """目标函数：最大化遮蔽时间（最小化负遮蔽时间）"""
        speed, direction_angle, drop_time, explosion_delay = params
        
        # 基本约束检查
        if not (70 <= speed <= 140):
            return 1000
        if not (0.1 <= drop_time <= 30):
            return 1000
        if not (0.1 <= explosion_delay <= 15):
            return 1000
        
        try:
            explosion_pos = model.smoke_explosion_position(
                drop_time, explosion_delay, model.FY1_init, speed, direction_angle
            )
            explosion_time = drop_time + explosion_delay
            
            # 检查物理约束
            if not model.check_constraints(model.M1_init, explosion_time, explosion_pos):
                return 1000
            
            # 计算遮蔽时间
            shielding_time = model.calculate_shielding_time(
                model.M1_init, explosion_time, explosion_pos
            )
            
            return -shielding_time  # 最小化负值等于最大化正值
            
        except Exception as e:
            return 1000
    
    print("问题2求解中（更新逻辑）...")
    
    # 参数边界
    bounds = [
        (70, 140),      # speed
        (0, 2*np.pi),   # direction_angle
        (0.1, 30),      # drop_time
        (0.1, 15)       # explosion_delay
    ]
    
    # 使用差分进化算法进行全局优化
    result = differential_evolution(
        objective_function, 
        bounds, 
        seed=42,
        maxiter=200,
        popsize=20,
        atol=1e-6,
        tol=1e-6
    )
    
    if result.success:
        optimal_params = result.x
        best_shielding_time = -result.fun
    else:
        print("全局优化失败，尝试网格搜索...")
        # 网格搜索备选方案
        best_params = None
        best_shielding_time = 0
        
        for speed in np.linspace(70, 140, 8):
            for angle in np.linspace(0, 2*np.pi, 12):
                for drop_time in np.linspace(0.5, 10, 10):
                    for explosion_delay in np.linspace(1, 10, 10):
                        params = [speed, angle, drop_time, explosion_delay]
                        score = -objective_function(params)
                        if score > best_shielding_time:
                            best_shielding_time = score
                            best_params = params
        
        if best_params:
            optimal_params = best_params
        else:
            print("优化失败，使用默认参数")
            optimal_params = [120.0, np.pi, 2.0, 4.0]
            best_shielding_time = -objective_function(optimal_params)
    
    speed_opt, angle_opt, drop_time_opt, explosion_delay_opt = optimal_params
    
    # 计算最优解的详细信息
    explosion_pos_opt = model.smoke_explosion_position(
        drop_time_opt, explosion_delay_opt, model.FY1_init, speed_opt, angle_opt
    )
    explosion_time_opt = drop_time_opt + explosion_delay_opt
    
    final_shielding_time = model.calculate_shielding_time(
        model.M1_init, explosion_time_opt, explosion_pos_opt
    )
    
    print("\n问题2结果（更新逻辑）：")
    print(f"最优飞行速度: {speed_opt:.2f} m/s")
    print(f"最优飞行方向角: {angle_opt:.4f} rad ({np.degrees(angle_opt):.2f}°)")
    print(f"最优投放时间: {drop_time_opt:.2f} s")
    print(f"最优起爆延迟: {explosion_delay_opt:.2f} s")
    print(f"烟幕弹投放点: {model.smoke_drop_position(drop_time_opt, model.FY1_init, speed_opt, angle_opt)}")
    print(f"烟幕弹起爆点: {explosion_pos_opt}")
    print(f"起爆时间: {explosion_time_opt:.2f} s")
    print(f"最大有效遮蔽时长: {final_shielding_time:.2f} s")
    
    return {
        'speed': speed_opt,
        'direction_angle': angle_opt,
        'drop_time': drop_time_opt,
        'explosion_delay': explosion_delay_opt,
        'explosion_pos': explosion_pos_opt,
        'explosion_time': explosion_time_opt,
        'shielding_time': final_shielding_time
    }

def analyze_problem_updated():
    """分析问题的关键因素（更新版本）"""
    model = SmokeInterferenceModel()
    
    print("问题分析（更新逻辑）:")
    print(f"导弹M1初始位置: {model.M1_init}")
    print(f"无人机FY1初始位置: {model.FY1_init}")
    print(f"真目标底面视线点: {model.target_bottom_view}")
    print(f"真目标顶面视线点: {model.target_top_view}")
    print(f"假目标位置: {model.fake_target}")
    
    # 计算导弹到假目标的距离和飞行时间
    missile_to_target_dist = np.linalg.norm(model.M1_init - model.fake_target)
    missile_flight_time = missile_to_target_dist / model.missile_speed
    print(f"导弹到假目标距离: {missile_to_target_dist:.2f} m")
    print(f"导弹飞行时间: {missile_flight_time:.2f} s")
    
    # 分析最佳拦截位置
    ideal_intercept_x = (model.M1_init[0] + model.target_bottom_view[0]) / 2
    print(f"理想拦截x坐标: {ideal_intercept_x:.2f} m")
    
    print("\n关键变化:")
    print("1. 真目标视线点从(0,200,0)更新为(0,193,0)")
    print("2. 使用更精确的点到直线段距离计算")
    print("3. 强化了位置有效性约束条件")

if __name__ == "__main__":
    print("=" * 60)
    print("烟幕干扰弹投放策略数学建模（更新逻辑）")
    print("=" * 60)
    
    # 分析问题
    analyze_problem_updated()
    
    print("\n" + "=" * 60)
    
    # 求解问题1
    shielding_time_1 = solve_problem1()
    
    print("\n" + "=" * 60)
    
    # 求解问题2
    optimal_params = solve_problem2()
    
    print("\n" + "=" * 60)
    print("总结:")
    print(f"问题1遮蔽时间: {shielding_time_1:.2f}s")
    print(f"问题2最优遮蔽时间: {optimal_params['shielding_time']:.2f}s")
    if shielding_time_1 > 0:
        print(f"优化提升: {optimal_params['shielding_time'] - shielding_time_1:.2f}s")
        print(f"提升比例: {((optimal_params['shielding_time'] - shielding_time_1) / shielding_time_1 * 100):.1f}%")
    else:
        print(f"优化提升: {optimal_params['shielding_time']:.2f}s (从0开始)")