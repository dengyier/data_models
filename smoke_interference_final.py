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
        
        # 更新的视线点
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
        
        # 计算导弹相关参数
        self.missile_to_target_dist = np.linalg.norm(self.M1_init - self.fake_target)
        self.missile_flight_time = self.missile_to_target_dist / self.missile_speed
        
        # 计算导弹速度角度
        missile_direction = self.fake_target - self.M1_init
        self.missile_beta = np.arccos(-missile_direction[2] / np.linalg.norm(missile_direction))  # 与z轴夹角
        self.missile_alpha = np.arctan2(missile_direction[1], missile_direction[0])  # 水平面与x轴夹角
    
    def missile_position(self, t, missile_init):
        """计算导弹在时刻t的位置"""
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
        horizontal_speed_x = speed * np.cos(direction_angle)
        horizontal_speed_y = speed * np.sin(direction_angle)
        
        x = drop_pos[0] + horizontal_speed_x * explosion_delay
        y = drop_pos[1] + horizontal_speed_y * explosion_delay
        z = drop_pos[2] - 0.5 * self.g * explosion_delay**2
        
        return np.array([x, y, z])
    
    def check_explosion_before_ground(self, drop_time, explosion_delay, drone_init, speed, direction_angle):
        """检查烟幕弹是否在落地前起爆"""
        drop_pos = self.smoke_drop_position(drop_time, drone_init, speed, direction_angle)
        
        # 计算落地时间：z = z0 - 0.5*g*t^2 = 0
        # t = sqrt(2*z0/g)
        time_to_ground = np.sqrt(2 * drop_pos[2] / self.g)
        
        return explosion_delay < time_to_ground
    
    def smoke_cloud_center(self, t, explosion_time, explosion_pos):
        """计算烟幕云团中心在时刻t的位置"""
        if t < explosion_time:
            return None
        
        time_since_explosion = t - explosion_time
        if time_since_explosion > self.effective_time:
            return None
        
        z_offset = -self.smoke_sink_speed * time_since_explosion
        return explosion_pos + np.array([0, 0, z_offset])
    
    def point_to_line_segment_distance(self, point, line_start, line_end):
        """计算点到直线段的距离"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_length_sq = np.dot(line_vec, line_vec)
        if line_length_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = np.dot(point_vec, line_vec) / line_length_sq
        
        if t < 0:
            return np.linalg.norm(point_vec)
        elif t > 1:
            return np.linalg.norm(point - line_end)
        else:
            projection = line_start + t * line_vec
            return np.linalg.norm(point - projection)
    
    def is_effective_shielding(self, t, missile_init, explosion_time, explosion_pos):
        """判断在时刻t是否形成有效遮蔽"""
        if t < explosion_time or t > explosion_time + self.effective_time:
            return False
        
        cloud_center = self.smoke_cloud_center(t, explosion_time, explosion_pos)
        if cloud_center is None:
            return False
        
        missile_pos = self.missile_position(t, missile_init)
        
        # 位置有效性检查：云团需要位于导弹与真目标之间
        if not (missile_pos[0] >= cloud_center[0] >= self.target_bottom_view[0]):
            return False
        
        # 空间有效性检查
        dist_to_bottom_line = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_bottom_view
        )
        dist_to_top_line = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_top_view
        )
        
        return (dist_to_bottom_line <= self.effective_radius or 
                dist_to_top_line <= self.effective_radius)
    
    def calculate_shielding_time(self, missile_init, explosion_time, explosion_pos, 
                               time_start=0, time_end=100, dt=0.01):
        """计算有效遮蔽时间"""
        total_time = 0
        t = max(time_start, explosion_time)
        end_time = min(time_end, explosion_time + self.effective_time)
        
        while t <= end_time:
            if self.is_effective_shielding(t, missile_init, explosion_time, explosion_pos):
                total_time += dt
            t += dt
        
        return total_time
    
    def check_detailed_constraints(self, speed, direction_angle, drop_time, explosion_delay, 
                                 missile_init, drone_init):
        """检查详细约束条件（根据更新逻辑）"""
        
        # 基本参数约束
        if not (70 <= speed <= 140):
            return False, "速度超出范围"
        
        if not (0 <= direction_angle <= 2*np.pi):
            return False, "方向角超出范围"
        
        if drop_time < 0:
            return False, "投放时间不能为负"
        
        if explosion_delay <= 0:
            return False, "起爆延迟必须为正"
        
        # 检查烟幕弹是否在落地前起爆
        if not self.check_explosion_before_ground(drop_time, explosion_delay, drone_init, speed, direction_angle):
            return False, "烟幕弹在落地后才起爆"
        
        explosion_pos = self.smoke_explosion_position(drop_time, explosion_delay, drone_init, speed, direction_angle)
        explosion_time = drop_time + explosion_delay
        
        # 时间约束：导弹飞行时间需满足条件
        if self.missile_flight_time < explosion_time + self.effective_time:
            return False, "导弹飞行时间不足"
        
        # x轴方向约束：云团需位于导弹与真目标之间
        missile_at_explosion = self.missile_position(explosion_time, missile_init)
        if not (missile_at_explosion[0] >= explosion_pos[0] >= self.target_bottom_view[0]):
            return False, "云团不在导弹与目标之间（x方向）"
        
        # y轴方向约束：云团y坐标不超过真目标边界
        if abs(explosion_pos[1]) > 207:  # 根据文档中的约束
            return False, "云团y坐标超出边界"
        
        # z轴方向约束：爆破时导弹高度应不低于云团高度
        missile_z_at_explosion = missile_at_explosion[2]
        if missile_z_at_explosion < explosion_pos[2]:
            return False, "导弹高度低于云团高度"
        
        # 云团消散时导弹高度不应高于云团最高点
        missile_at_end = self.missile_position(explosion_time + self.effective_time, missile_init)
        cloud_at_end = explosion_pos[2] - self.smoke_sink_speed * self.effective_time
        if missile_at_end[2] > explosion_pos[2]:  # 云团最高点就是起爆点
            return False, "导弹最终高度过高"
        
        return True, "所有约束满足"

# 问题1实现（保持不变）
def solve_problem1():
    """问题1：FY1以120m/s朝假目标飞行，1.5s后投放，3.6s后起爆"""
    model = SmokeInterferenceModel()
    
    drone_speed = 120.0
    direction_vector = model.fake_target - model.FY1_init
    direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
    drop_time = 1.5
    explosion_delay = 3.6
    
    explosion_pos = model.smoke_explosion_position(
        drop_time, explosion_delay, model.FY1_init, drone_speed, direction_angle
    )
    explosion_time = drop_time + explosion_delay
    
    print("问题1结果（最终版本）：")
    print(f"无人机飞行方向角: {direction_angle:.4f} rad ({np.degrees(direction_angle):.2f}°)")
    print(f"烟幕弹投放点: {model.smoke_drop_position(drop_time, model.FY1_init, drone_speed, direction_angle)}")
    print(f"烟幕弹起爆点: {explosion_pos}")
    print(f"起爆时间: {explosion_time}s")
    
    # 检查详细约束条件
    constraints_ok, constraint_msg = model.check_detailed_constraints(
        drone_speed, direction_angle, drop_time, explosion_delay, model.M1_init, model.FY1_init
    )
    print(f"约束条件检查: {constraints_ok} - {constraint_msg}")
    
    shielding_time = model.calculate_shielding_time(
        model.M1_init, explosion_time, explosion_pos
    )
    print(f"有效遮蔽时长: {shielding_time:.2f}s")
    
    return shielding_time

# 问题2实现（更新版本）
def solve_problem2_updated():
    """问题2：使用更新的约束条件和分层优化思想"""
    model = SmokeInterferenceModel()
    
    def objective_function(params):
        """目标函数：最大化遮蔽时间"""
        speed, direction_angle, drop_time, explosion_delay = params
        
        try:
            # 检查详细约束条件
            constraints_ok, _ = model.check_detailed_constraints(
                speed, direction_angle, drop_time, explosion_delay, model.M1_init, model.FY1_init
            )
            
            if not constraints_ok:
                return 1000  # 惩罚项
            
            explosion_pos = model.smoke_explosion_position(
                drop_time, explosion_delay, model.FY1_init, speed, direction_angle
            )
            explosion_time = drop_time + explosion_delay
            
            shielding_time = model.calculate_shielding_time(
                model.M1_init, explosion_time, explosion_pos
            )
            
            return -shielding_time  # 最小化负值等于最大化正值
            
        except Exception as e:
            return 1000
    
    print("问题2求解中（更新约束条件）...")
    
    # 参数边界（根据更新的约束条件）
    bounds = [
        (70, 140),      # speed
        (0, 2*np.pi),   # direction_angle
        (0.1, 30),      # drop_time
        (0.1, 15)       # explosion_delay
    ]
    
    # 分层优化思想的简化实现
    best_result = None
    best_shielding_time = 0
    
    # 第一层：粗搜索
    print("第一层优化：粗搜索...")
    result1 = differential_evolution(
        objective_function, 
        bounds, 
        seed=42,
        maxiter=100,
        popsize=15,
        atol=1e-4,
        tol=1e-4
    )
    
    if result1.success and -result1.fun > best_shielding_time:
        best_result = result1
        best_shielding_time = -result1.fun
        print(f"第一层最优解: {best_shielding_time:.3f}s")
    
    # 第二层：在最优解附近精细搜索
    if best_result is not None:
        print("第二层优化：精细搜索...")
        center = best_result.x
        
        # 在最优解附近定义更小的搜索范围
        refined_bounds = [
            (max(70, center[0]-10), min(140, center[0]+10)),
            (max(0, center[1]-0.5), min(2*np.pi, center[1]+0.5)),
            (max(0.1, center[2]-1), min(30, center[2]+1)),
            (max(0.1, center[3]-1), min(15, center[3]+1))
        ]
        
        result2 = differential_evolution(
            objective_function, 
            refined_bounds, 
            seed=123,
            maxiter=100,
            popsize=10,
            atol=1e-6,
            tol=1e-6
        )
        
        if result2.success and -result2.fun > best_shielding_time:
            best_result = result2
            best_shielding_time = -result2.fun
            print(f"第二层最优解: {best_shielding_time:.3f}s")
    
    if best_result is None:
        print("优化失败，使用默认参数")
        optimal_params = [120.0, np.pi, 2.0, 4.0]
        best_shielding_time = -objective_function(optimal_params)
    else:
        optimal_params = best_result.x
    
    speed_opt, angle_opt, drop_time_opt, explosion_delay_opt = optimal_params
    
    # 计算最优解的详细信息
    explosion_pos_opt = model.smoke_explosion_position(
        drop_time_opt, explosion_delay_opt, model.FY1_init, speed_opt, angle_opt
    )
    explosion_time_opt = drop_time_opt + explosion_delay_opt
    
    final_shielding_time = model.calculate_shielding_time(
        model.M1_init, explosion_time_opt, explosion_pos_opt
    )
    
    print("\n问题2结果（更新约束条件）：")
    print(f"最优飞行速度: {speed_opt:.2f} m/s")
    print(f"最优飞行方向角: {angle_opt:.4f} rad ({np.degrees(angle_opt):.2f}°)")
    print(f"最优投放时间: {drop_time_opt:.2f} s")
    print(f"最优起爆延迟: {explosion_delay_opt:.2f} s")
    print(f"烟幕弹投放点: {model.smoke_drop_position(drop_time_opt, model.FY1_init, speed_opt, angle_opt)}")
    print(f"烟幕弹起爆点: {explosion_pos_opt}")
    print(f"起爆时间: {explosion_time_opt:.2f} s")
    print(f"最大有效遮蔽时长: {final_shielding_time:.2f} s")
    
    # 验证最优解的约束条件
    constraints_ok, constraint_msg = model.check_detailed_constraints(
        speed_opt, angle_opt, drop_time_opt, explosion_delay_opt, model.M1_init, model.FY1_init
    )
    print(f"最优解约束检查: {constraints_ok} - {constraint_msg}")
    
    return {
        'speed': speed_opt,
        'direction_angle': angle_opt,
        'drop_time': drop_time_opt,
        'explosion_delay': explosion_delay_opt,
        'explosion_pos': explosion_pos_opt,
        'explosion_time': explosion_time_opt,
        'shielding_time': final_shielding_time
    }

def analyze_constraints():
    """分析约束条件的影响"""
    model = SmokeInterferenceModel()
    
    print("\n【约束条件分析】")
    print(f"导弹飞行总时间: {model.missile_flight_time:.2f}s")
    print(f"导弹速度角度β: {np.degrees(model.missile_beta):.2f}°")
    print(f"导弹水平角度α: {np.degrees(model.missile_alpha):.2f}°")
    
    print("\n关键约束条件:")
    print("1. 烟幕弹必须在落地前起爆")
    print("2. 导弹飞行时间 ≥ 起爆时间 + 20s")
    print("3. 云团x坐标位于导弹与目标之间")
    print("4. 云团y坐标 ≤ 207m")
    print("5. 爆破时导弹高度 ≥ 云团高度")
    print("6. 消散时导弹高度 ≤ 云团最高点")

if __name__ == "__main__":
    print("=" * 80)
    print("烟幕干扰弹投放策略数学建模（最终版本）")
    print("=" * 80)
    
    # 分析约束条件
    analyze_constraints()
    
    print("\n" + "=" * 60)
    
    # 求解问题1
    shielding_time_1 = solve_problem1()
    
    print("\n" + "=" * 60)
    
    # 求解问题2（更新版本）
    optimal_params = solve_problem2_updated()
    
    print("\n" + "=" * 60)
    print("总结:")
    print(f"问题1遮蔽时间: {shielding_time_1:.2f}s")
    print(f"问题2最优遮蔽时间: {optimal_params['shielding_time']:.2f}s")
    if shielding_time_1 > 0:
        improvement = optimal_params['shielding_time'] - shielding_time_1
        improvement_pct = (improvement / shielding_time_1) * 100
        print(f"优化提升: {improvement:.2f}s ({improvement_pct:.1f}%)")
    else:
        print(f"优化提升: {optimal_params['shielding_time']:.2f}s (从0开始)")
    
    print("\n关键改进:")
    print("1. 增加了烟幕弹落地前起爆的约束")
    print("2. 完善了x、y、z轴方向的详细约束条件")
    print("3. 采用分层优化策略提高求解效率")
    print("4. 强化了物理合理性检查")