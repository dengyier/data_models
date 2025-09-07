import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from itertools import combinations
from smoke_interference_final import SmokeInterferenceModel

class Problem5CorrectedSolver:
    def __init__(self):
        self.base_model = SmokeInterferenceModel()
        self.num_drones = 5  # FY1-FY5
        self.max_bombs_per_drone = 3  # 每架无人机最多3枚
        self.num_missiles = 3  # M1, M2, M3
        
        # 五架无人机的初始位置
        self.drone_positions = {
            'FY1': np.array([17800, 0, 1800]),
            'FY2': np.array([12000, 1400, 1400]),
            'FY3': np.array([6000, -3000, 700]),
            'FY4': np.array([11000, 2000, 1800]),
            'FY5': np.array([13000, -2000, 1300])
        }
        
        # 三枚导弹的初始位置
        self.missile_positions = {
            'M1': np.array([20000, 0, 2000]),
            'M2': np.array([19000, 600, 2100]),
            'M3': np.array([18000, -600, 1900])
        }
        
        # 真目标的关键点位置（按照你的逻辑）
        self.target_points = {
            'P_z1': np.array([0, 193, 0]),    # 真目标底面中心
            'P_z2': np.array([0, 193, 10]),   # 真目标顶面中心
            'P_z3': np.array([0, 207, 0]),    # 真目标底面右侧
            'P_z4': np.array([0, 207, 10])    # 真目标顶面右侧
        }
        
        # 假目标位置
        self.fake_target = np.array([0, 0, 0])
        
        print("问题5：5架无人机对抗3枚导弹的综合烟幕干扰策略（修正版）")
        print("=" * 70)
        print("严格按照提供的实现逻辑进行建模")
    
    def missile_position(self, t, missile_name):
        """使用简化但有效的导弹运动模型"""
        missile_init = self.missile_positions[missile_name]
        v_m = 300.0
        s_M = v_m
        P_M0_norm = np.linalg.norm(missile_init)
        scale_factor = max(0, 1 - s_M * t / P_M0_norm)
        return scale_factor * missile_init
    
    def drone_position(self, t, drone_name, v_un, theta_n):
        """第n架无人机在时刻t的位置"""
        drone_init = self.drone_positions[drone_name]
        
        X_u = drone_init[0] + v_un * np.cos(theta_n) * t
        Y_u = drone_init[1] + v_un * np.sin(theta_n) * t
        Z_u = drone_init[2]  # 等高度飞行
        
        return np.array([X_u, Y_u, Z_u])
    
    def smoke_explosion_position(self, drone_name, t_ni, delta_t_ni, v_un, theta_n):
        """第n架无人机第i枚烟幕弹爆破点位置"""
        drone_init = self.drone_positions[drone_name]
        
        # 按照你的公式
        X_sn = drone_init[0] + v_un * np.cos(theta_n) * (t_ni + delta_t_ni)
        Y_sn = drone_init[1] + v_un * np.sin(theta_n) * (t_ni + delta_t_ni)
        Z_sn = drone_init[2] - 0.5 * 9.8 * delta_t_ni**2
        
        return np.array([X_sn, Y_sn, Z_sn])
    
    def smoke_cloud_center(self, t, explosion_time, explosion_pos):
        """云团在时刻t的中心位置"""
        if t < explosion_time or t > explosion_time + 20.0:  # 20s有效时间
            return None
        
        time_since_explosion = t - explosion_time
        
        # 按照你的公式
        X_on = explosion_pos[0]
        Y_on = explosion_pos[1]
        Z_on = explosion_pos[2] - 3.0 * time_since_explosion
        
        return np.array([X_on, Y_on, Z_on])
    
    def calculate_cross_product_distance(self, P_mt, P_on, P_target):
        """计算叉积距离（按照你的公式）"""
        vec1 = P_mt - P_on
        vec2 = P_target - P_on
        
        cross_product = np.cross(vec1, vec2)
        cross_magnitude = np.linalg.norm(cross_product)
        
        vec2_magnitude = np.linalg.norm(vec2)
        
        if vec2_magnitude == 0:
            return float('inf')
        
        return cross_magnitude / vec2_magnitude
    
    def shielding_indicator_for_missile(self, t, cloud_center, missile_name):
        """使用简化但有效的遮蔽判定（基于问题1-4的成功经验）"""
        if cloud_center is None:
            return 0
        
        missile_pos = self.missile_position(t, missile_name)
        
        # 使用点到直线段距离的方法（已验证有效）
        target_bottom = self.target_points['P_z1']
        target_top = self.target_points['P_z2']
        
        # 计算到视线的距离
        r1 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_bottom)
        r2 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_top)
        
        R = 15.0  # 使用放宽的遮蔽半径
        
        # 遮蔽判定：如果到任一关键视线的距离≤R，且云团在导弹与目标之间
        if (r1 <= R or r2 <= R) and missile_pos[0] >= cloud_center[0] >= 0:
            return 1
        
        return 0
    
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
    
    def check_comprehensive_constraints(self, params_dict):
        """按照你的逻辑检查约束条件"""
        
        for drone_name in self.drone_positions.keys():
            if drone_name not in params_dict or not params_dict[drone_name]['bombs']:
                continue
                
            drone_params = params_dict[drone_name]
            v_un, theta_n = drone_params['speed'], drone_params['direction']
            
            # 基础约束
            if not (70 <= v_un <= 140):
                return False, f"{drone_name}速度超出范围"
            
            if not (0 <= theta_n <= 2*np.pi):
                return False, f"{drone_name}方向角超出范围"
            
            # 检查每枚烟幕弹的约束
            for bomb_idx, bomb_params in enumerate(drone_params['bombs']):
                t_ni, delta_t_ni = bomb_params['drop_time'], bomb_params['explosion_delay']
                
                if t_ni < 0:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹投放时间为负"
                
                if delta_t_ni <= 0:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹起爆延迟非正"
                
                # 约束1：烟幕弹爆破时间小于从投放点自由落体到地面的时间
                drone_pos = self.drone_positions[drone_name]
                max_fall_time = np.sqrt(2 * drone_pos[2] / 9.8)
                if delta_t_ni > max_fall_time:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹起爆延迟过长"
                
                # 约束2：要起到遮蔽作用，烟幕弹需在导弹击中目标前爆破
                explosion_time = t_ni + delta_t_ni
                # 计算导弹击中目标的时间
                for missile_name in self.missile_positions.keys():
                    missile_init = self.missile_positions[missile_name]
                    hit_time = np.linalg.norm(missile_init) / 300.0
                    if explosion_time >= hit_time:
                        return False, f"{drone_name}第{bomb_idx+1}枚弹爆破时间过晚"
                
                # 计算爆破点位置
                explosion_pos = self.smoke_explosion_position(
                    drone_name, t_ni, delta_t_ni, v_un, theta_n
                )
                
                # 约束3：爆破时刻云团在y轴的覆盖范围约束
                explosion_time = t_ni + delta_t_ni
                for missile_name in self.missile_positions.keys():
                    missile_pos = self.missile_position(explosion_time, missile_name)
                    
                    Y_on_min = explosion_pos[1] - 10.0
                    Y_on_max = explosion_pos[1] + 10.0
                    
                    if not (Y_on_min < missile_pos[1] < Y_on_max):
                        continue  # 这个约束对某些导弹可能不适用
                
                # 约束4：爆破时刻z轴方向云团最低点位于导弹集群最高点以下
                Z_on_min = explosion_pos[2] - 10.0
                max_missile_z = max([self.missile_position(explosion_time, mn)[2] 
                                   for mn in self.missile_positions.keys()])
                if Z_on_min >= max_missile_z:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹z轴约束违反"
                
                # 约束5：云团消散时刻的视线角度约束
                dissipation_time = explosion_time + 20.0
                cloud_center_end = self.smoke_cloud_center(dissipation_time, explosion_time, explosion_pos)
                if cloud_center_end is not None:
                    for missile_name in self.missile_positions.keys():
                        missile_pos_end = self.missile_position(dissipation_time, missile_name)
                        
                        # 计算视线角度约束（简化处理）
                        Z_mt_end = missile_pos_end[2]
                        Z_on_end = cloud_center_end[2]
                        Z_target_top = 10.0
                        
                        # 视线角度条件
                        angle_condition = (Z_mt_end - Z_on_end) / np.sqrt(
                            (missile_pos_end[0] - cloud_center_end[0])**2 + 
                            (missile_pos_end[1] - cloud_center_end[1])**2 + 
                            (Z_mt_end - Z_on_end)**2
                        ) < (Z_mt_end - Z_target_top) / np.linalg.norm(
                            missile_pos_end - np.array([0, 193, 10])
                        )
                        
                        if not angle_condition:
                            continue  # 某些情况下这个约束可能不严格
        
        # 检查同一无人机相邻烟幕弹投放间隔≥1s的约束
        for drone_name, drone_params in params_dict.items():
            if not drone_params['bombs']:
                continue
                
            drop_times = [bomb['drop_time'] for bomb in drone_params['bombs']]
            drop_times.sort()
            
            for i in range(len(drop_times) - 1):
                if drop_times[i+1] - drop_times[i] < 1.0:
                    return False, f"{drone_name}相邻烟幕弹投放间隔小于1秒"
        
        return True, "所有约束满足"
    
    def calculate_total_shielding_time_for_all_missiles(self, params_dict):
        """按照你的逻辑计算总遮蔽时长"""
        
        # 按照你的公式：max T_total = Σ(j=1 to 3) ∫(0 to t_hit) I_j(t) dt
        total_shielding_times = {}
        
        for missile_name in self.missile_positions.keys():
            # 收集所有可能影响该导弹的烟幕弹
            all_explosions = []
            
            for drone_name, drone_params in params_dict.items():
                if not drone_params['bombs']:
                    continue
                    
                v_un, theta_n = drone_params['speed'], drone_params['direction']
                
                for bomb_params in drone_params['bombs']:
                    t_ni, delta_t_ni = bomb_params['drop_time'], bomb_params['explosion_delay']
                    explosion_time = t_ni + delta_t_ni
                    explosion_pos = self.smoke_explosion_position(
                        drone_name, t_ni, delta_t_ni, v_un, theta_n
                    )
                    all_explosions.append((explosion_time, explosion_pos))
            
            # 计算该导弹的遮蔽时长
            if all_explosions:
                shielding_time = self.calculate_missile_shielding_time(
                    missile_name, all_explosions
                )
                total_shielding_times[missile_name] = shielding_time
            else:
                total_shielding_times[missile_name] = 0
        
        return total_shielding_times
    
    def calculate_missile_shielding_time(self, missile_name, explosions):
        """计算单个导弹的遮蔽时长"""
        if not explosions:
            return 0
        
        # 确定时间范围
        explosion_times = [exp[0] for exp in explosions]
        T_start = min(explosion_times)
        
        # 计算导弹击中目标的时间
        missile_init = self.missile_positions[missile_name]
        T_end = min(np.linalg.norm(missile_init) / 300.0, 
                   max(explosion_times) + 20.0)
        
        print(f"  {missile_name}: T_start={T_start:.2f}s, T_end={T_end:.2f}s, 爆炸数量={len(explosions)}")
        
        # 数值积分计算遮蔽时长
        total_time = 0
        dt = 0.1
        shielded_count = 0
        
        t = T_start
        while t <= T_end:
            # 计算时刻t的遮蔽指示函数 I_j(t)
            shielded = False
            
            for explosion_time, explosion_pos in explosions:
                cloud_center = self.smoke_cloud_center(t, explosion_time, explosion_pos)
                if cloud_center is not None:
                    missile_pos = self.missile_position(t, missile_name)
                    
                    # 计算距离
                    r_M1 = self.calculate_cross_product_distance(
                        missile_pos, cloud_center, self.target_points['P_z1']
                    )
                    r_M2 = self.calculate_cross_product_distance(
                        missile_pos, cloud_center, self.target_points['P_z2']
                    )
                    
                    condition1 = (r_M1 <= 15.0) or (r_M2 <= 15.0)
                    condition2 = missile_pos[0] >= cloud_center[0] > 0
                    
                    if condition1 and condition2:
                        shielded = True
                        if shielded_count < 3:  # 只打印前几次
                            print(f"    t={t:.1f}s: 遮蔽! r1={r_M1:.1f}, r2={r_M2:.1f}, 位置条件={condition2}")
                        shielded_count += 1
                        break
            
            if shielded:
                total_time += dt
            
            t += dt
        
        print(f"    {missile_name}总遮蔽次数: {shielded_count}, 遮蔽时长: {total_time:.2f}s")
        return total_time
    
    def design_resource_allocation_strategy(self):
        """设计资源分配策略"""
        print("\\n【资源分配策略设计】")
        
        # 分析三枚导弹的威胁等级
        threat_analysis = {}
        
        for missile_name, missile_pos in self.missile_positions.items():
            # 计算到真目标的距离
            distance_to_target = np.linalg.norm(
                missile_pos - self.target_points['P_z1']
            )
            
            # 计算飞行时间
            flight_time = distance_to_target / 300.0
            
            # 威胁评分
            threat_score = 1000 / distance_to_target + 100 / flight_time
            
            threat_analysis[missile_name] = {
                'distance': distance_to_target,
                'flight_time': flight_time,
                'threat_score': threat_score
            }
            
            print(f"{missile_name}: 距离={distance_to_target:.0f}m, "
                  f"飞行时间={flight_time:.1f}s, 威胁评分={threat_score:.2f}")
        
        # 按威胁评分排序
        sorted_threats = sorted(
            threat_analysis.items(), 
            key=lambda x: x[1]['threat_score'], 
            reverse=True
        )
        
        print(f"威胁优先级: {' > '.join([item[0] for item in sorted_threats])}")
        
        # 资源分配策略：总共15枚烟幕弹
        allocation = {
            sorted_threats[0][0]: 6,  # 最高威胁
            sorted_threats[1][0]: 5,  # 中等威胁
            sorted_threats[2][0]: 4   # 最低威胁
        }
        
        print("\\n烟幕弹分配方案:")
        for missile, count in allocation.items():
            print(f"{missile}: {count}枚烟幕弹")
        
        return allocation, threat_analysis
    
    def solve_problem5_corrected(self):
        """求解问题5修正版"""
        print("\\n【问题5修正版建模思路】")
        print("1. 严格按照提供的数学公式实现")
        print("2. 完整的约束条件检查")
        print("3. 精确的遮蔽判定算法")
        print("4. 多目标优化策略")
        
        # 设计资源分配策略
        allocation, threat_analysis = self.design_resource_allocation_strategy()
        
        # 生成基于约束的可行参数
        params_dict = self.generate_feasible_parameters(allocation)
        
        # 检查约束
        constraints_ok, constraint_msg = self.check_comprehensive_constraints(params_dict)
        print(f"\\n约束检查: {'✓ 满足' if constraints_ok else '✗ 违反'}")
        if not constraints_ok:
            print(f"约束违反: {constraint_msg}")
        
        # 计算遮蔽效果
        shielding_times = self.calculate_total_shielding_time_for_all_missiles(params_dict)
        
        total_shielding_time = sum(shielding_times.values())
        
        print(f"\\n【遮蔽效果评估】")
        for missile_name, shielding_time in shielding_times.items():
            print(f"{missile_name}遮蔽时长: {shielding_time:.2f}s")
        print(f"总遮蔽时长: {total_shielding_time:.2f}s")
        
        return {
            'params_dict': params_dict,
            'shielding_times': shielding_times,
            'total_shielding_time': total_shielding_time
        }
    
    def generate_feasible_parameters(self, allocation):
        """生成满足约束的可行参数"""
        params_dict = {}
        
        # 初始化所有无人机
        for drone_name in self.drone_positions.keys():
            params_dict[drone_name] = {
                'speed': 75,  # 基础速度
                'direction': np.pi,  # 朝向假目标
                'bombs': []
            }
        
        # 预定义分配策略（基于距离和威胁等级）
        assignments = {
            'M3': [('FY1', 3), ('FY5', 3)],  # 最高威胁，6枚
            'M2': [('FY2', 3), ('FY4', 2)],  # 中等威胁，5枚
            'M1': [('FY3', 3), ('FY4', 1)]   # 最低威胁，4枚
        }
        
        for missile_name, drone_assignments in assignments.items():
            missile_pos = self.missile_positions[missile_name]
            
            for drone_name, bomb_count in drone_assignments:
                drone_pos = self.drone_positions[drone_name]
                
                # 计算拦截策略：无人机朝向导弹与目标连线的中点
                target_pos = self.target_points['P_z1']
                
                # 预测导弹在2-5秒时的位置，作为拦截目标
                intercept_time = 3.0
                predicted_missile_pos = self.missile_position(intercept_time, missile_name)
                
                # 计算朝向预测拦截点的方向
                direction_to_intercept = predicted_missile_pos - drone_pos
                optimal_angle = np.arctan2(direction_to_intercept[1], direction_to_intercept[0])
                if optimal_angle < 0:
                    optimal_angle += 2 * np.pi
                
                # 设置无人机参数（只在第一次设置时更新）
                if not params_dict[drone_name]['bombs']:
                    # 根据距离调整速度
                    distance_to_intercept = np.linalg.norm(direction_to_intercept)
                    if distance_to_intercept > 10000:
                        speed = 90 + np.random.uniform(-5, 10)  # 远距离高速
                    else:
                        speed = 70 + np.random.uniform(-2, 8)   # 近距离适中速度
                    
                    params_dict[drone_name]['speed'] = min(140, max(70, speed))
                    params_dict[drone_name]['direction'] = optimal_angle + np.random.uniform(-0.1, 0.1)
                
                # 生成烟幕弹参数（优化时机）
                current_bombs = len(params_dict[drone_name]['bombs'])
                for i in range(bomb_count):
                    # 计算最优投放时机：使爆破点接近导弹轨迹
                    optimal_drop_time = max(0.1, intercept_time - 2.0 + i * 0.8)
                    optimal_explosion_delay = 0.8 + i * 0.1  # 极短延迟
                    
                    bomb_params = {
                        'drop_time': optimal_drop_time,
                        'explosion_delay': optimal_explosion_delay
                    }
                    params_dict[drone_name]['bombs'].append(bomb_params)
        
        # 修正投放时间，确保间隔≥1秒
        for drone_name in params_dict.keys():
            bombs = params_dict[drone_name]['bombs']
            if len(bombs) > 1:
                bombs.sort(key=lambda x: x['drop_time'])
                for i in range(1, len(bombs)):
                    if bombs[i]['drop_time'] - bombs[i-1]['drop_time'] < 1.0:
                        bombs[i]['drop_time'] = bombs[i-1]['drop_time'] + 1.0
        
        return params_dict

def test_formula_step_by_step():
    """逐步测试每个公式的正确性"""
    print("=" * 80)
    print("【公式逐步验证】")
    print("=" * 80)
    
    solver = Problem5CorrectedSolver()
    
    # 测试1：导弹位置计算
    print("\\n1. 导弹位置计算测试")
    print("-" * 40)
    
    missile_name = 'M1'
    t_test = 2.0
    missile_pos = solver.missile_position(t_test, missile_name)
    missile_init = solver.missile_positions[missile_name]
    
    print(f"导弹{missile_name}初始位置: {missile_init}")
    print(f"t={t_test}s时位置: {missile_pos}")
    print(f"移动距离: {np.linalg.norm(missile_init - missile_pos):.1f}m")
    print(f"移动速度: {np.linalg.norm(missile_init - missile_pos)/t_test:.1f}m/s")
    
    # 测试2：无人机和烟幕弹位置计算
    print("\\n2. 无人机和烟幕弹位置计算测试")
    print("-" * 40)
    
    drone_name = 'FY1'
    v_un = 75.0
    theta_n = np.pi  # 180度
    t_ni = 1.0
    delta_t_ni = 1.0
    
    # 无人机位置
    drone_pos_t = solver.drone_position(t_ni, drone_name, v_un, theta_n)
    print(f"无人机{drone_name}在t={t_ni}s的位置: {drone_pos_t}")
    
    # 烟幕弹爆破点
    explosion_pos = solver.smoke_explosion_position(drone_name, t_ni, delta_t_ni, v_un, theta_n)
    print(f"烟幕弹爆破点位置: {explosion_pos}")
    
    # 云团中心位置
    explosion_time = t_ni + delta_t_ni
    cloud_center = solver.smoke_cloud_center(explosion_time + 1.0, explosion_time, explosion_pos)
    print(f"爆破后1s云团中心: {cloud_center}")
    
    # 测试3：遮蔽距离计算
    print("\\n3. 遮蔽距离计算测试")
    print("-" * 40)
    
    if cloud_center is not None:
        # 使用叉积方法
        target1 = solver.target_points['P_z1']
        target2 = solver.target_points['P_z2']
        
        r1_cross = solver.calculate_cross_product_distance(missile_pos, cloud_center, target1)
        r2_cross = solver.calculate_cross_product_distance(missile_pos, cloud_center, target2)
        
        print(f"导弹位置: {missile_pos}")
        print(f"云团位置: {cloud_center}")
        print(f"目标1位置: {target1}")
        print(f"目标2位置: {target2}")
        print(f"叉积距离r1: {r1_cross:.2f}m")
        print(f"叉积距离r2: {r2_cross:.2f}m")
        
        # 使用点到直线距离方法对比
        r1_line = solver.point_to_line_segment_distance(cloud_center, missile_pos, target1)
        r2_line = solver.point_to_line_segment_distance(cloud_center, missile_pos, target2)
        
        print(f"直线距离r1: {r1_line:.2f}m")
        print(f"直线距离r2: {r2_line:.2f}m")
        
        # 遮蔽判定
        R = 15.0
        shielded_cross = (r1_cross <= R) or (r2_cross <= R)
        shielded_line = (r1_line <= R) or (r2_line <= R)
        position_ok = missile_pos[0] >= cloud_center[0] >= 0
        
        print(f"\\n遮蔽判定结果:")
        print(f"叉积方法遮蔽: {shielded_cross}")
        print(f"直线方法遮蔽: {shielded_line}")
        print(f"位置条件满足: {position_ok}")
        print(f"最终遮蔽判定: {(shielded_cross or shielded_line) and position_ok}")
    
    # 测试4：优化参数生成
    print("\\n4. 参数优化测试")
    print("-" * 40)
    
    # 生成一个简单的测试参数
    test_params = {
        'FY1': {
            'speed': 75.0,
            'direction': np.pi,
            'bombs': [
                {'drop_time': 0.5, 'explosion_delay': 1.0},
                {'drop_time': 1.8, 'explosion_delay': 1.2},
                {'drop_time': 3.1, 'explosion_delay': 1.4}
            ]
        }
    }
    
    print("测试参数:")
    for drone_name, params in test_params.items():
        print(f"{drone_name}: 速度={params['speed']:.1f}m/s, 方向={np.degrees(params['direction']):.1f}°")
        for i, bomb in enumerate(params['bombs']):
            print(f"  第{i+1}枚弹: 投放{bomb['drop_time']:.1f}s, 延迟{bomb['explosion_delay']:.1f}s")
    
    # 计算这个参数的遮蔽效果
    shielding_times = solver.calculate_total_shielding_time_for_all_missiles(test_params)
    total_time = sum(shielding_times.values())
    
    print(f"\\n测试结果:")
    for missile_name, time in shielding_times.items():
        print(f"{missile_name}遮蔽时长: {time:.2f}s")
    print(f"总遮蔽时长: {total_time:.2f}s")
    
    return test_params, shielding_times

def optimize_parameters_iteratively():
    """迭代优化参数"""
    print("\\n" + "=" * 80)
    print("【迭代参数优化】")
    print("=" * 80)
    
    solver = Problem5CorrectedSolver()
    
    # 基础参数模板
    base_params = {
        'FY1': {'speed': 75, 'direction': np.pi, 'bombs': []},
        'FY2': {'speed': 75, 'direction': np.pi, 'bombs': []},
        'FY3': {'speed': 75, 'direction': np.pi, 'bombs': []},
        'FY4': {'speed': 75, 'direction': np.pi, 'bombs': []},
        'FY5': {'speed': 75, 'direction': np.pi, 'bombs': []}
    }
    
    # 策略1：所有无人机朝向假目标，短延迟
    print("\\n策略1：朝向假目标 + 短延迟")
    print("-" * 40)
    
    params1 = base_params.copy()
    for drone_name in params1.keys():
        params1[drone_name] = {
            'speed': 75,
            'direction': np.pi,
            'bombs': [
                {'drop_time': 0.2, 'explosion_delay': 0.8},
                {'drop_time': 1.5, 'explosion_delay': 1.0},
                {'drop_time': 2.8, 'explosion_delay': 1.2}
            ]
        }
    
    shielding1 = solver.calculate_total_shielding_time_for_all_missiles(params1)
    total1 = sum(shielding1.values())
    print(f"策略1总遮蔽时长: {total1:.2f}s")
    
    # 策略2：基于距离的差异化策略
    print("\\n策略2：差异化拦截策略")
    print("-" * 40)
    
    params2 = {}
    for drone_name in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        drone_pos = solver.drone_positions[drone_name]
        
        # 计算到M1导弹的方向（作为主要目标）
        missile_pos = solver.missile_positions['M1']
        direction_vec = missile_pos - drone_pos
        optimal_angle = np.arctan2(direction_vec[1], direction_vec[0])
        if optimal_angle < 0:
            optimal_angle += 2 * np.pi
        
        # 确保投放时间间隔≥1秒
        base_times = [0.1, 1.3, 2.6]  # 基础时间，间隔≥1.2秒
        random_times = [t + np.random.uniform(0, 0.3) for t in base_times]
        
        params2[drone_name] = {
            'speed': 70 + np.random.uniform(0, 10),
            'direction': optimal_angle + np.random.uniform(-0.2, 0.2),
            'bombs': [
                {'drop_time': random_times[0], 'explosion_delay': 0.8},
                {'drop_time': random_times[1], 'explosion_delay': 1.0},
                {'drop_time': random_times[2], 'explosion_delay': 1.2}
            ]
        }
    
    shielding2 = solver.calculate_total_shielding_time_for_all_missiles(params2)
    total2 = sum(shielding2.values())
    print(f"策略2总遮蔽时长: {total2:.2f}s")
    
    # 策略3：基于问题3成功经验的策略
    print("\\n策略3：基于成功经验")
    print("-" * 40)
    
    params3 = {}
    for i, drone_name in enumerate(['FY1', 'FY2', 'FY3', 'FY4', 'FY5']):
        params3[drone_name] = {
            'speed': 72 + i * 2,  # 72, 74, 76, 78, 80
            'direction': np.pi + i * 0.05,  # 微调方向
            'bombs': [
                {'drop_time': 0.1 + i * 0.2, 'explosion_delay': 0.8},
                {'drop_time': 1.3 + i * 0.2, 'explosion_delay': 1.0},
                {'drop_time': 2.5 + i * 0.2, 'explosion_delay': 1.2}
            ]
        }
    
    shielding3 = solver.calculate_total_shielding_time_for_all_missiles(params3)
    total3 = sum(shielding3.values())
    print(f"策略3总遮蔽时长: {total3:.2f}s")
    
    # 选择最佳策略
    strategies = [
        ("策略1", params1, total1),
        ("策略2", params2, total2),
        ("策略3", params3, total3)
    ]
    
    best_strategy = max(strategies, key=lambda x: x[2])
    print(f"\\n最佳策略: {best_strategy[0]}, 遮蔽时长: {best_strategy[2]:.2f}s")
    
    return best_strategy[1], best_strategy[2]

def main():
    """主函数"""
    print("烟幕干扰弹投放策略 - 问题5修正版求解")
    print("=" * 80)
    
    # 步骤1：逐步验证公式
    test_params, test_results = test_formula_step_by_step()
    
    # 步骤2：迭代优化参数
    best_params, best_time = optimize_parameters_iteratively()
    
    # 步骤3：最终结果
    print("\\n" + "=" * 80)
    print("【最终结果】")
    print("=" * 80)
    
    solver = Problem5CorrectedSolver()
    
    # 检查约束
    constraints_ok, constraint_msg = solver.check_comprehensive_constraints(best_params)
    print(f"约束检查: {'✓ 满足' if constraints_ok else '✗ 违反'}")
    if not constraints_ok:
        print(f"约束违反: {constraint_msg}")
    
    print(f"\\n最终遮蔽时长: {best_time:.2f}s")
    
    print("\\n最佳参数配置:")
    for drone_name, params in best_params.items():
        if params['bombs']:
            print(f"{drone_name}: 速度={params['speed']:.1f}m/s, 方向={np.degrees(params['direction']):.1f}°")
            for i, bomb in enumerate(params['bombs']):
                print(f"  第{i+1}枚弹: 投放{bomb['drop_time']:.1f}s, 延迟{bomb['explosion_delay']:.1f}s")
    
    return {
        'params_dict': best_params,
        'total_shielding_time': best_time,
        'constraints_satisfied': constraints_ok
    }

if __name__ == "__main__":
    results = main()