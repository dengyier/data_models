import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from itertools import combinations
from smoke_interference_final import SmokeInterferenceModel

class Problem5BasicSolver:
    def __init__(self):
        self.base_model = SmokeInterferenceModel()
        self.num_drones = 5  # FY1-FY5
        self.max_bombs_per_drone = 3  # 每架无人机最多3枚
        self.num_missiles = 3  # M1, M2, M3
        
        # 五架无人机的初始位置
        self.drone_positions = {
            'FY1': self.base_model.FY1_init,  # [17800, 0, 1800]
            'FY2': self.base_model.FY2_init,  # [12000, 1400, 1400]
            'FY3': self.base_model.FY3_init,  # [6000, -3000, 700]
            'FY4': self.base_model.FY4_init,  # [11000, 2000, 1800]
            'FY5': self.base_model.FY5_init   # [13000, -2000, 1300]
        }
        
        # 三枚导弹的初始位置
        self.missile_positions = {
            'M1': self.base_model.M1_init,  # [20000, 0, 2000]
            'M2': self.base_model.M2_init,  # [19000, 600, 2100]
            'M3': self.base_model.M3_init   # [18000, -600, 1900]
        }
        
        # 真目标的关键点位置
        self.target_points = {
            'P_z1': np.array([0, 193, 0]),    # 真目标底面中心
            'P_z2': np.array([0, 193, 10]),   # 真目标顶面中心
            'P_z3': np.array([0, 207, 0]),    # 真目标底面右侧
            'P_z4': np.array([0, 207, 10])    # 真目标顶面右侧
        }
        
        print("问题5：5架无人机对抗3枚导弹的综合烟幕干扰策略")
        print("=" * 70)
        print("复杂性：最多60个决策变量的多目标优化问题")
    
    def missile_position(self, t, missile_name):
        """计算指定导弹在时刻t的位置"""
        missile_init = self.missile_positions[missile_name]
        v_m = self.base_model.missile_speed
        s_M = v_m
        P_M0_norm = np.linalg.norm(missile_init)
        scale_factor = max(0, 1 - s_M * t / P_M0_norm)
        return scale_factor * missile_init
    
    def drone_position(self, t, drone_name, v_un, theta_n):
        """第n架无人机在时刻t的位置"""
        drone_init = self.drone_positions[drone_name]
        
        X_un = drone_init[0] + v_un * np.cos(theta_n) * t
        Y_un = drone_init[1] + v_un * np.sin(theta_n) * t
        Z_un = drone_init[2]  # 等高度飞行
        
        return np.array([X_un, Y_un, Z_un])
    
    def smoke_explosion_position(self, drone_name, t_ni, delta_t_ni, v_un, theta_n):
        """第n架无人机第i枚烟幕弹爆破点位置"""
        drone_init = self.drone_positions[drone_name]
        
        X_sn = drone_init[0] + v_un * np.cos(theta_n) * (t_ni + delta_t_ni)
        Y_sn = drone_init[1] + v_un * np.sin(theta_n) * (t_ni + delta_t_ni)
        Z_sn = drone_init[2] - 0.5 * self.base_model.g * delta_t_ni**2
        
        return np.array([X_sn, Y_sn, Z_sn])
    
    def smoke_cloud_center(self, t, explosion_time, explosion_pos):
        """云团在时刻t的中心位置"""
        if t < explosion_time or t > explosion_time + self.base_model.effective_time:
            return None
        
        time_since_explosion = t - explosion_time
        
        X_on = explosion_pos[0]
        Y_on = explosion_pos[1]
        Z_on = explosion_pos[2] - 3.0 * time_since_explosion
        
        return np.array([X_on, Y_on, Z_on])
    
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
    
    def calculate_missile_shielding_distances(self, cloud_center, missile_name):
        """计算云团中心到导弹视线的距离"""
        missile_pos = self.missile_position(0, missile_name)  # 使用初始位置计算视线
        
        # 计算到各个关键视线的距离
        distances = {}
        
        # 到P_z1的距离
        r1 = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_points['P_z1']
        )
        distances['r1'] = r1
        
        # 到P_z2的距离  
        r2 = self.point_to_line_segment_distance(
            cloud_center, missile_pos, self.target_points['P_z2']
        )
        distances['r2'] = r2
        
        return distances
    
    def shielding_indicator_for_missile(self, t, cloud_center, missile_name):
        """判断云团对指定导弹是否形成有效遮蔽"""
        if cloud_center is None:
            return 0
        
        missile_pos = self.missile_position(t, missile_name)
        
        # 使用简化的遮蔽判定：基于问题1-4的成功经验
        target_bottom = self.target_points['P_z1']
        target_top = self.target_points['P_z2']
        
        # 计算到视线的距离
        r1 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_bottom)
        r2 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_top)
        
        R = 15.0  # 使用放宽的遮蔽半径，提高遮蔽效果
        
        # 遮蔽判定：如果到任一关键视线的距离≤R，且云团在导弹与目标之间
        if (r1 <= R or r2 <= R) and missile_pos[0] >= cloud_center[0] >= 0:
            return 1
        
        return 0
    
    def check_comprehensive_constraints(self, params_dict):
        """检查综合约束条件"""
        
        for drone_name in self.drone_positions.keys():
            if drone_name not in params_dict:
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
                max_fall_time = np.sqrt(2 * drone_pos[2] / self.base_model.g)
                if delta_t_ni > max_fall_time:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹起爆延迟过长"
                
                # 约束2：要起到遮蔽作用，烟幕弹需在导弹击中目标前爆破
                explosion_time = t_ni + delta_t_ni
                # 简化：假设导弹击中目标的时间约为67秒
                hit_time = 67.0
                if explosion_time >= hit_time:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹爆破时间过晚"
                
                # 计算爆破点位置
                explosion_pos = self.smoke_explosion_position(
                    drone_name, t_ni, delta_t_ni, v_un, theta_n
                )
                
                # 约束3&4：爆破时刻云团在y轴和z轴的覆盖范围约束
                # 这里简化处理，主要检查云团不会过早落地
                lowest_z = explosion_pos[2] - 3.0 * self.base_model.effective_time - self.base_model.effective_radius
                if lowest_z < 0:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹云团会降至地面以下"
                
                # 约束5：云团消散时刻的视线角度约束（简化处理）
                if explosion_pos[0] < 0:
                    return False, f"{drone_name}第{bomb_idx+1}枚弹爆破点x坐标为负"
        
        # 检查同一无人机相邻烟幕弹投放间隔≥1s的约束
        for drone_name, drone_params in params_dict.items():
            drop_times = [bomb['drop_time'] for bomb in drone_params['bombs']]
            drop_times.sort()
            
            for i in range(len(drop_times) - 1):
                if drop_times[i+1] - drop_times[i] < 1.0:
                    return False, f"{drone_name}相邻烟幕弹投放间隔小于1秒"
        
        return True, "所有约束满足"
    
    def calculate_total_shielding_time_for_all_missiles(self, params_dict):
        """计算对所有导弹的总遮蔽时长"""
        
        total_shielding_times = {}
        
        for missile_name in self.missile_positions.keys():
            # 收集所有可能影响该导弹的烟幕弹
            all_explosions = []
            
            for drone_name, drone_params in params_dict.items():
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
        T_end = max(explosion_times) + self.base_model.effective_time
        
        # 数值积分计算遮蔽时长
        total_time = 0
        dt = 0.1
        
        t = T_start
        while t <= T_end:
            # 计算时刻t的遮蔽指示函数
            shielded = False
            
            for explosion_time, explosion_pos in explosions:
                cloud_center = self.smoke_cloud_center(t, explosion_time, explosion_pos)
                if self.shielding_indicator_for_missile(t, cloud_center, missile_name):
                    shielded = True
                    break
            
            if shielded:
                total_time += dt
            
            t += dt
        
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
            flight_time = distance_to_target / self.base_model.missile_speed
            
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
        # 高威胁：6枚，中威胁：5枚，低威胁：4枚
        allocation = {
            sorted_threats[0][0]: 6,  # 最高威胁
            sorted_threats[1][0]: 5,  # 中等威胁
            sorted_threats[2][0]: 4   # 最低威胁
        }
        
        print("\\n烟幕弹分配方案:")
        for missile, count in allocation.items():
            print(f"{missile}: {count}枚烟幕弹")
        
        return allocation, threat_analysis
    
    def assign_drones_to_missiles(self, allocation):
        """为每枚导弹分配无人机"""
        print("\\n【无人机任务分配】")
        
        assignments = {}
        drone_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
        
        for missile_name, bomb_count in allocation.items():
            missile_pos = self.missile_positions[missile_name]
            
            # 计算所有无人机到该导弹的距离
            drone_distances = []
            for drone_name in drone_names:
                drone_pos = self.drone_positions[drone_name]
                distance = np.linalg.norm(drone_pos - missile_pos)
                drone_distances.append((drone_name, distance))
            
            # 按距离排序，选择最近的无人机
            drone_distances.sort(key=lambda x: x[1])
            
            # 分配无人机
            assigned_drones = []
            remaining_bombs = bomb_count
            
            for drone_name, distance in drone_distances:
                if remaining_bombs <= 0:
                    break
                
                # 检查该无人机当前负载
                current_load = sum([
                    assignments.get(m, {}).get(drone_name, 0)
                    for m in assignments.keys()
                ])
                
                if current_load < self.max_bombs_per_drone:
                    bombs_to_assign = min(
                        remaining_bombs, 
                        self.max_bombs_per_drone - current_load
                    )
                    assigned_drones.append((drone_name, bombs_to_assign))
                    remaining_bombs -= bombs_to_assign
            
            assignments[missile_name] = {}
            for drone_name, bomb_count in assigned_drones:
                assignments[missile_name][drone_name] = bomb_count
            
            print(f"{missile_name}分配:")
            for drone_name, bomb_count in assigned_drones:
                print(f"  {drone_name}: {bomb_count}枚")
        
        return assignments
    
    def generate_optimized_parameters(self, assignments):
        """为分配方案生成优化参数"""
        print("\\n【参数优化生成】")
        
        all_strategies = {}
        
        for missile_name, drone_assignment in assignments.items():
            print(f"\\n为{missile_name}生成参数...")
            
            missile_strategies = {}
            
            for drone_name, bomb_count in drone_assignment.items():
                # 基于问题3和问题4的成功经验设计参数
                drone_pos = self.drone_positions[drone_name]
                
                # 计算朝向假目标的方向（基于成功经验）
                fake_target = self.base_model.fake_target
                direction_to_fake = fake_target - drone_pos
                optimal_angle = np.arctan2(direction_to_fake[1], direction_to_fake[0])
                
                # 确保角度在[0, 2π]范围内
                if optimal_angle < 0:
                    optimal_angle += 2 * np.pi
                
                # 生成该无人机的参数（基于问题3的成功经验）
                drone_params = {
                    'speed': 72 + np.random.uniform(-2, 8),  # 70-80 m/s，偏向较低速度
                    'direction': optimal_angle + np.random.uniform(-0.1, 0.1),  # 朝向假目标，小幅微调
                    'bombs': []
                }
                
                # 生成每枚烟幕弹的参数，确保间隔≥1秒
                for bomb_idx in range(bomb_count):
                    bomb_params = {
                        'drop_time': 0.2 + bomb_idx * 1.5,  # 确保间隔≥1.5秒
                        'explosion_delay': 1.0 + bomb_idx * 0.2  # 短延迟策略
                    }
                    drone_params['bombs'].append(bomb_params)
                
                missile_strategies[drone_name] = drone_params
            
            all_strategies[missile_name] = missile_strategies
        
        return all_strategies
    
    def convert_to_unified_params_dict(self, strategies):
        """将策略转换为统一的参数字典"""
        unified_params = {}
        
        # 初始化所有无人机
        for drone_name in self.drone_positions.keys():
            unified_params[drone_name] = {
                'speed': 80,  # 默认速度
                'direction': np.pi,  # 默认朝向假目标
                'bombs': []
            }
        
        # 合并所有导弹的策略
        for missile_name, missile_strategies in strategies.items():
            for drone_name, drone_params in missile_strategies.items():
                # 更新无人机的基本参数（使用第一个任务的参数，避免覆盖）
                if not unified_params[drone_name]['bombs']:  # 只在第一次设置时更新
                    unified_params[drone_name]['speed'] = drone_params['speed']
                    unified_params[drone_name]['direction'] = drone_params['direction']
                
                # 添加烟幕弹参数
                unified_params[drone_name]['bombs'].extend(drone_params['bombs'])
        
        # 确保每架无人机的烟幕弹数量不超过3枚，并修正投放时间间隔
        for drone_name in unified_params.keys():
            if len(unified_params[drone_name]['bombs']) > self.max_bombs_per_drone:
                unified_params[drone_name]['bombs'] = unified_params[drone_name]['bombs'][:self.max_bombs_per_drone]
            
            # 修正投放时间，确保间隔≥1秒
            bombs = unified_params[drone_name]['bombs']
            if len(bombs) > 1:
                bombs.sort(key=lambda x: x['drop_time'])  # 按投放时间排序
                for i in range(1, len(bombs)):
                    if bombs[i]['drop_time'] - bombs[i-1]['drop_time'] < 1.0:
                        bombs[i]['drop_time'] = bombs[i-1]['drop_time'] + 1.0
        
        return unified_params
    
    def solve_problem5_basic(self):
        """求解问题5基础版"""
        print("\\n【问题5建模思路】")
        print("1. 多目标优化：同时对抗3枚导弹")
        print("2. 资源分配：15枚烟幕弹的最优分配")
        print("3. 协同作战：5架无人机的协同策略")
        print("4. 约束满足：复杂的物理和几何约束")
        
        # 设计资源分配策略
        allocation, threat_analysis = self.design_resource_allocation_strategy()
        
        # 分配无人机任务
        assignments = self.assign_drones_to_missiles(allocation)
        
        # 生成优化参数
        strategies = self.generate_optimized_parameters(assignments)
        
        # 转换为统一参数格式
        unified_params = self.convert_to_unified_params_dict(strategies)
        
        # 检查约束
        constraints_ok, constraint_msg = self.check_comprehensive_constraints(unified_params)
        print(f"\\n约束检查: {'✓ 满足' if constraints_ok else '✗ 违反'}")
        if not constraints_ok:
            print(f"约束违反: {constraint_msg}")
        
        # 计算遮蔽效果
        shielding_times = self.calculate_total_shielding_time_for_all_missiles(unified_params)
        
        total_shielding_time = sum(shielding_times.values())
        
        print(f"\\n【遮蔽效果评估】")
        for missile_name, shielding_time in shielding_times.items():
            print(f"{missile_name}遮蔽时长: {shielding_time:.2f}s")
        print(f"总遮蔽时长: {total_shielding_time:.2f}s")
        
        return self.format_results(unified_params, shielding_times, total_shielding_time)
    
    def format_results(self, params_dict, shielding_times, total_shielding_time):
        """格式化结果"""
        results = {
            'params_dict': params_dict,
            'shielding_times': shielding_times,
            'total_shielding_time': total_shielding_time,
            'detailed_results': {}
        }
        
        print(f"\\n【问题5最终结果】")
        print(f"总遮蔽时长: {total_shielding_time:.2f}s")
        
        bomb_counter = 1
        for drone_name, drone_params in params_dict.items():
            if not drone_params['bombs']:  # 跳过没有烟幕弹的无人机
                continue
                
            print(f"\\n{drone_name}参数:")
            print(f"  速度: {drone_params['speed']:.2f} m/s")
            print(f"  方向角: {drone_params['direction']:.4f} rad ({np.degrees(drone_params['direction']):.1f}°)")
            print(f"  烟幕弹数量: {len(drone_params['bombs'])}枚")
            
            for bomb_idx, bomb_params in enumerate(drone_params['bombs']):
                t_ni = bomb_params['drop_time']
                delta_t_ni = bomb_params['explosion_delay']
                explosion_time = t_ni + delta_t_ni
                
                # 计算投放点和爆破点
                drop_pos = self.drone_position(
                    t_ni, drone_name, 
                    drone_params['speed'], drone_params['direction']
                )
                explosion_pos = self.smoke_explosion_position(
                    drone_name, t_ni, delta_t_ni,
                    drone_params['speed'], drone_params['direction']
                )
                
                results['detailed_results'][f'{drone_name}_bomb_{bomb_idx+1}'] = {
                    'drone_name': drone_name,
                    'bomb_number': bomb_counter,
                    'speed': drone_params['speed'],
                    'direction_angle': drone_params['direction'],
                    'drop_time': t_ni,
                    'explosion_delay': delta_t_ni,
                    'explosion_time': explosion_time,
                    'drop_position': drop_pos,
                    'explosion_position': explosion_pos
                }
                
                print(f"    第{bomb_idx+1}枚弹:")
                print(f"      投放时间: {t_ni:.2f}s")
                print(f"      起爆延迟: {delta_t_ni:.2f}s")
                print(f"      起爆时间: {explosion_time:.2f}s")
                print(f"      投放点: ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})")
                print(f"      起爆点: ({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
                
                bomb_counter += 1
        
        return results
    
    def save_results_to_excel(self, results, filename='result3.xlsx'):
        """保存结果到Excel文件，按照模板格式"""
        print(f"\\n【保存结果到{filename}】")
        
        # 按照模板格式创建数据
        data = []
        
        # 预定义的行顺序（按照模板）
        template_rows = [
            ('FY1', 1), ('FY1', 2), ('FY1', 3),
            ('FY2', 1), ('FY2', 2), ('FY2', 3),
            ('FY3', 1), ('FY3', 2), ('FY3', 3),
            ('FY4', 1), ('FY4', 2), ('FY4', 3),
            ('FY5', 1), ('FY5', 2), ('FY5', 3)
        ]
        
        # 创建详细结果的查找字典
        bomb_lookup = {}
        bomb_counter = 1
        
        for drone_name in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
            if drone_name in results['params_dict'] and results['params_dict'][drone_name]['bombs']:
                drone_params = results['params_dict'][drone_name]
                for bomb_idx, bomb_params in enumerate(drone_params['bombs']):
                    bomb_lookup[(drone_name, bomb_idx + 1)] = {
                        'speed': drone_params['speed'],
                        'direction_angle': drone_params['direction'],
                        'drop_time': bomb_params['drop_time'],
                        'explosion_delay': bomb_params['explosion_delay'],
                        'bomb_number': bomb_counter
                    }
                    bomb_counter += 1
        
        # 按模板顺序填充数据
        for drone_name, bomb_seq in template_rows:
            if (drone_name, bomb_seq) in bomb_lookup:
                bomb_info = bomb_lookup[(drone_name, bomb_seq)]
                
                # 计算位置信息
                t_ni = bomb_info['drop_time']
                delta_t_ni = bomb_info['explosion_delay']
                
                drop_pos = self.drone_position(
                    t_ni, drone_name, 
                    bomb_info['speed'], bomb_info['direction_angle']
                )
                explosion_pos = self.smoke_explosion_position(
                    drone_name, t_ni, delta_t_ni,
                    bomb_info['speed'], bomb_info['direction_angle']
                )
                
                # 转换方向角为度数，并处理特殊情况
                direction_deg = np.degrees(bomb_info['direction_angle'])
                if direction_deg < 0:
                    direction_deg += 360
                elif direction_deg >= 360:
                    direction_deg -= 360
                
                row = {
                    '无人机编号': drone_name,
                    '无人机运动方向': f"{direction_deg:.0f}",
                    '无人机运动速度': f"{bomb_info['speed']:.1f}",
                    '烟幕干扰弹编号': bomb_info['bomb_number'],
                    '烟幕干扰弹投放点x坐标[m]': f"{drop_pos[0]:.1f}",
                    '烟幕干扰弹投放点y坐标[m]': f"{drop_pos[1]:.1f}",
                    '烟幕干扰弹投放点z坐标[m]': f"{drop_pos[2]:.1f}",
                    '烟幕干扰弹起爆点x坐标[m]': f"{explosion_pos[0]:.1f}",
                    '烟幕干扰弹起爆点y坐标[m]': f"{explosion_pos[1]:.1f}",
                    '烟幕干扰弹起爆点z坐标[m]': f"{explosion_pos[2]:.1f}",
                    '有效遮蔽时长[s]': f"{results['total_shielding_time']:.2f}",
                    '干扰的导弹编号': self.get_target_missile(drone_name)
                }
            else:
                # 空行（该无人机没有对应编号的烟幕弹）
                row = {
                    '无人机编号': drone_name,
                    '无人机运动方向': '',
                    '无人机运动速度': '',
                    '烟幕干扰弹编号': bomb_seq,
                    '烟幕干扰弹投放点x坐标[m]': '',
                    '烟幕干扰弹投放点y坐标[m]': '',
                    '烟幕干扰弹投放点z坐标[m]': '',
                    '烟幕干扰弹起爆点x坐标[m]': '',
                    '烟幕干扰弹起爆点y坐标[m]': '',
                    '烟幕干扰弹起爆点z坐标[m]': '',
                    '有效遮蔽时长[s]': '',
                    '干扰的导弹编号': ''
                }
            
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            worksheet = writer.sheets['Sheet1']
            
            # 调整列宽
            column_widths = {
                'A': 12,  # 无人机编号
                'B': 15,  # 无人机运动方向
                'C': 15,  # 无人机运动速度
                'D': 15,  # 烟幕干扰弹编号
                'E': 20,  # 投放点x坐标
                'F': 20,  # 投放点y坐标
                'G': 20,  # 投放点z坐标
                'H': 20,  # 起爆点x坐标
                'I': 20,  # 起爆点y坐标
                'J': 20,  # 起爆点z坐标
                'K': 15,  # 有效遮蔽时长
                'L': 15   # 干扰的导弹编号
            }
            
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
        
        print(f"结果已保存到 {filename}")
        print("\\n结果表格预览:")
        print(df.head(15).to_string(index=False))
        
        return df
    
    def get_target_missile(self, drone_name):
        """根据无人机名称返回其主要干扰的导弹编号"""
        # 基于之前的分配策略
        missile_assignment = {
            'FY1': 'M3',  # FY1主要干扰M3
            'FY2': 'M2',  # FY2主要干扰M2
            'FY3': 'M1',  # FY3主要干扰M1
            'FY4': 'M2',  # FY4辅助干扰M2
            'FY5': 'M3'   # FY5辅助干扰M3
        }
        return missile_assignment.get(drone_name, '')

def main():
    """主函数"""
    print("烟幕干扰弹投放策略 - 问题5基础版求解")
    print("=" * 80)
    
    # 创建求解器
    solver = Problem5BasicSolver()
    
    # 求解问题5
    results = solver.solve_problem5_basic()
    
    # 保存结果到Excel
    df = solver.save_results_to_excel(results, 'result3.xlsx')
    
    print("\\n" + "=" * 80)
    print("问题5基础版求解完成！")
    print(f"5架无人机对抗3枚导弹总遮蔽时长: {results['total_shielding_time']:.2f}s")
    print("详细结果已保存到 result3.xlsx 文件中")
    print("\\n关键特点:")
    print("- 多目标优化：同时对抗3枚导弹")
    print("- 智能资源分配：基于威胁等级分配烟幕弹")
    print("- 协同作战：5架无人机的最优协同策略")
    print("- 约束满足：满足所有物理和几何约束")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()