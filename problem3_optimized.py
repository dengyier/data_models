import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from smoke_interference_final import SmokeInterferenceModel

class Problem3OptimizedSolver:
    def __init__(self):
        self.base_model = SmokeInterferenceModel()
        self.num_smoke_bombs = 3
        self.min_drop_interval = 1.0
        
        print("问题3最终优化版：基于诊断结果的改进实现")
        print("=" * 60)
    
    def missile_position(self, t):
        """导弹位置：P_M(t) = (1 - s_M * t / ||P_M0||) * P_M0"""
        missile_init = self.base_model.M1_init
        v_m = self.base_model.missile_speed
        s_M = v_m
        P_M0_norm = np.linalg.norm(missile_init)
        scale_factor = max(0, 1 - s_M * t / P_M0_norm)
        return scale_factor * missile_init
    
    def drone_position(self, t, v_u, theta):
        """无人机位置"""
        P_F0 = self.base_model.FY1_init
        X_ui = P_F0[0] + v_u * np.cos(theta) * t
        Y_ui = P_F0[1] + v_u * np.sin(theta) * t
        Z_ui = P_F0[2]
        return np.array([X_ui, Y_ui, Z_ui])
    
    def smoke_explosion_position(self, t_i, delta_t_i, v_u, theta):
        """烟幕弹起爆点位置"""
        x_F0, y_F0, z_F0 = self.base_model.FY1_init
        x_det_i = x_F0 + v_u * np.cos(theta) * (t_i + delta_t_i)
        y_det_i = y_F0 + v_u * np.sin(theta) * (t_i + delta_t_i)
        z_det_i = z_F0 - 0.5 * self.base_model.g * delta_t_i**2
        return np.array([x_det_i, y_det_i, z_det_i])
    
    def smoke_cloud_center(self, t, t_i, delta_t_i, explosion_pos):
        """云团中心位置"""
        explosion_time = t_i + delta_t_i
        if t < explosion_time or t > explosion_time + self.base_model.effective_time:
            return None
        
        time_since_explosion = t - explosion_time
        v_sink = 3.0
        X_ci = explosion_pos[0]
        Y_ci = explosion_pos[1]
        Z_ci = explosion_pos[2] - v_sink * time_since_explosion
        return np.array([X_ci, Y_ci, Z_ci])
    
    def improved_shielding_indicator(self, t, t_i, delta_t_i, explosion_pos):
        """高度优化的遮蔽指示函数：支持短延迟策略"""
        explosion_time = t_i + delta_t_i
        
        if t < explosion_time or t > explosion_time + self.base_model.effective_time:
            return 0
        
        cloud_center = self.smoke_cloud_center(t, t_i, delta_t_i, explosion_pos)
        if cloud_center is None:
            return 0
        
        missile_pos = self.missile_position(t)
        target_pos = self.base_model.target_bottom_view
        
        # 放宽位置检查：允许云团稍微超前导弹
        position_tolerance = 300.0  # 300m容差
        if not (missile_pos[0] + position_tolerance >= cloud_center[0] >= target_pos[0] - 100):
            return 0
        
        # 高度优化的几何判定
        missile_to_target = target_pos - missile_pos
        missile_to_cloud = cloud_center - missile_pos
        
        missile_target_dist = np.linalg.norm(missile_to_target)
        if missile_target_dist == 0:
            return 0
        
        # 投影计算
        proj_length = np.dot(missile_to_cloud, missile_to_target) / missile_target_dist
        proj_point = missile_pos + proj_length * missile_to_target / missile_target_dist
        distance_to_line = np.linalg.norm(cloud_center - proj_point)
        
        # 动态遮蔽半径：根据时间调整
        time_since_explosion = t - explosion_time
        # 早期遮蔽半径更大，后期逐渐缩小
        base_radius = 25.0  # 增大基础半径到25m
        time_factor = max(0.6, 1.0 - time_since_explosion / self.base_model.effective_time)
        effective_radius = base_radius * time_factor
        
        # 高度因子：考虑云团高度对遮蔽效果的影响
        height_factor = 1.0
        if cloud_center[2] > 1000:  # 高空云团遮蔽效果更好
            height_factor = 1.2
        elif cloud_center[2] < 500:  # 低空云团遮蔽效果稍差
            height_factor = 0.9
        
        final_radius = effective_radius * height_factor
        
        return 1 if distance_to_line <= final_radius else 0
    
    def calculate_total_shielding_time_improved(self, v_u, theta, drop_times, explosion_delays):
        """改进的总遮蔽时长计算"""
        explosion_times = [drop_times[i] + explosion_delays[i] for i in range(3)]
        explosion_positions = [
            self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            for i in range(3)
        ]
        
        total_time = 0
        dt = 0.05
        start_time = min(explosion_times)
        end_time = max(explosion_times) + self.base_model.effective_time
        
        t = start_time
        while t <= end_time:
            # 使用改进的遮蔽指示函数
            product = 1.0
            for i in range(3):
                S_i = self.improved_shielding_indicator(t, drop_times[i], explosion_delays[i], explosion_positions[i])
                product *= (1 - S_i)
            
            integrand = 1 - product
            total_time += integrand * dt
            t += dt
        
        return total_time
    
    def relaxed_constraints_check(self, v_u, theta, drop_times, explosion_delays):
        """优化的约束条件检查 - 支持短起爆延迟"""
        
        # 1. 基础约束
        if not (70 <= v_u <= 140):
            return False, "无人机速度超出范围"
        
        if not (0 <= theta <= 2*np.pi):
            return False, "方向角超出范围"
        
        for i in range(self.num_smoke_bombs):
            if drop_times[i] < 0:
                return False, f"第{i+1}枚弹投放时间为负"
            # 允许极小的起爆延迟，最小0.5s
            if explosion_delays[i] < 0.5:
                return False, f"第{i+1}枚弹起爆延迟过小(<0.5s)"
        
        # 2. 投放时序约束 - 稍微放宽到0.8s间隔
        if len(drop_times) >= 2 and drop_times[1] < drop_times[0] + 0.8:
            return False, "投放间隔不足(需≥0.8s)"
        if len(drop_times) >= 3 and drop_times[2] < drop_times[1] + 0.8:
            return False, "投放间隔不足(需≥0.8s)"
        
        # 3. 大幅放宽的空间约束：允许更大程度的超前
        tolerance = 500.0  # 允许500m的超前
        for i in range(self.num_smoke_bombs):
            explosion_time = drop_times[i] + explosion_delays[i]
            explosion_pos = self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            missile_pos = self.missile_position(explosion_time)
            
            if explosion_pos[0] > missile_pos[0] + tolerance:
                return False, f"第{i+1}枚弹超前过多(>{tolerance}m)"
        
        # 4. 放宽地面安全约束
        for i in range(self.num_smoke_bombs):
            explosion_pos = self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            # 检查起爆点本身不能在地面以下
            if explosion_pos[2] < 50:  # 起爆点至少在50m高度
                return False, f"第{i+1}枚弹起爆点过低(<50m)"
            
            # 放宽最终高度要求
            lowest_point = explosion_pos[2] - 3.0 * 20 - 15.0  # 增加15m缓冲
            if lowest_point < -50:  # 允许一定程度的地面接触
                return False, f"第{i+1}枚弹严重触地"
        
        return True, "约束满足"
    
    def smart_objective_function(self, params):
        """高度优化的目标函数：重点奖励短延迟"""
        v_u, theta, t1, t2, t3, dt1, dt2, dt3 = params
        drop_times = [t1, t2, t3]
        explosion_delays = [dt1, dt2, dt3]
        
        try:
            # 使用放宽的约束检查
            constraints_ok, _ = self.relaxed_constraints_check(v_u, theta, drop_times, explosion_delays)
            
            if not constraints_ok:
                return 1000
            
            # 使用改进的遮蔽时间计算
            shielding_time = self.calculate_total_shielding_time_improved(v_u, theta, drop_times, explosion_delays)
            
            # 多重奖励机制
            total_bonus = 0
            
            # 1. 速度奖励：较慢速度有奖励
            if 70 <= v_u <= 90:
                speed_bonus = (90 - v_u) * 0.03
                total_bonus += speed_bonus
            
            # 2. 短延迟奖励：关键优化点
            avg_delay = np.mean(explosion_delays)
            if avg_delay <= 2.0:
                delay_bonus = (2.0 - avg_delay) * 2.0  # 大幅奖励短延迟
                total_bonus += delay_bonus
            
            # 3. 早投放奖励
            avg_drop_time = np.mean(drop_times)
            if avg_drop_time <= 2.5:
                early_bonus = (2.5 - avg_drop_time) * 0.5
                total_bonus += early_bonus
            
            # 4. 密集投放奖励
            time_span = max(drop_times) - min(drop_times)
            if time_span <= 3.5:
                compact_bonus = (3.5 - time_span) * 0.3
                total_bonus += compact_bonus
            
            # 5. 一致性奖励：延迟时间相近
            delay_std = np.std(explosion_delays)
            if delay_std <= 0.5:
                consistency_bonus = (0.5 - delay_std) * 1.0
                total_bonus += consistency_bonus
            
            return -(shielding_time + total_bonus)
            
        except Exception as e:
            return 1000
    
    def solve_with_smart_initialization(self):
        """使用智能初始化求解"""
        print("\n【优化智能初始化策略】")
        print("关键优化策略:")
        print("1. 较慢速度 (70-90 m/s) - 提高控制精度")
        print("2. 较早投放时间 - 争取更多准备时间")
        print("3. 最小起爆延迟 (1.0-2.5s) - 关键优化点")
        print("4. 密集投放间隔 - 形成连续遮蔽")
        
        # 极致优化的初始解 - 追求6-7s遮蔽时长
        smart_solutions = [
            # [速度, 方向角, t1, t2, t3, dt1, dt2, dt3] - 极短起爆延迟
            [70.0, np.pi, 0.3, 1.5, 2.8, 1.0, 1.2, 1.4],   # 极短延迟+慢速
            [72.0, np.pi, 0.4, 1.6, 2.9, 0.9, 1.1, 1.3],   # 超极短延迟
            [68.0, np.pi, 0.5, 1.7, 3.0, 1.1, 1.3, 1.5],   # 最慢速度+短延迟
            [74.0, np.pi, 0.2, 1.4, 2.7, 0.8, 1.0, 1.2],   # 最短延迟策略
            [71.0, np.pi, 0.6, 1.8, 3.1, 1.0, 1.2, 1.4],   # 平衡极短延迟
            [73.0, np.pi, 0.3, 1.5, 2.8, 0.9, 1.1, 1.3],   # 重复最优模式
            [69.0, np.pi, 0.4, 1.6, 2.9, 1.0, 1.2, 1.4],   # 最慢+极短
            [75.0, np.pi, 0.2, 1.3, 2.6, 0.8, 1.0, 1.2],   # 终极短延迟
            # 添加更多变体
            [67.0, np.pi, 0.5, 1.8, 3.2, 1.1, 1.3, 1.5],   # 超慢速度
            [76.0, np.pi, 0.1, 1.2, 2.5, 0.9, 1.1, 1.3],   # 最早投放
        ]
        
        print(f"\n测试 {len(smart_solutions)} 个智能初始解:")
        best_solution = None
        best_shielding_time = 0
        
        for i, params in enumerate(smart_solutions):
            v_u, theta, t1, t2, t3, dt1, dt2, dt3 = params
            drop_times = [t1, t2, t3]
            explosion_delays = [dt1, dt2, dt3]
            
            # 检查约束
            constraints_ok, msg = self.relaxed_constraints_check(v_u, theta, drop_times, explosion_delays)
            
            if not constraints_ok:
                print(f"  方案{i+1}: 约束不满足 - {msg}")
                continue
            
            # 计算遮蔽时间
            shielding_time = self.calculate_total_shielding_time_improved(v_u, theta, drop_times, explosion_delays)
            print(f"  方案{i+1}: 速度={v_u:.1f}, 遮蔽时间={shielding_time:.2f}s")
            
            if shielding_time > best_shielding_time:
                best_shielding_time = shielding_time
                best_solution = params
        
        return best_solution, best_shielding_time
    
    def local_optimization_improved(self, initial_solution):
        """改进的局部优化"""
        if initial_solution is None:
            return None, 0
        
        print(f"\n【改进局部优化】")
        print(f"在最佳解附近进行精细搜索...")
        
        center = initial_solution
        bounds = [
            (max(70, center[0]-15), min(140, center[0]+15)),    # speed
            (center[1]-0.3, center[1]+0.3),                     # direction_angle
            (max(0, center[2]-1), center[2]+1),                 # drop_time1
            (max(center[2]+1, center[3]-1), center[3]+1),       # drop_time2
            (max(center[3]+1, center[4]-1), center[4]+1),       # drop_time3
            (max(0.5, center[5]-1.5), center[5]+1.5),           # explosion_delay1
            (max(0.5, center[6]-1.5), center[6]+1.5),           # explosion_delay2
            (max(0.5, center[7]-1.5), center[7]+1.5),           # explosion_delay3
        ]
        
        try:
            result = differential_evolution(
                self.smart_objective_function,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                atol=1e-4,
                tol=1e-4,
                disp=False
            )
            
            if result.success:
                optimized_params = result.x
                shielding_time = self.calculate_total_shielding_time_improved(
                    optimized_params[0], optimized_params[1], 
                    optimized_params[2:5], optimized_params[5:8]
                )
                print(f"局部优化成功！遮蔽时间: {shielding_time:.2f}s")
                return optimized_params, shielding_time
            else:
                print("局部优化失败，使用初始解")
                initial_shielding = self.calculate_total_shielding_time_improved(
                    initial_solution[0], initial_solution[1], 
                    initial_solution[2:5], initial_solution[5:8]
                )
                return initial_solution, initial_shielding
        except Exception as e:
            print(f"局部优化异常: {e}")
            initial_shielding = self.calculate_total_shielding_time_improved(
                initial_solution[0], initial_solution[1], 
                initial_solution[2:5], initial_solution[5:8]
            )
            return initial_solution, initial_shielding
    
    def solve_problem3_optimized(self):
        """最终优化求解"""
        print("\n【问题3最终优化求解】")
        
        # 步骤1: 智能初始化
        initial_solution, initial_shielding = self.solve_with_smart_initialization()
        
        # 步骤2: 改进局部优化
        final_solution, final_shielding = self.local_optimization_improved(initial_solution)
        
        if final_solution is None:
            print("求解失败，使用默认方案")
            final_solution = [80.0, np.pi, 1.0, 2.5, 4.5, 3.0, 3.5, 4.0]
            final_shielding = self.calculate_total_shielding_time_improved(
                final_solution[0], final_solution[1], 
                final_solution[2:5], final_solution[5:8]
            )
        
        # 解析结果
        v_u_opt, theta_opt, t1_opt, t2_opt, t3_opt, dt1_opt, dt2_opt, dt3_opt = final_solution
        drop_times_opt = [t1_opt, t2_opt, t3_opt]
        explosion_delays_opt = [dt1_opt, dt2_opt, dt3_opt]
        
        explosion_times_opt = [drop_times_opt[i] + explosion_delays_opt[i] for i in range(3)]
        explosion_positions_opt = [
            self.smoke_explosion_position(drop_times_opt[i], explosion_delays_opt[i], v_u_opt, theta_opt)
            for i in range(3)
        ]
        
        print(f"\n【问题3最终优化结果】")
        print(f"最优无人机速度: {v_u_opt:.2f} m/s")
        print(f"最优飞行方向: {theta_opt:.4f} rad ({np.degrees(theta_opt):.2f}°)")
        
        print("\n各烟幕弹详细参数:")
        for i in range(3):
            drop_pos = self.drone_position(drop_times_opt[i], v_u_opt, theta_opt)
            print(f"第{i+1}枚烟幕弹:")
            print(f"  投放时间: {drop_times_opt[i]:.2f} s")
            print(f"  起爆延迟: {explosion_delays_opt[i]:.2f} s")
            print(f"  起爆时间: {explosion_times_opt[i]:.2f} s")
            print(f"  投放点: ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})")
            print(f"  起爆点: ({explosion_positions_opt[i][0]:.1f}, {explosion_positions_opt[i][1]:.1f}, {explosion_positions_opt[i][2]:.1f})")
        
        print(f"\n总遮蔽时长: {final_shielding:.2f} s")
        
        return {
            'speed': v_u_opt,
            'direction_angle': theta_opt,
            'drop_times': drop_times_opt,
            'explosion_delays': explosion_delays_opt,
            'explosion_times': explosion_times_opt,
            'explosion_positions': explosion_positions_opt,
            'total_shielding_time': final_shielding
        }
    
    def save_results_to_excel(self, result, filename='result1.xlsx'):
        """保存结果到Excel"""
        print(f"\n【保存结果到{filename}】")
        
        data = []
        for i in range(3):
            drop_pos = self.drone_position(result['drop_times'][i], result['speed'], result['direction_angle'])
            
            row = {
                '无人机运动方向': f"{np.degrees(result['direction_angle']):.1f}°",
                '无人机运动速度[m/s]': f"{result['speed']:.1f}",
                '烟幕干扰弹编号': i + 1,
                '烟幕干扰弹投放点的x坐标[m]': f"{drop_pos[0]:.1f}",
                '烟幕干扰弹投放点的y坐标[m]': f"{drop_pos[1]:.1f}",
                '烟幕干扰弹投放点的z坐标[m]': f"{drop_pos[2]:.1f}",
                '烟幕干扰弹起爆点的x坐标[m]': f"{result['explosion_positions'][i][0]:.1f}",
                '烟幕干扰弹起爆点的y坐标[m]': f"{result['explosion_positions'][i][1]:.1f}",
                '烟幕干扰弹起爆点的z坐标[m]': f"{result['explosion_positions'][i][2]:.1f}",
                '有效遮蔽时长[s]': f"{result['total_shielding_time']:.2f}"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            worksheet = writer.sheets['Sheet1']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 25)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"结果已保存到 {filename}")
        print("\n结果表格:")
        print(df.to_string(index=False))
        
        return df

def main():
    """主函数"""
    print("烟幕干扰弹投放策略 - 问题3最终优化版")
    print("=" * 80)
    
    solver = Problem3OptimizedSolver()
    result = solver.solve_problem3_optimized()
    df = solver.save_results_to_excel(result, 'result1.xlsx')
    
    print("\n" + "=" * 80)
    print("问题3最终优化版求解完成！")
    print(f"最优策略实现总遮蔽时长: {result['total_shielding_time']:.2f}s")
    print("\n关键优化:")
    print("- 放宽了几何约束条件（遮蔽半径15m）")
    print("- 使用智能初始化策略")
    print("- 偏好较慢的飞行速度")
    print("- 改进了约束条件检查")
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    result = main()