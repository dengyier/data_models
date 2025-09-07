import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from smoke_interference_final import SmokeInterferenceModel

class Problem3UpdatedSolver:
    def __init__(self):
        self.base_model = SmokeInterferenceModel()
        self.num_smoke_bombs = 3
        self.min_drop_interval = 1.0
        
        print("问题3更新版：基于最新策略的实现")
        print("=" * 60)
    
    def missile_position(self, t):
        """导弹位置：P_M(t) = (1 - s_M * t / ||P_M0||) * P_M0"""
        # 根据策略文档中的精确公式
        missile_init = self.base_model.M1_init
        v_m = self.base_model.missile_speed
        
        # 计算 s_M = v_m (导弹速度)
        s_M = v_m
        
        # 计算 ||P_M0|| (导弹初始位置的模长)
        P_M0_norm = np.linalg.norm(missile_init)
        
        # 按照策略公式：P_M(t) = (1 - s_M * t / ||P_M0||) * P_M0
        scale_factor = 1 - s_M * t / P_M0_norm
        
        # 确保scale_factor不为负（导弹不会超过目标点）
        scale_factor = max(0, scale_factor)
        
        return scale_factor * missile_init
    
    def drone_position(self, t, v_u, theta):
        """无人机在时刻t的位置：P_FY(t; s_FY, θ) = P_F0 + [s_FY cos(θ), s_FY sin(θ), 0]^T · t"""
        # 根据策略文档的精确公式
        P_F0 = self.base_model.FY1_init  # 无人机初始位置
        
        # 位置向量
        X_ui = P_F0[0] + v_u * np.cos(theta) * t
        Y_ui = P_F0[1] + v_u * np.sin(theta) * t
        Z_ui = P_F0[2]  # 等高度飞行
        
        return np.array([X_ui, Y_ui, Z_ui])
    
    def smoke_explosion_position(self, t_i, delta_t_i, v_u, theta):
        """第i枚烟幕弹起爆点位置：P_det,i"""
        # 根据策略文档的公式
        x_F0, y_F0, z_F0 = self.base_model.FY1_init
        
        # 起爆点坐标
        x_det_i = x_F0 + v_u * np.cos(theta) * (t_i + delta_t_i)
        y_det_i = y_F0 + v_u * np.sin(theta) * (t_i + delta_t_i)
        z_det_i = z_F0 - 0.5 * self.base_model.g * delta_t_i**2
        
        return np.array([x_det_i, y_det_i, z_det_i])
    
    def smoke_cloud_center(self, t, t_i, delta_t_i, explosion_pos):
        """第i个云团在时刻t的中心位置：P_cloud,i(t) = P_det,i - [0, 0, v_sink(t - T_det,i)]^T"""
        explosion_time = t_i + delta_t_i  # T_det,i
        
        if t < explosion_time:
            return None  # 还未起爆
        
        time_since_explosion = t - explosion_time
        if time_since_explosion > self.base_model.effective_time:
            return None  # 已失效
        
        # 根据策略文档的精确公式
        v_sink = 3.0  # 下沉速度 3 m/s
        
        X_ci = explosion_pos[0]  # x坐标不变
        Y_ci = explosion_pos[1]  # y坐标不变
        Z_ci = explosion_pos[2] - v_sink * time_since_explosion  # 匀速下沉
        
        return np.array([X_ci, Y_ci, Z_ci])
    
    def shielding_indicator(self, t, t_i, delta_t_i, explosion_pos):
        """遮蔽指示函数S_i(t)：若在时刻t，云团i对导弹M1有效遮蔽则为1，其他为0"""
        explosion_time = t_i + delta_t_i  # T_det,i
        
        # 检查时间有效性
        if t < explosion_time or t > explosion_time + self.base_model.effective_time:
            return 0
        
        cloud_center = self.smoke_cloud_center(t, t_i, delta_t_i, explosion_pos)
        if cloud_center is None:
            return 0
        
        # 使用基础模型的遮蔽判定逻辑
        # 这里需要确保使用正确的导弹位置计算
        missile_pos = self.missile_position(t)
        
        # 检查云团是否位于导弹与真目标之间
        target_pos = self.base_model.target_bottom_view
        if not (missile_pos[0] >= cloud_center[0] >= target_pos[0]):
            return 0
        
        # 使用基础模型的精确遮蔽判定
        is_effective = self.base_model.is_effective_shielding(
            t, self.base_model.M1_init, explosion_time, explosion_pos
        )
        
        return 1 if is_effective else 0
    
    def total_shielding_indicator(self, t, drop_times, explosion_delays, explosion_positions):
        """总遮蔽指示函数：只要有一个云团有效遮蔽就为1"""
        for i in range(self.num_smoke_bombs):
            if self.shielding_indicator(t, drop_times[i], explosion_delays[i], explosion_positions[i]) == 1:
                return 1
        return 0
    
    def calculate_total_shielding_time(self, v_u, theta, drop_times, explosion_delays):
        """计算总遮蔽时长：T_total = ∫[0 to T] (1 - ∏(1 - S_i(t))) dt"""
        
        # 计算所有起爆点
        explosion_positions = [
            self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            for i in range(self.num_smoke_bombs)
        ]
        
        # 确定积分范围
        explosion_times = [drop_times[i] + explosion_delays[i] for i in range(self.num_smoke_bombs)]
        T_start = min(explosion_times)
        T_end = max(explosion_times) + self.base_model.effective_time
        
        # 数值积分
        total_time = 0
        dt = 0.05  # 时间步长
        
        t = T_start
        while t <= T_end:
            # 计算 1 - ∏(1 - S_i(t))
            product = 1.0
            for i in range(self.num_smoke_bombs):
                S_i = self.shielding_indicator(t, drop_times[i], explosion_delays[i], explosion_positions[i])
                product *= (1 - S_i)
            
            integrand = 1 - product
            total_time += integrand * dt
            t += dt
        
        return total_time
    
    def check_constraints(self, v_u, theta, drop_times, explosion_delays):
        """检查所有约束条件（根据策略文档）"""
        
        # 1. 基础性能约束
        if not (70 <= v_u <= 140):
            return False, "无人机速度超出范围 [70, 140]"
        
        if not (0 <= theta <= 360):  # 策略文档中是0-360度
            return False, "方向角超出范围 [0, 360]"
        
        for i in range(self.num_smoke_bombs):
            if drop_times[i] < 0:
                return False, f"第{i+1}枚弹投放时间T_L{i+1} < 0"
            if explosion_delays[i] <= 0:
                return False, f"第{i+1}枚弹起爆延迟t_det{i+1} <= 0"
        
        # 2. 投放时序约束
        if len(drop_times) >= 2 and drop_times[1] < drop_times[0] + 1:
            return False, "T_L2 < T_L1 + 1"
        if len(drop_times) >= 3 and drop_times[2] < drop_times[1] + 1:
            return False, "T_L3 < T_L2 + 1"
        
        # 3. 空间追及约束 (X-Y平面)
        for i in range(self.num_smoke_bombs):
            explosion_time = drop_times[i] + explosion_delays[i]  # T_det,i
            explosion_pos = self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            missile_pos = self.missile_position(explosion_time)
            
            # x_c,i(T_det,i) ≤ x_M(T_det,i)
            if explosion_pos[0] > missile_pos[0]:
                return False, f"第{i+1}枚弹违反空间追及约束：云团x={explosion_pos[0]:.1f} > 导弹x={missile_pos[0]:.1f}"
        
        # 4. 垂直窗口约束 (Z平面)
        for i in range(self.num_smoke_bombs):
            explosion_time = drop_times[i] + explosion_delays[i]
            explosion_pos = self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            missile_pos = self.missile_position(explosion_time)
            
            # 引爆高度上限：z_c,i(T_det,i) ≤ z_M(T_det,i)
            if explosion_pos[2] > missile_pos[2]:
                return False, f"第{i+1}枚弹违反引爆高度上限：云团z={explosion_pos[2]:.1f} > 导弹z={missile_pos[2]:.1f}"
            
            # 消散高度下限：z_c,i(T_det,i + 20) ≥ z_M(T_det,i + 20)
            end_time = explosion_time + 20
            cloud_end_z = explosion_pos[2] - 3.0 * 20  # 20秒后云团高度
            missile_end_pos = self.missile_position(end_time)
            
            if cloud_end_z < missile_end_pos[2]:
                return False, f"第{i+1}枚弹违反消散高度下限：云团终点z={cloud_end_z:.1f} < 导弹终点z={missile_end_pos[2]:.1f}"
        
        # 6. 地面安全约束
        R_cloud = 10.0  # 云团半径
        for i in range(self.num_smoke_bombs):
            explosion_time = drop_times[i] + explosion_delays[i]
            explosion_pos = self.smoke_explosion_position(drop_times[i], explosion_delays[i], v_u, theta)
            
            # z_c,i(T_det,i + 20) - R_cloud ≥ 0
            lowest_point = explosion_pos[2] - 3.0 * 20 - R_cloud
            if lowest_point < 0:
                return False, f"第{i+1}枚弹违反地面安全约束：最低点={lowest_point:.1f} < 0"
        
        return True, "所有约束满足"
    
    def objective_function(self, params):
        """目标函数：最大化总遮蔽时长"""
        v_u, theta, t1, t2, t3, dt1, dt2, dt3 = params
        drop_times = [t1, t2, t3]
        explosion_delays = [dt1, dt2, dt3]
        
        try:
            # 检查约束条件
            constraints_ok, constraint_msg = self.check_constraints(v_u, theta, drop_times, explosion_delays)
            
            if not constraints_ok:
                return 1000  # 惩罚项
            
            # 计算总遮蔽时长
            total_shielding_time = self.calculate_total_shielding_time(v_u, theta, drop_times, explosion_delays)
            
            return -total_shielding_time  # 最小化负值等于最大化正值
            
        except Exception as e:
            return 1000
    
    def verify_missile_model(self):
        """验证导弹运动模型"""
        print("\n【导弹运动模型验证】")
        
        # 测试几个时间点
        test_times = [0, 10, 20, 30, 40, 50, 60]
        
        print("时间(s) | 旧模型位置 | 新模型位置 | 距离原点")
        print("-" * 60)
        
        for t in test_times:
            # 旧模型（线性运动）
            old_pos = self.base_model.missile_position(t, self.base_model.M1_init)
            
            # 新模型（归一化运动）
            new_pos = self.missile_position(t)
            
            # 距离原点
            old_dist = np.linalg.norm(old_pos)
            new_dist = np.linalg.norm(new_pos)
            
            print(f"{t:6.1f} | ({old_pos[0]:6.0f},{old_pos[1]:3.0f},{old_pos[2]:6.0f}) | ({new_pos[0]:6.0f},{new_pos[1]:3.0f},{new_pos[2]:6.0f}) | {old_dist:6.0f} -> {new_dist:6.0f}")
        
        # 计算导弹到达原点的时间
        missile_init = self.base_model.M1_init
        v_m = self.base_model.missile_speed
        P_M0_norm = np.linalg.norm(missile_init)
        arrival_time = P_M0_norm / v_m
        
        print(f"\n导弹理论到达时间: {arrival_time:.2f}s")
        print(f"导弹初始距离: {P_M0_norm:.1f}m")
        print(f"导弹速度: {v_m:.1f}m/s")
    
    def solve_problem3_updated(self):
        """求解更新版问题3"""
        print("\n【更新策略要点】")
        print("1. 使用策略文档的精确运动学公式")
        print("2. 实现完整的约束条件检查")
        print("3. 改进的遮蔽指示函数")
        print("4. 基于积分的总遮蔽时长计算")
        
        # 验证导弹模型
        self.verify_missile_model()
        
        # 参数边界：[v_u, theta, t1, t2, t3, dt1, dt2, dt3]
        bounds = [
            (70, 140),          # 无人机速度 s_FY
            (0, 360),           # 方向角 θ (度)
            (0, 8),             # 投放时间1 T_L1
            (1.5, 9.5),         # 投放时间2 T_L2 (至少比T_L1大1s)
            (3, 11),            # 投放时间3 T_L3 (至少比T_L2大1s)
            (1.0, 6.0),         # 起爆延迟1 t_det1
            (1.0, 6.0),         # 起爆延迟2 t_det2
            (1.0, 6.0)          # 起爆延迟3 t_det3
        ]
        
        print(f"\n参数搜索范围:")
        print(f"- 无人机速度: {bounds[0][0]}-{bounds[0][1]} m/s")
        print(f"- 方向角: {bounds[1][0]:.1f}-{bounds[1][1]:.1f} rad")
        print(f"- 投放时间: 0-13 s (间隔≥1s)")
        print(f"- 起爆延迟: 1-8 s")
        
        # 多轮优化
        print("\n开始多轮优化...")
        best_result = None
        best_shielding_time = 0
        
        for attempt in range(5):
            print(f"第 {attempt + 1} 轮优化...")
            try:
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    seed=42 + attempt * 25,
                    maxiter=150,
                    popsize=20,
                    atol=1e-4,
                    tol=1e-4,
                    disp=False
                )
                
                if result.success and -result.fun > best_shielding_time:
                    best_result = result
                    best_shielding_time = -result.fun
                    print(f"  成功！遮蔽时间: {best_shielding_time:.3f}s")
                elif result.success:
                    print(f"  成功但效果一般: {-result.fun:.3f}s")
                else:
                    print(f"  失败")
            except Exception as e:
                print(f"  异常: {e}")
        
        # 如果优化失败，使用启发式参数
        if best_result is None or best_shielding_time <= 0:
            print("\n优化效果不理想，使用启发式参数...")
            
            heuristic_params_list = [
                [100.0, np.pi, 1.0, 2.5, 4.5, 3.0, 3.5, 4.0],  # 方案1
                [90.0, np.pi, 0.8, 2.2, 4.2, 2.8, 3.2, 3.8],   # 方案2
                [110.0, np.pi, 1.2, 2.8, 4.8, 3.2, 3.6, 4.2],  # 方案3
            ]
            
            for i, params in enumerate(heuristic_params_list):
                score = -self.objective_function(params)
                print(f"启发式方案{i+1}: {score:.3f}s")
                if score > best_shielding_time:
                    best_shielding_time = score
                    optimal_params = params
        else:
            optimal_params = best_result.x
        
        # 解析最优参数
        v_u_opt, theta_opt, t1_opt, t2_opt, t3_opt, dt1_opt, dt2_opt, dt3_opt = optimal_params
        drop_times_opt = [t1_opt, t2_opt, t3_opt]
        explosion_delays_opt = [dt1_opt, dt2_opt, dt3_opt]
        
        explosion_times_opt = [drop_times_opt[i] + explosion_delays_opt[i] for i in range(3)]
        explosion_positions_opt = [
            self.smoke_explosion_position(drop_times_opt[i], explosion_delays_opt[i], v_u_opt, theta_opt)
            for i in range(3)
        ]
        
        print(f"\n【问题3更新版结果】")
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
        
        print(f"\n总遮蔽时长: {best_shielding_time:.2f} s")
        
        # 验证约束条件
        constraints_ok, constraint_msg = self.check_constraints(
            v_u_opt, theta_opt, drop_times_opt, explosion_delays_opt
        )
        print(f"约束条件检查: {constraints_ok} - {constraint_msg}")
        
        return {
            'speed': v_u_opt,
            'direction_angle': theta_opt,
            'drop_times': drop_times_opt,
            'explosion_delays': explosion_delays_opt,
            'explosion_times': explosion_times_opt,
            'explosion_positions': explosion_positions_opt,
            'total_shielding_time': best_shielding_time
        }
    
    def analyze_shielding_process(self, result):
        """分析遮蔽过程"""
        print("\n【遮蔽过程详细分析】")
        
        v_u = result['speed']
        theta = result['direction_angle']
        drop_times = result['drop_times']
        explosion_delays = result['explosion_delays']
        explosion_positions = result['explosion_positions']
        
        # 分析时间窗口
        explosion_times = result['explosion_times']
        start_time = min(explosion_times)
        end_time = max(explosion_times) + self.base_model.effective_time
        
        print(f"分析时间窗口: {start_time:.1f}s - {end_time:.1f}s")
        
        # 详细分析关键时刻
        key_times = np.arange(start_time, min(start_time + 15, end_time), 2.0)
        
        for t in key_times:
            missile_pos = self.missile_position(t)
            total_indicator = self.total_shielding_indicator(t, drop_times, explosion_delays, explosion_positions)
            
            print(f"\nt={t:.1f}s: 总体遮蔽={total_indicator}")
            print(f"  导弹位置: ({missile_pos[0]:.0f}, {missile_pos[1]:.0f}, {missile_pos[2]:.0f})")
            
            for i in range(3):
                cloud_center = self.smoke_cloud_center(t, drop_times[i], explosion_delays[i], explosion_positions[i])
                if cloud_center is not None:
                    individual_indicator = self.shielding_indicator(t, drop_times[i], explosion_delays[i], explosion_positions[i])
                    print(f"  第{i+1}枚弹: S_{i+1}(t)={individual_indicator}, 云团位置=({cloud_center[0]:.0f},{cloud_center[1]:.0f},{cloud_center[2]:.0f})")
    
    def save_results_to_excel(self, result, filename='result1.xlsx'):
        """按照模板格式保存结果到Excel文件"""
        print(f"\n【保存结果到{filename}】")
        
        data = []
        
        # 为每枚烟幕弹创建一行数据
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
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            worksheet = writer.sheets['Sheet1']
            
            # 调整列宽
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
        
        print(f"结果已按模板格式保存到 {filename}")
        print("\n结果表格（按模板格式）:")
        print(df.to_string(index=False))
        
        return df

def main():
    """主函数"""
    print("烟幕干扰弹投放策略 - 问题3更新版求解")
    print("=" * 80)
    
    # 创建求解器
    solver = Problem3UpdatedSolver()
    
    # 求解问题3
    result = solver.solve_problem3_updated()
    
    # 分析遮蔽过程
    solver.analyze_shielding_process(result)
    
    # 保存结果到Excel
    df = solver.save_results_to_excel(result, 'result1.xlsx')
    
    print("\n" + "=" * 80)
    print("问题3更新版求解完成！")
    print(f"最优策略实现总遮蔽时长: {result['total_shielding_time']:.2f}s")
    print("详细结果已保存到 result1.xlsx 文件中")
    print("\n更新策略的改进:")
    print("- 使用了更精确的运动学公式")
    print("- 改进了遮蔽指示函数的计算")
    print("- 采用了基于积分的总遮蔽时长计算")
    print("- 强化了约束条件的检查")
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    result = main()