import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from smoke_interference_final import SmokeInterferenceModel

class Problem4BasicSolver:
    def __init__(self):
        self.base_model = SmokeInterferenceModel()
        self.num_drones = 3  # FY1, FY2, FY3
        self.num_smoke_bombs = 3  # 每架无人机投放1枚
        
        # 三架无人机的初始位置
        self.drone_positions = {
            'FY1': self.base_model.FY1_init,  # [17800, 0, 1800]
            'FY2': self.base_model.FY2_init,  # [12000, 1400, 1400]
            'FY3': self.base_model.FY3_init   # [6000, -3000, 700]
        }
        
        print("问题4：FY1、FY2、FY3三架无人机各投放1枚烟幕弹干扰M1")
        print("=" * 60)
        print("建模思路：先对每架无人机进行独立优化，再协同配合")
    
    def missile_position(self, t):
        """导弹位置：使用统一的导弹运动模型"""
        # 使用问题3中验证过的导弹运动模型
        missile_init = self.base_model.M1_init
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
    
    def smoke_explosion_position(self, drone_name, t_in, delta_t_n, v_un, theta_n):
        """第n枚烟幕弹爆破点位置"""
        drone_init = self.drone_positions[drone_name]
        
        # 根据策略文档的公式
        X_sn = drone_init[0] + v_un * np.cos(theta_n) * (t_in + delta_t_n)
        Y_sn = drone_init[1] + v_un * np.sin(theta_n) * (t_in + delta_t_n)
        Z_sn = drone_init[2] - 0.5 * self.base_model.g * delta_t_n**2
        
        return np.array([X_sn, Y_sn, Z_sn])
    
    def smoke_cloud_center(self, t, drone_name, t_in, delta_t_n, explosion_pos):
        """第n个云团在时刻t的中心位置"""
        explosion_time = t_in + delta_t_n
        
        if t < explosion_time or t > explosion_time + self.base_model.effective_time:
            return None
        
        time_since_explosion = t - explosion_time
        
        # 根据策略文档公式：Z_on = Z_sn - 3(t - t_in - Δt_n)
        X_on = explosion_pos[0]  # x坐标不变
        Y_on = explosion_pos[1]  # y坐标不变
        Z_on = explosion_pos[2] - 3.0 * time_since_explosion  # 以3m/s下沉
        
        return np.array([X_on, Y_on, Z_on])
    
    def shielding_indicator(self, t, drone_name, t_in, delta_t_n, explosion_pos):
        """遮蔽指示函数：判断第n个云团是否对导弹M1有效遮蔽"""
        cloud_center = self.smoke_cloud_center(t, drone_name, t_in, delta_t_n, explosion_pos)
        
        if cloud_center is None:
            return 0
        
        missile_pos = self.missile_position(t)
        
        # 根据策略文档的判定条件
        # (1) 计算云团中心到两视线段距离
        target_bottom = self.base_model.target_bottom_view  # [0, 193, 0]
        target_top = self.base_model.target_top_view        # [0, 193, 10]
        
        # 计算距离 r1 和 r2
        r1 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_bottom)
        r2 = self.point_to_line_segment_distance(cloud_center, missile_pos, target_top)
        
        R = self.base_model.effective_radius  # 10m
        
        # 如果 r1 ≤ R 或 r2 ≤ R，则形成有效遮蔽
        if r1 <= R or r2 <= R:
            # (2) 爆破时刻云团位于导弹与目标之间
            if missile_pos[0] >= cloud_center[0]:
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
    
    def total_shielding_indicator(self, t, drone_params):
        """总遮蔽指示函数：I(t) = 1 - ∏(1 - S_n(t))"""
        product = 1.0
        
        drone_names = ['FY1', 'FY2', 'FY3']
        
        for i, drone_name in enumerate(drone_names):
            v_un, theta_n, t_in, delta_t_n = drone_params[i*4:(i+1)*4]
            explosion_pos = self.smoke_explosion_position(drone_name, t_in, delta_t_n, v_un, theta_n)
            S_n = self.shielding_indicator(t, drone_name, t_in, delta_t_n, explosion_pos)
            product *= (1 - S_n)
        
        return 1 - product
    
    def calculate_total_shielding_time(self, drone_params):
        """计算总遮蔽时长：T_total = ∫ I(t) dt"""
        
        # 解析参数
        drone_names = ['FY1', 'FY2', 'FY3']
        explosion_times = []
        
        for i, drone_name in enumerate(drone_names):
            v_un, theta_n, t_in, delta_t_n = drone_params[i*4:(i+1)*4]
            explosion_times.append(t_in + delta_t_n)
        
        # 确定积分范围
        T_start = min(explosion_times)
        T_end = max(explosion_times) + self.base_model.effective_time
        
        # 数值积分
        total_time = 0
        dt = 0.05
        
        t = T_start
        while t <= T_end:
            I_t = self.total_shielding_indicator(t, drone_params)
            total_time += I_t * dt
            t += dt
        
        return total_time
    
    def check_constraints(self, drone_params):
        """检查约束条件"""
        drone_names = ['FY1', 'FY2', 'FY3']
        
        for i, drone_name in enumerate(drone_names):
            v_un, theta_n, t_in, delta_t_n = drone_params[i*4:(i+1)*4]
            
            # (1) 基础条件
            if not (70 <= v_un <= 140):
                return False, f"{drone_name}速度超出范围"
            
            if not (0 <= theta_n <= 2*np.pi):
                return False, f"{drone_name}方向角超出范围"
            
            if t_in < 0:
                return False, f"{drone_name}投放时间为负"
            
            if delta_t_n <= 0:
                return False, f"{drone_name}起爆延迟非正"
            
            # 计算爆破点位置
            explosion_pos = self.smoke_explosion_position(drone_name, t_in, delta_t_n, v_un, theta_n)
            explosion_time = t_in + delta_t_n
            
            # (2) 下降范围：Z_on - R ≥ 0
            R = self.base_model.effective_radius
            lowest_z = explosion_pos[2] - 3.0 * self.base_model.effective_time - R
            if lowest_z < 0:
                return False, f"{drone_name}云团会降至地面以下"
            
            # (3) x轴方向：X_on ≥ 0（爆破点在正半轴）
            if explosion_pos[0] < 0:
                return False, f"{drone_name}违反x轴约束"
            
            # 放宽约束条件，只检查关键的物理约束
            # 导弹位置约束在遮蔽判定中处理，这里不强制要求
        
        return True, "所有约束满足"
    
    def objective_function(self, params):
        """目标函数：最大化总遮蔽时长"""
        try:
            # 检查约束条件
            constraints_ok, constraint_msg = self.check_constraints(params)
            
            if not constraints_ok:
                return 1000  # 惩罚项
            
            # 计算总遮蔽时长
            total_shielding_time = self.calculate_total_shielding_time(params)
            
            return -total_shielding_time  # 最小化负值等于最大化正值
            
        except Exception as e:
            return 1000
    
    def solve_problem4_basic(self):
        """求解问题4基础版"""
        print("\n【问题4建模思路】")
        print("1. 决策变量：3架无人机×4个参数 = 12个决策变量")
        print("2. 目标函数：max T_total = ∫ I(t) dt")
        print("3. 约束条件：基础约束+空间约束+物理约束")
        print("4. 求解策略：基于问题3经验的启发式初始化")
        
        # 参数边界：[v_u1, θ_1, t_i1, Δt_1, v_u2, θ_2, t_i2, Δt_2, v_u3, θ_3, t_i3, Δt_3]
        bounds = [
            # FY1参数
            (70, 140), (0, 2*np.pi), (0, 8), (0.8, 4.0),
            # FY2参数  
            (70, 140), (0, 2*np.pi), (0, 8), (0.8, 4.0),
            # FY3参数
            (70, 140), (0, 2*np.pi), (0, 8), (0.8, 4.0)
        ]
        
        print(f"\n参数搜索范围:")
        print(f"- 每架无人机速度: 70-140 m/s")
        print(f"- 每架无人机方向角: 0-2π rad")
        print(f"- 投放时间: 0-8 s")
        print(f"- 起爆延迟: 0.8-4.0 s")
        
        # 基于问题3经验的启发式初始化
        print("\n【启发式初始化】")
        print("基于问题3的成功经验：")
        print("- 较慢速度效果更好")
        print("- 短起爆延迟是关键")
        print("- 早期投放争取时间")
        
        # 分析各无人机的最优方向
        print("\n分析各无人机位置:")
        for drone_name, pos in self.drone_positions.items():
            print(f"{drone_name}: ({pos[0]}, {pos[1]}, {pos[2]})")
        
        # 计算各无人机到导弹拦截点的最优方向
        missile_init = self.base_model.M1_init
        fake_target = self.base_model.fake_target
        
        optimal_angles = {}
        for drone_name, drone_pos in self.drone_positions.items():
            # 计算朝向假目标的方向
            direction_to_fake = fake_target - drone_pos
            angle_to_fake = np.arctan2(direction_to_fake[1], direction_to_fake[0])
            # 确保角度在[0, 2π]范围内
            if angle_to_fake < 0:
                angle_to_fake += 2 * np.pi
            optimal_angles[drone_name] = angle_to_fake
            print(f"{drone_name}朝向假目标角度: {np.degrees(angle_to_fake):.1f}°")
        
        smart_solutions = [
            # [v_u1, θ_1, t_i1, Δt_1, v_u2, θ_2, t_i2, Δt_2, v_u3, θ_3, t_i3, Δt_3]
            # 方案1：所有无人机都朝向假目标，短延迟
            [76, optimal_angles['FY1'], 0.2, 1.0, 
             80, optimal_angles['FY2'], 0.5, 1.2, 
             85, optimal_angles['FY3'], 0.8, 1.4],
            
            # 方案2：FY1主力，其他辅助
            [72, optimal_angles['FY1'], 0.1, 0.9, 
             85, optimal_angles['FY2'] + 0.3, 1.0, 1.5, 
             90, optimal_angles['FY3'] - 0.3, 1.5, 1.8],
            
            # 方案3：错时投放，形成连续遮蔽
            [75, optimal_angles['FY1'], 0.3, 1.1, 
             78, optimal_angles['FY2'], 1.2, 1.3, 
             82, optimal_angles['FY3'], 2.0, 1.5],
            
            # 方案4：基于距离调整策略
            [74, optimal_angles['FY1'] + 0.1, 0.2, 1.0, 
             80, optimal_angles['FY2'] - 0.2, 0.8, 1.2, 
             88, optimal_angles['FY3'] + 0.2, 1.3, 1.6],
            
            # 方案5：极短延迟策略
            [70, optimal_angles['FY1'], 0.1, 0.8, 
             75, optimal_angles['FY2'], 0.6, 0.9, 
             80, optimal_angles['FY3'], 1.1, 1.0],
        ]
        
        print(f"\n测试 {len(smart_solutions)} 个启发式方案:")
        best_solution = None
        best_shielding_time = 0
        
        for i, params in enumerate(smart_solutions):
            # 检查约束
            constraints_ok, msg = self.check_constraints(params)
            
            if not constraints_ok:
                print(f"  方案{i+1}: 约束不满足 - {msg}")
                continue
            
            # 计算遮蔽时间
            shielding_time = self.calculate_total_shielding_time(params)
            print(f"  方案{i+1}: 遮蔽时间={shielding_time:.2f}s")
            
            if shielding_time > best_shielding_time:
                best_shielding_time = shielding_time
                best_solution = params
        
        if best_solution is None:
            print("所有启发式方案都不可行，使用默认参数")
            best_solution = [80, np.pi, 1.0, 2.0, 85, np.pi, 1.5, 2.5, 90, np.pi, 2.0, 3.0]
            best_shielding_time = self.calculate_total_shielding_time(best_solution)
        
        # 局部优化
        print(f"\n【局部优化】")
        print(f"在最佳启发式解附近进行精细搜索...")
        
        try:
            # 在最佳解附近定义搜索范围
            center = best_solution
            refined_bounds = []
            
            for j in range(12):
                if j % 4 == 0:  # 速度
                    refined_bounds.append((max(70, center[j]-10), min(140, center[j]+10)))
                elif j % 4 == 1:  # 方向角
                    refined_bounds.append((max(0, center[j]-0.5), min(2*np.pi, center[j]+0.5)))
                elif j % 4 == 2:  # 投放时间
                    refined_bounds.append((max(0, center[j]-1), center[j]+1))
                else:  # 起爆延迟
                    refined_bounds.append((max(0.8, center[j]-0.5), center[j]+0.5))
            
            result = differential_evolution(
                self.objective_function,
                refined_bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                atol=1e-4,
                tol=1e-4,
                disp=False
            )
            
            if result.success and -result.fun > best_shielding_time:
                best_solution = result.x
                best_shielding_time = -result.fun
                print(f"局部优化成功！遮蔽时间: {best_shielding_time:.2f}s")
            else:
                print("局部优化未改进，使用启发式解")
        
        except Exception as e:
            print(f"局部优化异常: {e}")
        
        return self.format_results(best_solution, best_shielding_time)
    
    def format_results(self, optimal_params, total_shielding_time):
        """格式化结果"""
        drone_names = ['FY1', 'FY2', 'FY3']
        results = {}
        
        print(f"\n【问题4最终结果】")
        print(f"总遮蔽时长: {total_shielding_time:.2f} s")
        
        for i, drone_name in enumerate(drone_names):
            v_un, theta_n, t_in, delta_t_n = optimal_params[i*4:(i+1)*4]
            explosion_time = t_in + delta_t_n
            explosion_pos = self.smoke_explosion_position(drone_name, t_in, delta_t_n, v_un, theta_n)
            drop_pos = self.drone_position(t_in, drone_name, v_un, theta_n)
            
            results[drone_name] = {
                'speed': v_un,
                'direction_angle': theta_n,
                'drop_time': t_in,
                'explosion_delay': delta_t_n,
                'explosion_time': explosion_time,
                'drop_position': drop_pos,
                'explosion_position': explosion_pos
            }
            
            print(f"\n{drone_name}参数:")
            print(f"  速度: {v_un:.2f} m/s")
            print(f"  方向角: {theta_n:.4f} rad ({np.degrees(theta_n):.1f}°)")
            print(f"  投放时间: {t_in:.2f} s")
            print(f"  起爆延迟: {delta_t_n:.2f} s")
            print(f"  起爆时间: {explosion_time:.2f} s")
            print(f"  投放点: ({drop_pos[0]:.1f}, {drop_pos[1]:.1f}, {drop_pos[2]:.1f})")
            print(f"  起爆点: ({explosion_pos[0]:.1f}, {explosion_pos[1]:.1f}, {explosion_pos[2]:.1f})")
        
        results['total_shielding_time'] = total_shielding_time
        results['optimal_params'] = optimal_params
        
        return results
    
    def analyze_drone_contributions(self, results):
        """分析各无人机的贡献"""
        print(f"\n【各无人机贡献分析】")
        
        drone_names = ['FY1', 'FY2', 'FY3']
        
        # 分析各无人机的独立贡献
        for drone_name in drone_names:
            drone_result = results[drone_name]
            
            # 计算单独的遮蔽时间
            single_params = [0] * 12
            drone_index = drone_names.index(drone_name)
            single_params[drone_index*4:(drone_index+1)*4] = [
                drone_result['speed'],
                drone_result['direction_angle'],
                drone_result['drop_time'],
                drone_result['explosion_delay']
            ]
            
            # 其他无人机设置为无效参数
            for j in range(3):
                if j != drone_index:
                    single_params[j*4:(j+1)*4] = [100, np.pi, 100, 1.0]  # 无效时间
            
            try:
                single_shielding = self.calculate_total_shielding_time(single_params)
                print(f"{drone_name}独立贡献: {single_shielding:.2f}s")
            except:
                print(f"{drone_name}独立贡献: 计算失败")
        
        print(f"协同总效果: {results['total_shielding_time']:.2f}s")
    
    def save_results_to_excel(self, results, filename='result2.xlsx'):
        """保存结果到Excel文件 - 按照竞赛模板格式"""
        print(f"\n【保存结果到{filename}】")
        
        data = []
        drone_names = ['FY1', 'FY2', 'FY3']
        
        for i, drone_name in enumerate(drone_names):
            drone_result = results[drone_name]
            
            # 将角度转换为度数，并确保为正值
            angle_degrees = np.degrees(drone_result['direction_angle'])
            if angle_degrees < 0:
                angle_degrees += 360
            
            row = {
                '无人机编号': drone_name,
                '无人机运动方向': f"{angle_degrees:.1f}",  # 只保留数值，不加度符号
                '无人机运动速度(m/s)': f"{drone_result['speed']:.1f}",
                '烟幕干扰弹投放点的x坐标(m)': f"{drone_result['drop_position'][0]:.1f}",
                '烟幕干扰弹投放点的y坐标(m)': f"{drone_result['drop_position'][1]:.1f}",
                '烟幕干扰弹投放点的z坐标(m)': f"{drone_result['drop_position'][2]:.1f}",
                '烟幕干扰弹起爆点的x坐标(m)': f"{drone_result['explosion_position'][0]:.1f}",
                '烟幕干扰弹起爆点的y坐标(m)': f"{drone_result['explosion_position'][1]:.1f}",
                '烟幕干扰弹起爆点的z坐标(m)': f"{drone_result['explosion_position'][2]:.1f}",
                '有效遮蔽时长(s)': f"{results['total_shielding_time']:.2f}"
            }
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            worksheet = writer.sheets['Sheet1']
            
            # 调整列宽以适应内容
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # 添加注释信息
            note_row = len(data) + 3
            worksheet[f'A{note_row}'] = "注：以x轴为正向，"
            worksheet[f'A{note_row+1}'] = "逆时针方向为正，"
            worksheet[f'A{note_row+2}'] = "取值0~360（度）。"
        
        print(f"结果已保存到 {filename}")
        print("\n结果表格:")
        print(df.to_string(index=False))
        
        return df

def main():
    """主函数"""
    print("烟幕干扰弹投放策略 - 问题4基础版求解")
    print("=" * 80)
    
    # 创建求解器
    solver = Problem4BasicSolver()
    
    # 求解问题4
    results = solver.solve_problem4_basic()
    
    # 分析各无人机贡献
    solver.analyze_drone_contributions(results)
    
    # 保存结果到Excel
    df = solver.save_results_to_excel(results, 'result2.xlsx')
    
    print("\n" + "=" * 80)
    print("问题4基础版求解完成！")
    print(f"三架无人机协同遮蔽时长: {results['total_shielding_time']:.2f}s")
    print("详细结果已保存到 result2.xlsx 文件中")
    print("\n关键特点:")
    print("- 三架无人机独立决策，协同作战")
    print("- 基于问题3的成功经验进行参数设计")
    print("- 考虑了各无人机的初始位置差异")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()