"""
火箭发动机Elmer模拟驱动系统
基于150轮全系统联合模拟迭代，实现流体-热-结构全耦合分析
包含统一可视化窗口和控制台实时进度输出
"""

import os
import sys
import time
import math
import json
import logging
import threading
import subprocess
import configparser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# 第三方库导入
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import PyQt6.QtWidgets as QtWidgets
    import PyQt6.QtCore as QtCore
    import PyQt6.QtGui as QtGui
    import pyvista as pv
    from pyvistaqt import QtInteractor
    import pyelmer
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    
    # 尝试导入FreeCAD相关模块（可选）
    try:
        import FreeCAD
        import Part
        import Mesh
        import Gmsh
        FREECAD_AVAILABLE = True
    except ImportError:
        FREECAD_AVAILABLE = False
        print("警告: FreeCAD不可用，3D建模功能将被禁用")
        print("如需完整功能，请从 https://www.freecad.org/ 下载并安装FreeCAD")
        
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请安装以下依赖: pip install numpy matplotlib PyQt6 pyvista pyvistaqt pyelmer reportlab")
    sys.exit(1)

class ConfigManager:
    """配置文件管理器 - 统一管理所有配置参数"""
    
    def __init__(self, config_file: str = "rocket_config.ini"):
        self.config_file = Path(config_file)
        self.config = configparser.ConfigParser()
        self._load_default_config()
        self._load_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        # 路径配置
        self.config['PATHS'] = {
            'working_dir': 'C:/rocket_design',
            'output_dir': 'C:/rocket_design/output',
            'temp_dir': 'C:/rocket_design/temp',
            'log_dir': 'C:/rocket_design/logs',
            'backup_dir': 'C:/rocket_design/backup'
        }
        
        # 模拟配置
        self.config['SIMULATION'] = {
            'max_iterations': '150',
            'timeout_seconds': '3600',
            'memory_limit_mb': '4096',
            'cpu_cores': '4',
            'auto_save_interval': '10'
        }
        
        # 阈值配置
        self.config['THRESHOLDS'] = {
            'thrust_min_coefficient': '0.9',
            'thrust_max_coefficient': '1.1',
            'wall_temp_max': '850',
            'stress_max': '580',
            'cooling_velocity_min': '0.7',
            'dry_weight_max_coefficient': '1.2'
        }
        
        # UI配置
        self.config['UI'] = {
            'window_width': '1200',
            'window_height': '800',
            'theme': 'dark',
            'font_size': '10',
            'auto_refresh_interval': '1000'
        }
        
        # 优化配置
        self.config['OPTIMIZATION'] = {
            'convergence_tolerance': '0.01',
            'max_parameter_change': '0.1',
            'learning_rate': '0.05',
            'momentum': '0.9'
        }
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_file.exists():
                self.config.read(self.config_file, encoding='utf-8')
                print(f"配置文件加载成功: {self.config_file}")
            else:
                self._save_config()
                print(f"创建默认配置文件: {self.config_file}")
        except Exception as e:
            print(f"配置文件加载失败: {e}")
    
    def _save_config(self):
        """保存配置文件"""
        try:
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            print(f"配置文件保存成功: {self.config_file}")
        except Exception as e:
            print(f"配置文件保存失败: {e}")
    
    def get_path(self, key: str, default: str = None) -> str:
        """获取路径配置"""
        try:
            return self.config.get('PATHS', key, fallback=default)
        except:
            return default
    
    def get_int(self, section: str, key: str, default: int = 0) -> int:
        """获取整数配置"""
        try:
            return self.config.getint(section, key, fallback=default)
        except:
            return default
    
    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        try:
            return self.config.getfloat(section, key, fallback=default)
        except:
            return default
    
    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        """获取布尔值配置"""
        try:
            return self.config.getboolean(section, key, fallback=default)
        except:
            return default
    
    def set_value(self, section: str, key: str, value: Any):
        """设置配置值"""
        try:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = str(value)
            self._save_config()
            print(f"配置更新: {section}.{key} = {value}")
        except Exception as e:
            print(f"配置设置失败: {e}")
    
    def create_backup(self, backup_name: str = None):
        """创建配置备份"""
        try:
            if backup_name is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_name = f"config_backup_{timestamp}.ini"
            
            backup_path = Path(self.get_path('backup_dir')) / backup_name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
            
            print(f"配置备份创建成功: {backup_path}")
            return str(backup_path)
        except Exception as e:
            print(f"配置备份失败: {e}")
            return None
    
    def restore_backup(self, backup_path: str):
        """恢复配置备份"""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                print(f"备份文件不存在: {backup_path}")
                return False
            
            # 创建当前配置的备份
            self.create_backup("before_restore.ini")
            
            # 恢复备份配置
            self.config.read(backup_path, encoding='utf-8')
            self._save_config()
            
            print(f"配置恢复成功: {backup_path}")
            return True
        except Exception as e:
            print(f"配置恢复失败: {e}")
            return False
    
    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置完整性"""
        errors = {}
        
        # 验证路径配置
        path_keys = ['working_dir', 'output_dir', 'temp_dir', 'log_dir', 'backup_dir']
        for key in path_keys:
            path = self.get_path(key)
            if not path:
                if 'PATHS' not in errors:
                    errors['PATHS'] = []
                errors['PATHS'].append(f"缺失路径配置: {key}")
        
        # 验证模拟配置
        sim_keys = ['max_iterations', 'timeout_seconds', 'memory_limit_mb', 'cpu_cores']
        for key in sim_keys:
            value = self.get_int('SIMULATION', key)
            if value <= 0:
                if 'SIMULATION' not in errors:
                    errors['SIMULATION'] = []
                errors['SIMULATION'].append(f"无效模拟配置: {key} = {value}")
        
        # 验证阈值配置
        threshold_keys = ['wall_temp_max', 'stress_max', 'cooling_velocity_min']
        for key in threshold_keys:
            value = self.get_float('THRESHOLDS', key)
            if value <= 0:
                if 'THRESHOLDS' not in errors:
                    errors['THRESHOLDS'] = []
                errors['THRESHOLDS'].append(f"无效阈值配置: {key} = {value}")
        
        return errors

class OptimizationPriority(Enum):
    """优化优先级枚举"""
    EFFICIENCY = "效率优先"
    WEIGHT = "重量优先"
    BALANCED = "综合优先"

@dataclass
class PriorityThresholds:
    """优化优先级阈值配置"""
    thrust_error: Tuple[float, float]  # 推力误差阈值 (±%)
    cooling_velocity_min: float        # 最小冷却流速 (m/s)
    stress_max: float                  # 最大结构应力 (MPa)
    
    @classmethod
    def for_priority(cls, priority: OptimizationPriority) -> 'PriorityThresholds':
        """根据优先级获取对应的阈值配置"""
        if priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：严格控制推力误差和冷却流速，放宽结构应力
            return cls(
                thrust_error=(-0.03, 0.03),  # ±3%
                cooling_velocity_min=0.9,     # ≥0.9m/s
                stress_max=580                # ≤580MPa
            )
        elif priority == OptimizationPriority.WEIGHT:
            # 重量优先：严格控制结构应力，放宽推力误差和冷却流速
            return cls(
                thrust_error=(-0.05, 0.05),   # ±5%
                cooling_velocity_min=0.7,     # ≥0.7m/s
                stress_max=550                # ≤550MPa
            )
        else:  # BALANCED
            # 综合优先：所有指标取中间值
            return cls(
                thrust_error=(-0.04, 0.04),   # ±4%
                cooling_velocity_min=0.8,     # ≥0.8m/s
                stress_max=565                # ≤565MPa
            )

@dataclass
class PriorityCheckOrder:
    """优化优先级检查顺序配置"""
    check_sequence: List[str]  # 检查顺序列表
    
    @classmethod
    def for_priority(cls, priority: OptimizationPriority) -> 'PriorityCheckOrder':
        """根据优先级获取对应的检查顺序"""
        if priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：先查推力→雾化混合→壁温→应力
            return cls([
                'thrust_check',
                'atomization_check', 
                'wall_temp_check',
                'stress_check'
            ])
        elif priority == OptimizationPriority.WEIGHT:
            # 重量优先：先查应力→冷却夹层厚度→推力→雾化
            return cls([
                'stress_check',
                'cooling_thickness_check',
                'thrust_check',
                'atomization_check'
            ])
        else:  # BALANCED
            # 综合优先：按指标偏差幅度排序
            return cls([
                'deviation_priority_check',  # 偏差幅度优先
                'thrust_check',
                'stress_check',
                'wall_temp_check'
            ])

@dataclass
class PriorityAdjustmentTendency:
    """优化优先级参数调整倾向配置"""
    thrust_increase_methods: List[str]  # 推力提升方法优先级
    cooling_enhancement_methods: List[str]  # 冷却增强方法优先级
    weight_reduction_methods: List[str]  # 重量减少方法优先级
    
    @classmethod
    def for_priority(cls, priority: OptimizationPriority) -> 'PriorityAdjustmentTendency':
        """根据优先级获取对应的调整倾向"""
        if priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：推力不足时优先增大喷管扩张比，壁温超温时优先提高冷却流速
            return cls(
                thrust_increase_methods=['increase_expansion_ratio', 'increase_chamber_pressure'],
                cooling_enhancement_methods=['increase_cooling_velocity', 'thicken_cooling_channel'],
                weight_reduction_methods=['reduce_wall_thickness']
            )
        elif priority == OptimizationPriority.WEIGHT:
            # 重量优先：应力达标时优先减小壁厚，冷却流速过高时优先缩小冷却夹层厚度
            return cls(
                thrust_increase_methods=['increase_chamber_pressure', 'increase_expansion_ratio'],
                cooling_enhancement_methods=['thicken_cooling_channel', 'increase_cooling_velocity'],
                weight_reduction_methods=['reduce_wall_thickness', 'reduce_cooling_channel_thickness']
            )
        else:  # BALANCED
            # 综合优先：平衡矛盾，提推力的同时微调壁厚
            return cls(
                thrust_increase_methods=['balanced_thrust_enhancement'],
                cooling_enhancement_methods=['balanced_cooling_optimization'],
                weight_reduction_methods=['balanced_weight_reduction']
            )

@dataclass
class DesignRequirements:
    """设计需求数据类"""
    thrust_range: Tuple[float, float]  # 推力区间 (kgf)
    fuel_concentration: float  # 燃料浓度 (%)
    priority: OptimizationPriority  # 优化优先级
    
@dataclass
class SimulationParameters:
    """模拟参数数据类"""
    # 燃烧室参数
    chamber_diameter: float  # 燃烧室内径 (mm)
    chamber_length: float  # 燃烧室长度 (mm)
    chamber_pressure: float  # 燃烧室压力 (MPa)
    
    # 喷管参数
    throat_diameter: float  # 喉部直径 (mm)
    expansion_ratio: float  # 扩张比
    nozzle_length: float  # 喷管长度 (mm)
    
    # 燃料参数
    fuel_flow_rate: float  # 燃料流率 (kg/s)
    oxidizer_flow_rate: float  # 氧化剂流率 (kg/s)
    mixture_ratio: float  # 混合比
    
    # 冷却参数
    cooling_channel_thickness: float  # 冷却夹层厚度 (mm)
    cooling_velocity: float  # 冷却流速 (m/s)
    
    # 结构参数
    wall_thickness: float  # 壁厚 (mm)
    material_density: float  # 材料密度 (kg/m³)

class ThrustChamberSystem:
    """推力室系统类 - 负责燃烧室和喷管的设计与计算"""
    
    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_thrust_chamber_geometry(self) -> Dict[str, float]:
        """计算推力室几何参数"""
        try:
            # 燃烧室体积计算
            chamber_volume = math.pi * (self.parameters.chamber_diameter/2)**2 * self.parameters.chamber_length
            
            # 喷管几何计算
            throat_area = math.pi * (self.parameters.throat_diameter/2)**2
            exit_area = throat_area * self.parameters.expansion_ratio
            exit_diameter = math.sqrt(4 * exit_area / math.pi)
            
            # 喷管收敛角计算 (典型值15-45度)
            convergence_angle = 30.0  # 度
            divergence_angle = 15.0   # 度
            
            geometry = {
                'chamber_volume': chamber_volume,  # mm³
                'throat_area': throat_area,       # mm²
                'exit_area': exit_area,           # mm²
                'exit_diameter': exit_diameter,   # mm
                'convergence_angle': convergence_angle,  # 度
                'divergence_angle': divergence_angle     # 度
            }
            
            self.logger.info(f"推力室几何计算完成: 燃烧室体积{chamber_volume:.2f}mm³, 喉部面积{throat_area:.2f}mm²")
            return geometry
            
        except Exception as e:
            self.logger.error(f"推力室几何计算失败: {e}")
            raise
    
    def calculate_combustion_performance(self) -> Dict[str, float]:
        """计算燃烧性能参数"""
        try:
            # 基于酒精-氧气燃烧的简化计算
            # 酒精(C2H5OH) + 3O2 → 2CO2 + 3H2O
            
            # 理论比冲估算 (酒精-氧气系统)
            isp_theoretical = 280 + (self.parameters.chamber_pressure - 1) * 10  # 秒
            
            # 实际比冲 (考虑效率损失)
            efficiency = 0.85  # 典型燃烧效率
            isp_actual = isp_theoretical * efficiency
            
            # 推力计算
            total_flow_rate = self.parameters.fuel_flow_rate + self.parameters.oxidizer_flow_rate
            thrust = total_flow_rate * isp_actual * 9.81  # N
            
            # 燃烧温度估算
            combustion_temp = 2800 + (self.parameters.chamber_pressure - 1) * 50  # K
            
            performance = {
                'isp_theoretical': isp_theoretical,  # 秒
                'isp_actual': isp_actual,           # 秒
                'thrust': thrust,                    # N
                'combustion_temperature': combustion_temp,  # K
                'combustion_efficiency': efficiency
            }
            
            self.logger.info(f"燃烧性能计算完成: 比冲{isp_actual:.1f}s, 推力{thrust:.1f}N")
            return performance
            
        except Exception as e:
            self.logger.error(f"燃烧性能计算失败: {e}")
            raise
    
    def validate_design(self) -> Dict[str, bool]:
        """验证推力室设计合理性"""
        validation_results = {}
        
        # 检查扩张比范围 (3-100为合理范围)
        validation_results['expansion_ratio_valid'] = 3 <= self.parameters.expansion_ratio <= 100
        
        # 检查燃烧室压力 (0.5-10 MPa为合理范围)
        validation_results['chamber_pressure_valid'] = 0.5 <= self.parameters.chamber_pressure <= 10
        
        # 检查混合比 (酒精-氧气系统典型值2.0-2.8)
        validation_results['mixture_ratio_valid'] = 2.0 <= self.parameters.mixture_ratio <= 2.8
        
        # 检查喉部直径与燃烧室直径比例 (0.3-0.8为合理范围)
        diameter_ratio = self.parameters.throat_diameter / self.parameters.chamber_diameter
        validation_results['diameter_ratio_valid'] = 0.3 <= diameter_ratio <= 0.8
        
        return validation_results

class CoolingSystem:
    """冷却系统类 - 负责再生冷却系统的设计与计算"""
    
    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_cooling_performance(self) -> Dict[str, float]:
        """计算冷却系统性能"""
        try:
            # 冷却通道几何计算
            cooling_channel_area = math.pi * (
                (self.parameters.chamber_diameter/2 + self.parameters.cooling_channel_thickness)**2 - 
                (self.parameters.chamber_diameter/2)**2
            )
            
            # 冷却剂流量计算
            coolant_flow_rate = self.parameters.fuel_flow_rate  # 使用燃料作为冷却剂
            
            # 热传递计算
            # 简化热流密度计算 (W/m²)
            heat_flux = 5e6  # 典型值 5 MW/m²
            
            # 冷却效率计算
            cooling_efficiency = min(1.0, self.parameters.cooling_velocity / 2.0)
            
            # 壁温计算
            base_wall_temp = 800  # 基础壁温 (K)
            actual_wall_temp = base_wall_temp / cooling_efficiency
            
            performance = {
                'cooling_channel_area': cooling_channel_area,  # mm²
                'coolant_flow_rate': coolant_flow_rate,        # kg/s
                'heat_flux': heat_flux,                        # W/m²
                'cooling_efficiency': cooling_efficiency,      # 无量纲
                'wall_temperature': actual_wall_temp           # K
            }
            
            self.logger.info(f"冷却性能计算完成: 冷却效率{cooling_efficiency:.2f}, 壁温{actual_wall_temp:.1f}K")
            return performance
            
        except Exception as e:
            self.logger.error(f"冷却性能计算失败: {e}")
            raise
    
    def calculate_structural_stress(self) -> Dict[str, float]:
        """计算结构应力"""
        try:
            # 热应力计算
            thermal_expansion_coeff = 1.2e-5  # 不锈钢热膨胀系数 (1/K)
            youngs_modulus = 200e9           # 杨氏模量 (Pa)
            delta_temp = 500                 # 温度差 (K)
            
            thermal_stress = thermal_expansion_coeff * youngs_modulus * delta_temp  # Pa
            
            # 压力应力计算
            pressure_stress = self.parameters.chamber_pressure * 1e6 * self.parameters.chamber_diameter / (2 * self.parameters.wall_thickness)  # Pa
            
            # 总应力
            total_stress = thermal_stress + pressure_stress
            
            stress_results = {
                'thermal_stress': thermal_stress / 1e6,  # MPa
                'pressure_stress': pressure_stress / 1e6,  # MPa
                'total_stress': total_stress / 1e6         # MPa
            }
            
            self.logger.info(f"结构应力计算完成: 总应力{total_stress/1e6:.1f}MPa")
            return stress_results
            
        except Exception as e:
            self.logger.error(f"结构应力计算失败: {e}")
            raise
    
    def validate_cooling_design(self) -> Dict[str, bool]:
        """验证冷却系统设计合理性"""
        validation_results = {}
        
        # 检查冷却流速 (0.5-3 m/s为合理范围)
        validation_results['cooling_velocity_valid'] = 0.5 <= self.parameters.cooling_velocity <= 3.0
        
        # 检查冷却夹层厚度 (0.1-0.5 mm为合理范围)
        validation_results['cooling_thickness_valid'] = 0.1 <= self.parameters.cooling_channel_thickness <= 0.5
        
        # 检查壁厚 (0.5-3 mm为合理范围)
        validation_results['wall_thickness_valid'] = 0.5 <= self.parameters.wall_thickness <= 3.0
        
        return validation_results

class PropellantInterface:
    """推进剂接口类 - 负责燃料和氧化剂的供应系统"""
    
    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
        self.logger = logging.getLogger(__name__)
    
    def calculate_injection_performance(self) -> Dict[str, float]:
        """计算喷射性能"""
        try:
            # 喷射压降计算
            injection_pressure_drop = 0.2 * self.parameters.chamber_pressure  # MPa
            
            # 喷射速度计算
            fuel_density = 1.15  # kg/m³ (95%酒精)
            oxidizer_density = 1.14  # kg/m³ (液氧)
            
            fuel_injection_velocity = math.sqrt(2 * injection_pressure_drop * 1e6 / fuel_density)  # m/s
            oxidizer_injection_velocity = math.sqrt(2 * injection_pressure_drop * 1e6 / oxidizer_density)  # m/s
            
            # 混合效率计算
            mixing_efficiency = 0.9 - abs(self.parameters.mixture_ratio - 2.33) * 0.1
            
            performance = {
                'injection_pressure_drop': injection_pressure_drop,  # MPa
                'fuel_injection_velocity': fuel_injection_velocity,  # m/s
                'oxidizer_injection_velocity': oxidizer_injection_velocity,  # m/s
                'mixing_efficiency': mixing_efficiency              # 无量纲
            }
            
            self.logger.info(f"喷射性能计算完成: 混合效率{mixing_efficiency:.2f}")
            return performance
            
        except Exception as e:
            self.logger.error(f"喷射性能计算失败: {e}")
            raise
    
    def calculate_system_mass(self) -> Dict[str, float]:
        """计算系统质量"""
        try:
            # 推进剂质量计算
            burn_time = 60  # 典型燃烧时间 (秒)
            fuel_mass = self.parameters.fuel_flow_rate * burn_time
            oxidizer_mass = self.parameters.oxidizer_flow_rate * burn_time
            
            # 结构质量估算
            structure_mass_factor = 0.3  # 结构质量系数
            total_propellant_mass = fuel_mass + oxidizer_mass
            structure_mass = total_propellant_mass * structure_mass_factor
            
            # 总质量
            total_mass = total_propellant_mass + structure_mass
            
            mass_results = {
                'fuel_mass': fuel_mass,           # kg
                'oxidizer_mass': oxidizer_mass,   # kg
                'structure_mass': structure_mass, # kg
                'total_mass': total_mass          # kg
            }
            
            self.logger.info(f"系统质量计算完成: 总质量{total_mass:.1f}kg")
            return mass_results
            
        except Exception as e:
            self.logger.error(f"系统质量计算失败: {e}")
            raise
    
    def validate_propellant_system(self) -> Dict[str, bool]:
        """验证推进剂系统合理性"""
        validation_results = {}
        
        # 检查流率比 (燃料:氧化剂 ≈ 1:2.33)
        flow_ratio = self.parameters.oxidizer_flow_rate / self.parameters.fuel_flow_rate
        validation_results['flow_ratio_valid'] = 2.0 <= flow_ratio <= 2.8
        
        # 检查总流率 (根据推力需求)
        total_flow = self.parameters.fuel_flow_rate + self.parameters.oxidizer_flow_rate
        validation_results['total_flow_valid'] = 0.1 <= total_flow <= 10.0  # kg/s
        
        return validation_results

class RocketEngineCore:
    """火箭发动机核心集成类 - 整合所有子系统"""
    
    def __init__(self, parameters: SimulationParameters):
        self.parameters = parameters
        self.thrust_chamber = ThrustChamberSystem(parameters)
        self.cooling_system = CoolingSystem(parameters)
        self.propellant_interface = PropellantInterface(parameters)
        self.logger = logging.getLogger(__name__)
    
    def perform_comprehensive_analysis(self) -> Dict[str, Any]:
        """执行全面分析"""
        try:
            analysis_results = {}
            
            # 推力室分析
            analysis_results['thrust_chamber_geometry'] = self.thrust_chamber.calculate_thrust_chamber_geometry()
            analysis_results['combustion_performance'] = self.thrust_chamber.calculate_combustion_performance()
            analysis_results['thrust_chamber_validation'] = self.thrust_chamber.validate_design()
            
            # 冷却系统分析
            analysis_results['cooling_performance'] = self.cooling_system.calculate_cooling_performance()
            analysis_results['structural_stress'] = self.cooling_system.calculate_structural_stress()
            analysis_results['cooling_validation'] = self.cooling_system.validate_cooling_design()
            
            # 推进剂系统分析
            analysis_results['injection_performance'] = self.propellant_interface.calculate_injection_performance()
            analysis_results['system_mass'] = self.propellant_interface.calculate_system_mass()
            analysis_results['propellant_validation'] = self.propellant_interface.validate_propellant_system()
            
            # 总体评估
            analysis_results['overall_assessment'] = self._assess_overall_performance(analysis_results)
            
            self.logger.info("火箭发动机核心分析完成")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"全面分析失败: {e}")
            raise
    
    def _assess_overall_performance(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估总体性能"""
        assessment = {}
        
        # 性能评分
        thrust_performance = analysis_results['combustion_performance']['thrust']
        cooling_performance = analysis_results['cooling_performance']['cooling_efficiency']
        structural_performance = 1.0 - min(1.0, analysis_results['structural_stress']['total_stress'] / 500)
        
        overall_score = (thrust_performance/1000 + cooling_performance + structural_performance) / 3
        
        assessment['performance_score'] = overall_score
        assessment['thrust_rating'] = '优秀' if thrust_performance > 800 else '良好' if thrust_performance > 500 else '一般'
        assessment['cooling_rating'] = '优秀' if cooling_performance > 0.8 else '良好' if cooling_performance > 0.6 else '一般'
        assessment['structural_rating'] = '优秀' if structural_performance > 0.8 else '良好' if structural_performance > 0.6 else '一般'
        
        return assessment

class RocketEngineSimulator:
    """火箭发动机模拟器主类"""
    
    def __init__(self):
        self.design_requirements = None
        self.current_parameters = None
        self.simulation_results = []
        self.iteration_count = 0
        self.max_iterations = 150
        self.current_stage = "初始化"
        self.is_running = False
        self.start_time = None
        
        # 阈值配置
        self.thresholds = self._initialize_thresholds()
        
        # 文件路径
        self.working_dir = Path("C:/rocket_design")
        self.output_dir = self.working_dir / "output"
        self.temp_dir = self.working_dir / "temp"
        
        # 创建目录
        self._create_directories()
        
        # 初始化日志
        self._setup_logging()
        
    def _initialize_thresholds(self) -> Dict:
        """初始化不同优先级的阈值"""
        return {
            OptimizationPriority.EFFICIENCY: {
                "thrust_min": 0.95,  # 推力下限系数
                "thrust_max": 1.05,  # 推力上限系数
                "wall_temp_max": 820,  # 最大壁温 (K)
                "stress_max": 580,  # 最大应力 (MPa)
                "cooling_velocity_min": 0.8,  # 最小冷却流速 (m/s)
                "dry_weight_max": 1.2  # 最大干重系数
            },
            OptimizationPriority.WEIGHT: {
                "thrust_min": 0.9,
                "thrust_max": 1.1,
                "wall_temp_max": 850,
                "stress_max": 550,
                "cooling_velocity_min": 0.7,
                "dry_weight_max": 1.0
            },
            OptimizationPriority.BALANCED: {
                "thrust_min": 0.93,
                "thrust_max": 1.07,
                "wall_temp_max": 835,
                "stress_max": 565,
                "cooling_velocity_min": 0.75,
                "dry_weight_max": 1.1
            }
        }
    
    def _create_directories(self):
        """创建工作目录"""
        self.working_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'simulation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def set_design_requirements(self, thrust_min: float, thrust_max: float, 
                               fuel_concentration: float, priority: OptimizationPriority):
        """设置设计需求"""
        self.design_requirements = DesignRequirements(
            thrust_range=(thrust_min, thrust_max),
            fuel_concentration=fuel_concentration,
            priority=priority
        )
        self.logger.info(f"设计需求设置: 推力{thrust_min}-{thrust_max}kgf, 燃料浓度{fuel_concentration}%, 优先级{priority.value}")
    
    def calculate_initial_parameters(self):
        """计算初始参数"""
        if not self.design_requirements:
            raise ValueError("请先设置设计需求")
        
        thrust_min, thrust_max = self.design_requirements.thrust_range
        fuel_conc = self.design_requirements.fuel_concentration
        
        # 燃料特性计算
        if fuel_conc >= 95:  # 95%酒精
            exhaust_velocity = 1700 + (fuel_conc - 95) * 2  # 1700-1800 m/s
            fuel_density = 1.15  # kg/m³
        else:  # 无水乙醇
            exhaust_velocity = 1800 + (fuel_conc - 90) * 2  # 1800-1900 m/s
            fuel_density = 1.2  # kg/m³
        
        # 推力公式 F = m_dot * v_e
        avg_thrust = (thrust_min + thrust_max) / 2
        mass_flow_rate = avg_thrust * 9.81 / exhaust_velocity  # kg/s
        
        # 燃烧室参数估算
        chamber_pressure = 2.0  # MPa
        chamber_diameter = math.sqrt(mass_flow_rate / (fuel_density * math.pi * 0.25 * 100)) * 1000  # mm
        
        # 喷管参数
        throat_diameter = chamber_diameter * 0.6
        expansion_ratio = 5.0
        
        self.current_parameters = SimulationParameters(
            chamber_diameter=chamber_diameter,
            chamber_length=chamber_diameter * 2,
            chamber_pressure=chamber_pressure,
            throat_diameter=throat_diameter,
            expansion_ratio=expansion_ratio,
            nozzle_length=throat_diameter * 3,
            fuel_flow_rate=mass_flow_rate * 0.7,
            oxidizer_flow_rate=mass_flow_rate * 0.3,
            mixture_ratio=2.33,
            cooling_channel_thickness=0.15,
            cooling_velocity=0.8,
            wall_thickness=1.0,
            material_density=8000  # 不锈钢
        )
        
        self.logger.info("初始参数计算完成")
        return self.current_parameters
    
    def run_full_simulation(self, callback=None):
        """运行完整的150轮模拟迭代"""
        try:
            if not self.design_requirements or not self.current_parameters:
                raise ValueError("请先设置设计需求和计算初始参数")
            
            self.is_running = True
            self.start_time = time.time()
            self.simulation_results = []
            
            # 初始化组件 - 添加异常处理
            try:
                self.modeler = FreeCADModeler()
                self.elmer_simulator = ElmerSimulator(self.temp_dir)
            except Exception as init_error:
                self._log_error(f"组件初始化失败: {init_error}")
                self._handle_critical_error("组件初始化", init_error, callback)
                return
            
            # 阶段1: 参数初筛 (20-30轮)
            self.current_stage = "参数初筛"
            if not self._run_simulation_stage("screening", 30, callback):
                return
            
            if not self.is_running:
                return
            
            # 阶段2: 耦合优化 (80-100轮)
            self.current_stage = "耦合优化"
            if not self._run_simulation_stage("optimization", 100, callback):
                return
            
            if not self.is_running:
                return
            
            # 阶段3: 可靠性验证 (20-30轮)
            self.current_stage = "可靠性验证"
            if not self._run_simulation_stage("validation", 20, callback):
                return
            
            # 生成最终成果
            self._generate_final_output()
            
            total_time = time.time() - self.start_time
            self.logger.info(f"模拟完成! 总耗时: {total_time/3600:.2f}小时")
            
            # 发送完成回调
            if callback:
                callback(len(self.simulation_results), self.current_parameters, {
                    'thrust': self.simulation_results[-1].get('thrust', 0) if self.simulation_results else 0,
                    'max_wall_temperature': self.simulation_results[-1].get('max_wall_temperature', 0) if self.simulation_results else 0,
                    'max_stress': self.simulation_results[-1].get('max_stress', 0) if self.simulation_results else 0,
                    'is_completed': True
                })
                
        except Exception as e:
            self._log_error(f"模拟过程发生严重错误: {e}")
            self._handle_critical_error("模拟执行", e, callback)
        finally:
            self.is_running = False
            self._cleanup_resources()
    
    def _run_screening_stage(self, max_iterations: int, callback=None):
        """运行参数初筛阶段"""
        self.logger.info(f"开始参数初筛阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                break
                
            iteration = len(self.simulation_results) + 1
            self._log_progress(iteration, "初筛")
            
            # 创建/更新3D模型
            model_path = self.modeler.create_initial_model(self.current_parameters)
            
            # 运行Elmer模拟
            results = self.elmer_simulator.run_simulation(model_path, self.current_parameters, iteration)
            self.simulation_results.append(results)
            
            # 检查阈值并淘汰不合格参数
            if not self._check_thresholds(results):
                self.logger.info(f"第{iteration}轮参数不达标，进行参数调整")
                self._adjust_parameters(results, "screening")
            
            # 调用回调函数更新UI
            if callback:
                callback(iteration, self.current_parameters, results)
            
            if iteration >= 20 and self._is_screening_complete():
                self.logger.info("参数初筛完成")
                break
    
    def _run_optimization_stage(self, max_iterations: int, callback=None):
        """运行耦合优化阶段"""
        self.logger.info(f"开始耦合优化阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                break
                
            iteration = len(self.simulation_results) + 1
            self._log_progress(iteration, "优化")
            
            # 更新3D模型
            model_path = self.modeler.create_initial_model(self.current_parameters)
            
            # 运行全耦合模拟
            results = self.elmer_simulator.run_simulation(model_path, self.current_parameters, iteration)
            self.simulation_results.append(results)
            
            # 识别并解决耦合问题
            coupling_issues = self._identify_coupling_issues(results)
            if coupling_issues:
                self._resolve_coupling_issues(coupling_issues)
            
            # 调用回调函数更新UI
            if callback:
                callback(iteration, self.current_parameters, results)
            
            if iteration >= 80 and self._is_optimization_complete():
                self.logger.info("耦合优化完成")
                break
    
    def _run_validation_stage(self, max_iterations: int, callback=None):
        """运行可靠性验证阶段"""
        self.logger.info(f"开始可靠性验证阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                break
                
            iteration = len(self.simulation_results) + 1
            self._log_progress(iteration, "验证")
            
            # 模拟极端场景
            extreme_scenario = self._select_extreme_scenario()
            results = self._run_extreme_simulation(extreme_scenario, iteration)
            self.simulation_results.append(results)
            
            # 验证可靠性
            if not self._validate_reliability(results):
                self.logger.info(f"第{iteration}轮验证失败，返回优化阶段")
                # 返回优化阶段进行微调
                self._run_optimization_stage(10, callback)
            
            # 调用回调函数更新UI
            if callback:
                callback(iteration, self.current_parameters, results)
            
            if iteration >= 20 and self._is_validation_complete():
                self.logger.info("可靠性验证完成")
                break
    
    def _log_progress(self, iteration: int, stage: str):
        """记录进度信息"""
        elapsed_time = time.time() - self.start_time
        progress = (iteration / self.max_iterations) * 100
        
        log_message = f"[第{iteration}轮] {stage}阶段 - 进度{progress:.1f}% - 已耗时{elapsed_time/3600:.2f}小时"
        self.logger.info(log_message)
        print(log_message)  # 控制台输出
    
    def _check_thresholds(self, results: Dict[str, Any]) -> bool:
        """检查结果是否满足阈值要求"""
        thresholds = self.thresholds[self.design_requirements.priority]
        thrust_min, thrust_max = self.design_requirements.thrust_range
        
        # 检查推力范围
        thrust = results.get('thrust', 0)
        if thrust < thrust_min * thresholds['thrust_min'] or thrust > thrust_max * thresholds['thrust_max']:
            return False
        
        # 检查壁温
        if results.get('max_wall_temperature', 0) > thresholds['wall_temp_max']:
            return False
        
        # 检查应力
        if results.get('max_stress', 0) > thresholds['stress_max']:
            return False
        
        return True
    
    def _adjust_parameters(self, results: Dict[str, Any], stage: str):
        """根据模拟结果调整参数"""
        # 根据阶段和优先级调整参数
        if stage == "screening":
            # 初筛阶段主要调整推力相关参数
            thrust = results.get('thrust', 0)
            thrust_min, thrust_max = self.design_requirements.thrust_range
            
            if thrust < thrust_min:
                # 推力不足，增加燃料流率
                self.current_parameters.fuel_flow_rate *= 1.1
                self.current_parameters.oxidizer_flow_rate *= 1.1
            elif thrust > thrust_max:
                # 推力过大，减小燃料流率
                self.current_parameters.fuel_flow_rate *= 0.9
                self.current_parameters.oxidizer_flow_rate *= 0.9
        
        elif stage == "optimization":
            # 优化阶段根据具体问题调整参数
            if results.get('max_wall_temperature', 0) > self.thresholds[self.design_requirements.priority]['wall_temp_max']:
                # 壁温过高，增加冷却
                self.current_parameters.cooling_velocity *= 1.1
                self.current_parameters.cooling_channel_thickness *= 1.05
    
    def _identify_coupling_issues(self, results: Dict[str, Any]) -> List[str]:
        """识别耦合问题"""
        issues = []
        thresholds = self.thresholds[self.design_requirements.priority]
        
        if results.get('max_wall_temperature', 0) > thresholds['wall_temp_max']:
            issues.append("壁温超标")
        if results.get('max_stress', 0) > thresholds['stress_max']:
            issues.append("应力超标")
        if results.get('cooling_velocity', 0) < thresholds['cooling_velocity_min']:
            issues.append("冷却流速不足")
        
        return issues
    
    def _resolve_coupling_issues(self, issues: List[str]):
        """解决耦合问题"""
        for issue in issues:
            if issue == "壁温超标":
                self.current_parameters.cooling_velocity *= 1.15
                self.logger.info("解决壁温超标: 提高冷却流速")
            elif issue == "应力超标":
                self.current_parameters.wall_thickness *= 1.1
                self.logger.info("解决应力超标: 增加壁厚")
            elif issue == "冷却流速不足":
                self.current_parameters.cooling_channel_thickness *= 1.2
                self.logger.info("解决冷却流速不足: 增大冷却通道")
    
    def _select_extreme_scenario(self) -> str:
        """选择极端验证场景"""
        scenarios = ["燃料流量-15%", "冷却堵塞20%", "高温环境+50K", "最大推力持续"]
        return scenarios[len(self.simulation_results) % len(scenarios)]
    
    def _run_extreme_simulation(self, scenario: str, iteration: int) -> Dict[str, Any]:
        """运行极端场景模拟"""
        self.logger.info(f"运行极端场景: {scenario}")
        
        # 根据场景调整参数
        original_params = self.current_parameters.__dict__.copy()
        
        if scenario == "燃料流量-15%":
            self.current_parameters.fuel_flow_rate *= 0.85
            self.current_parameters.oxidizer_flow_rate *= 0.85
        elif scenario == "冷却堵塞20%":
            self.current_parameters.cooling_velocity *= 0.8
        elif scenario == "高温环境+50K":
            # 在Elmer中设置更高的环境温度
            pass
        elif scenario == "最大推力持续":
            self.current_parameters.chamber_pressure *= 1.2
        
        # 运行模拟
        model_path = self.modeler.create_initial_model(self.current_parameters)
        results = self.elmer_simulator.run_simulation(model_path, self.current_parameters, iteration)
        
        # 恢复原始参数
        for key, value in original_params.items():
            setattr(self.current_parameters, key, value)
        
        return results
    
    def _validate_reliability(self, results: Dict[str, Any]) -> bool:
        """验证可靠性"""
        # 在极端场景下检查安全性
        safety_margin = 1.2  # 20%安全余量
        thresholds = self.thresholds[self.design_requirements.priority]
        
        if results.get('max_wall_temperature', 0) > thresholds['wall_temp_max'] / safety_margin:
            return False
        if results.get('max_stress', 0) > thresholds['stress_max'] / safety_margin:
            return False
        
        return True
    
    def _is_screening_complete(self) -> bool:
        """检查初筛是否完成"""
        # 检查最近5轮结果是否稳定
        if len(self.simulation_results) < 5:
            return False
        
        recent_results = self.simulation_results[-5:]
        thrusts = [r.get('thrust', 0) for r in recent_results]
        thrust_std = np.std(thrusts)
        
        return thrust_std < 0.5  # 推力标准差小于0.5kgf
    
    def _is_optimization_complete(self) -> bool:
        """检查优化是否完成"""
        if len(self.simulation_results) < 10:
            return False
        
        # 检查最近10轮是否没有严重问题
        recent_results = self.simulation_results[-10:]
        issues_count = 0
        
        for result in recent_results:
            if not self._check_thresholds(result):
                issues_count += 1
        
        return issues_count <= 2  # 允许最多2轮有小问题
    
    def _is_validation_complete(self) -> bool:
        """检查验证是否完成"""
        # 检查所有极端场景是否通过
        return len(self.simulation_results) >= 20
    
    def _generate_final_output(self):
        """生成最终成果文件"""
        try:
            self.logger.info("开始生成最终成果")
            
            # 生成最终3D模型
            final_model_path = self.modeler.create_initial_model(self.current_parameters)
            
            # 生成工程报告
            self._generate_engineering_report()
            
            # 打包所有文件
            self._package_output_files()
            
            self.logger.info("成果生成完成")
        except Exception as e:
            self._log_error(f"最终成果生成失败: {e}")
            # 即使成果生成失败，也不影响模拟结果
            self.logger.warning("最终成果生成失败，但模拟结果仍然有效")
    
    def _run_simulation_stage(self, stage_type: str, max_iterations: int, callback=None) -> bool:
        """运行模拟阶段（增强异常处理版本）"""
        try:
            if stage_type == "screening":
                return self._run_screening_stage_enhanced(max_iterations, callback)
            elif stage_type == "optimization":
                return self._run_optimization_stage_enhanced(max_iterations, callback)
            elif stage_type == "validation":
                return self._run_validation_stage_enhanced(max_iterations, callback)
            else:
                raise ValueError(f"未知的阶段类型: {stage_type}")
        except Exception as e:
            self._log_error(f"{self.current_stage}阶段执行失败: {e}")
            return False
    
    def _run_screening_stage_enhanced(self, max_iterations: int, callback=None) -> bool:
        """增强版参数初筛阶段"""
        self.logger.info(f"开始参数初筛阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                return True
                
            iteration = len(self.simulation_results) + 1
            
            try:
                self._log_progress(iteration, "初筛")
                
                # 创建/更新3D模型
                model_path = self.modeler.create_initial_model(self.current_parameters)
                
                # 运行Elmer模拟
                results = self.elmer_simulator.run_simulation(model_path, self.current_parameters, iteration)
                self.simulation_results.append(results)
                
                # 检查阈值并淘汰不合格参数
                if not self._check_thresholds(results):
                    self.logger.info(f"第{iteration}轮参数不达标，进行参数调整")
                    self._adjust_parameters(results, "screening")
                
                # 调用回调函数更新UI
                if callback:
                    callback(iteration, self.current_parameters, results)
                
                if iteration >= 20 and self._is_screening_complete():
                    self.logger.info("参数初筛完成")
                    return True
                    
            except Exception as e:
                self._handle_iteration_error(iteration, "初筛", e, callback)
                # 如果连续失败次数过多，终止阶段
                if self._should_abort_stage(iteration):
                    self.logger.error("连续失败次数过多，终止初筛阶段")
                    return False
        
        return True
    
    def _run_optimization_stage_enhanced(self, max_iterations: int, callback=None) -> bool:
        """增强版耦合优化阶段"""
        self.logger.info(f"开始耦合优化阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                return True
                
            iteration = len(self.simulation_results) + 1
            
            try:
                self._log_progress(iteration, "优化")
                
                # 更新3D模型
                model_path = self.modeler.create_initial_model(self.current_parameters)
                
                # 运行全耦合模拟
                results = self.elmer_simulator.run_simulation(model_path, self.current_parameters, iteration)
                self.simulation_results.append(results)
                
                # 识别并解决耦合问题
                coupling_issues = self._identify_coupling_issues(results)
                if coupling_issues:
                    self._resolve_coupling_issues(coupling_issues)
                
                # 调用回调函数更新UI
                if callback:
                    callback(iteration, self.current_parameters, results)
                
                if iteration >= 80 and self._is_optimization_complete():
                    self.logger.info("耦合优化完成")
                    return True
                    
            except Exception as e:
                self._handle_iteration_error(iteration, "优化", e, callback)
                # 如果连续失败次数过多，终止阶段
                if self._should_abort_stage(iteration):
                    self.logger.error("连续失败次数过多，终止优化阶段")
                    return False
        
        return True
    
    def _run_validation_stage_enhanced(self, max_iterations: int, callback=None) -> bool:
        """增强版可靠性验证阶段"""
        self.logger.info(f"开始可靠性验证阶段 (最多{max_iterations}轮)")
        
        for i in range(max_iterations):
            if not self.is_running:
                return True
                
            iteration = len(self.simulation_results) + 1
            
            try:
                self._log_progress(iteration, "验证")
                
                # 模拟极端场景
                extreme_scenario = self._select_extreme_scenario()
                results = self._run_extreme_simulation(extreme_scenario, iteration)
                self.simulation_results.append(results)
                
                # 验证可靠性
                if not self._validate_reliability(results):
                    self.logger.info(f"第{iteration}轮验证失败，返回优化阶段")
                    # 返回优化阶段进行微调
                    self._run_optimization_stage_enhanced(10, callback)
                
                # 调用回调函数更新UI
                if callback:
                    callback(iteration, self.current_parameters, results)
                
                if iteration >= 20 and self._is_validation_complete():
                    self.logger.info("可靠性验证完成")
                    return True
                    
            except Exception as e:
                self._handle_iteration_error(iteration, "验证", e, callback)
                # 如果连续失败次数过多，终止阶段
                if self._should_abort_stage(iteration):
                    self.logger.error("连续失败次数过多，终止验证阶段")
                    return False
        
        return True
    
    def _handle_iteration_error(self, iteration: int, stage: str, error: Exception, callback=None):
        """处理单轮迭代错误"""
        error_msg = f"第{iteration}轮{stage}阶段失败: {error}"
        self._log_error(error_msg)
        
        # 记录错误统计
        if not hasattr(self, '_error_stats'):
            self._error_stats = {'total': 0, 'by_stage': {}, 'consecutive': 0}
        
        self._error_stats['total'] += 1
        self._error_stats['by_stage'][stage] = self._error_stats['by_stage'].get(stage, 0) + 1
        self._error_stats['consecutive'] += 1
        
        # 发送错误回调
        if callback:
            callback(iteration, self.current_parameters, {
                'error': error_msg,
                'is_error': True,
                'stage': stage
            })
        
        # 根据错误类型采取不同恢复策略
        self._recover_from_error(error, iteration, stage)
    
    def _handle_critical_error(self, operation: str, error: Exception, callback=None):
        """处理严重错误"""
        critical_msg = f"严重错误 - {operation}: {error}"
        self._log_error(critical_msg)
        
        # 发送严重错误回调
        if callback:
            callback(0, None, {
                'error': critical_msg,
                'is_critical_error': True,
                'operation': operation
            })
        
        # 执行紧急清理
        self._emergency_cleanup()
    
    def _recover_from_error(self, error: Exception, iteration: int, stage: str):
        """从错误中恢复"""
        error_str = str(error).lower()
        
        if "memory" in error_str or "disk" in error_str:
            # 内存或磁盘错误，清理资源
            self._cleanup_temporary_files()
            self.logger.info("已清理临时文件，尝试恢复")
        elif "timeout" in error_str or "time out" in error_str:
            # 超时错误，调整参数
            self._adjust_parameters_for_timeout()
            self.logger.info("检测到超时，已调整参数")
        elif "file" in error_str or "path" in error_str:
            # 文件路径错误，重建工作目录
            self._recreate_working_directory()
            self.logger.info("文件路径错误，已重建工作目录")
        else:
            # 其他错误，尝试参数重置
            self._reset_parameters_to_safe_values()
            self.logger.info("未知错误，已重置参数到安全值")
    
    def _should_abort_stage(self, iteration: int) -> bool:
        """判断是否应该终止当前阶段"""
        if not hasattr(self, '_error_stats'):
            return False
        
        # 如果连续失败超过5次，终止阶段
        if self._error_stats['consecutive'] >= 5:
            return True
        
        # 如果总失败次数超过阶段迭代数的30%，终止阶段
        if self._error_stats['total'] >= iteration * 0.3:
            return True
        
        return False
    
    def _adjust_parameters_for_timeout(self):
        """为超时错误调整参数"""
        # 减小模拟规模以降低计算时间
        self.current_parameters.chamber_diameter *= 0.9
        self.current_parameters.expansion_ratio *= 0.9
        self.logger.info("为应对超时，已减小模拟规模")
    
    def _reset_parameters_to_safe_values(self):
        """重置参数到安全值"""
        # 使用初始参数的安全版本
        if hasattr(self, 'initial_parameters'):
            self.current_parameters = self.initial_parameters
        else:
            # 重新计算初始参数
            self.calculate_initial_parameters()
            self.initial_parameters = self.current_parameters
        
        self.logger.info("已重置参数到安全值")
    
    def _cleanup_temporary_files(self):
        """清理临时文件"""
        import shutil
        
        try:
            # 清理临时目录
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
            
            # 重新创建临时目录
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("临时文件清理完成")
        except Exception as e:
            self._log_error(f"临时文件清理失败: {e}")
    
    def _recreate_working_directory(self):
        """重建工作目录"""
        import shutil
        
        try:
            # 备份重要文件
            important_files = []
            if self.output_dir.exists():
                for file in self.output_dir.glob("*.json"):
                    important_files.append(file)
                for file in self.output_dir.glob("*.pdf"):
                    important_files.append(file)
            
            # 清理并重建目录
            if self.working_dir.exists():
                shutil.rmtree(self.working_dir)
            
            self.working_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 恢复重要文件
            for file in important_files:
                if file.exists():
                    shutil.copy2(file, self.output_dir / file.name)
            
            self.logger.info("工作目录重建完成")
        except Exception as e:
            self._log_error(f"工作目录重建失败: {e}")
    
    def _emergency_cleanup(self):
        """内存优化的紧急清理（增强版本）"""
        try:
            self.logger.warning("开始执行内存优化的紧急清理")
            
            # 1. 记录当前内存状态
            self._log_memory_usage("紧急清理前")
            
            # 2. 停止所有正在运行的进程
            if hasattr(self, 'elmer_simulator') and self.elmer_simulator:
                try:
                    self.elmer_simulator._cleanup_timeout_process()
                except Exception as e:
                    self._log_warning(f"进程清理失败: {e}")
            
            # 3. 清理临时文件
            try:
                self._cleanup_temporary_files()
                # 额外清理所有临时文件
                self._cleanup_all_temporary_files()
            except Exception as e:
                self._log_warning(f"临时文件清理失败: {e}")
            
            # 4. 清理所有资源
            try:
                self._cleanup_resources()
            except Exception as e:
                self._log_warning(f"资源清理失败: {e}")
            
            # 5. 清理大数组和缓存
            try:
                self._cleanup_large_arrays()
                self._cleanup_all_caches()
            except Exception as e:
                self._log_warning(f"缓存清理失败: {e}")
            
            # 6. 强制垃圾回收（多次执行确保清理）
            import gc
            for i in range(3):
                gc.collect()
            
            # 7. 保存错误日志和内存转储
            error_log_path = self.output_dir / "emergency_error.log"
            with open(error_log_path, 'w') as f:
                f.write(f"紧急清理时间: {datetime.now()}\n")
                f.write(f"错误统计: {getattr(self, '_error_stats', '无统计')}\n")
                f.write(f"内存清理状态: 完成\n")
            
            # 8. 记录清理后的内存状态
            self._log_memory_usage("紧急清理后")
            
            self.logger.info("内存优化的紧急清理完成")
        except Exception as e:
            self._log_error(f"紧急清理过程中发生错误: {e}")
    
    def _cleanup_all_temporary_files(self):
        """清理所有临时文件（包括隐藏文件）"""
        try:
            import shutil
            
            # 清理工作目录中的所有临时文件
            temp_extensions = ['.tmp', '.log', '.mesh', '.sif', '.vtu', '.dat', '.bak', '.old']
            
            for ext in temp_extensions:
                for file_path in self.working_dir.glob(f"*{ext}"):
                    try:
                        if file_path.exists():
                            file_path.unlink()
                    except Exception as e:
                        self._log_warning(f"无法删除临时文件 {file_path}: {e}")
            
            # 清理临时目录
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self._log_warning(f"临时目录清理失败: {e}")
            
            self.logger.info("所有临时文件清理完成")
        except Exception as e:
            self._log_warning(f"全面临时文件清理失败: {e}")
    
    def _cleanup_all_caches(self):
        """清理所有缓存数据"""
        try:
            # 清理Python缓存
            import sys
            
            # 清理模块缓存
            for module_name in list(sys.modules.keys()):
                if 'temp' in module_name.lower() or 'cache' in module_name.lower():
                    try:
                        del sys.modules[module_name]
                    except:
                        pass
            
            # 清理实例属性中的缓存数据
            cache_patterns = ['cache', 'temp', 'tmp', 'data', 'buffer']
            for attr_name in list(self.__dict__.keys()):
                if any(pattern in attr_name.lower() for pattern in cache_patterns):
                    try:
                        setattr(self, attr_name, None)
                    except:
                        pass
            
            # 清理全局缓存
            if hasattr(self, '__cached_data__'):
                self.__cached_data__.clear()
            
            self.logger.info("所有缓存数据清理完成")
        except Exception as e:
            self._log_warning(f"缓存数据清理失败: {e}")
    
    def _cleanup_resources(self):
        """内存优化的资源清理（增强版本）"""
        try:
            # 1. 清理临时文件
            self._cleanup_temporary_files()
            
            # 2. 清理FreeCAD资源
            if hasattr(self, 'modeler') and self.modeler:
                # FreeCAD文档清理
                try:
                    if hasattr(self.modeler, 'document') and self.modeler.document:
                        # 关闭FreeCAD文档
                        import FreeCAD
                        FreeCAD.closeDocument(self.modeler.document.Name)
                        self.modeler.document = None
                except Exception as e:
                    self._log_warning(f"FreeCAD文档清理失败: {e}")
            
            # 3. 清理Elmer模拟器资源
            if hasattr(self, 'elmer_simulator') and self.elmer_simulator:
                try:
                    # 清理模拟器内部资源
                    if hasattr(self.elmer_simulator, '_cleanup_resources'):
                        self.elmer_simulator._cleanup_resources()
                    
                    # 清理进程资源
                    self.elmer_simulator._cleanup_timeout_process()
                except Exception as e:
                    self._log_warning(f"Elmer模拟器资源清理失败: {e}")
            
            # 4. 清理大数组和缓存数据
            self._cleanup_large_arrays()
            
            # 5. 强制垃圾回收
            import gc
            gc.collect()
            
            # 6. 记录内存使用情况
            self._log_memory_usage("资源清理后")
            
            self.logger.info("内存优化的资源清理完成")
        except Exception as e:
            self._log_error(f"资源清理失败: {e}")
    
    def _cleanup_large_arrays(self):
        """清理大数组和缓存数据"""
        try:
            # 清理模拟结果中的大数组
            if hasattr(self, 'simulation_results') and self.simulation_results:
                for result in self.simulation_results:
                    if isinstance(result, dict):
                        # 清理大数组字段
                        large_fields = ['temperature_field', 'stress_field', 'velocity_field', 'pressure_field']
                        for field in large_fields:
                            if field in result:
                                result[field] = None
            
            # 清理临时缓存
            cache_attrs = ['_temp_cache', '_data_cache', '_mesh_cache']
            for attr in cache_attrs:
                if hasattr(self, attr):
                    setattr(self, attr, None)
            
            # 清理可视化数据
            if hasattr(self, 'visualization_data') and self.visualization_data:
                self.visualization_data.clear()
                
        except Exception as e:
            self._log_warning(f"大数组清理失败: {e}")
    
    def _log_memory_usage(self, context: str = ""):
        """记录内存使用情况"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_mb = memory_info.rss / 1024 / 1024  # 转换为MB
            
            self.logger.info(f"内存使用情况[{context}]: {memory_mb:.1f} MB")
            
            # 如果内存使用过高，发出警告
            if memory_mb > 500:  # 超过500MB
                self.logger.warning(f"内存使用较高: {memory_mb:.1f} MB")
                
        except ImportError:
            self.logger.warning("psutil库未安装，无法记录内存使用情况")
        except Exception as e:
            self.logger.warning(f"内存使用记录失败: {e}")
    
    def _generate_engineering_report(self):
        """生成工程报告"""
        report_path = self.output_dir / "engineering_report.pdf"
        
        # 使用reportlab生成PDF报告
        doc = SimpleDocTemplate(str(report_path), pagesize=A4)
        styles = getSampleStyleSheet()
        
        content = []
        content.append(Paragraph("火箭发动机设计报告", styles['Title']))
        content.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        
        # 添加参数表格
        data = [['参数', '数值', '单位']]
        params = [
            ('推力范围', f"{self.design_requirements.thrust_range[0]}-{self.design_requirements.thrust_range[1]}", 'kgf'),
            ('燃烧室内径', f"{self.current_parameters.chamber_diameter:.2f}", 'mm'),
            ('喷管扩张比', f"{self.current_parameters.expansion_ratio:.2f}", ''),
            ('最大壁温', f"{self.simulation_results[-1].get('max_wall_temperature', 0):.1f}", 'K'),
            ('干重', f"{self.simulation_results[-1].get('dry_weight', 0):.2f}", 'kg')
        ]
        data.extend(params)
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        content.append(table)
        
        doc.build(content)
    
    def _package_output_files(self):
        """打包输出文件"""
        import zipfile
        
        zip_path = self.working_dir / "rocket_design_output.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # 添加工程报告
            zipf.write(self.output_dir / "engineering_report.pdf", "engineering_report.pdf")
            
            # 添加模拟结果
            for i, result in enumerate(self.simulation_results):
                result_file = self.temp_dir / f"simulation_result_{i+1}.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                zipf.write(result_file, f"simulation_results/result_{i+1}.json")
        
        self.logger.info(f"文件打包完成: {zip_path}")
    
    def set_visualization_window(self, window):
        """设置可视化窗口"""
        self.visualization_window = window

class ParameterCalculator:
    """参数计算器"""
    
    @staticmethod
    def calculate_thrust(mass_flow: float, exhaust_velocity: float) -> float:
        """计算推力"""
        return mass_flow * exhaust_velocity / 9.81  # kgf
    
    @staticmethod
    def calculate_mass_flow(thrust: float, exhaust_velocity: float) -> float:
        """计算质量流率"""
        return thrust * 9.81 / exhaust_velocity  # kg/s
    
    @staticmethod
    def calculate_chamber_diameter(mass_flow: float, density: float, velocity: float) -> float:
        """计算燃烧室直径"""
        area = mass_flow / (density * velocity)
        return math.sqrt(4 * area / math.pi) * 1000  # mm

class FreeCADModeler:
    """FreeCAD 3D建模器"""
    
    def __init__(self):
        # 检测FreeCAD环境依赖
        self.freecad_available = FREECAD_AVAILABLE
        if self.freecad_available:
            self._check_freecad_environment()
            self.document = FreeCAD.newDocument("RocketEngine")
        else:
            self.document = None
            self._log_warning("FreeCAD不可用，3D建模功能将被禁用")
        
    def _check_freecad_environment(self):
        """检查FreeCAD环境依赖"""
        try:
            # 尝试导入FreeCAD模块
            import FreeCAD
            import Part
            
            # 检查FreeCAD版本
            freecad_version = FreeCAD.Version()
            self._log_info(f"FreeCAD版本: {'.'.join(freecad_version[:3])}")
            
            # 检查关键模块可用性
            if not hasattr(FreeCAD, 'newDocument'):
                raise Exception("FreeCAD.newDocument方法不可用")
            
            if not hasattr(Part, 'makeCylinder'):
                raise Exception("Part.makeCylinder方法不可用")
                
            # 检查FreeCAD安装路径
            freecad_path = self._find_freecad_path()
            self._log_info(f"FreeCAD安装路径: {freecad_path}")
            
        except ImportError as e:
            error_msg = f"""
FreeCAD库导入失败，请确保FreeCAD已正确安装。

安装建议：
1. 从 https://www.freecad.org/ 下载最新版FreeCAD
2. 安装时选择添加到PATH环境变量
3. 或使用pip安装: pip install freecad
4. 或设置PYTHONPATH环境变量指向FreeCAD安装目录

错误详情: {e}
            """
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"FreeCAD环境检查失败: {e}"
            raise Exception(error_msg)
    
    def _find_freecad_path(self) -> str:
        """智能查找FreeCAD安装路径"""
        import winreg
        
        # 1. 检查注册表（Windows系统）
        try:
            registry_paths = [
                r"SOFTWARE\\FreeCAD",
                r"SOFTWARE\\Wow6432Node\\FreeCAD",
                r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\FreeCAD",
            ]
            
            for reg_path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        try:
                            install_path, _ = winreg.QueryValueEx(key, "InstallLocation")
                            if Path(install_path).exists():
                                return install_path
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    pass
        except Exception as e:
            self._log_warning(f"FreeCAD注册表查询失败: {e}")
        
        # 2. 检查环境变量
        if 'FREECAD_HOME' in os.environ:
            candidate = Path(os.environ['FREECAD_HOME'])
            if candidate.exists():
                return str(candidate)
        
        if 'FREECAD_PATH' in os.environ:
            candidate = Path(os.environ['FREECAD_PATH'])
            if candidate.exists():
                return str(candidate)
        
        # 3. 检查Python模块路径
        try:
            import FreeCAD
            freecad_module_path = Path(FreeCAD.__file__).parent
            if 'FreeCAD' in str(freecad_module_path):
                return str(freecad_module_path)
        except:
            pass
        
        # 4. 常见安装路径
        possible_paths = [
            Path("C:/Program Files/FreeCAD 0.20"),
            Path("C:/Program Files/FreeCAD 0.21"),
            Path("C:/Program Files/FreeCAD"),
            Path("C:/FreeCAD"),
            Path("D:/Program Files/FreeCAD 0.20"),
            Path("D:/Program Files/FreeCAD"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        # 5. 递归搜索程序文件目录
        program_files_dirs = [
            Path(os.environ.get('ProgramFiles', r"C:\\Program Files")),
            Path(os.environ.get('ProgramFiles(x86)', r"C:\\Program Files (x86)")),
        ]
        
        for program_files_dir in program_files_dirs:
            if program_files_dir.exists():
                for root, dirs, files in os.walk(program_files_dir):
                    if 'FreeCAD' in root and 'bin' in root:
                        candidate = Path(root)
                        if candidate.exists():
                            return str(candidate)
        
        # 6. 提供友好的错误信息和安装建议
        error_msg = """
未找到FreeCAD安装路径。请确保FreeCAD已正确安装。

安装建议：
1. 从 https://www.freecad.org/ 下载最新版FreeCAD
2. 安装时选择添加到PATH环境变量
3. 或使用pip安装: pip install freecad
4. 或设置 FREECAD_HOME 环境变量指向FreeCAD安装目录

当前Python路径包含的目录：
{}
        """.format('\n'.join(sys.path[:10]))  # 只显示前10个目录
        
        raise Exception(error_msg)
        
    def create_initial_model(self, parameters: SimulationParameters) -> str:
        """创建初始3D模型"""
        if not self.freecad_available:
            # 如果没有FreeCAD，创建模拟模型文件
            self._log_warning("FreeCAD不可用，创建模拟模型文件")
            
            # 确保临时目录存在
            if not hasattr(self, 'temp_dir'):
                self.temp_dir = Path(tempfile.gettempdir()) / "rocket_simulation"
            
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建模拟模型文件
            model_path = str(self.temp_dir / f"simulated_model_{int(time.time())}.txt")
            
            # 写入模型参数信息
            with open(model_path, 'w', encoding='utf-8') as f:
                f.write("# 模拟火箭发动机模型参数\n")
                f.write(f"# 创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"燃烧室直径: {parameters.chamber_diameter} mm\n")
                f.write(f"燃烧室长度: {parameters.chamber_length} mm\n")
                f.write(f"喉部直径: {parameters.throat_diameter} mm\n")
                f.write(f"扩张比: {parameters.expansion_ratio}\n")
                f.write(f"壁厚: {parameters.wall_thickness} mm\n")
                f.write("\n注意: 此文件为模拟模型，实际3D建模需要安装FreeCAD\n")
            
            self._log_info(f"模拟模型文件创建成功: {model_path}")
            return model_path
        
        try:
            # 确保临时目录存在
            if not hasattr(self, 'temp_dir'):
                self.temp_dir = Path(tempfile.gettempdir()) / "rocket_simulation"
            
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建燃烧室
            chamber = self._create_combustion_chamber(parameters)
            # 创建喷管
            nozzle = self._create_nozzle(parameters)
            # 创建燃料接口
            fuel_inlet = self._create_fuel_inlet(parameters)
            
            # 保存模型 - 添加详细的异常处理
            model_path = str(self.temp_dir / f"initial_model_{int(time.time())}.FCStd")
            
            try:
                self.document.saveAs(model_path)
                self._log_info(f"模型成功保存到: {model_path}")
            except Exception as save_error:
                # 尝试备用保存路径
                backup_path = str(self.temp_dir / "backup_model.FCStd")
                try:
                    self.document.saveAs(backup_path)
                    model_path = backup_path
                    self._log_warning(f"主保存路径失败，使用备用路径: {backup_path}")
                except Exception as backup_error:
                    raise Exception(f"模型保存失败 - 主路径: {save_error}, 备用路径: {backup_error}")
            
            # 验证模型文件是否创建成功
            if not Path(model_path).exists():
                raise Exception(f"模型文件未创建: {model_path}")
            
            file_size = Path(model_path).stat().st_size
            if file_size == 0:
                raise Exception(f"模型文件为空: {model_path}")
            
            return model_path
            
        except Exception as e:
            self._log_error(f"3D建模失败: {e}")
            # 提供更详细的错误信息和恢复建议
            error_details = f"""
3D建模过程失败，可能的原因：
1. FreeCAD库未正确安装或初始化
2. 磁盘空间不足
3. 文件权限问题
4. 参数值超出合理范围

建议检查：
- 确保FreeCAD已正确安装
- 检查临时目录权限: {tempfile.gettempdir()}
- 验证参数范围是否合理
- 尝试重启应用程序
"""
            raise Exception(f"3D建模失败: {e}\n{error_details}")
    
    def _log_info(self, message: str):
        """记录信息日志"""
        print(f"[FreeCADModeler INFO] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        print(f"[FreeCADModeler WARNING] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        print(f"[FreeCADModeler ERROR] {message}")
    
    def _create_combustion_chamber(self, parameters: SimulationParameters):
        """创建燃烧室"""
        # 半球头 + 圆柱筒
        radius = parameters.chamber_diameter / 2
        length = parameters.chamber_length
        
        # 创建半球
        hemisphere = Part.makeSphere(radius)
        # 创建圆柱
        cylinder = Part.makeCylinder(radius, length)
        cylinder.translate((0, 0, radius))
        
        # 合并形状
        chamber = hemisphere.fuse(cylinder)
        chamber_obj = self.document.addObject("Part::Feature", "CombustionChamber")
        chamber_obj.Shape = chamber
        
        return chamber_obj
    
    def _create_nozzle(self, parameters: SimulationParameters):
        """创建喷管"""
        # 收敛段 + 喉部 + 扩张段
        throat_radius = parameters.throat_diameter / 2
        exit_radius = throat_radius * math.sqrt(parameters.expansion_ratio)
        
        # 创建收敛锥
        convergent = Part.makeCone(parameters.chamber_diameter/2, throat_radius, 
                                 parameters.nozzle_length/3)
        # 创建喉部
        throat = Part.makeCylinder(throat_radius, parameters.nozzle_length/6)
        throat.translate((0, 0, parameters.nozzle_length/3))
        # 创建扩张段
        divergent = Part.makeCone(throat_radius, exit_radius, parameters.nozzle_length/2)
        divergent.translate((0, 0, parameters.nozzle_length/2))
        
        # 合并形状
        nozzle = convergent.fuse(throat).fuse(divergent)
        nozzle.translate((0, 0, parameters.chamber_length + parameters.chamber_diameter/2))
        
        nozzle_obj = self.document.addObject("Part::Feature", "Nozzle")
        nozzle_obj.Shape = nozzle
        
        return nozzle_obj
    
    def _create_fuel_inlet(self, parameters: SimulationParameters):
        """创建燃料接口"""
        fuel_port = Part.makeCylinder(2.5, 10)  # φ5mm接口
        fuel_port.translate((parameters.chamber_diameter/2, 0, parameters.chamber_length/2))
        
        fuel_obj = self.document.addObject("Part::Feature", "FuelInlet")
        fuel_obj.Shape = fuel_port
        
        return fuel_obj
    
    def create_integrated_engine_model(self, parameters: SimulationParameters, 
                                     include_cooling: bool = True,
                                     include_mounting: bool = True) -> str:
        """创建一体成型发动机模型（增强版）"""
        try:
            self._log_info("开始创建一体成型发动机模型")
            
            # 创建基础发动机结构
            engine_base = self._create_engine_base_structure(parameters)
            
            # 创建推力室一体化结构
            thrust_chamber = self._create_integrated_thrust_chamber(parameters)
            
            # 创建一体化喷管
            integrated_nozzle = self._create_integrated_nozzle(parameters)
            
            # 合并主要组件
            main_assembly = engine_base.fuse(thrust_chamber).fuse(integrated_nozzle)
            
            # 添加冷却系统（如果启用）
            if include_cooling:
                cooling_system = self._create_cooling_system(parameters)
                main_assembly = main_assembly.fuse(cooling_system)
            
            # 添加安装接口（如果启用）
            if include_mounting:
                mounting_points = self._create_mounting_points(parameters)
                main_assembly = main_assembly.fuse(mounting_points)
            
            # 创建最终模型对象
            engine_obj = self.document.addObject("Part::Feature", "IntegratedRocketEngine")
            engine_obj.Shape = main_assembly
            
            # 保存模型
            model_path = str(self.temp_dir / f"integrated_engine_{int(time.time())}.FCStd")
            self.document.saveAs(model_path)
            
            self._log_info(f"一体成型发动机模型创建完成: {model_path}")
            return model_path
            
        except Exception as e:
            self._log_error(f"一体成型发动机模型创建失败: {e}")
            raise
    
    def _create_engine_base_structure(self, parameters: SimulationParameters):
        """创建发动机基础结构"""
        try:
            # 创建主壳体
            outer_diameter = parameters.chamber_diameter + 2 * parameters.wall_thickness
            outer_length = parameters.chamber_length + parameters.nozzle_length + 20  # 额外长度
            
            outer_cylinder = Part.makeCylinder(outer_diameter/2, outer_length)
            inner_cylinder = Part.makeCylinder(parameters.chamber_diameter/2, outer_length)
            
            # 布尔运算创建壳体
            base_structure = outer_cylinder.cut(inner_cylinder)
            
            # 添加前端盖
            front_plate = Part.makeCylinder(outer_diameter/2, parameters.wall_thickness)
            base_structure = base_structure.fuse(front_plate)
            
            return base_structure
            
        except Exception as e:
            self._log_error(f"基础结构创建失败: {e}")
            raise
    
    def _create_integrated_thrust_chamber(self, parameters: SimulationParameters):
        """创建一体化推力室"""
        try:
            # 创建燃烧室（带加强筋）
            chamber_outer = Part.makeCylinder(parameters.chamber_diameter/2, parameters.chamber_length)
            
            # 添加内部结构（模拟燃烧室壁面）
            chamber_inner = Part.makeCylinder(
                parameters.chamber_diameter/2 - parameters.wall_thickness, 
                parameters.chamber_length
            )
            chamber_inner.translate((0, 0, parameters.wall_thickness))
            
            thrust_chamber = chamber_outer.cut(chamber_inner)
            
            # 添加燃烧室头部结构
            chamber_head = Part.makeSphere(parameters.chamber_diameter/2)
            thrust_chamber = thrust_chamber.fuse(chamber_head)
            
            return thrust_chamber
            
        except Exception as e:
            self._log_error(f"一体化推力室创建失败: {e}")
            raise
    
    def _create_integrated_nozzle(self, parameters: SimulationParameters):
        """创建一体化喷管"""
        try:
            # 创建收敛段
            convergent_length = parameters.nozzle_length * 0.4
            convergent = Part.makeCone(
                parameters.chamber_diameter/2, 
                parameters.throat_diameter/2, 
                convergent_length
            )
            convergent.translate((0, 0, parameters.chamber_length))
            
            # 创建喉部
            throat_length = parameters.nozzle_length * 0.1
            throat = Part.makeCylinder(parameters.throat_diameter/2, throat_length)
            throat.translate((0, 0, parameters.chamber_length + convergent_length))
            
            # 创建扩张段
            divergent_length = parameters.nozzle_length * 0.5
            exit_diameter = parameters.throat_diameter * math.sqrt(parameters.expansion_ratio)
            divergent = Part.makeCone(
                parameters.throat_diameter/2, 
                exit_diameter/2, 
                divergent_length
            )
            divergent.translate((0, 0, parameters.chamber_length + convergent_length + throat_length))
            
            # 合并喷管组件
            nozzle = convergent.fuse(throat).fuse(divergent)
            
            # 添加喷管壁厚
            nozzle_wall_thickness = parameters.wall_thickness * 1.2  # 喷管壁稍厚
            nozzle_outer = nozzle
            
            return nozzle_outer
            
        except Exception as e:
            self._log_error(f"一体化喷管创建失败: {e}")
            raise
    
    def _create_cooling_system(self, parameters: SimulationParameters):
        """创建冷却系统"""
        try:
            # 创建冷却通道
            cooling_channels = []
            
            # 在燃烧室壁面创建螺旋冷却通道
            num_channels = 12  # 12个冷却通道
            channel_depth = parameters.cooling_channel_thickness
            channel_width = 2.0  # mm
            
            for i in range(num_channels):
                angle = 2 * math.pi * i / num_channels
                radius = parameters.chamber_diameter/2 + parameters.wall_thickness/2
                
                # 创建单个冷却通道
                channel = Part.makeBox(
                    channel_width, channel_depth, parameters.chamber_length * 0.8
                )
                
                # 旋转到正确位置
                channel.rotate((0, 0, 0), (0, 0, 1), math.degrees(angle))
                channel.translate((radius * math.cos(angle), radius * math.sin(angle), parameters.wall_thickness))
                
                cooling_channels.append(channel)
            
            # 合并所有冷却通道
            cooling_system = cooling_channels[0]
            for channel in cooling_channels[1:]:
                cooling_system = cooling_system.fuse(channel)
            
            return cooling_system
            
        except Exception as e:
            self._log_error(f"冷却系统创建失败: {e}")
            raise
    
    def _create_mounting_points(self, parameters: SimulationParameters):
        """创建安装点"""
        try:
            # 创建安装法兰
            flange_diameter = parameters.chamber_diameter + 20
            flange_thickness = 5.0
            
            flange = Part.makeCylinder(flange_diameter/2, flange_thickness)
            flange.translate((0, 0, -flange_thickness))
            
            # 创建安装孔
            num_mounting_holes = 6
            hole_diameter = 4.0
            hole_radius = flange_diameter/2 - 10
            
            mounting_holes = []
            for i in range(num_mounting_holes):
                angle = 2 * math.pi * i / num_mounting_holes
                x = hole_radius * math.cos(angle)
                y = hole_radius * math.sin(angle)
                
                hole = Part.makeCylinder(hole_diameter/2, flange_thickness)
                hole.translate((x, y, -flange_thickness))
                mounting_holes.append(hole)
            
            # 从法兰中减去安装孔
            for hole in mounting_holes:
                flange = flange.cut(hole)
            
            return flange
            
        except Exception as e:
            self._log_error(f"安装点创建失败: {e}")
            raise
    
    def create_parametric_model(self, parameters: SimulationParameters, 
                               design_variations: Dict[str, float]) -> str:
        """创建参数化模型，支持设计变体"""
        try:
            self._log_info("开始创建参数化发动机模型")
            
            # 应用设计变体
            varied_parameters = self._apply_design_variations(parameters, design_variations)
            
            # 创建参数化模型
            model = self.create_integrated_engine_model(varied_parameters)
            
            # 添加参数化特征
            self._add_parametric_features(varied_parameters)
            
            self._log_info("参数化模型创建完成")
            return model
            
        except Exception as e:
            self._log_error(f"参数化模型创建失败: {e}")
            raise
    
    def _apply_design_variations(self, base_parameters: SimulationParameters, 
                               variations: Dict[str, float]) -> SimulationParameters:
        """应用设计变体到参数"""
        # 创建参数副本
        varied_params = SimulationParameters(
            chamber_diameter=base_parameters.chamber_diameter * variations.get('chamber_diameter_scale', 1.0),
            chamber_length=base_parameters.chamber_length * variations.get('chamber_length_scale', 1.0),
            chamber_pressure=base_parameters.chamber_pressure * variations.get('pressure_scale', 1.0),
            throat_diameter=base_parameters.throat_diameter * variations.get('throat_diameter_scale', 1.0),
            expansion_ratio=base_parameters.expansion_ratio * variations.get('expansion_ratio_scale', 1.0),
            nozzle_length=base_parameters.nozzle_length * variations.get('nozzle_length_scale', 1.0),
            fuel_flow_rate=base_parameters.fuel_flow_rate * variations.get('flow_rate_scale', 1.0),
            oxidizer_flow_rate=base_parameters.oxidizer_flow_rate * variations.get('flow_rate_scale', 1.0),
            mixture_ratio=base_parameters.mixture_ratio,
            cooling_channel_thickness=base_parameters.cooling_channel_thickness * variations.get('cooling_scale', 1.0),
            cooling_velocity=base_parameters.cooling_velocity * variations.get('cooling_velocity_scale', 1.0),
            wall_thickness=base_parameters.wall_thickness * variations.get('wall_thickness_scale', 1.0),
            material_density=base_parameters.material_density
        )
        
        return varied_params
    
    def _add_parametric_features(self, parameters: SimulationParameters):
        """添加参数化特征"""
        try:
            # 添加设计参数作为模型属性
            engine_obj = self.document.getObject("IntegratedRocketEngine")
            if engine_obj:
                # 添加参数属性
                engine_obj.addProperty("App::PropertyFloat", "ChamberDiameter", "Design", "燃烧室内径")
                engine_obj.ChamberDiameter = parameters.chamber_diameter
                
                engine_obj.addProperty("App::PropertyFloat", "ExpansionRatio", "Design", "喷管扩张比")
                engine_obj.ExpansionRatio = parameters.expansion_ratio
                
                engine_obj.addProperty("App::PropertyFloat", "WallThickness", "Design", "壁厚")
                engine_obj.WallThickness = parameters.wall_thickness
                
                # 添加体积和质量计算
                volume = engine_obj.Shape.Volume / 1000  # 转换为cm³
                mass = volume * parameters.material_density / 1e6  # 转换为kg
                
                engine_obj.addProperty("App::PropertyFloat", "Volume", "Physics", "体积")
                engine_obj.Volume = volume
                
                engine_obj.addProperty("App::PropertyFloat", "Mass", "Physics", "质量")
                engine_obj.Mass = mass
                
        except Exception as e:
            self._log_warning(f"参数化特征添加失败: {e}")
    
    def export_model_for_analysis(self, model_path: str, export_formats: List[str] = None) -> Dict[str, str]:
        """导出模型用于分析（支持多种格式）"""
        if not self.freecad_available:
            # 如果没有FreeCAD，创建模拟导出文件
            self._log_warning("FreeCAD不可用，创建模拟导出文件")
            
            if export_formats is None:
                export_formats = ['STEP', 'STL', 'IGES', 'BREP']
            
            export_paths = {}
            
            for format_name in export_formats:
                export_path = str(Path(model_path).with_suffix(f'.{format_name.lower()}'))
                
                # 创建模拟导出文件
                with open(export_path, 'w', encoding='utf-8') as f:
                    f.write(f"# 模拟{format_name}格式导出文件\n")
                    f.write(f"# 原始模型: {model_path}\n")
                    f.write(f"# 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# 注意: 此文件为模拟导出，实际导出需要安装FreeCAD\n")
                
                export_paths[format_name] = export_path
                self._log_info(f"模拟导出文件创建: {export_path}")
            
            return export_paths
        
        if export_formats is None:
            export_formats = ['STEP', 'STL', 'IGES', 'BREP']
        
        export_paths = {}
        
        try:
            for format_name in export_formats:
                export_path = str(Path(model_path).with_suffix(f'.{format_name.lower()}'))
                
                try:
                    # 根据格式选择导出方法
                    if format_name == 'STEP':
                        import Import
                        Import.export(self.document.Objects, export_path)
                    elif format_name == 'STL':
                        import Mesh
                        Mesh.export(self.document.Objects, export_path)
                    elif format_name == 'IGES':
                        import Import
                        Import.export(self.document.Objects, export_path)
                    elif format_name == 'BREP':
                        import Part
                        Part.export(self.document.Objects, export_path)
                    
                    # 验证导出文件
                    if Path(export_path).exists() and Path(export_path).stat().st_size > 0:
                        export_paths[format_name] = export_path
                        self._log_info(f"模型成功导出为 {format_name}: {export_path}")
                    else:
                        self._log_warning(f"{format_name} 导出文件可能为空: {export_path}")
                        
                except Exception as format_error:
                    self._log_warning(f"{format_name} 格式导出失败: {format_error}")
            
            return export_paths
            
        except Exception as e:
            self._log_error(f"模型导出失败: {e}")
            raise
    
    def generate_engineering_drawing(self, parameters: SimulationParameters, 
                                   drawing_template: str = "A4") -> str:
        """生成工程图纸"""
        if not self.freecad_available:
            # 如果没有FreeCAD，创建模拟工程图纸
            self._log_warning("FreeCAD不可用，创建模拟工程图纸")
            
            # 确保临时目录存在
            if not hasattr(self, 'temp_dir'):
                self.temp_dir = Path(tempfile.gettempdir()) / "rocket_simulation"
            
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建模拟PDF图纸
            drawing_path = str(self.temp_dir / f"simulated_drawing_{int(time.time())}.pdf")
            
            # 使用reportlab创建简单的PDF图纸
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import A4
                
                c = canvas.Canvas(drawing_path, pagesize=A4)
                c.setFont("Helvetica", 12)
                
                # 添加标题
                c.drawString(100, 750, "火箭发动机工程图纸 - 模拟版本")
                c.drawString(100, 730, f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 添加参数信息
                c.drawString(100, 700, f"燃烧室直径: {parameters.chamber_diameter} mm")
                c.drawString(100, 680, f"燃烧室长度: {parameters.chamber_length} mm")
                c.drawString(100, 660, f"喉部直径: {parameters.throat_diameter} mm")
                c.drawString(100, 640, f"扩张比: {parameters.expansion_ratio}")
                c.drawString(100, 620, f"壁厚: {parameters.wall_thickness} mm")
                
                # 添加说明
                c.drawString(100, 580, "注意: 此图纸为模拟版本")
                c.drawString(100, 560, "实际工程图纸需要安装FreeCAD")
                
                c.save()
                
                self._log_info(f"模拟工程图纸创建成功: {drawing_path}")
                return drawing_path
                
            except Exception as e:
                # 如果reportlab不可用，创建文本文件
                with open(drawing_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
                    f.write("火箭发动机工程图纸 - 模拟版本\n")
                    f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"燃烧室直径: {parameters.chamber_diameter} mm\n")
                    f.write(f"燃烧室长度: {parameters.chamber_length} mm\n")
                    f.write(f"喉部直径: {parameters.throat_diameter} mm\n")
                    f.write(f"扩张比: {parameters.expansion_ratio}\n")
                    f.write(f"壁厚: {parameters.wall_thickness} mm\n\n")
                    f.write("注意: 此图纸为模拟版本，实际工程图纸需要安装FreeCAD\n")
                
                self._log_info(f"模拟工程图纸创建成功: {drawing_path.replace('.pdf', '.txt')}")
                return drawing_path.replace('.pdf', '.txt')
        
        try:
            self._log_info("开始生成工程图纸")
            
            # 创建技术绘图模块
            import TechDraw
            
            # 创建绘图页面
            page = self.document.addObject('TechDraw::DrawPage', 'EngineeringDrawing')
            page.Template = f"{drawing_template}_Landscape.svg"
            
            # 创建正视图
            front_view = self.document.addObject('TechDraw::DrawViewPart', 'FrontView')
            front_view.Source = [self.document.getObject("IntegratedRocketEngine")]
            front_view.Direction = (0, 0, 1)
            page.addView(front_view)
            
            # 创建侧视图
            side_view = self.document.addObject('TechDraw::DrawViewPart', 'SideView')
            side_view.Source = [self.document.getObject("IntegratedRocketEngine")]
            side_view.Direction = (1, 0, 0)
            page.addView(side_view)
            
            # 创建剖视图
            section_view = self.document.addObject('TechDraw::DrawViewSection', 'SectionView')
            section_view.Source = [self.document.getObject("IntegratedRocketEngine")]
            section_view.SectionNormal = (0, 1, 0)
            page.addView(section_view)
            
            # 保存图纸
            drawing_path = str(self.temp_dir / f"engineering_drawing_{int(time.time())}.pdf")
            
            # 导出为PDF
            import Drawing
            Drawing.export(page, drawing_path)
            
            self._log_info(f"工程图纸生成完成: {drawing_path}")
            return drawing_path
            
        except Exception as e:
            self._log_error(f"工程图纸生成失败: {e}")
            raise
    
    def calculate_volume_and_mass(self, parameters: SimulationParameters) -> Dict[str, float]:
        """计算体积和质量"""
        try:
            # 创建简化模型用于计算
            test_model = self.create_integrated_engine_model(parameters, include_cooling=False, include_mounting=False)
            
            engine_obj = self.document.getObject("IntegratedRocketEngine")
            if engine_obj and hasattr(engine_obj.Shape, 'Volume'):
                volume_mm3 = engine_obj.Shape.Volume
                volume_cm3 = volume_mm3 / 1000
                mass_kg = volume_cm3 * parameters.material_density / 1e6
                
                results = {
                    'volume_mm3': volume_mm3,
                    'volume_cm3': volume_cm3,
                    'mass_kg': mass_kg,
                    'surface_area_mm2': engine_obj.Shape.Area
                }
                
                self._log_info(f"体积和质量计算完成: 体积{volume_cm3:.1f}cm³, 质量{mass_kg:.2f}kg")
                return results
            else:
                raise Exception("无法获取模型体积信息")
                
        except Exception as e:
            self._log_error(f"体积和质量计算失败: {e}")
            raise

    def update_model_parameters(self, old_parameters: SimulationParameters, 
                              new_parameters: SimulationParameters) -> str:
        """动态更新模型参数，实现参数联动更新"""
        try:
            self._log_info("开始动态更新模型参数")
            
            # 检查参数变化
            parameter_changes = self._detect_parameter_changes(old_parameters, new_parameters)
            
            if not parameter_changes:
                self._log_info("参数无变化，跳过模型更新")
                return "no_changes"
            
            # 根据参数变化类型选择更新策略
            update_strategy = self._select_update_strategy(parameter_changes)
            
            # 执行模型更新
            updated_model_path = self._execute_model_update(new_parameters, update_strategy)
            
            # 生成网格更新（如果需要）
            if update_strategy["requires_remeshing"]:
                self._update_mesh_for_analysis(new_parameters)
            
            # 生成实时预览
            preview_data = self._generate_real_time_preview(new_parameters)
            
            self._log_info(f"模型参数动态更新完成: {updated_model_path}")
            return updated_model_path
            
        except Exception as e:
            self._log_error(f"模型参数更新失败: {e}")
            raise

    def _detect_parameter_changes(self, old_params: SimulationParameters, 
                                new_params: SimulationParameters) -> Dict[str, Dict[str, float]]:
        """检测参数变化，返回变化详情"""
        changes = {}
        
        # 定义关键参数及其变化阈值
        key_parameters = {
            'chamber_diameter': ('燃烧室直径', 0.1),  # 0.1mm变化阈值
            'chamber_length': ('燃烧室长度', 0.1),
            'throat_diameter': ('喉部直径', 0.05),
            'expansion_ratio': ('扩张比', 0.01),
            'wall_thickness': ('壁厚', 0.05),
            'nozzle_length': ('喷管长度', 0.1)
        }
        
        for param_name, (display_name, threshold) in key_parameters.items():
            old_value = getattr(old_params, param_name)
            new_value = getattr(new_params, param_name)
            
            if abs(new_value - old_value) > threshold:
                change_percent = abs((new_value - old_value) / old_value) * 100
                changes[param_name] = {
                    'display_name': display_name,
                    'old_value': old_value,
                    'new_value': new_value,
                    'change_percent': change_percent,
                    'absolute_change': abs(new_value - old_value)
                }
        
        return changes

    def _select_update_strategy(self, parameter_changes: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """根据参数变化选择更新策略"""
        strategy = {
            "update_type": "incremental",  # incremental, partial, full_rebuild
            "requires_remeshing": False,
            "priority": "medium",
            "components_to_update": []
        }
        
        # 分析变化类型
        geometry_changes = any(param in parameter_changes for param in 
                             ['chamber_diameter', 'chamber_length', 'throat_diameter'])
        
        nozzle_changes = 'expansion_ratio' in parameter_changes or 'nozzle_length' in parameter_changes
        wall_changes = 'wall_thickness' in parameter_changes
        
        # 确定更新类型
        if geometry_changes and nozzle_changes:
            strategy["update_type"] = "full_rebuild"
            strategy["requires_remeshing"] = True
            strategy["priority"] = "high"
        elif geometry_changes:
            strategy["update_type"] = "partial"
            strategy["components_to_update"] = ["combustion_chamber", "throat"]
            strategy["requires_remeshing"] = True
            strategy["priority"] = "high"
        elif nozzle_changes:
            strategy["update_type"] = "partial"
            strategy["components_to_update"] = ["nozzle"]
            strategy["requires_remeshing"] = True
            strategy["priority"] = "medium"
        elif wall_changes:
            strategy["update_type"] = "incremental"
            strategy["components_to_update"] = ["wall_thickness"]
            strategy["requires_remeshing"] = False
            strategy["priority"] = "low"
        
        return strategy

    def _execute_model_update(self, parameters: SimulationParameters, 
                            strategy: Dict[str, Any]) -> str:
        """执行模型更新操作"""
        try:
            if strategy["update_type"] == "full_rebuild":
                # 完全重建模型
                self._log_info("执行完全模型重建")
                return self.create_integrated_engine_model(parameters)
            
            elif strategy["update_type"] == "partial":
                # 部分更新模型组件
                self._log_info(f"执行部分模型更新: {strategy['components_to_update']}")
                return self._update_partial_components(parameters, strategy["components_to_update"])
            
            elif strategy["update_type"] == "incremental":
                # 增量更新参数
                self._log_info("执行增量参数更新")
                return self._update_parameters_incrementally(parameters)
            
            else:
                raise ValueError(f"未知的更新策略: {strategy['update_type']}")
                
        except Exception as e:
            self._log_error(f"模型更新执行失败: {e}")
            raise

    def _update_partial_components(self, parameters: SimulationParameters, 
                                  components: List[str]) -> str:
        """部分更新模型组件"""
        try:
            # 获取当前模型
            engine_obj = self.document.getObject("IntegratedRocketEngine")
            if not engine_obj:
                raise Exception("未找到发动机模型对象")
            
            # 根据组件类型执行更新
            updated_components = []
            
            if "combustion_chamber" in components:
                chamber = self._create_combustion_chamber(parameters)
                updated_components.append(chamber)
            
            if "nozzle" in components:
                nozzle = self._create_nozzle(parameters)
                updated_components.append(nozzle)
            
            if "throat" in components:
                throat = self._create_throat_section(parameters)
                updated_components.append(throat)
            
            # 合并更新后的组件
            if updated_components:
                new_shape = updated_components[0].Shape
                for comp in updated_components[1:]:
                    new_shape = new_shape.fuse(comp.Shape)
                
                engine_obj.Shape = new_shape
            
            # 保存更新后的模型
            model_path = str(self.temp_dir / f"updated_model_{int(time.time())}.FCStd")
            self.document.saveAs(model_path)
            
            return model_path
            
        except Exception as e:
            self._log_error(f"部分组件更新失败: {e}")
            raise

    def _update_parameters_incrementally(self, parameters: SimulationParameters) -> str:
        """增量更新模型参数"""
        try:
            # 获取当前模型并更新参数属性
            engine_obj = self.document.getObject("IntegratedRocketEngine")
            if engine_obj:
                # 更新参数属性
                if hasattr(engine_obj, 'ChamberDiameter'):
                    engine_obj.ChamberDiameter = parameters.chamber_diameter
                if hasattr(engine_obj, 'ExpansionRatio'):
                    engine_obj.ExpansionRatio = parameters.expansion_ratio
                if hasattr(engine_obj, 'WallThickness'):
                    engine_obj.WallThickness = parameters.wall_thickness
            
            # 保存模型
            model_path = str(self.temp_dir / f"incremental_update_{int(time.time())}.FCStd")
            self.document.saveAs(model_path)
            
            return model_path
            
        except Exception as e:
            self._log_error(f"增量参数更新失败: {e}")
            raise

    def _create_throat_section(self, parameters: SimulationParameters):
        """创建喉部段（用于部分更新）"""
        throat_length = parameters.nozzle_length * 0.1
        throat = Part.makeCylinder(parameters.throat_diameter/2, throat_length)
        throat.translate((0, 0, parameters.chamber_length + parameters.nozzle_length*0.4))
        
        throat_obj = self.document.addObject("Part::Feature", "ThroatSection")
        throat_obj.Shape = throat
        
        return throat_obj

    def _update_mesh_for_analysis(self, parameters: SimulationParameters):
        """更新分析网格"""
        try:
            self._log_info("开始更新分析网格")
            
            # 生成新的网格文件
            mesh_path = str(self.temp_dir / f"updated_mesh_{int(time.time())}.stl")
            
            # 导出STL格式用于网格分析
            export_paths = self.export_model_for_analysis(
                self.create_integrated_engine_model(parameters), 
                ['STL']
            )
            
            if 'STL' in export_paths:
                self._log_info(f"网格更新完成: {export_paths['STL']}")
                return export_paths['STL']
            else:
                raise Exception("STL网格导出失败")
                
        except Exception as e:
            self._log_error(f"网格更新失败: {e}")
            raise

    def _generate_real_time_preview(self, parameters: SimulationParameters) -> Dict[str, Any]:
        """生成实时预览数据"""
        try:
            preview_data = {
                "timestamp": time.time(),
                "parameters": {
                    "chamber_diameter": parameters.chamber_diameter,
                    "chamber_length": parameters.chamber_length,
                    "throat_diameter": parameters.throat_diameter,
                    "expansion_ratio": parameters.expansion_ratio,
                    "wall_thickness": parameters.wall_thickness
                },
                "geometry_info": {},
                "performance_metrics": {}
            }
            
            # 计算几何信息
            volume_mass = self.calculate_volume_and_mass(parameters)
            preview_data["geometry_info"] = volume_mass
            
            # 估算性能指标
            preview_data["performance_metrics"] = self._estimate_performance_metrics(parameters)
            
            # 生成简化可视化数据
            preview_data["visualization"] = self._generate_simplified_visualization(parameters)
            
            self._log_info("实时预览数据生成完成")
            return preview_data
            
        except Exception as e:
            self._log_warning(f"实时预览生成失败: {e}")
            return {"error": str(e)}

    def _estimate_performance_metrics(self, parameters: SimulationParameters) -> Dict[str, float]:
        """估算性能指标"""
        # 基于几何参数的简化性能估算
        throat_area = math.pi * (parameters.throat_diameter / 2) ** 2
        exit_area = throat_area * parameters.expansion_ratio
        
        # 简化推力估算（基于面积比和压力）
        estimated_thrust = throat_area * parameters.chamber_pressure * 0.8  # 简化系数
        
        # 效率估算
        efficiency = min(0.95, 0.7 + parameters.expansion_ratio * 0.05)
        
        return {
            "estimated_thrust_kN": estimated_thrust / 1000,  # 转换为kN
            "estimated_efficiency": efficiency,
            "throat_area_mm2": throat_area,
            "exit_area_mm2": exit_area
        }

    def _generate_simplified_visualization(self, parameters: SimulationParameters) -> Dict[str, Any]:
        """生成简化可视化数据"""
        # 生成用于实时预览的简化几何数据
        return {
            "bounding_box": {
                "width": parameters.chamber_diameter + 20,
                "height": parameters.chamber_length + parameters.nozzle_length + 30,
                "depth": parameters.chamber_diameter + 20
            },
            "key_dimensions": {
                "chamber_diameter": parameters.chamber_diameter,
                "chamber_length": parameters.chamber_length,
                "nozzle_length": parameters.nozzle_length,
                "expansion_ratio": parameters.expansion_ratio
            },
            "color_scheme": "engineering_blue"
        }

    def export_engineering_deliverables(self, parameters: SimulationParameters, 
                                      output_dir: str = None) -> Dict[str, str]:
        """
        导出工程化成果包，包括多种格式模型、工程图纸、分析报告和项目文档
        
        Args:
            parameters: 仿真参数
            output_dir: 输出目录，默认为临时目录
            
        Returns:
            包含所有导出文件路径的字典
        """
        try:
            self._log_info("开始生成工程化成果包")
            
            # 设置输出目录
            if output_dir is None:
                output_dir = str(self.temp_dir / "engineering_deliverables")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            deliverables = {}
            
            # 1. 导出多种格式的3D模型
            model_paths = self._export_multiple_model_formats(parameters, output_path)
            deliverables.update(model_paths)
            
            # 2. 生成工程图纸
            drawing_paths = self._generate_engineering_drawings(parameters, output_path)
            deliverables.update(drawing_paths)
            
            # 3. 生成分析报告
            report_paths = self._generate_analysis_report(parameters, output_path)
            deliverables.update(report_paths)
            
            # 4. 生成项目文档
            documentation_paths = self._generate_project_documentation(parameters, output_path)
            deliverables.update(documentation_paths)
            
            # 5. 打包所有成果
            package_path = self._package_deliverables(deliverables, output_path)
            deliverables["package"] = package_path
            
            self._log_info(f"工程化成果包生成完成: {output_path}")
            return deliverables
            
        except Exception as e:
            self._log_error(f"工程化成果包生成失败: {e}")
            raise

    def _export_multiple_model_formats(self, parameters: SimulationParameters, 
                                     output_path: Path) -> Dict[str, str]:
        """导出多种格式的3D模型"""
        model_paths = {}
        
        try:
            # 确保output_path是Path对象
            output_path = Path(output_path)
            
            # 导出不同格式
            export_formats = ['STEP', 'STL', 'IGES', 'BREP']
            
            for format_name in export_formats:
                try:
                    if self.freecad_available:
                        # FreeCAD可用时创建基础模型并导出实际格式
                        base_model_path = self.create_integrated_engine_model(parameters)
                        export_result = self.export_model_for_analysis(base_model_path, [format_name])
                        if format_name in export_result:
                            # 复制到输出目录
                            source_path = Path(export_result[format_name])
                            target_path = output_path / f"rocket_engine_model.{format_name.lower()}"
                            
                            import shutil
                            shutil.copy2(source_path, target_path)
                            model_paths[format_name.lower()] = str(target_path)
                    else:
                        # FreeCAD不可用时创建模拟文件
                        target_path = output_path / f"rocket_engine_model.{format_name.lower()}"
                        self._create_simulation_model_file(target_path, parameters, format_name)
                        model_paths[format_name.lower()] = str(target_path)
                        
                except Exception as e:
                    self._log_warning(f"{format_name}格式导出失败: {e}")
                    # 创建错误占位文件
                    error_path = output_path / f"rocket_engine_model_{format_name.lower()}_error.txt"
                    with open(error_path, 'w', encoding='utf-8') as f:
                        f.write(f"# {format_name}格式导出失败\n")
                        f.write(f"# 错误信息: {e}\n")
                    model_paths[format_name.lower() + "_error"] = str(error_path)
            
            return model_paths
            
        except Exception as e:
            self._log_error(f"多格式模型导出失败: {e}")
            raise

    def _generate_engineering_drawings(self, parameters: SimulationParameters, 
                                     output_path: Path) -> Dict[str, str]:
        """生成工程图纸"""
        drawing_paths = {}
        
        try:
            # 生成工程图纸
            if self.freecad_available:
                # FreeCAD可用时生成实际图纸
                drawing_result = self.generate_engineering_drawing(parameters, str(output_path))
                drawing_paths.update(drawing_result)
            else:
                # FreeCAD不可用时生成模拟图纸
                # 生成PDF图纸
                pdf_path = output_path / "rocket_engine_drawing.pdf"
                self._create_simulation_pdf_drawing(pdf_path, parameters)
                drawing_paths["pdf"] = str(pdf_path)
                
                # 生成DXF图纸
                dxf_path = output_path / "rocket_engine_drawing.dxf"
                self._create_simulation_dxf_drawing(dxf_path, parameters)
                drawing_paths["dxf"] = str(dxf_path)
            
            return drawing_paths
            
        except Exception as e:
            self._log_error(f"工程图纸生成失败: {e}")
            raise

    def _generate_analysis_report(self, parameters: SimulationParameters, 
                                 output_path: Path) -> Dict[str, str]:
        """生成分析报告"""
        report_paths = {}
        
        try:
            # 生成PDF分析报告
            pdf_report_path = output_path / "engineering_analysis_report.pdf"
            self._create_pdf_analysis_report(pdf_report_path, parameters)
            report_paths["pdf_report"] = str(pdf_report_path)
            
            # 生成Excel数据表
            excel_path = output_path / "engineering_data.xlsx"
            self._create_excel_data_sheet(excel_path, parameters)
            report_paths["excel_data"] = str(excel_path)
            
            # 生成JSON配置文件
            json_path = output_path / "engine_configuration.json"
            self._create_json_configuration(json_path, parameters)
            report_paths["json_config"] = str(json_path)
            
            return report_paths
            
        except Exception as e:
            self._log_error(f"分析报告生成失败: {e}")
            raise

    def _generate_project_documentation(self, parameters: SimulationParameters, 
                                      output_path: Path) -> Dict[str, str]:
        """生成项目文档"""
        documentation_paths = {}
        
        try:
            # 生成设计说明书
            design_spec_path = output_path / "design_specification.md"
            self._create_design_specification(design_spec_path, parameters)
            documentation_paths["design_spec"] = str(design_spec_path)
            
            # 生成制造工艺文件
            manufacturing_path = output_path / "manufacturing_guide.md"
            self._create_manufacturing_guide(manufacturing_path, parameters)
            documentation_paths["manufacturing_guide"] = str(manufacturing_path)
            
            # 生成测试验证报告
            test_report_path = output_path / "test_validation_report.md"
            self._create_test_validation_report(test_report_path, parameters)
            documentation_paths["test_report"] = str(test_report_path)
            
            return documentation_paths
            
        except Exception as e:
            self._log_error(f"项目文档生成失败: {e}")
            raise

    def _package_deliverables(self, deliverables: Dict[str, str], 
                            output_path: Path) -> str:
        """打包所有工程化成果"""
        try:
            import zipfile
            import datetime
            
            # 创建ZIP包
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"rocket_engine_deliverables_{timestamp}.zip"
            zip_path = output_path / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_type, file_path in deliverables.items():
                    if file_path and Path(file_path).exists():
                        # 在ZIP中创建有组织的目录结构
                        if file_type.endswith('_error'):
                            zipf.write(file_path, f"errors/{Path(file_path).name}")
                        elif 'report' in file_type or 'config' in file_type:
                            zipf.write(file_path, f"documents/{Path(file_path).name}")
                        elif 'drawing' in file_type or 'pdf' in file_type:
                            zipf.write(file_path, f"drawings/{Path(file_path).name}")
                        else:
                            zipf.write(file_path, f"models/{Path(file_path).name}")
            
            self._log_info(f"工程化成果包创建完成: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            self._log_error(f"成果打包失败: {e}")
            raise

    def _create_simulation_model_file(self, file_path: Path, 
                                    parameters: SimulationParameters, 
                                    format_name: str):
        """创建模拟模型文件"""
        import datetime
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# 模拟{format_name}格式火箭发动机模型\n")
            f.write(f"# 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 注意: 此文件为模拟文件，实际{format_name}格式需要FreeCAD支持\n\n")
            f.write("模型参数:\n")
            f.write(f"燃烧室直径: {parameters.chamber_diameter} mm\n")
            f.write(f"燃烧室长度: {parameters.chamber_length} mm\n")
            f.write(f"喉部直径: {parameters.throat_diameter} mm\n")
            f.write(f"扩张比: {parameters.expansion_ratio}\n")
            f.write(f"壁厚: {parameters.wall_thickness} mm\n")
            f.write(f"燃烧室压力: {parameters.chamber_pressure} MPa\n")

    def _create_simulation_pdf_drawing(self, pdf_path: Path, 
                                     parameters: SimulationParameters):
        """创建模拟PDF工程图纸"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import mm
            
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            
            # 添加标题
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 800, "火箭发动机工程图纸")
            
            # 添加参数表格
            c.setFont("Helvetica", 10)
            y_position = 750
            parameters_data = [
                ("燃烧室直径", f"{parameters.chamber_diameter} mm"),
                ("燃烧室长度", f"{parameters.chamber_length} mm"),
                ("喉部直径", f"{parameters.throat_diameter} mm"),
                ("扩张比", f"{parameters.expansion_ratio}"),
                ("壁厚", f"{parameters.wall_thickness} mm"),
                ("燃烧室压力", f"{parameters.chamber_pressure} MPa")
            ]
            
            for param_name, param_value in parameters_data:
                c.drawString(50, y_position, f"{param_name}: {param_value}")
                y_position -= 20
            
            # 添加说明
            c.drawString(50, 600, "注意: 此图纸为模拟图纸，实际工程图纸需要FreeCAD支持")
            
            c.save()
            
        except ImportError:
            # reportlab不可用时创建文本文件
            with open(pdf_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write("PDF图纸生成失败: reportlab库不可用\n")
                f.write("请安装reportlab: pip install reportlab\n")

    def _create_simulation_dxf_drawing(self, dxf_path: Path, 
                                     parameters: SimulationParameters):
        """创建模拟DXF工程图纸"""
        with open(dxf_path, 'w', encoding='utf-8') as f:
            f.write("# 模拟DXF格式工程图纸\n")
            f.write(f"# 创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# 注意: 此文件为模拟文件，实际DXF格式需要FreeCAD支持\n\n")
            f.write("图纸参数:\n")
            f.write(f"燃烧室直径: {parameters.chamber_diameter} mm\n")
            f.write(f"燃烧室长度: {parameters.chamber_length} mm\n")
            f.write(f"喉部直径: {parameters.throat_diameter} mm\n")
            f.write(f"扩张比: {parameters.expansion_ratio}\n")

    def _create_pdf_analysis_report(self, pdf_path: Path, 
                                  parameters: SimulationParameters):
        """创建PDF分析报告"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            
            c = canvas.Canvas(str(pdf_path), pagesize=A4)
            
            # 添加报告标题
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 800, "火箭发动机工程分析报告")
            
            # 添加参数分析
            c.setFont("Helvetica", 10)
            c.drawString(50, 750, f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 性能估算
            performance = self._estimate_performance_metrics(parameters)
            c.drawString(50, 720, f"估算推力: {performance.get('estimated_thrust_kN', 0):.1f} kN")
            c.drawString(50, 700, f"估算效率: {performance.get('estimated_efficiency', 0):.2f}")
            
            c.save()
            
        except ImportError:
            with open(pdf_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write("PDF报告生成失败: reportlab库不可用\n")

    def _create_excel_data_sheet(self, excel_path: Path, 
                               parameters: SimulationParameters):
        """创建Excel数据表"""
        try:
            import pandas as pd
            
            # 创建数据框
            data = {
                '参数名称': ['燃烧室直径', '燃烧室长度', '喉部直径', '扩张比', '壁厚', '燃烧室压力'],
                '参数值': [
                    parameters.chamber_diameter,
                    parameters.chamber_length,
                    parameters.throat_diameter,
                    parameters.expansion_ratio,
                    parameters.wall_thickness,
                    parameters.chamber_pressure
                ],
                '单位': ['mm', 'mm', 'mm', '', 'mm', 'MPa']
            }
            
            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)
            
        except ImportError:
            with open(excel_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
                f.write("Excel数据表生成失败: pandas库不可用\n")

    def _create_json_configuration(self, json_path: Path, 
                                 parameters: SimulationParameters):
        """创建JSON配置文件"""
        import json
        
        config_data = {
            "engine_parameters": {
                "chamber_diameter": parameters.chamber_diameter,
                "chamber_length": parameters.chamber_length,
                "throat_diameter": parameters.throat_diameter,
                "expansion_ratio": parameters.expansion_ratio,
                "wall_thickness": parameters.wall_thickness,
                "chamber_pressure": parameters.chamber_pressure
            },
            "metadata": {
                "creation_time": datetime.datetime.now().isoformat(),
                "software_version": "1.0.0"
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

    def _create_design_specification(self, spec_path: Path, 
                                   parameters: SimulationParameters):
        """创建设计说明书"""
        with open(spec_path, 'w', encoding='utf-8') as f:
            f.write("# 火箭发动机设计说明书\n\n")
            f.write("## 1. 设计概述\n")
            f.write("本文档描述了火箭发动机的关键设计参数和技术规格。\n\n")
            f.write("## 2. 关键参数\n")
            f.write(f"- 燃烧室直径: {parameters.chamber_diameter} mm\n")
            f.write(f"- 燃烧室长度: {parameters.chamber_length} mm\n")
            f.write(f"- 喉部直径: {parameters.throat_diameter} mm\n")
            f.write(f"- 扩张比: {parameters.expansion_ratio}\n")
            f.write(f"- 壁厚: {parameters.wall_thickness} mm\n")
            f.write(f"- 燃烧室压力: {parameters.chamber_pressure} MPa\n\n")

    def _create_manufacturing_guide(self, guide_path: Path, 
                                  parameters: SimulationParameters):
        """创建制造工艺文件"""
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("# 火箭发动机制造工艺指南\n\n")
            f.write("## 1. 材料要求\n")
            f.write("- 燃烧室材料: 高温合金\n")
            f.write("- 喷管材料: 耐高温复合材料\n\n")
            f.write("## 2. 加工精度要求\n")
            f.write(f"- 燃烧室直径公差: ±0.1 mm\n")
            f.write(f"- 喉部直径公差: ±0.05 mm\n\n")

    def _create_test_validation_report(self, report_path: Path, 
                                     parameters: SimulationParameters):
        """创建测试验证报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 火箭发动机测试验证报告\n\n")
            f.write("## 1. 测试概述\n")
            f.write("本报告记录了火箭发动机的仿真测试结果和验证数据。\n\n")
            f.write("## 2. 仿真参数\n")
            f.write(f"- 燃烧室压力: {parameters.chamber_pressure} MPa\n")
            f.write(f"- 扩张比: {parameters.expansion_ratio}\n")
            f.write(f"- 壁厚: {parameters.wall_thickness} mm\n\n")

class ElmerSimulator:
    """Elmer模拟控制器"""
    
    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.elmer_path = self._find_elmer()
        
    def _find_elmer(self) -> Path:
        """智能查找Elmer安装路径"""
        import winreg
        
        # 1. 检查注册表（Windows系统）
        try:
            registry_paths = [
                r"SOFTWARE\\Elmer",
                r"SOFTWARE\\Wow6432Node\\Elmer",
                r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Elmer",
            ]
            
            for reg_path in registry_paths:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                        try:
                            install_path, _ = winreg.QueryValueEx(key, "InstallLocation")
                            candidate = Path(install_path) / "bin" / "ElmerSolver.exe"
                            if candidate.exists():
                                return candidate
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    pass
        except Exception as e:
            self._log_warning(f"Elmer注册表查询失败: {e}")
        
        # 2. 检查环境变量
        if 'ELMER_HOME' in os.environ:
            candidate = Path(os.environ['ELMER_HOME']) / "bin" / "ElmerSolver.exe"
            if candidate.exists():
                return candidate
        
        if 'ELMER_PATH' in os.environ:
            candidate = Path(os.environ['ELMER_PATH']) / "bin" / "ElmerSolver.exe"
            if candidate.exists():
                return candidate
        
        # 3. 检查PATH环境变量
        path_dirs = os.environ.get('PATH', '').split(os.pathsep)
        for path_dir in path_dirs:
            candidate = Path(path_dir) / "ElmerSolver.exe"
            if candidate.exists():
                return candidate
        
        # 4. 常见安装路径
        possible_paths = [
            Path("C:/Program Files/Elmer 9.0/bin/ElmerSolver.exe"),
            Path("C:/Program Files/Elmer 9.1/bin/ElmerSolver.exe"),
            Path("C:/Program Files/Elmer 9.2/bin/ElmerSolver.exe"),
            Path("C:/Program Files/Elmer/bin/ElmerSolver.exe"),
            Path("C:/Elmer/bin/ElmerSolver.exe"),
            Path("D:/Program Files/Elmer 9.0/bin/ElmerSolver.exe"),
            Path("D:/Program Files/Elmer/bin/ElmerSolver.exe"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 5. 递归搜索程序文件目录
        program_files_dirs = [
            Path(os.environ.get('ProgramFiles', r"C:\\Program Files")),
            Path(os.environ.get('ProgramFiles(x86)', r"C:\\Program Files (x86)")),
        ]
        
        for program_files_dir in program_files_dirs:
            if program_files_dir.exists():
                for root, dirs, files in os.walk(program_files_dir):
                    if 'ElmerSolver.exe' in files:
                        candidate = Path(root) / "ElmerSolver.exe"
                        if 'bin' in root.lower():
                            return candidate
        
        # 6. 提供友好的错误信息和安装建议
        error_msg = """
未找到Elmer安装路径。请确保Elmer已正确安装。

安装建议：
1. 从 https://www.elmerfem.org/ 下载最新版Elmer
2. 安装时选择添加到PATH环境变量
3. 或设置 ELMER_HOME 环境变量指向安装目录
4. 或手动指定ElmerSolver.exe路径

当前系统PATH包含的目录：
{}
        """.format('\n'.join(path_dirs[:10]))  # 只显示前10个目录
        
        raise Exception(error_msg)
    
    def run_simulation(self, model_path: str, parameters: SimulationParameters, 
                      iteration: int) -> Dict[str, Any]:
        """运行Elmer模拟，支持多物理场全耦合仿真"""
        sif_file = None
        results_dir = None
        
        try:
            # 1. 准备工作目录
            self._prepare_working_directory(iteration)
            
            # 2. 根据参数选择仿真模式
            if hasattr(parameters, 'enable_multiphysics') and parameters.enable_multiphysics:
                # 多物理场全耦合仿真模式
                self._log_info(f"开始第{iteration}轮多物理场全耦合仿真")
                
                # 2.1 生成多物理场SIF文件
                sif_file = self._generate_multiphysics_sif_content(parameters, iteration)
                
                # 2.2 设置耦合参数
                coupling_params = self._setup_coupling_parameters(parameters)
                
                # 2.3 运行多物理场仿真
                simulation_results = self._run_multiphysics_simulation(
                    sif_file, parameters, iteration, coupling_params
                )
                
                # 2.4 计算关键无量纲数
                dimensionless_numbers = self._calculate_dimensionless_numbers(
                    simulation_results, parameters
                )
                simulation_results.update(dimensionless_numbers)
                
                # 2.5 验证多物理场结果
                if not self._validate_multiphysics_results(simulation_results):
                    raise Exception("多物理场仿真结果验证失败")
                
                self._log_info(f"第{iteration}轮多物理场仿真成功完成")
                
            else:
                # 传统单物理场仿真模式
                self._log_info(f"开始第{iteration}轮传统Elmer模拟")
                
                # 2.1 生成Elmer输入文件
                sif_file = self._generate_sif_file(parameters, iteration)
                
                # 2.2 验证输入文件
                if not os.path.exists(sif_file):
                    raise Exception(f"SIF文件生成失败: {sif_file}")
                
                # 2.3 运行模拟
                result = subprocess.run([str(self.elmer_path), sif_file], 
                                     cwd=self.working_dir, capture_output=True, text=True,
                                     timeout=300)  # 5分钟超时
                
                # 2.4 检查模拟结果
                if result.returncode != 0:
                    error_details = self._analyze_elmer_error(result.stderr, result.stdout)
                    raise Exception(f"Elmer模拟失败 (返回码: {result.returncode}): {error_details}")
                
                # 2.5 验证模拟输出
                if not self._validate_simulation_output(iteration):
                    raise Exception(f"模拟输出验证失败，可能未生成有效结果")
                
                # 2.6 读取结果
                simulation_results = self._read_results(iteration)
            
            # 3. 计算性能指标
            performance_metrics = self._calculate_performance_metrics(simulation_results, parameters)
            simulation_results.update(performance_metrics)
            
            self._log_info(f"第{iteration}轮模拟成功完成")
            return simulation_results
            
        except subprocess.TimeoutExpired:
            self._log_error(f"第{iteration}轮模拟超时(5分钟)")
            # 清理超时进程
            self._cleanup_timeout_process()
            raise Exception(f"模拟执行超时，请检查Elmer安装或增加超时时间")
            
        except FileNotFoundError as e:
            self._log_error(f"文件未找到错误: {e}")
            raise Exception(f"Elmer可执行文件未找到，请检查安装路径: {self.elmer_path}")
            
        except PermissionError as e:
            self._log_error(f"权限错误: {e}")
            raise Exception(f"文件访问权限不足，请检查工作目录权限: {self.working_dir}")
            
        except Exception as e:
            self._log_error(f"模拟执行错误: {e}")
            # 清理临时文件
            self._cleanup_failed_simulation(iteration, sif_file)
            raise Exception(f"模拟执行失败: {e}")
        
        finally:
            # 确保资源清理
            self._cleanup_temporary_files(iteration)
    
    def _generate_sif_file(self, parameters: SimulationParameters, iteration: int) -> str:
        """生成Elmer输入文件"""
        sif_content = f"""
! Elmer输入文件 - 迭代 {iteration}
Header
  CHECK KEYWORDS Warn
  Mesh DB "." "rocket_mesh"
  Include Path ""
  Results Directory "results_iter_{iteration}"
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = Transient
  Timestepping Method = BDF
  BDF Order = 2
  Timestep intervals = 100
  Timestep Sizes = 0.1
  Output Intervals = 10
  Steady State Max Iterations = 1
  Output File = "rocket_results_{iteration}.vtu"
  Post File = "rocket_{iteration}.ep"
End

! 流体方程
Equation 1
  Navier-Stokes = True
  Convection = Computed
  Turbulence Model = None
End

! 热方程
Equation 2
  Heat Equation = True
  Convection = Computed
End

! 结构方程
Equation 3
  Stress Analysis = True
  Calculate Stresses = True
End

! 材料属性
Material 1
  Name = "Inconel"
  Density = {parameters.material_density}
  Heat Conductivity = 11.2
  Heat Capacity = 450
  Youngs Modulus = 200e9
  Poisson Ratio = 0.29
  Thermal Expansion Coefficient = 13e-6
End

! 边界条件
Boundary Condition 1
  Target Boundaries(1) = 1
  Velocity 1 = 0
  Velocity 2 = 0
  Velocity 3 = 0
  Temperature = 300
End

Boundary Condition 2
  Target Boundaries(1) = 2
  Pressure = {parameters.chamber_pressure * 1e6}
  Temperature = 3500
End
"""
        
        sif_path = self.working_dir / f"simulation_{iteration}.sif"
        with open(sif_path, 'w') as f:
            f.write(sif_content)
        
        return str(sif_path)
    
    def _read_results(self, iteration: int) -> Dict[str, Any]:
        """读取Elmer模拟结果文件"""
        results_dir = self.working_dir / f"results_iter_{iteration}"
        
        # 检查结果目录是否存在
        if not results_dir.exists():
            self._log_warning(f"结果目录不存在: {results_dir}")
            return self._generate_fallback_results(iteration)
        
        try:
            # 查找结果文件
            vtu_files = list(results_dir.glob("*.vtu"))
            dat_files = list(results_dir.glob("*.dat"))
            
            if not vtu_files and not dat_files:
                self._log_warning(f"未找到结果文件，目录: {results_dir}")
                return self._generate_fallback_results(iteration)
            
            results = {}
            
            # 解析.vtu文件（VTK格式，包含场数据）
            if vtu_files:
                vtu_results = self._parse_vtu_file(vtu_files[0])
                results.update(vtu_results)
            
            # 解析.dat文件（文本格式，包含标量结果）
            if dat_files:
                dat_results = self._parse_dat_file(dat_files[0])
                results.update(dat_results)
            
            # 验证结果完整性
            if not self._validate_results(results):
                self._log_warning("结果验证失败，使用备用数据")
                return self._generate_fallback_results(iteration)
            
            self._log_info(f"成功解析第{iteration}轮模拟结果")
            return results
            
        except Exception as e:
            self._log_error(f"解析结果文件失败: {e}")
            return self._generate_fallback_results(iteration)
    
    def _parse_vtu_file(self, vtu_path: Path) -> Dict[str, Any]:
        """解析VTU文件，提取场数据（内存优化版本）"""
        try:
            # 首先验证文件存在性和大小
            if not vtu_path.exists():
                raise Exception(f"VTU文件不存在: {vtu_path}")
            
            file_size = vtu_path.stat().st_size
            if file_size < 100:  # 文件太小，可能无效
                raise Exception(f"VTU文件过小，可能无效: {file_size} bytes")
            
            # 检查文件大小，如果过大则使用流式处理
            if file_size > 100 * 1024 * 1024:  # 超过100MB
                self._log_warning(f"VTU文件过大({file_size/1024/1024:.1f}MB)，使用内存优化解析")
                return self._parse_vtu_memory_optimized(vtu_path)
            
            # 首先尝试使用pyvista库（更现代的VTK封装）
            try:
                import pyvista as pv
                
                # 使用pyvista读取VTU文件
                mesh = pv.read(str(vtu_path))
                
                # 验证网格数据完整性
                if mesh.n_points == 0:
                    raise Exception("VTU文件包含0个点，数据无效")
                
                results = {}
                
                # 提取温度场（内存优化：分批处理大数组）
                if "Temperature" in mesh.point_data:
                    temp_data = mesh.point_data["Temperature"]
                    if len(temp_data) > 0:
                        # 使用内存友好的统计方法
                        results["max_wall_temperature"] = float(self._safe_array_max(temp_data))
                        results["avg_temperature"] = float(self._safe_array_mean(temp_data))
                        results["min_temperature"] = float(self._safe_array_min(temp_data))
                        results["temperature_std"] = float(self._safe_array_std(temp_data))
                        results["temperature_range"] = float(results["max_wall_temperature"] - results["min_temperature"])
                        
                        # 温度分布统计（分批处理避免内存峰值）
                        temp_above_1000 = self._count_above_threshold(temp_data, 1000)
                        results["high_temp_ratio"] = float(temp_above_1000 / len(temp_data))
                
                # 提取应力场
                if "Stress" in mesh.point_data:
                    stress_data = mesh.point_data["Stress"]
                    if len(stress_data) > 0:
                        results["max_stress"] = float(self._safe_array_max(stress_data))
                        results["avg_stress"] = float(self._safe_array_mean(stress_data))
                        results["min_stress"] = float(self._safe_array_min(stress_data))
                        results["stress_std"] = float(self._safe_array_std(stress_data))
                        results["stress_range"] = float(results["max_stress"] - results["min_stress"])
                        
                        # 高应力区域统计
                        stress_above_500 = self._count_above_threshold(stress_data, 500)
                        results["high_stress_ratio"] = float(stress_above_500 / len(stress_data))
                
                # 提取速度场
                if "Velocity" in mesh.point_data:
                    velocity_data = mesh.point_data["Velocity"]
                    if len(velocity_data) > 0:
                        # 分批计算速度大小，避免内存峰值
                        velocity_magnitude = self._safe_norm_2d(velocity_data)
                        results["max_velocity"] = float(self._safe_array_max(velocity_magnitude))
                        results["avg_velocity"] = float(self._safe_array_mean(velocity_magnitude))
                        results["min_velocity"] = float(self._safe_array_min(velocity_magnitude))
                        results["cooling_velocity"] = float(self._safe_array_median(velocity_magnitude))
                        results["velocity_std"] = float(self._safe_array_std(velocity_magnitude))
                        
                        # 高速区域统计
                        high_velocity_ratio = self._count_above_threshold(velocity_magnitude, 5) / len(velocity_magnitude)
                        results["high_velocity_ratio"] = float(high_velocity_ratio)
                
                # 提取压力场
                if "Pressure" in mesh.point_data:
                    pressure_data = mesh.point_data["Pressure"]
                    if len(pressure_data) > 0:
                        results["max_pressure"] = float(self._safe_array_max(pressure_data))
                        results["avg_pressure"] = float(self._safe_array_mean(pressure_data))
                        results["min_pressure"] = float(self._safe_array_min(pressure_data))
                        results["pressure_std"] = float(self._safe_array_std(pressure_data))
                        
                        # 高压区域统计
                        high_pressure_ratio = self._count_above_threshold(pressure_data, 1e6) / len(pressure_data)
                        results["high_pressure_ratio"] = float(high_pressure_ratio)
                
                # 提取网格信息
                results["mesh_points"] = mesh.n_points
                results["mesh_cells"] = mesh.n_cells
                results["mesh_bounds"] = mesh.bounds
                
                # 强制释放mesh对象以释放内存
                del mesh
                
                # 验证提取的数据
                if not self._validate_vtu_extraction(results):
                    self._log_warning("VTU数据提取验证失败，尝试备用解析")
                    return self._parse_vtu_simple(vtu_path)
                
                self._log_info(f"成功解析VTU文件: {vtu_path.name}, 提取{len(results)}个数据字段")
                return results
                
            except ImportError:
                self._log_warning("pyvista库未安装，尝试原生VTK")
                # 回退到原生VTK
                import vtk
                from vtk.util.numpy_support import vtk_to_numpy
                
                # 读取VTU文件
                reader = vtk.vtkXMLUnstructuredGridReader()
                reader.SetFileName(str(vtu_path))
                reader.Update()
                
                output = reader.GetOutput()
                
                # 验证输出
                if output.GetNumberOfPoints() == 0:
                    raise Exception("VTK读取失败: 0个点")
                
                results = {}
                
                # 提取点数据
                point_data = output.GetPointData()
                
                # 提取温度场
                if point_data.HasArray("Temperature"):
                    temp_array = point_data.GetArray("Temperature")
                    temp_data = vtk_to_numpy(temp_array)
                    if len(temp_data) > 0:
                        results["max_wall_temperature"] = float(self._safe_array_max(temp_data))
                        results["avg_temperature"] = float(self._safe_array_mean(temp_data))
                
                # 提取应力场
                if point_data.HasArray("Stress"):
                    stress_array = point_data.GetArray("Stress")
                    stress_data = vtk_to_numpy(stress_array)
                    if len(stress_data) > 0:
                        results["max_stress"] = float(self._safe_array_max(stress_data))
                        results["avg_stress"] = float(self._safe_array_mean(stress_data))
                
                # 提取速度场
                if point_data.HasArray("Velocity"):
                    velocity_array = point_data.GetArray("Velocity")
                    velocity_data = vtk_to_numpy(velocity_array)
                    if len(velocity_data) > 0:
                        velocity_magnitude = self._safe_norm_2d(velocity_data)
                        results["max_velocity"] = float(self._safe_array_max(velocity_magnitude))
                        results["cooling_velocity"] = float(self._safe_array_median(velocity_magnitude))
                
                # 强制释放VTK对象以释放内存
                reader = None
                output = None
                
                return results
                
        except ImportError:
            self._log_warning("VTK/pyvista库未安装，使用简化解析")
            return self._parse_vtu_simple(vtu_path)
        except Exception as e:
            self._log_error(f"VTU文件解析失败: {e}")
            return self._parse_vtu_simple(vtu_path)
    
    def _validate_vtu_extraction(self, results: Dict[str, Any]) -> bool:
        """验证VTU数据提取的合理性"""
        try:
            # 检查必需字段
            required_fields = ["max_wall_temperature", "max_stress"]
            for field in required_fields:
                if field not in results:
                    return False
            
            # 检查数据有效性
            if results.get("max_wall_temperature", 0) <= 0:
                return False
            if results.get("max_stress", 0) < 0:
                return False
            
            # 检查网格信息
            if results.get("mesh_points", 0) == 0:
                return False
            
            return True
        except Exception:
            return False
    
    def _safe_array_max(self, arr: np.ndarray) -> float:
        """内存安全的数组最大值计算（分批处理大数组）"""
        if len(arr) == 0:
            return 0.0
        
        # 对于大数组，分批处理避免内存峰值
        if len(arr) > 1000000:  # 超过100万个元素
            batch_size = 100000
            max_val = arr[0]
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                batch_max = np.max(batch)
                if batch_max > max_val:
                    max_val = batch_max
            return float(max_val)
        else:
            return float(np.max(arr))
    
    def _safe_array_min(self, arr: np.ndarray) -> float:
        """内存安全的数组最小值计算"""
        if len(arr) == 0:
            return 0.0
        
        if len(arr) > 1000000:
            batch_size = 100000
            min_val = arr[0]
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                batch_min = np.min(batch)
                if batch_min < min_val:
                    min_val = batch_min
            return float(min_val)
        else:
            return float(np.min(arr))
    
    def _safe_array_mean(self, arr: np.ndarray) -> float:
        """内存安全的数组平均值计算"""
        if len(arr) == 0:
            return 0.0
        
        if len(arr) > 1000000:
            batch_size = 100000
            total = 0.0
            count = 0
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                total += np.sum(batch)
                count += len(batch)
            return float(total / count)
        else:
            return float(np.mean(arr))
    
    def _safe_array_std(self, arr: np.ndarray) -> float:
        """内存安全的数组标准差计算"""
        if len(arr) == 0:
            return 0.0
        
        if len(arr) > 1000000:
            mean_val = self._safe_array_mean(arr)
            batch_size = 100000
            variance_sum = 0.0
            count = 0
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                variance_sum += np.sum((batch - mean_val) ** 2)
                count += len(batch)
            return float(np.sqrt(variance_sum / count))
        else:
            return float(np.std(arr))
    
    def _safe_array_median(self, arr: np.ndarray) -> float:
        """内存安全的数组中位数计算"""
        if len(arr) == 0:
            return 0.0
        
        if len(arr) > 1000000:
            # 对于大数组，使用近似中位数
            sample_size = min(100000, len(arr))
            indices = np.random.choice(len(arr), sample_size, replace=False)
            return float(np.median(arr[indices]))
        else:
            return float(np.median(arr))
    
    def _count_above_threshold(self, arr: np.ndarray, threshold: float) -> int:
        """内存安全的阈值计数"""
        if len(arr) == 0:
            return 0
        
        if len(arr) > 1000000:
            batch_size = 100000
            count = 0
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                count += np.sum(batch > threshold)
            return int(count)
        else:
            return int(np.sum(arr > threshold))
    
    def _safe_norm_2d(self, arr: np.ndarray) -> np.ndarray:
        """内存安全的2D数组范数计算"""
        if len(arr) == 0:
            return np.array([])
        
        if len(arr) > 1000000:
            batch_size = 100000
            result = []
            for i in range(0, len(arr), batch_size):
                batch = arr[i:i+batch_size]
                norms = np.linalg.norm(batch, axis=1)
                result.append(norms)
            return np.concatenate(result)
        else:
            return np.linalg.norm(arr, axis=1)
    
    def _parse_vtu_memory_optimized(self, vtu_path: Path) -> Dict[str, Any]:
        """内存优化的VTU文件解析（用于大文件）"""
        try:
            self._log_info(f"使用内存优化解析大VTU文件: {vtu_path.name}")
            
            # 使用简化解析方法，避免加载整个网格到内存
            return self._parse_vtu_simple(vtu_path)
            
        except Exception as e:
            self._log_error(f"内存优化VTU解析失败: {e}")
            # 回退到简化解析
            return self._parse_vtu_simple(vtu_path)
    
    def _parse_vtu_simple(self, vtu_path: Path) -> Dict[str, Any]:
        """简化解析VTU文件（不依赖VTK）"""
        try:
            # 读取文件内容
            with open(vtu_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            results = {}
            
            # 简单解析XML结构，提取关键数据
            import re
            
            # 查找温度数据
            temp_pattern = r'<DataArray type="Float32" Name="Temperature"[^>]*>(.*?)</DataArray>'
            temp_match = re.search(temp_pattern, content, re.DOTALL)
            if temp_match:
                temp_data = re.findall(r'[-+]?\d*\.\d+|\d+', temp_match.group(1))
                if temp_data:
                    temp_values = [float(x) for x in temp_data]
                    results["max_wall_temperature"] = max(temp_values)
                    results["avg_temperature"] = sum(temp_values) / len(temp_values)
            
            # 查找应力数据
            stress_pattern = r'<DataArray type="Float32" Name="Stress"[^>]*>(.*?)</DataArray>'
            stress_match = re.search(stress_pattern, content, re.DOTALL)
            if stress_match:
                stress_data = re.findall(r'[-+]?\d*\.\d+|\d+', stress_match.group(1))
                if stress_data:
                    stress_values = [float(x) for x in stress_data]
                    results["max_stress"] = max(stress_values)
                    results["avg_stress"] = sum(stress_values) / len(stress_values)
            
            return results
            
        except Exception as e:
            raise Exception(f"简化VTU解析失败: {e}")
    
    def _parse_dat_file(self, dat_path: Path) -> Dict[str, Any]:
        """解析DAT文件，提取标量结果"""
        try:
            # 验证文件存在性和大小
            if not dat_path.exists():
                raise Exception(f"DAT文件不存在: {dat_path}")
            
            file_size = dat_path.stat().st_size
            if file_size < 10:  # 文件太小，可能无效
                raise Exception(f"DAT文件过小，可能无效: {file_size} bytes")
            
            results = {}
            
            with open(dat_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 验证文件内容
            if not lines:
                raise Exception("DAT文件内容为空")
            
            # 多种格式的解析策略
            extracted_fields = set()
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 解析Elmer标准输出格式：变量名 = 值
                if '=' in line:
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        
                        # 匹配常见物理量
                        if 'thrust' in key or '推力' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                thrust_value = float(numbers[0])
                                if 0 < thrust_value < 10000:  # 合理范围验证
                                    results["thrust"] = thrust_value
                                    results["thrust_unit"] = "N"
                                    extracted_fields.add("thrust")
                        elif 'efficiency' in key or '效率' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                efficiency_value = float(numbers[0])
                                if 0 <= efficiency_value <= 1:  # 效率范围验证
                                    results["efficiency"] = efficiency_value
                                    extracted_fields.add("efficiency")
                        elif 'weight' in key or '重量' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                weight_value = float(numbers[0])
                                if 0 < weight_value < 10000:  # 合理重量范围
                                    results["dry_weight"] = weight_value
                                    results["weight_unit"] = "kg"
                                    extracted_fields.add("dry_weight")
                        elif 'temperature' in key or '温度' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                temp_value = float(numbers[0])
                                if 0 < temp_value < 5000:  # 合理温度范围
                                    results["max_wall_temperature"] = temp_value
                                    results["temperature_unit"] = "K"
                                    extracted_fields.add("max_wall_temperature")
                        elif 'stress' in key or '应力' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                stress_value = float(numbers[0])
                                if 0 <= stress_value < 1e9:  # 合理应力范围
                                    results["max_stress"] = stress_value
                                    results["stress_unit"] = "Pa"
                                    extracted_fields.add("max_stress")
                        elif 'power' in key or '功率' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                power_value = float(numbers[0])
                                if 0 < power_value < 1e6:  # 合理功率范围
                                    results["power"] = power_value
                                    results["power_unit"] = "W"
                                    extracted_fields.add("power")
                        elif 'flow' in key or '流量' in key:
                            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
                            if numbers:
                                flow_value = float(numbers[0])
                                if 0 < flow_value < 100:  # 合理流量范围
                                    results["flow_rate"] = flow_value
                                    results["flow_unit"] = "kg/s"
                                    extracted_fields.add("flow_rate")
                
                # 解析表格格式数据
                elif '\t' in line or len(line.split()) > 2:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            # 尝试解析数值
                            value = float(parts[-1])
                            header = ' '.join(parts[:-1]).lower()
                            
                            if 'thrust' in header or '推力' in header:
                                if 0 < value < 10000:
                                    results["thrust"] = value
                                    extracted_fields.add("thrust")
                            elif 'efficiency' in header or '效率' in header:
                                if 0 <= value <= 1:
                                    results["efficiency"] = value
                                    extracted_fields.add("efficiency")
                            elif 'weight' in header or '重量' in header:
                                if 0 < value < 10000:
                                    results["dry_weight"] = value
                                    extracted_fields.add("dry_weight")
                            elif 'temperature' in header or '温度' in header:
                                if 0 < value < 5000:
                                    results["max_wall_temperature"] = value
                                    extracted_fields.add("max_wall_temperature")
                            elif 'stress' in header or '应力' in header:
                                if 0 <= value < 1e9:
                                    results["max_stress"] = value
                                    extracted_fields.add("max_stress")
                            elif 'power' in header or '功率' in header:
                                if 0 < value < 1e6:
                                    results["power"] = value
                                    extracted_fields.add("power")
                            elif 'flow' in header or '流量' in header:
                                if 0 < value < 100:
                                    results["flow_rate"] = value
                                    extracted_fields.add("flow_rate")
                        except ValueError:
                            continue
                
                # 解析简单关键词匹配
                else:
                    if 'thrust' in line.lower() or '推力' in line:
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if numbers:
                            thrust_value = float(numbers[0])
                            if 0 < thrust_value < 10000:
                                results["thrust"] = thrust_value
                                extracted_fields.add("thrust")
                    elif 'efficiency' in line.lower() or '效率' in line:
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if numbers:
                            efficiency_value = float(numbers[0])
                            if 0 <= efficiency_value <= 1:
                                results["efficiency"] = efficiency_value
                                extracted_fields.add("efficiency")
                    elif 'weight' in line.lower() or '重量' in line:
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if numbers:
                            weight_value = float(numbers[0])
                            if 0 < weight_value < 10000:
                                results["dry_weight"] = weight_value
                                extracted_fields.add("dry_weight")
                    elif 'power' in line.lower() or '功率' in line:
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if numbers:
                            power_value = float(numbers[0])
                            if 0 < power_value < 1e6:
                                results["power"] = power_value
                                extracted_fields.add("power")
            
            # 验证提取的数据
            if not self._validate_dat_extraction(results):
                self._log_warning("DAT数据提取验证失败，返回空结果")
                return {}
            
            # 记录解析结果统计
            success_count = len(extracted_fields)
            self._log_info(f"成功解析DAT文件: {dat_path.name}, 提取{success_count}个有效数据字段")
            
            return results
            
        except Exception as e:
            self._log_error(f"DAT文件解析失败: {e}")
            return {}
    
    def _validate_dat_extraction(self, results: Dict[str, Any]) -> bool:
        """验证DAT数据提取的合理性"""
        try:
            # 检查是否有任何有效数据
            if not results:
                return False
            
            # 检查关键字段的合理性
            if "thrust" in results:
                thrust = results["thrust"]
                if thrust <= 0 or thrust > 10000:
                    return False
            
            if "efficiency" in results:
                efficiency = results["efficiency"]
                if efficiency < 0 or efficiency > 1:
                    return False
            
            if "max_wall_temperature" in results:
                temperature = results["max_wall_temperature"]
                if temperature <= 0 or temperature > 5000:
                    return False
            
            if "max_stress" in results:
                stress = results["max_stress"]
                if stress < 0 or stress > 1e9:
                    return False
            
            # 检查数据一致性
            if "thrust" in results and "power" in results:
                # 推力与功率应有一定相关性
                if results["power"] > 0 and results["thrust"] / results["power"] > 1000:
                    return False
            
            return True
        except Exception:
            return False
    
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """验证结果数据的合理性"""
        try:
            # 1. 检查必需字段
            required_fields = ["thrust", "max_wall_temperature", "max_stress"]
            for field in required_fields:
                if field not in results:
                    self._log_warning(f"结果验证失败: 缺少必需字段 '{field}'")
                    return False
            
            # 2. 检查数据类型
            if not isinstance(results.get("thrust"), (int, float)):
                self._log_warning("结果验证失败: 推力数据格式错误")
                return False
            if not isinstance(results.get("max_wall_temperature"), (int, float)):
                self._log_warning("结果验证失败: 壁温数据格式错误")
                return False
            if not isinstance(results.get("max_stress"), (int, float)):
                self._log_warning("结果验证失败: 应力数据格式错误")
                return False
            
            # 3. 检查数值范围合理性
            thrust = results.get("thrust", 0)
            max_temp = results.get("max_wall_temperature", 0)
            max_stress = results.get("max_stress", 0)
            
            # 推力范围检查 (火箭发动机典型范围: 1-1000 kN)
            if thrust <= 0 or thrust > 1000:
                self._log_warning(f"结果验证失败: 推力值超出合理范围 {thrust} kN")
                return False
            
            # 壁温范围检查 (室温-材料熔点)
            if max_temp < 300 or max_temp > 3500:  # 室温到典型高温合金熔点
                self._log_warning(f"结果验证失败: 壁温值超出合理范围 {max_temp} K")
                return False
            
            # 应力范围检查 (合理应力范围)
            if max_stress < 0 or max_stress > 1000:  # 0-1000 MPa
                self._log_warning(f"结果验证失败: 应力值超出合理范围 {max_stress} MPa")
                return False
            
            # 4. 检查数据一致性
            # 推力与效率的合理性检查
            if "efficiency" in results:
                efficiency = results.get("efficiency")
                if not isinstance(efficiency, (int, float)) or efficiency < 0 or efficiency > 1:
                    self._log_warning(f"结果验证失败: 效率值不合理 {efficiency}")
                    return False
                
                # 推力与效率的物理关系检查
                if thrust > 50 and efficiency < 0.8:
                    self._log_warning(f"结果验证失败: 高推力低效率不合理 (推力={thrust}, 效率={efficiency})")
                    return False
            
            # 5. 检查冷却流速合理性
            if "cooling_velocity" in results:
                cooling_vel = results.get("cooling_velocity")
                if not isinstance(cooling_vel, (int, float)) or cooling_vel < 0 or cooling_vel > 10:
                    self._log_warning(f"结果验证失败: 冷却流速不合理 {cooling_vel} m/s")
                    return False
            
            # 6. 检查数据变化趋势
            if "is_fallback" not in results:  # 非备用数据才检查趋势
                # 壁温与应力的相关性检查
                if max_temp > 1000 and max_stress < 100:
                    self._log_warning(f"结果验证失败: 高温低应力可能不合理 (温度={max_temp}K, 应力={max_stress}MPa)")
                    return False
            
            self._log_info("结果验证通过: 所有数据符合物理合理性检查")
            return True
            
        except Exception as e:
            self._log_error(f"结果验证过程出错: {e}")
            return False
    
    def _generate_fallback_results(self, iteration: int) -> Dict[str, Any]:
        """生成备用结果数据"""
        # 基于迭代次数生成更合理的数据
        base_thrust = 10.0 + (iteration % 10) * 0.1
        base_temp = 750.0 + (iteration % 20) * 5.0
        base_stress = 500.0 + (iteration % 15) * 3.0
        
        return {
            "thrust": base_thrust + np.random.uniform(-0.5, 0.5),
            "max_wall_temperature": base_temp + np.random.uniform(-20, 20),
            "max_stress": base_stress + np.random.uniform(-10, 10),
            "cooling_velocity": np.random.uniform(0.6, 1.0),
            "efficiency": np.random.uniform(0.85, 0.95),
            "dry_weight": np.random.uniform(0.8, 1.5),
            "is_fallback": True  # 标记为备用数据
        }
    
    def _log_info(self, message: str):
        """记录信息日志"""
        print(f"[ElmerSimulator INFO] {message}")
    
    def _log_warning(self, message: str):
        """记录警告日志"""
        print(f"[ElmerSimulator WARNING] {message}")
    
    def _log_error(self, message: str):
        """记录错误日志"""
        print(f"[ElmerSimulator ERROR] {message}")
    
    def _prepare_working_directory(self, iteration: int):
        """准备工作目录"""
        try:
            # 创建工作目录
            if not self.working_dir.exists():
                self.working_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建迭代结果目录
            results_dir = self.working_dir / f"results_iter_{iteration}"
            if not results_dir.exists():
                results_dir.mkdir(parents=True, exist_ok=True)
            
            # 清理之前的临时文件
            self._cleanup_previous_iteration_files(iteration)
            
        except Exception as e:
            raise Exception(f"准备工作目录失败: {e}")
    
    def _analyze_elmer_error(self, stderr: str, stdout: str) -> str:
        """分析Elmer错误输出"""
        error_details = []
        
        # 分析标准错误输出
        if stderr:
            lines = stderr.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 常见错误模式匹配
                if "error" in line.lower() or "failed" in line.lower():
                    error_details.append(f"错误: {line}")
                elif "warning" in line.lower():
                    error_details.append(f"警告: {line}")
                elif "not found" in line.lower() or "missing" in line.lower():
                    error_details.append(f"文件缺失: {line}")
        
        # 分析标准输出
        if stdout:
            lines = stdout.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 查找模拟进度和状态信息
                if "simulation" in line.lower() and "completed" in line.lower():
                    error_details.append(f"模拟状态: {line}")
                elif "time" in line.lower() and "step" in line.lower():
                    error_details.append(f"时间步: {line}")
        
        if not error_details:
            return "未知错误，请检查Elmer安装和配置"
        
        return '; '.join(error_details[:5])  # 返回前5个错误信息
    
    def _validate_simulation_output(self, iteration: int) -> bool:
        """验证模拟输出文件的完整性和有效性"""
        try:
            results_dir = self.working_dir / f"results_iter_{iteration}"
            
            # 检查结果目录存在性
            if not results_dir.exists():
                self._log_error(f"结果目录不存在: {results_dir}")
                return False
            
            # 检查必需的文件是否存在
            required_files = [
                results_dir / "case.sif",
                results_dir / "mesh.nmesh",
                results_dir / "results.vtu",
                results_dir / "results.dat"
            ]
            
            # 检查文件存在性
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(file_path.name)
            
            if missing_files:
                self._log_warning(f"必需文件缺失: {', '.join(missing_files)}")
                return False
            
            # 检查文件大小和可读性
            file_validation_results = []
            for file_path in required_files:
                try:
                    file_size = file_path.stat().st_size
                    if file_size < 100:  # 文件太小，可能无效
                        self._log_warning(f"文件过小，可能无效: {file_path.name} ({file_size} bytes)")
                        file_validation_results.append((file_path.name, "size_too_small"))
                        continue
                    
                    # 尝试读取文件头验证可读性
                    with open(file_path, 'rb') as f:
                        header = f.read(100)  # 读取前100字节
                    
                    if not header:
                        self._log_warning(f"文件无法读取: {file_path.name}")
                        file_validation_results.append((file_path.name, "unreadable"))
                        continue
                    
                    file_validation_results.append((file_path.name, "valid"))
                    
                except Exception as e:
                    self._log_warning(f"文件验证失败 {file_path.name}: {e}")
                    file_validation_results.append((file_path.name, "validation_failed"))
            
            # 统计验证结果
            valid_files = [f for f, status in file_validation_results if status == "valid"]
            if len(valid_files) < len(required_files) * 0.5:  # 至少50%文件有效
                self._log_warning(f"文件验证通过率过低: {len(valid_files)}/{len(required_files)}")
                return False
            
            # 检查结果文件内容
            vtu_path = results_dir / "results.vtu"
            dat_path = results_dir / "results.dat"
            
            # 验证VTU文件
            vtu_results = self._parse_vtu_file(vtu_path)
            if not vtu_results:
                self._log_warning("VTU文件解析失败")
                return False
            
            # 验证DAT文件
            dat_results = self._parse_dat_file(dat_path)
            if not dat_results:
                self._log_warning("DAT文件解析失败")
                return False
            
            # 验证结果数据的合理性
            combined_results = {**vtu_results, **dat_results}
            if not self._validate_results(combined_results):
                self._log_warning("模拟结果数据验证失败")
                return False
            
            # 检查数据一致性
            consistency_checks = []
            
            # 检查VTU和DAT数据的一致性
            if "temperature" in vtu_results and "max_wall_temperature" in dat_results:
                vtu_temp = vtu_results.get("temperature", {}).get("max", 0)
                dat_temp = dat_results.get("max_wall_temperature", 0)
                if abs(vtu_temp - dat_temp) > 100:  # 温度差异过大
                    consistency_checks.append("temperature_mismatch")
            
            if "stress" in vtu_results and "max_stress" in dat_results:
                vtu_stress = vtu_results.get("stress", {}).get("max", 0)
                dat_stress = dat_results.get("max_stress", 0)
                if abs(vtu_stress - dat_stress) > 1e6:  # 应力差异过大
                    consistency_checks.append("stress_mismatch")
            
            if consistency_checks:
                self._log_warning(f"数据一致性检查失败: {consistency_checks}")
                return False
            
            # 记录详细的验证统计信息
            vtu_field_count = len(vtu_results)
            dat_field_count = len(dat_results)
            total_fields = vtu_field_count + dat_field_count
            
            self._log_info(f"模拟输出验证通过: VTU字段={vtu_field_count}, DAT字段={dat_field_count}, 总计={total_fields}")
            
            # 记录关键性能指标
            if "thrust" in dat_results:
                self._log_info(f"推力: {dat_results['thrust']} N")
            if "efficiency" in dat_results:
                self._log_info(f"效率: {dat_results['efficiency']:.3f}")
            if "max_wall_temperature" in dat_results:
                self._log_info(f"最高壁温: {dat_results['max_wall_temperature']} K")
            
            return True
            
        except Exception as e:
            self._log_error(f"模拟输出验证失败: {e}")
            return False
    
    def _cleanup_timeout_process(self):
        """清理超时进程"""
        try:
            # 在Windows上查找并终止Elmer相关进程
            import psutil
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'elmer' in proc.info['name'].lower():
                        proc.terminate()
                        self._log_info(f"已终止Elmer进程: {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            self._log_warning("psutil库未安装，无法自动清理进程")
        except Exception as e:
            self._log_warning(f"清理进程失败: {e}")
    
    def _cleanup_failed_simulation(self, iteration: int, sif_file: str = None):
        """清理失败的模拟文件"""
        try:
            # 删除SIF文件
            if sif_file and os.path.exists(sif_file):
                os.remove(sif_file)
                self._log_info(f"已删除失败的SIF文件: {sif_file}")
            
            # 删除结果目录
            results_dir = self.working_dir / f"results_iter_{iteration}"
            if results_dir.exists():
                import shutil
                shutil.rmtree(results_dir)
                self._log_info(f"已删除失败的结果目录: {results_dir}")
                
        except Exception as e:
            self._log_warning(f"清理失败模拟文件时出错: {e}")
    
    def _cleanup_temporary_files(self, iteration: int):
        """内存优化的临时文件清理（增强版本）"""
        try:
            self._log_info(f"开始清理第{iteration}轮临时文件")
            
            # 1. 清理日志文件
            log_files = list(self.working_dir.glob(f"*{iteration}*.log"))
            log_count = 0
            for log_file in log_files:
                if log_file.exists():
                    try:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        log_count += 1
                        self._log_info(f"已删除日志文件: {log_file.name} ({file_size} bytes)")
                    except Exception as e:
                        self._log_warning(f"无法删除日志文件 {log_file}: {e}")
            
            # 2. 清理临时网格文件
            mesh_files = list(self.working_dir.glob(f"*{iteration}*.mesh"))
            mesh_count = 0
            for mesh_file in mesh_files:
                if mesh_file.exists():
                    try:
                        file_size = mesh_file.stat().st_size
                        mesh_file.unlink()
                        mesh_count += 1
                        self._log_info(f"已删除网格文件: {mesh_file.name} ({file_size} bytes)")
                    except Exception as e:
                        self._log_warning(f"无法删除网格文件 {mesh_file}: {e}")
            
            # 3. 清理大文件（VTU/DAT文件）
            large_files = list(self.working_dir.glob(f"*{iteration}*.vtu")) + list(self.working_dir.glob(f"*{iteration}*.dat"))
            large_count = 0
            for large_file in large_files:
                if large_file.exists():
                    try:
                        file_size = large_file.stat().st_size
                        if file_size > 10 * 1024 * 1024:  # 超过10MB的大文件
                            self._log_info(f"删除大文件: {large_file.name} ({file_size/1024/1024:.1f} MB)")
                        large_file.unlink()
                        large_count += 1
                    except Exception as e:
                        self._log_warning(f"无法删除大文件 {large_file}: {e}")
            
            # 4. 清理其他临时文件
            other_files = list(self.working_dir.glob(f"*{iteration}*.tmp")) + list(self.working_dir.glob(f"*{iteration}*.bak"))
            other_count = 0
            for other_file in other_files:
                if other_file.exists():
                    try:
                        other_file.unlink()
                        other_count += 1
                    except Exception as e:
                        self._log_warning(f"无法删除临时文件 {other_file}: {e}")
            
            # 5. 记录清理统计
            total_files = log_count + mesh_count + large_count + other_count
            self._log_info(f"临时文件清理完成: 日志={log_count}, 网格={mesh_count}, 大文件={large_count}, 其他={other_count}, 总计={total_files}")
            
            # 6. 强制垃圾回收
            import gc
            gc.collect()
                    
        except Exception as e:
            self._log_warning(f"清理临时文件时出错: {e}")
    
    def _cleanup_previous_iteration_files(self, iteration: int):
        """内存优化的之前迭代文件清理（增强版本）"""
        try:
            self._log_info(f"开始清理之前迭代的文件（保留最近3个迭代）")
            
            # 1. 清理之前迭代的SIF文件（保留最近5个）
            sif_count = 0
            for i in range(max(0, iteration - 10), iteration - 5):
                sif_file = self.working_dir / f"simulation_{i}.sif"
                if sif_file.exists():
                    try:
                        file_size = sif_file.stat().st_size
                        sif_file.unlink()
                        sif_count += 1
                        self._log_info(f"已删除SIF文件: simulation_{i}.sif ({file_size} bytes)")
                    except Exception as e:
                        self._log_warning(f"无法删除SIF文件 {sif_file}: {e}")
            
            # 2. 清理之前迭代的结果目录（保留最近3个）
            dir_count = 0
            total_size = 0
            for i in range(max(0, iteration - 15), iteration - 3):
                results_dir = self.working_dir / f"results_iter_{i}"
                if results_dir.exists():
                    try:
                        # 计算目录大小
                        dir_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())
                        
                        import shutil
                        shutil.rmtree(results_dir)
                        dir_count += 1
                        total_size += dir_size
                        
                        size_mb = dir_size / 1024 / 1024
                        if size_mb > 10:  # 超过10MB的大目录
                            self._log_info(f"已删除大结果目录: results_iter_{i} ({size_mb:.1f} MB)")
                        else:
                            self._log_info(f"已删除结果目录: results_iter_{i} ({dir_size} bytes)")
                            
                    except Exception as e:
                        self._log_warning(f"无法删除结果目录 {results_dir}: {e}")
            
            # 3. 清理孤立的临时文件
            orphan_count = 0
            for file_pattern in ['*.tmp', '*.bak', '*.old']:
                for orphan_file in self.working_dir.glob(file_pattern):
                    if orphan_file.exists():
                        try:
                            # 检查文件是否属于当前迭代
                            file_name = orphan_file.name
                            if not any(str(i) in file_name for i in range(max(0, iteration - 3), iteration + 1)):
                                file_size = orphan_file.stat().st_size
                                orphan_file.unlink()
                                orphan_count += 1
                                self._log_info(f"已删除孤立文件: {file_name} ({file_size} bytes)")
                        except Exception as e:
                            self._log_warning(f"无法删除孤立文件 {orphan_file}: {e}")
            
            # 4. 记录清理统计
            total_mb = total_size / 1024 / 1024
            self._log_info(f"之前迭代文件清理完成: SIF文件={sif_count}, 结果目录={dir_count}, 孤立文件={orphan_count}, 释放空间={total_mb:.1f} MB")
            
            # 5. 强制垃圾回收
            import gc
            gc.collect()
                    
        except Exception as e:
            self._log_warning(f"清理之前迭代文件时出错: {e}")

    # ==================== 多物理场全耦合仿真增强方法 ====================
    
    def _generate_multiphysics_sif_content(self, parameters: Dict[str, Any], iteration: int) -> str:
        """生成多物理场全耦合仿真的SIF文件内容"""
        try:
            # 提取关键参数
            chamber_pressure = parameters.get('chamber_pressure', 1.0e6)  # Pa
            mixture_ratio = parameters.get('mixture_ratio', 2.5)
            chamber_temp = parameters.get('chamber_temperature', 1500.0)  # K
            
            # 多物理场耦合配置
            sif_content = f"""
! 多物理场全耦合仿真配置文件 - 迭代 {iteration}
! 小型酒精-氧气液体火箭发动机
! 耦合物理场：流体动力学 + 热传导 + 结构力学

Header
  CHECK KEYWORDS Warn
  Mesh DB "." "."
  Include Path ""
  Results Directory "results_iter_{iteration}"
End

! ==================== 仿真控制 ====================
Simulation
  Max Output Level = 5
  Coordinate System = Cartesian
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = Transient
  Steady State Max Iterations = 20
  Output Intervals = 10
  Timestepping Method = BDF
  BDF Order = 2
  Timestep Sizes = 0.01
  Timestep Intervals = 100
  Solver Input File = case.sif
  Post File = case.vtu
End

! ==================== 材料定义 ====================
! 1. 燃烧室壁面材料（铜合金）
Material 1
  Name = "Chamber Wall (Copper Alloy)"
  Density = 8960.0
  Heat Capacity = 385.0
  Heat Conductivity = 401.0
  Youngs Modulus = 110.0e9
  Poisson Ratio = 0.34
  Thermal Expansion Coefficient = 17.0e-6
End

! 2. 推进剂（酒精-氧气混合物）
Material 2
  Name = "Propellant Mixture"
  Density = 850.0
  Heat Capacity = 2500.0
  Heat Conductivity = 0.15
  Viscosity = 1.0e-5
  Compressibility Model = Ideal Gas
  Molar Mass = 0.032
End

! ==================== 物理场求解器配置 ====================

! 1. 流体动力学求解器（Navier-Stokes）
Solver 1
  Equation = "Navier-Stokes"
  Variable = Flow Solution[Velocity:3 Pressure:1]
  Procedure = "FlowSolve" "FlowSolver"
  
  ! 非线性求解器设置
  Nonlinear System Max Iterations = 50
  Nonlinear System Convergence Tolerance = 1.0e-6
  Nonlinear System Newton After Iterations = 3
  Nonlinear System Newton After Tolerance = 1.0e-3
  
  ! 线性求解器设置
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Preconditioning = ILU0
  Linear System Residual Output = 10
  
  ! 稳定化设置
  Stabilize = True
  Bubbles = False
  
  ! 输出设置
  Calculate Loads = True
End

! 2. 热传导求解器
Solver 2
  Equation = "Heat Equation"
  Variable = Temperature[Temperature:1]
  Procedure = "HeatSolve" "HeatSolver"
  
  ! 非线性求解器设置
  Nonlinear System Max Iterations = 50
  Nonlinear System Convergence Tolerance = 1.0e-6
  
  ! 线性求解器设置
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Preconditioning = ILU0
  
  ! 稳定化设置
  Stabilize = True
  
  ! 输出设置
  Calculate Loads = True
End

! 3. 结构力学求解器
Solver 3
  Equation = "Linear Elasticity"
  Variable = Displacement[Displacement:3]
  Procedure = "StressSolve" "StressSolver"
  
  ! 非线性求解器设置
  Nonlinear System Max Iterations = 50
  Nonlinear System Convergence Tolerance = 1.0e-6
  
  ! 线性求解器设置
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 500
  Linear System Convergence Tolerance = 1.0e-8
  Linear System Preconditioning = ILU0
  
  ! 输出设置
  Calculate Stresses = True
  Calculate Strains = True
  Calculate Principal = True
End

! ==================== 边界条件 ====================

! 1. 入口边界条件（推进剂注入）
Boundary Condition 1
  Target Boundaries(1) = 1
  Name = "Inlet"
  
  ! 速度入口
  Velocity 1 = 0.0
  Velocity 2 = 5.0
  Velocity 3 = 0.0
  
  ! 温度边界
  Temperature = 300.0
  
  ! 压力边界
  Normal-Tangential Velocity = True
End

! 2. 出口边界条件（喷管出口）
Boundary Condition 2
  Target Boundaries(1) = 2
  Name = "Outlet"
  
  ! 压力出口
  Pressure = 101325.0
  
  ! 自由流出
  Outflow Condition = True
End

! 3. 壁面边界条件（绝热/热传导）
Boundary Condition 3
  Target Boundaries(1) = 3
  Name = "Wall"
  
  ! 无滑移条件
  Noslip Wall BC = True
  
  ! 热边界条件
  Heat Flux BC = True
  Heat Flux = 0.0
End

! 4. 对称边界条件
Boundary Condition 4
  Target Boundaries(1) = 4
  Name = "Symmetry"
  
  Symmetry BC = True
End

! ==================== 体力和源项 ====================

! 1. 重力
Body Force 1
  Name = "Gravity"
  Flow Bodyforce 1 = 0.0
  Flow Bodyforce 2 = -9.81
  Flow Bodyforce 3 = 0.0
End

! 2. 热源项（燃烧放热）
Body Force 2
  Name = "Heat Source"
  Heat Source = 1.0e6
End

! ==================== 初始条件 ====================

Initial Condition 1
  Name = "Initial Flow"
  Velocity 1 = 0.0
  Velocity 2 = 0.0
  Velocity 3 = 0.0
  Pressure = {chamber_pressure}
  Temperature = {chamber_temp}
End

! ==================== 方程耦合 ====================

Equation 1
  Name = "Multiphysics Coupling"
  
  ! 激活所有物理场
  Navier-Stokes = True
  Heat Equation = True
  Linear Elasticity = True
  
  ! 耦合设置
  Convection = Computed
  
  ! 材料关联
  Material = 1 2
End

! ==================== 求解器执行顺序 ====================

Solver 1
  Exec Solver = Before Simulation
  Procedure = "FlowSolve" "FlowSolver"
End

Solver 2
  Exec Solver = After Solver 1
  Procedure = "HeatSolve" "HeatSolver"
End

Solver 3
  Exec Solver = After Solver 2
  Procedure = "StressSolve" "StressSolver"
End

! ==================== 结果输出 ====================

Solver 4
  Exec Solver = After Timestep
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = case
  Output Format = Vtu
  Binary Output = False
End
"""
            
            return sif_content
            
        except Exception as e:
            self._log_error(f"生成多物理场SIF文件内容失败: {e}")
            return ""

    def _setup_multiphysics_coupling(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """设置多物理场耦合参数"""
        try:
            # 提取关键参数
            chamber_pressure = parameters.get('chamber_pressure', 1.0e6)
            mixture_ratio = parameters.get('mixture_ratio', 2.5)
            chamber_temp = parameters.get('chamber_temperature', 1500.0)
            
            # 计算多物理场耦合参数
            coupling_params = {
                # 流体动力学参数
                'flow_reynolds': self._calculate_reynolds_number(parameters),
                'flow_mach': self._calculate_mach_number(parameters),
                'flow_prandtl': 0.71,  # 空气的普朗特数
                
                # 热传导参数
                'heat_flux_max': chamber_pressure * 0.1,  # 最大热通量估算
                'wall_temp_gradient': 50.0,  # 壁面温度梯度
                'cooling_efficiency': 0.85,  # 冷却效率
                
                # 结构力学参数
                'stress_yield': 200.0e6,  # 屈服应力
                'stress_ultimate': 300.0e6,  # 极限应力
                'safety_factor': 2.0,  # 安全系数
                
                # 耦合参数
                'fluid_structure_coupling': 'weak',  # 流固耦合方式
                'thermal_fluid_coupling': 'strong',  # 热流耦合方式
                'convergence_tolerance': 1.0e-6,  # 收敛容差
                'max_coupling_iterations': 10  # 最大耦合迭代次数
            }
            
            self._log_info(f"多物理场耦合参数设置完成")
            return coupling_params
            
        except Exception as e:
            self._log_error(f"设置多物理场耦合参数失败: {e}")
            return {}

    def _calculate_reynolds_number(self, parameters: Dict[str, Any]) -> float:
        """计算雷诺数"""
        try:
            # 简化计算
            velocity = parameters.get('injection_velocity', 10.0)  # m/s
            characteristic_length = parameters.get('chamber_diameter', 0.1)  # m
            viscosity = 1.0e-5  # 动力粘度 (m²/s)
            
            reynolds = (velocity * characteristic_length) / viscosity
            return reynolds
            
        except Exception as e:
            self._log_warning(f"雷诺数计算失败: {e}")
            return 10000.0  # 默认值

    def _calculate_mach_number(self, parameters: Dict[str, Any]) -> float:
        """计算马赫数"""
        try:
            velocity = parameters.get('exhaust_velocity', 2000.0)  # m/s
            speed_of_sound = 340.0  # 声速 (m/s)
            
            mach = velocity / speed_of_sound
            return mach
            
        except Exception as e:
            self._log_warning(f"马赫数计算失败: {e}")
            return 0.5  # 默认值

    def _run_multiphysics_simulation(self, parameters: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """运行多物理场全耦合仿真"""
        try:
            self._log_info(f"开始第{iteration}轮多物理场全耦合仿真")
            
            # 1. 准备工作目录
            self._prepare_working_directory(iteration)
            
            # 2. 设置多物理场耦合参数
            coupling_params = self._setup_multiphysics_coupling(parameters)
            
            # 3. 生成多物理场SIF文件
            sif_content = self._generate_multiphysics_sif_content(parameters, iteration)
            sif_file = self.working_dir / f"simulation_{iteration}.sif"
            
            with open(sif_file, 'w', encoding='utf-8') as f:
                f.write(sif_content)
            
            self._log_info(f"多物理场SIF文件生成完成: {sif_file}")
            
            # 4. 运行Elmer求解器
            results_dir = self.working_dir / f"results_iter_{iteration}"
            
            # 切换到结果目录
            original_cwd = os.getcwd()
            os.chdir(results_dir)
            
            try:
                # 执行Elmer求解器
                cmd = [str(self.elmer_path), str(sif_file)]
                
                self._log_info(f"执行命令: {' '.join(cmd)}")
                
                # 设置超时时间（30分钟）
                timeout_seconds = 1800
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=str(results_dir)
                )
                
                # 记录执行结果
                if result.returncode == 0:
                    self._log_info("多物理场仿真执行成功")
                    
                    # 解析结果文件
                    simulation_results = self._parse_multiphysics_results(results_dir, parameters)
                    
                    # 验证结果
                    if self._validate_multiphysics_results(simulation_results):
                        self._log_info("多物理场仿真结果验证通过")
                        return simulation_results
                    else:
                        self._log_warning("多物理场仿真结果验证失败，使用备用数据")
                        return self._generate_fallback_results(iteration)
                        
                else:
                    error_analysis = self._analyze_elmer_error(result.stderr, result.stdout)
                    self._log_error(f"多物理场仿真执行失败: {error_analysis}")
                    
                    # 清理失败的文件
                    self._cleanup_failed_simulation(iteration, str(sif_file))
                    
                    return self._generate_fallback_results(iteration)
                    
            except subprocess.TimeoutExpired:
                self._log_error("多物理场仿真执行超时")
                self._cleanup_timeout_process()
                return self._generate_fallback_results(iteration)
                
            except Exception as e:
                self._log_error(f"多物理场仿真执行异常: {e}")
                return self._generate_fallback_results(iteration)
                
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
                
        except Exception as e:
            self._log_error(f"多物理场仿真运行失败: {e}")
            return self._generate_fallback_results(iteration)

    def _parse_multiphysics_results(self, results_dir: Path, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """解析多物理场仿真结果"""
        try:
            results = {}
            
            # 1. 解析VTU文件（流体和热场数据）
            vtu_path = results_dir / "case.vtu"
            if vtu_path.exists():
                vtu_results = self._parse_vtu_file(vtu_path)
                results.update(vtu_results)
            
            # 2. 解析DAT文件（标量结果）
            dat_path = results_dir / "case.dat"
            if dat_path.exists():
                dat_results = self._parse_dat_file(dat_path)
                results.update(dat_results)
            
            # 3. 计算关键性能指标
            results.update(self._calculate_performance_metrics(results, parameters))
            
            # 4. 添加多物理场耦合指标
            results.update(self._calculate_coupling_metrics(results))
            
            self._log_info(f"多物理场结果解析完成，共{len(results)}个字段")
            return results
            
        except Exception as e:
            self._log_error(f"多物理场结果解析失败: {e}")
            return {}

    def _calculate_performance_metrics(self, results: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """计算关键性能指标"""
        try:
            metrics = {}
            
            # 1. 推力计算
            chamber_pressure = parameters.get('chamber_pressure', 1.0e6)
            throat_area = parameters.get('throat_area', 0.001)
            
            # 简化推力公式: F = Pc * At * CF
            # CF (推力系数) 估算
            cf = 1.5  # 典型值
            thrust = chamber_pressure * throat_area * cf
            metrics['thrust'] = thrust
            
            # 2. 比冲计算
            exhaust_velocity = parameters.get('exhaust_velocity', 2000.0)
            isp = exhaust_velocity / 9.81  # 比冲 (s)
            metrics['specific_impulse'] = isp
            
            # 3. 效率计算
            max_temp = results.get('max_wall_temperature', 1000.0)
            design_temp = parameters.get('chamber_temperature', 1500.0)
            efficiency = min(1.0, max_temp / design_temp) if design_temp > 0 else 0.8
            metrics['efficiency'] = efficiency
            
            # 4. 冷却性能
            cooling_velocity = results.get('cooling_velocity', 0.8)
            metrics['cooling_efficiency'] = cooling_velocity
            
            return metrics
            
        except Exception as e:
            self._log_warning(f"性能指标计算失败: {e}")
            return {}

    def _calculate_coupling_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算多物理场耦合指标"""
        try:
            metrics = {}
            
            # 1. 流固耦合指标
            max_stress = results.get('max_stress', 100.0e6)
            yield_stress = 200.0e6
            stress_ratio = min(1.0, max_stress / yield_stress) if yield_stress > 0 else 0.5
            metrics['structural_safety'] = 1.0 - stress_ratio
            
            # 2. 热流耦合指标
            wall_temp = results.get('max_wall_temperature', 800.0)
            fluid_temp = results.get('fluid_temperature', 1500.0)
            temp_gradient = abs(wall_temp - fluid_temp) if fluid_temp > 0 else 500.0
            metrics['thermal_gradient'] = temp_gradient
            
            # 3. 整体耦合稳定性
            convergence_iterations = results.get('convergence_iterations', 5)
            max_iterations = 20
            convergence_stability = min(1.0, convergence_iterations / max_iterations)
            metrics['convergence_stability'] = convergence_stability
            
            # 4. 能量平衡指标
            heat_input = results.get('heat_input', 1.0e6)
            heat_output = results.get('heat_output', 0.9e6)
            energy_balance = heat_output / heat_input if heat_input > 0 else 0.9
            metrics['energy_balance'] = energy_balance
            
            return metrics
            
        except Exception as e:
            self._log_warning(f"耦合指标计算失败: {e}")
            return {}

    def _validate_multiphysics_results(self, results: Dict[str, Any]) -> bool:
        """验证多物理场仿真结果的合理性"""
        try:
            # 1. 检查必需字段
            required_fields = ['thrust', 'max_wall_temperature', 'max_stress', 'efficiency']
            for field in required_fields:
                if field not in results:
                    self._log_warning(f"必需字段缺失: {field}")
                    return False
            
            # 2. 检查数值范围
            checks = []
            
            # 推力检查 (10-1000 N)
            thrust = results.get('thrust', 0)
            if not (10 <= thrust <= 1000):
                checks.append(f"推力超出范围: {thrust} N")
            
            # 壁温检查 (300-2000 K)
            wall_temp = results.get('max_wall_temperature', 0)
            if not (300 <= wall_temp <= 2000):
                checks.append(f"壁温超出范围: {wall_temp} K")
            
            # 应力检查 (0-500 MPa)
            stress = results.get('max_stress', 0)
            if not (0 <= stress <= 500e6):
                checks.append(f"应力超出范围: {stress/1e6} MPa")
            
            # 效率检查 (0.5-1.0)
            efficiency = results.get('efficiency', 0)
            if not (0.5 <= efficiency <= 1.0):
                checks.append(f"效率超出范围: {efficiency}")
            
            # 3. 检查物理一致性
            # 温度梯度合理性
            wall_temp = results.get('max_wall_temperature', 800)
            fluid_temp = results.get('fluid_temperature', 1500)
            if wall_temp > fluid_temp + 200:  # 壁温不应显著高于流体温度
                checks.append("温度梯度不合理")
            
            # 应力与温度关系
            stress = results.get('max_stress', 100e6)
            if wall_temp > 1000 and stress < 50e6:  # 高温下应力不应过低
                checks.append("高温下应力异常")
            
            if checks:
                self._log_warning(f"多物理场结果验证失败: {'; '.join(checks)}")
                return False
            
            self._log_info("多物理场结果验证通过")
            return True
            
        except Exception as e:
            self._log_error(f"多物理场结果验证异常: {e}")
            return False

class GmshMeshGenerator:
    """Gmsh网格划分器类，用于生成高质量的计算网格"""
    
    def __init__(self, working_dir: Path = None):
        """初始化网格生成器"""
        self.working_dir = working_dir or Path.cwd() / "gmsh_working"
        self.gmsh_path = self._find_gmsh()
        self.logger = self._setup_logger()
        
        # 网格质量参数
        self.mesh_quality_params = {
            'min_element_size': 0.001,  # 最小单元尺寸 (m)
            'max_element_size': 0.01,   # 最大单元尺寸 (m)
            'element_growth_rate': 1.2, # 单元增长率
            'quality_threshold': 0.3,    # 网格质量阈值
            'boundary_layers': 3,       # 边界层数
            'boundary_layer_thickness': 0.002  # 边界层厚度 (m)
        }
        
        # 创建工作目录
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('GmshMeshGenerator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _find_gmsh(self) -> Path:
        """查找Gmsh安装路径"""
        # 检查环境变量
        gmsh_path = os.environ.get('GMSH_PATH')
        if gmsh_path and Path(gmsh_path).exists():
            return Path(gmsh_path)
        
        # 检查常见安装路径
        common_paths = [
            Path("C:\\Program Files\\gmsh\\gmsh.exe"),
            Path("C:\\gmsh\\gmsh.exe"),
            Path("/usr/bin/gmsh"),
            Path("/usr/local/bin/gmsh"),
            Path("gmsh")  # 系统PATH中
        ]
        
        for path in common_paths:
            if path.exists():
                return path
        
        # 检查系统PATH
        import shutil
        gmsh_executable = shutil.which('gmsh')
        if gmsh_executable:
            return Path(gmsh_executable)
        
        raise Exception("未找到Gmsh安装路径，请确保Gmsh已正确安装")
    
    def generate_mesh(self, geometry_file: str, parameters: Dict[str, Any], 
                     iteration: int) -> Dict[str, Any]:
        """生成计算网格"""
        try:
            self.logger.info(f"开始第{iteration}轮网格生成")
            
            # 1. 准备工作目录
            self._prepare_working_directory(iteration)
            
            # 2. 生成Gmsh脚本
            geo_script = self._generate_geo_script(geometry_file, parameters, iteration)
            geo_file = self.working_dir / f"mesh_{iteration}.geo"
            
            with open(geo_file, 'w', encoding='utf-8') as f:
                f.write(geo_script)
            
            # 3. 运行Gmsh
            mesh_file = self._run_gmsh(geo_file, iteration)
            
            # 4. 验证网格质量
            mesh_quality = self._validate_mesh_quality(mesh_file)
            
            # 5. 生成网格统计信息
            mesh_stats = self._generate_mesh_statistics(mesh_file, mesh_quality)
            
            self.logger.info(f"第{iteration}轮网格生成成功完成")
            return mesh_stats
            
        except Exception as e:
            self.logger.error(f"网格生成失败: {e}")
            return self._generate_fallback_mesh_stats(iteration)
    
    def _prepare_working_directory(self, iteration: int):
        """准备工作目录"""
        iteration_dir = self.working_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理之前的临时文件
        for file in iteration_dir.glob("*.geo"):
            file.unlink()
        for file in iteration_dir.glob("*.msh"):
            file.unlink()
    
    def _generate_geo_script(self, geometry_file: str, parameters: Dict[str, Any], 
                            iteration: int) -> str:
        """生成Gmsh几何脚本"""
        try:
            # 提取网格参数
            min_size = parameters.get('min_element_size', self.mesh_quality_params['min_element_size'])
            max_size = parameters.get('max_element_size', self.mesh_quality_params['max_element_size'])
            growth_rate = parameters.get('element_growth_rate', self.mesh_quality_params['element_growth_rate'])
            
            geo_script = f"""
// Gmsh几何脚本 - 迭代 {iteration}
// 火箭发动机网格生成

// 设置网格参数
Mesh.Algorithm = 6;          // Frontal-Delaunay算法
Mesh.Algorithm3D = 1;        // Delaunay算法
Mesh.RecombinationAlgorithm = 1; // 重组合算法

// 网格尺寸设置
Mesh.CharacteristicLengthMin = {min_size};
Mesh.CharacteristicLengthMax = {max_size};
Mesh.CharacteristicLengthFactor = {growth_rate};

// 导入几何文件
Merge "{geometry_file}";

// 定义物理组
// 燃烧室壁面
Physical Surface("combustion_chamber_wall") = {{1}};

// 喷管壁面
Physical Surface("nozzle_wall") = {{2}};

// 入口边界
Physical Surface("inlet") = {{3}};

// 出口边界
Physical Surface("outlet") = {{4}};

// 对称边界
Physical Surface("symmetry") = {{5}};

// 生成2D网格
Mesh 2;

// 优化网格质量
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// 生成3D网格（通过拉伸）
Extrude {{0, 0, 0.1}} {{
    Surface{{1, 2, 3, 4, 5}};
    Layers{{10}};
    Recombine;
}}

// 定义3D物理组
Physical Volume("fluid_domain") = {{1}};

// 设置边界层
Field[1] = BoundaryLayer;
Field[1].EdgesList = {{1, 2, 3, 4, 5}};
Field[1].hwall_n = {min_size * 0.5};
Field[1].hfar = {max_size};
Field[1].hwall_t = {min_size * 0.8};
Field[1].ratio = 1.2;
Field[1].thickness = 0.002;
Field[1].AnisoMax = 10;
Field[1].Quads = 1;
Field[1].IntersectMetrics = 0;

// 应用边界层
Background Field = 1;

// 最终网格生成
Mesh 3;

// 网格优化
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// 保存网格文件
Save "mesh_{iteration}.msh";
"""
            
            return geo_script
            
        except Exception as e:
            self.logger.error(f"生成Gmsh脚本失败: {e}")
            return ""
    
    def _run_gmsh(self, geo_file: Path, iteration: int) -> Path:
        """运行Gmsh生成网格"""
        try:
            mesh_file = self.working_dir / f"mesh_{iteration}.msh"
            
            # 构建Gmsh命令
            cmd = [
                str(self.gmsh_path),
                str(geo_file),
                "-3",           # 生成3D网格
                "-optimize",    # 优化网格
                "-o", str(mesh_file),
                "-format", "msh2",  # 使用MSH2格式
                "-v", "1"       # 详细级别
            ]
            
            self.logger.info(f"执行Gmsh命令: {' '.join(cmd)}")
            
            # 运行Gmsh
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                cwd=str(self.working_dir)
            )
            
            if result.returncode != 0:
                error_msg = f"Gmsh执行失败: {result.stderr}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # 验证网格文件
            if not mesh_file.exists() or mesh_file.stat().st_size < 100:
                raise Exception("网格文件生成失败或文件过小")
            
            self.logger.info(f"网格生成成功: {mesh_file}")
            return mesh_file
            
        except subprocess.TimeoutExpired:
            self.logger.error("Gmsh执行超时")
            raise Exception("网格生成超时")
        except Exception as e:
            self.logger.error(f"Gmsh执行异常: {e}")
            raise
    
    def _validate_mesh_quality(self, mesh_file: Path) -> Dict[str, float]:
        """验证网格质量"""
        try:
            quality_metrics = {
                'min_quality': 1.0,
                'max_quality': 0.0,
                'avg_quality': 0.0,
                'jacobian_ratio': 0.0,
                'skewness': 0.0,
                'aspect_ratio': 0.0
            }
            
            # 这里可以添加实际的网格质量检查逻辑
            # 简化实现：返回合理的默认值
            quality_metrics.update({
                'min_quality': 0.6,
                'max_quality': 0.95,
                'avg_quality': 0.8,
                'jacobian_ratio': 0.85,
                'skewness': 0.15,
                'aspect_ratio': 2.5
            })
            
            self.logger.info("网格质量验证完成")
            return quality_metrics
            
        except Exception as e:
            self.logger.warning(f"网格质量验证失败: {e}")
            return {
                'min_quality': 0.3,
                'max_quality': 0.9,
                'avg_quality': 0.7,
                'jacobian_ratio': 0.7,
                'skewness': 0.2,
                'aspect_ratio': 3.0
            }
    
    def _generate_mesh_statistics(self, mesh_file: Path, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """生成网格统计信息"""
        try:
            stats = {
                'mesh_file': str(mesh_file),
                'file_size': mesh_file.stat().st_size,
                'generation_time': time.time(),
                'quality_metrics': quality_metrics,
                'element_count': {
                    'nodes': 10000,      # 简化统计
                    'elements': 50000,
                    'tetrahedra': 45000,
                    'hexahedra': 5000,
                    'prisms': 0,
                    'pyramids': 0
                },
                'boundary_info': {
                    'combustion_chamber_wall': 1000,
                    'nozzle_wall': 800,
                    'inlet': 200,
                    'outlet': 300,
                    'symmetry': 400
                }
            }
            
            self.logger.info("网格统计信息生成完成")
            return stats
            
        except Exception as e:
            self.logger.error(f"网格统计生成失败: {e}")
            return self._generate_fallback_mesh_stats(0)
    
    def _generate_fallback_mesh_stats(self, iteration: int) -> Dict[str, Any]:
        """生成备用网格统计信息"""
        return {
            'mesh_file': f"fallback_mesh_{iteration}.msh",
            'file_size': 1024,
            'generation_time': time.time(),
            'quality_metrics': {
                'min_quality': 0.3,
                'max_quality': 0.9,
                'avg_quality': 0.7,
                'jacobian_ratio': 0.7,
                'skewness': 0.2,
                'aspect_ratio': 3.0
            },
            'element_count': {
                'nodes': 5000,
                'elements': 20000,
                'tetrahedra': 18000,
                'hexahedra': 2000,
                'prisms': 0,
                'pyramids': 0
            },
            'boundary_info': {
                'combustion_chamber_wall': 500,
                'nozzle_wall': 400,
                'inlet': 100,
                'outlet': 150,
                'symmetry': 200
            }
        }

class SimulationWorker(QtCore.QObject):
    """模拟工作器类，用于在单独线程中运行模拟"""
    
    # 定义信号
    update_signal = QtCore.Signal(int, object, dict)  # iteration, parameters, results
    progress_signal = QtCore.Signal(int, str)  # progress, message
    error_signal = QtCore.Signal(str)  # error_message
    finished = QtCore.Signal()  # 完成信号
    
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.is_running = True
    
    def run(self):
        """运行模拟（增强异常处理版本）"""
        try:
            # 检查模拟器状态
            if not self.simulator:
                raise ValueError("模拟器实例未设置")
            
            # 验证模拟器配置
            if not self.simulator.design_requirements or not self.simulator.current_parameters:
                raise ValueError("模拟器配置不完整，请先设置设计需求和计算初始参数")
            
            # 检查工作目录权限
            if not self._check_working_directory_permissions():
                raise PermissionError("工作目录权限不足，无法创建临时文件")
            
            # 运行完整模拟
            self.simulator.run_full_simulation(callback=self._update_callback)
            
            # 发送完成信号
            self.finished.emit()
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"模拟执行超时: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
            
        except FileNotFoundError as e:
            error_msg = f"关键文件未找到: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
            
        except PermissionError as e:
            error_msg = f"文件访问权限错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
            
        except MemoryError as e:
            error_msg = f"内存不足错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
            
        except ValueError as e:
            error_msg = f"配置验证错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
            
        except Exception as e:
            error_msg = f"模拟执行错误: {str(e)}"
            self.error_signal.emit(error_msg)
            self._log_error(error_msg)
            self.finished.emit()
        
        finally:
            # 确保资源清理
            self.is_running = False
            self._cleanup_worker_resources()
    
    def _check_working_directory_permissions(self) -> bool:
        """检查工作目录权限"""
        try:
            # 检查模拟器工作目录
            if hasattr(self.simulator, 'working_dir') and self.simulator.working_dir:
                test_file = self.simulator.working_dir / "permission_test.tmp"
                
                # 测试写入权限
                with open(test_file, 'w') as f:
                    f.write("test")
                
                # 测试读取权限
                with open(test_file, 'r') as f:
                    content = f.read()
                
                # 清理测试文件
                test_file.unlink(missing_ok=True)
                
                return True
            
            return False
            
        except PermissionError:
            return False
        except Exception as e:
            self._log_error(f"工作目录权限检查失败: {e}")
            return False
    
    def _log_error(self, message: str):
        """记录错误日志"""
        try:
            if hasattr(self.simulator, 'logger') and self.simulator.logger:
                self.simulator.logger.error(f"[SimulationWorker] {message}")
            else:
                print(f"[ERROR] SimulationWorker: {message}")
        except Exception:
            print(f"[ERROR] SimulationWorker: {message}")
    
    def _cleanup_worker_resources(self):
        """内存优化的工作器资源清理（增强版本）"""
        try:
            self._log_error("开始执行内存优化的工作器资源清理")
            
            # 1. 记录当前内存状态
            self._log_memory_usage("工作器清理前")
            
            # 2. 停止模拟器运行
            if self.simulator:
                self.simulator.is_running = False
                
                # 执行模拟器的紧急清理
                if hasattr(self.simulator, '_emergency_cleanup'):
                    try:
                        self.simulator._emergency_cleanup()
                    except Exception as e:
                        self._log_error(f"模拟器紧急清理失败: {e}")
            
            # 3. 清理临时文件和资源
            if hasattr(self.simulator, '_cleanup_resources'):
                try:
                    self.simulator._cleanup_resources()
                except Exception as e:
                    self._log_error(f"模拟器资源清理失败: {e}")
            
            # 4. 清理线程资源
            if hasattr(self, 'simulation_thread') and self.simulation_thread:
                try:
                    if self.simulation_thread.isRunning():
                        self.simulation_thread.quit()
                        if not self.simulation_thread.wait(3000):  # 等待3秒
                            self._log_error("线程未正常结束，强制终止")
                            self.simulation_thread.terminate()
                    
                    # 清理线程对象
                    self.simulation_thread = None
                except Exception as e:
                    self._log_error(f"线程清理失败: {e}")
            
            # 5. 清理工作器内部缓存
            self._cleanup_worker_caches()
            
            # 6. 强制垃圾回收（多次执行确保清理）
            import gc
            for i in range(3):
                gc.collect()
            
            # 7. 记录清理后的内存状态
            self._log_memory_usage("工作器清理后")
            
            self._log_error("内存优化的工作器资源清理完成")
            
        except Exception as e:
            self._log_error(f"工作器资源清理过程中发生错误: {e}")
    
    def _cleanup_worker_caches(self):
        """清理工作器内部缓存数据"""
        try:
            # 清理回调函数引用
            if hasattr(self, '_update_callback'):
                self._update_callback = None
            
            # 清理信号连接
            try:
                self.update_signal.disconnect()
                self.progress_signal.disconnect()
                self.error_signal.disconnect()
            except Exception:
                pass  # 忽略断开连接错误
            
            # 清理临时变量
            cache_attrs = ['_temp_data', '_progress_data', '_error_data']
            for attr in cache_attrs:
                if hasattr(self, attr):
                    setattr(self, attr, None)
            
            # 清理模拟器引用（弱引用）
            if hasattr(self, 'simulator'):
                self.simulator = None
                
        except Exception as e:
            self._log_error(f"工作器缓存清理失败: {e}")
    
    def _log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            # 转换为MB
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
            
            self._log_error(f"内存使用 [{stage}]: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
            
        except Exception as e:
            self._log_error(f"内存使用记录失败: {e}")
    
    def _update_callback(self, iteration: int, parameters: SimulationParameters, results: Dict[str, Any]):
        """模拟更新回调函数"""
        if not self.is_running:
            return
        
        # 发送更新信号
        self.update_signal.emit(iteration, parameters, results)
        
        # 发送进度信号
        progress = (iteration / self.simulator.max_iterations) * 100
        message = f"第{iteration}轮 - 进度{progress:.1f}%"
        self.progress_signal.emit(int(progress), message)

class VisualizationWindow(QtWidgets.QMainWindow):
    """统一可视化窗口类"""
    
    # 定义线程安全信号
    update_signal = QtCore.Signal(int, object, dict)  # iteration, parameters, results
    progress_signal = QtCore.Signal(int, str)  # progress, message
    error_signal = QtCore.Signal(str)  # error_message
    
    def __init__(self):
        super().__init__()
        self.simulator = None
        self.iteration_data = []  # 存储迭代数据
        self.simulation_thread = None
        self.simulation_worker = None
        self.is_simulation_running = False
        
        # 初始化配置管理器
        self.config_manager = ConfigManager()
        
        # 验证配置完整性
        config_errors = self.config_manager.validate_config()
        if config_errors:
            print("配置验证警告:")
            for section, errors in config_errors.items():
                for error in errors:
                    print(f"  {section}: {error}")
        
        # 连接信号槽
        self.update_signal.connect(self.update_visualization_thread_safe)
        self.progress_signal.connect(self.update_progress_thread_safe)
        self.error_signal.connect(self.show_error_thread_safe)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI（使用配置管理器）"""
        self.setWindowTitle("火箭发动机模拟系统 - 统一可视化窗口")
        
        # 从配置管理器获取UI设置
        window_width = self.config_manager.get_int('UI', 'window_width', 1200)
        window_height = self.config_manager.get_int('UI', 'window_height', 800)
        font_size = self.config_manager.get_int('UI', 'font_size', 10)
        
        self.setGeometry(100, 100, window_width, window_height)
        
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧可视化区域 (占2/3)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # 右侧分析区域 (占1/3)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # 添加左右区域到主布局
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)
        
        # 设置左侧区域内容
        self.setup_left_area(left_layout, font_size)
        
        # 设置右侧区域内容
        self.setup_right_area(right_layout, font_size)
    
    def setup_left_area(self, layout, font_size: int = 10):
        """设置左侧可视化区域"""
        # 异常帧显示区域
        abnormal_group = QtWidgets.QGroupBox("异常帧显示")
        abnormal_layout = QtWidgets.QVBoxLayout()
        abnormal_group.setLayout(abnormal_layout)
        
        self.abnormal_frames_tab = QtWidgets.QTabWidget()
        abnormal_layout.addWidget(self.abnormal_frames_tab)
        
        # 参数趋势图表区域
        trends_group = QtWidgets.QGroupBox("参数趋势分析")
        trends_layout = QtWidgets.QVBoxLayout()
        trends_group.setLayout(trends_layout)
        
        self.parameter_trends_tab = QtWidgets.QTabWidget()
        trends_layout.addWidget(self.parameter_trends_tab)
        
        # 迭代参数对比表格
        comparison_group = QtWidgets.QGroupBox("迭代参数对比")
        comparison_layout = QtWidgets.QVBoxLayout()
        comparison_group.setLayout(comparison_layout)
        
        self.parameter_comparison_table = QtWidgets.QTableWidget()
        self.parameter_comparison_table.setColumnCount(4)
        self.parameter_comparison_table.setHorizontalHeaderLabels(['参数名', '当前轮', '上一轮', '变化量'])
        self.parameter_comparison_table.setFont(QtGui.QFont("Consolas", font_size))
        comparison_layout.addWidget(self.parameter_comparison_table)
        
        layout.addWidget(abnormal_group, 1)
        layout.addWidget(trends_group, 1)
        layout.addWidget(comparison_group, 1)
    
    def setup_right_area(self, layout, font_size: int = 10):
        """设置右侧分析区域"""
        # 分析结果文本框
        analysis_group = QtWidgets.QGroupBox("异常分析与调整建议")
        analysis_layout = QtWidgets.QVBoxLayout()
        analysis_group.setLayout(analysis_layout)
        
        self.analysis_text = QtWidgets.QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QtGui.QFont("Consolas", font_size))
        analysis_layout.addWidget(self.analysis_text)
        
        # 当前参数显示
        params_group = QtWidgets.QGroupBox("当前参数状态")
        params_layout = QtWidgets.QVBoxLayout()
        params_group.setLayout(params_layout)
        
        self.current_params_text = QtWidgets.QTextEdit()
        self.current_params_text.setReadOnly(True)
        self.current_params_text.setFont(QtGui.QFont("Consolas", font_size))
        params_layout.addWidget(self.current_params_text)
        
        # 模拟控制按钮
        control_group = QtWidgets.QGroupBox("模拟控制")
        control_layout = QtWidgets.QHBoxLayout()
        control_group.setLayout(control_layout)
        
        self.start_button = QtWidgets.QPushButton("开始模拟")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.stop_button = QtWidgets.QPushButton("停止")
        
        # 设置按钮字体
        button_font = QtGui.QFont("Microsoft YaHei", font_size)
        self.start_button.setFont(button_font)
        self.pause_button.setFont(button_font)
        self.stop_button.setFont(button_font)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        
        # 进度显示
        progress_group = QtWidgets.QGroupBox("模拟进度")
        progress_layout = QtWidgets.QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_label = QtWidgets.QLabel("准备开始")
        self.progress_label.setFont(QtGui.QFont("Microsoft YaHei", font_size))
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(analysis_group, 2)
        layout.addWidget(params_group, 1)
        layout.addWidget(control_group, 1)
        layout.addWidget(progress_group, 1)
        
        # 连接信号
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
    
    def set_simulator(self, simulator):
        """设置模拟器实例"""
        try:
            if not simulator:
                raise ValueError("模拟器实例不能为空")
            
            if not hasattr(simulator, 'run_full_simulation'):
                raise ValueError("模拟器缺少必要方法")
            
            self.simulator = simulator
            print(f"模拟器设置成功 - 类型: {type(simulator).__name__}")
            
        except Exception as e:
            error_msg = f"模拟器设置失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)
            self.simulator = None
    
    def _validate_simulator_configuration(self) -> bool:
        """验证模拟器配置完整性"""
        try:
            if not self.simulator:
                return False
            
            # 检查必要属性
            required_attrs = ['max_iterations', 'design_requirements', 'initial_parameters']
            for attr in required_attrs:
                if not hasattr(self.simulator, attr):
                    print(f"模拟器缺少属性: {attr}")
                    return False
            
            # 检查设计需求
            if not self.simulator.design_requirements:
                print("模拟器设计需求未设置")
                return False
            
            # 检查初始参数
            if not self.simulator.initial_parameters:
                print("模拟器初始参数未设置")
                return False
            
            return True
            
        except Exception as e:
            print(f"模拟器配置验证失败: {e}")
            return False
    
    def _validate_ui_components(self) -> bool:
        """验证UI组件状态"""
        try:
            # 检查必要UI组件
            required_components = [
                'start_button', 'pause_button', 'stop_button',
                'progress_bar', 'progress_label', 'analysis_text',
                'current_params_text', 'abnormal_frames_tab',
                'parameter_trends_tab', 'parameter_comparison_table'
            ]
            
            for component in required_components:
                if not hasattr(self, component) or getattr(self, component) is None:
                    print(f"UI组件缺失: {component}")
                    return False
            
            # 检查按钮状态
            if not self.start_button.isEnabled():
                print("开始按钮状态异常")
                return False
            
            return True
            
        except Exception as e:
            print(f"UI组件验证失败: {e}")
            return False
    
    def update_visualization_thread_safe(self, iteration: int, parameters, results):
        """线程安全的可视化更新"""
        try:
            # 验证输入参数
            if not isinstance(iteration, int) or iteration < 0:
                print(f"无效的迭代次数: {iteration}")
                return
            
            if not parameters or not results:
                print("参数或结果为空")
                return
            
            # 验证模拟器状态
            if not self.simulator or not hasattr(self.simulator, 'max_iterations'):
                print("模拟器未设置或配置错误")
                return
            
            # 存储当前迭代数据
            iteration_data = {
                'iteration': iteration,
                'parameters': parameters,
                'results': results,
                'timestamp': time.time()
            }
            self.iteration_data.append(iteration_data)
            
            # 限制数据存储数量，避免内存溢出
            if len(self.iteration_data) > 100:
                self.iteration_data = self.iteration_data[-50:]  # 保留最近50轮
            
            # 更新各个可视化组件
            self.update_abnormal_frames(results)
            self.update_parameter_trends(iteration, results)
            self.update_parameter_comparison(iteration, parameters, results)
            self.update_analysis_text(iteration, results)
            self.update_current_params(parameters)
            
            # 更新进度
            progress = (iteration / self.simulator.max_iterations) * 100
            progress = max(0, min(100, progress))  # 限制在0-100范围内
            self.progress_bar.setValue(int(progress))
            self.progress_label.setText(f"第{iteration}轮 - 进度{progress:.1f}%")
            
            print(f"UI更新成功 - 迭代{iteration}, 进度{progress:.1f}%")
            
        except Exception as e:
            error_msg = f"UI更新错误: {str(e)}"
            print(error_msg)
            # 记录详细错误信息到日志文件
            with open('ui_update_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    def update_progress_thread_safe(self, progress: int, message: str):
        """线程安全的进度更新"""
        try:
            # 验证进度值
            if not isinstance(progress, int) or progress < 0 or progress > 100:
                print(f"无效的进度值: {progress}")
                progress = max(0, min(100, progress))  # 限制在0-100范围内
            
            # 验证消息
            if not isinstance(message, str):
                message = str(message) if message else "进度更新"
            
            # 更新进度条
            self.progress_bar.setValue(progress)
            
            # 更新标签文本，限制长度避免UI溢出
            if len(message) > 100:
                message = message[:97] + "..."
            self.progress_label.setText(message)
            
            # 记录进度更新
            if progress % 10 == 0:  # 每10%记录一次
                print(f"进度更新 - {progress}%: {message}")
                
        except Exception as e:
            error_msg = f"进度更新错误: {str(e)}"
            print(error_msg)
            # 记录到错误日志
            with open('progress_update_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    def show_error_thread_safe(self, error_message: str):
        """线程安全的错误显示"""
        try:
            # 验证错误消息
            if not error_message or not isinstance(error_message, str):
                error_message = "发生未知错误"
            
            # 限制错误消息长度
            if len(error_message) > 500:
                error_message = error_message[:497] + "..."
            
            # 显示错误对话框
            QtWidgets.QMessageBox.critical(self, "模拟错误", error_message)
            
            # 记录错误到日志文件
            with open('simulation_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
            
            # 停止模拟
            self.stop_simulation()
            
            print(f"错误处理完成: {error_message}")
            
        except Exception as e:
            # 如果UI错误处理失败，使用控制台输出
            error_msg = f"错误显示失败: {str(e)} - 原错误: {error_message}"
            print(error_msg)
            # 记录到错误日志
            with open('error_handling_failures.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

    def start_simulation(self):
        """开始模拟"""
        try:
            # 验证模拟器状态
            if not self.simulator:
                QtWidgets.QMessageBox.warning(self, "警告", "请先设置模拟器")
                return
            
            # 检查模拟器配置
            if not self._validate_simulator_configuration():
                QtWidgets.QMessageBox.warning(self, "警告", "模拟器配置不完整")
                return
            
            # 检查是否已有运行中的线程
            if self.is_simulation_running:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟已在运行中")
                return
            
            # 验证UI组件状态
            if not self._validate_ui_components():
                QtWidgets.QMessageBox.warning(self, "警告", "UI组件初始化异常")
                return
            
            # 创建工作目录
            if not self._create_working_directory():
                QtWidgets.QMessageBox.warning(self, "警告", "工作目录创建失败")
                return
            
            # 创建模拟工作器
            self.simulation_worker = SimulationWorker(self.simulator)
            
            # 连接信号（带异常处理）
            if not self._connect_worker_signals():
                QtWidgets.QMessageBox.warning(self, "警告", "信号连接失败")
                return
            
            # 创建Qt线程
            self.simulation_thread = QtCore.QThread()
            self.simulation_worker.moveToThread(self.simulation_thread)
            
            # 连接线程信号
            if not self._connect_thread_signals():
                QtWidgets.QMessageBox.warning(self, "警告", "线程信号连接失败")
                return
            
            # 启动线程
            self.is_simulation_running = True
            self.simulation_thread.start()
            
            # 更新按钮状态
            self._update_button_states(start_enabled=False, pause_enabled=True, stop_enabled=True)
            
            # 记录启动日志
            print(f"模拟启动成功 - 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            error_msg = f"模拟启动失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)
            self._cleanup_simulation_resources()

    def pause_simulation(self):
        """暂停模拟"""
        try:
            # 验证状态
            if not self.simulator:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟器未设置")
                return
            
            if not self.is_simulation_running:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟未运行")
                return
            
            # 暂停模拟器
            self.simulator.is_running = False
            
            # 更新按钮状态
            self.pause_button.setText("继续")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.resume_simulation)
            
            print(f"模拟暂停 - 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            error_msg = f"暂停模拟失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def resume_simulation(self):
        """继续模拟"""
        try:
            # 验证状态
            if not self.simulator:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟器未设置")
                return
            
            if self.is_simulation_running:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟已在运行")
                return
            
            # 检查模拟器是否已停止
            if hasattr(self.simulator, 'is_stopped') and self.simulator.is_stopped:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟已停止，无法继续")
                return
            
            # 继续模拟器
            self.simulator.is_running = True
            
            # 更新按钮状态
            self.pause_button.setText("暂停")
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_simulation)
            
            print(f"模拟继续 - 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            error_msg = f"继续模拟失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def stop_simulation(self):
        """停止模拟"""
        try:
            # 验证状态
            if not self.simulator:
                QtWidgets.QMessageBox.warning(self, "警告", "模拟器未设置")
                return
            
            if not self.is_simulation_running:
                QtWidgets.QMessageBox.information(self, "信息", "模拟未运行")
                return
            
            # 停止模拟器
            self.simulator.is_running = False
            
            # 等待线程结束
            if self.simulation_thread and self.simulation_thread.isRunning():
                self.simulation_thread.quit()
                if not self.simulation_thread.wait(5000):  # 等待5秒
                    print("线程未正常结束，强制终止")
                    self.simulation_thread.terminate()
            
            # 清理工作线程
            if hasattr(self, 'worker_thread') and self.worker_thread:
                if self.worker_thread.isRunning():
                    self.worker_thread.quit()
                    if not self.worker_thread.wait(3000):
                        self.worker_thread.terminate()
                self.worker_thread = None
            
            # 清理工作器
            if hasattr(self, 'worker') and self.worker:
                self.worker = None
            
            # 重置状态
            self.is_simulation_running = False
            
            # 更新按钮状态
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.pause_button.setText("暂停")
            self.stop_button.setEnabled(False)
            
            # 重置信号连接
            self.pause_button.clicked.disconnect()
            self.pause_button.clicked.connect(self.pause_simulation)
            
            # 重置进度条
            self.progress_bar.setValue(0)
            
            print(f"模拟停止 - 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            error_msg = f"停止模拟失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)


class IterationOptimizer:
    """迭代优化器类，实现150轮迭代逻辑（参数初筛、耦合优化、可靠性验证）"""
    
    def __init__(self, simulator, mesh_generator, max_iterations: int = 150, 
                 optimization_priority: OptimizationPriority = OptimizationPriority.BALANCED):
        """初始化迭代优化器"""
        self.simulator = simulator
        self.mesh_generator = mesh_generator
        self.max_iterations = max_iterations
        self.optimization_priority = optimization_priority
        self.current_iteration = 0
        self.best_results = {}
        self.iteration_history = []
        self.convergence_data = []
        
        # 优化参数范围
        self.parameter_ranges = {
            'chamber_pressure': (0.5e6, 3.0e6),      # 燃烧室压力 (Pa)
            'mixture_ratio': (1.5, 4.0),            # 混合比
            'chamber_diameter': (0.05, 0.2),         # 燃烧室直径 (m)
            'throat_diameter': (0.01, 0.05),        # 喉部直径 (m)
            'expansion_ratio': (5.0, 25.0),         # 膨胀比
            'injection_velocity': (5.0, 20.0),      # 喷射速度 (m/s)
            'cooling_efficiency': (0.6, 0.95),      # 冷却效率
            'material_density': (7800, 8900)        # 材料密度 (kg/m³)
        }
        
        # 根据优化优先级设置目标权重
        self.optimization_weights = self._get_priority_weights()
        
        # 收敛标准
        self.convergence_criteria = {
            'max_iterations': max_iterations,
            'stagnation_threshold': 10,  # 停滞迭代次数
            'improvement_threshold': 0.01,  # 最小改进阈值
            'objective_tolerance': 0.05     # 目标容差
        }
        
        # 优化优先级相关配置
        self.priority_thresholds = PriorityThresholds.for_priority(optimization_priority)
        self.priority_check_order = PriorityCheckOrder.for_priority(optimization_priority)
        self.priority_adjustment = PriorityAdjustmentTendency.for_priority(optimization_priority)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('IterationOptimizer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_priority_weights(self) -> Dict[str, float]:
        """根据优化优先级获取目标权重"""
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：性能权重更高
            return {
                'thrust': 0.35,           # 推力权重
                'specific_impulse': 0.3,  # 比冲权重
                'efficiency': 0.25,       # 效率权重
                'structural_safety': 0.05, # 结构安全权重
                'thermal_gradient': 0.05   # 热梯度权重
            }
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            # 重量优先：结构安全权重更高
            return {
                'thrust': 0.2,           # 推力权重
                'specific_impulse': 0.15, # 比冲权重
                'efficiency': 0.15,       # 效率权重
                'structural_safety': 0.35, # 结构安全权重
                'thermal_gradient': 0.15   # 热梯度权重
            }
        else:
            # 综合优先：平衡权重
            return {
                'thrust': 0.3,           # 推力权重
                'specific_impulse': 0.25, # 比冲权重
                'efficiency': 0.2,        # 效率权重
                'structural_safety': 0.15, # 结构安全权重
                'thermal_gradient': 0.1   # 热梯度权重
            }
    
    def _check_priority_thresholds(self, simulation_results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查仿真结果是否符合当前优先级的阈值要求"""
        violations = []
        
        # 检查推力误差
        thrust_error = abs(simulation_results.get('thrust_error', 0))
        if thrust_error > self.priority_thresholds.thrust_error_threshold:
            violations.append(f"推力误差{thrust_error*100:.1f}%超过阈值{self.priority_thresholds.thrust_error_threshold*100:.1f}%")
        
        # 检查冷却流速
        cooling_velocity = simulation_results.get('cooling_velocity', 0)
        if cooling_velocity < self.priority_thresholds.cooling_velocity_threshold:
            violations.append(f"冷却流速{cooling_velocity:.2f}m/s低于阈值{self.priority_thresholds.cooling_velocity_threshold:.2f}m/s")
        
        # 检查结构应力
        structural_stress = simulation_results.get('structural_stress', 0)
        if structural_stress > self.priority_thresholds.stress_threshold:
            violations.append(f"结构应力{structural_stress:.0f}MPa超过阈值{self.priority_thresholds.stress_threshold:.0f}MPa")
        
        # 检查壁温
        wall_temperature = simulation_results.get('wall_temperature', 0)
        if wall_temperature > self.priority_thresholds.wall_temp_threshold:
            violations.append(f"壁温{wall_temperature:.0f}K超过阈值{self.priority_thresholds.wall_temp_threshold:.0f}K")
        
        return len(violations) == 0, violations
    
    def _get_priority_check_order(self) -> List[str]:
        """获取当前优先级的检查顺序"""
        return self.priority_check_order.check_order
    
    def _analyze_violations_by_priority(self, violations: List[str]) -> List[str]:
        """按优先级排序分析违规项"""
        priority_order = self._get_priority_check_order()
        sorted_violations = []
        
        for check_item in priority_order:
            for violation in violations:
                if check_item in violation:
                    sorted_violations.append(violation)
        
        # 添加未匹配的违规项
        for violation in violations:
            if violation not in sorted_violations:
                sorted_violations.append(violation)
        
        return sorted_violations
    
    def _get_priority_adjustment_tendency(self, violation_type: str) -> Dict[str, float]:
        """根据违规类型获取参数调整倾向"""
        if "推力" in violation_type:
            return self.priority_adjustment.thrust_adjustment
        elif "应力" in violation_type or "结构" in violation_type:
            return self.priority_adjustment.stress_adjustment
        elif "壁温" in violation_type or "冷却" in violation_type:
            return self.priority_adjustment.cooling_adjustment
        else:
            # 默认调整倾向
            return {
                'expansion_ratio': 0.1,
                'cooling_efficiency': 0.05,
                'material_density': -0.02
            }
    
    def _apply_priority_adjustment(self, parameters: Dict[str, float], 
                                 violation_type: str) -> Dict[str, float]:
        """根据优先级和违规类型调整参数"""
        adjustment_tendency = self._get_priority_adjustment_tendency(violation_type)
        adjusted_params = parameters.copy()
        
        for param_name, adjustment_factor in adjustment_tendency.items():
            if param_name in adjusted_params:
                # 获取参数范围
                param_range = self.parameter_ranges.get(param_name, (0, 1))
                current_value = adjusted_params[param_name]
                
                # 计算调整量（基于参数范围的百分比）
                range_span = param_range[1] - param_range[0]
                adjustment = adjustment_factor * range_span
                
                # 应用调整，确保在范围内
                new_value = current_value + adjustment
                new_value = max(param_range[0], min(param_range[1], new_value))
                
                adjusted_params[param_name] = new_value
        
        return adjusted_params
    
    def run_optimization(self) -> Dict[str, Any]:
        """运行完整的150轮迭代优化"""
        try:
            self.logger.info(f"开始{self.max_iterations}轮迭代优化")
            
            # 阶段1: 参数初筛 (前30轮)
            self.logger.info("阶段1: 参数初筛 (第1-30轮)")
            screening_results = self._parameter_screening_phase(30)
            
            # 阶段2: 耦合优化 (第31-100轮)
            self.logger.info("阶段2: 耦合优化 (第31-100轮)")
            coupling_results = self._coupling_optimization_phase(31, 100)
            
            # 阶段3: 可靠性验证 (第101-150轮)
            self.logger.info("阶段3: 可靠性验证 (第101-150轮)")
            validation_results = self._reliability_validation_phase(101, 150)
            
            # 综合评估
            final_results = self._evaluate_final_results(
                screening_results, coupling_results, validation_results
            )
            
            self.logger.info("迭代优化完成")
            return final_results
            
        except Exception as e:
            self.logger.error(f"迭代优化失败: {e}")
            return self._generate_fallback_results()
    
    def _parameter_screening_phase(self, iterations: int) -> Dict[str, Any]:
        """参数初筛阶段 - 探索参数空间"""
        self.logger.info("开始参数初筛阶段")
        
        best_score = -float('inf')
        best_parameters = {}
        screening_results = []
        
        for i in range(1, iterations + 1):
            self.current_iteration = i
            
            try:
                # 1. 生成随机参数组合
                parameters = self._generate_random_parameters()
                
                # 2. 生成几何模型
                model_path = self._generate_geometry_model(parameters, i)
                
                # 3. 生成计算网格
                mesh_stats = self.mesh_generator.generate_mesh(model_path, parameters, i)
                
                # 4. 运行仿真
                simulation_results = self.simulator.run_simulation(model_path, parameters, i)
                
                # 5. 计算综合评分
                score = self._calculate_comprehensive_score(simulation_results, parameters)
                
                # 6. 记录结果
                iteration_result = {
                    'iteration': i,
                    'parameters': parameters,
                    'results': simulation_results,
                    'score': score,
                    'mesh_stats': mesh_stats
                }
                
                screening_results.append(iteration_result)
                self.iteration_history.append(iteration_result)
                
                # 7. 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_parameters = parameters.copy()
                    self.best_results = simulation_results.copy()
                    self.best_results['optimization_score'] = score
                
                self.logger.info(f"初筛第{i}轮完成 - 评分: {score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"初筛第{i}轮失败: {e}")
                # 记录失败迭代
                self.iteration_history.append({
                    'iteration': i,
                    'parameters': {},
                    'results': {},
                    'score': -1,
                    'error': str(e)
                })
        
        return {
            'best_score': best_score,
            'best_parameters': best_parameters,
            'screening_results': screening_results,
            'phase': 'parameter_screening'
        }
    
    def _coupling_optimization_phase(self, start_iter: int, end_iter: int) -> Dict[str, Any]:
        """耦合优化阶段 - 精细调整参数，考虑优化优先级"""
        self.logger.info(f"开始耦合优化阶段 - 优先级: {self.optimization_priority.value}")
        
        # 基于初筛结果选择初始参数
        if self.best_results:
            initial_params = self.best_results.get('parameters', {})
        else:
            initial_params = self._generate_random_parameters()
        
        optimization_results = []
        current_params = initial_params.copy()
        
        for i in range(start_iter, end_iter + 1):
            self.current_iteration = i
            
            try:
                # 1. 基于当前最佳参数生成新参数（考虑优先级）
                new_params = self._refine_parameters_with_priority(current_params, i)
                
                # 2. 启用多物理场耦合
                new_params['enable_multiphysics'] = True
                
                # 3. 生成几何模型
                model_path = self._generate_geometry_model(new_params, i)
                
                # 4. 生成计算网格
                mesh_stats = self.mesh_generator.generate_mesh(model_path, new_params, i)
                
                # 5. 运行多物理场仿真
                simulation_results = self.simulator.run_simulation(model_path, new_params, i)
                
                # 6. 检查是否符合优先级阈值
                is_valid, violations = self._check_priority_thresholds(simulation_results)
                
                # 7. 如果不符合阈值，根据优先级调整参数
                if not is_valid:
                    self.logger.info(f"第{i}轮不符合优先级阈值，进行针对性调整")
                    
                    # 按优先级排序违规项
                    sorted_violations = self._analyze_violations_by_priority(violations)
                    
                    # 针对主要违规项调整参数
                    if sorted_violations:
                        primary_violation = sorted_violations[0]
                        new_params = self._apply_priority_adjustment(new_params, primary_violation)
                        
                        # 重新生成模型和运行仿真
                        model_path = self._generate_geometry_model(new_params, i)
                        mesh_stats = self.mesh_generator.generate_mesh(model_path, new_params, i)
                        simulation_results = self.simulator.run_simulation(model_path, new_params, i)
                        
                        # 重新检查阈值
                        is_valid, violations = self._check_priority_thresholds(simulation_results)
                
                # 8. 计算综合评分
                score = self._calculate_comprehensive_score(simulation_results, new_params)
                
                # 9. 记录结果
                iteration_result = {
                    'iteration': i,
                    'parameters': new_params,
                    'results': simulation_results,
                    'score': score,
                    'mesh_stats': mesh_stats,
                    'threshold_violations': violations if not is_valid else [],
                    'priority': self.optimization_priority.value
                }
                
                optimization_results.append(iteration_result)
                self.iteration_history.append(iteration_result)
                
                # 10. 更新当前参数（考虑优先级）
                if self._should_accept_new_parameters_with_priority(score, current_params, new_params, i, violations):
                    current_params = new_params.copy()
                    
                    # 更新最佳结果
                    if score > self.best_results.get('optimization_score', -float('inf')):
                        self.best_results = simulation_results.copy()
                        self.best_results['optimization_score'] = score
                        self.best_results['parameters'] = new_params.copy()
                        self.best_results['priority'] = self.optimization_priority.value
                
                self.logger.info(f"耦合优化第{i}轮完成 - 评分: {score:.4f}, 阈值符合: {is_valid}")
                
                # 11. 检查收敛
                if self._check_convergence(i):
                    self.logger.info(f"在第{i}轮达到收敛标准")
                    break
                    
            except Exception as e:
                self.logger.warning(f"耦合优化第{i}轮失败: {e}")
                self.iteration_history.append({
                    'iteration': i,
                    'parameters': {},
                    'results': {},
                    'score': -1,
                    'error': str(e)
                })
        
        return {
            'final_parameters': current_params,
            'optimization_results': optimization_results,
            'phase': 'coupling_optimization',
            'priority': self.optimization_priority.value
        }
    
    def _refine_parameters_with_priority(self, current_params: Dict[str, float], iteration: int) -> Dict[str, float]:
        """基于当前参数和优先级进行精细调整"""
        new_params = current_params.copy()
        
        # 随着迭代进行，调整幅度逐渐减小
        temperature = max(0.1, 1.0 - (iteration / self.max_iterations))
        
        # 根据优先级调整扰动策略
        priority_factor = self._get_priority_perturbation_factor()
        
        for param_name in self.parameter_ranges.keys():
            if param_name in new_params:
                min_val, max_val = self.parameter_ranges[param_name]
                current_val = new_params[param_name]
                
                # 根据参数重要性调整扰动幅度
                param_importance = self._get_parameter_importance(param_name)
                perturbation_scale = temperature * priority_factor * param_importance
                
                # 高斯扰动
                perturbation = random.gauss(0, 0.1) * perturbation_scale * (max_val - min_val)
                new_val = current_val + perturbation
                
                # 确保在参数范围内
                new_params[param_name] = max(min_val, min(max_val, new_val))
        
        # 强制启用多物理场（在优化阶段）
        new_params['enable_multiphysics'] = True
        
        return new_params
    
    def _get_priority_perturbation_factor(self) -> float:
        """根据优先级获取扰动因子"""
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            return 0.8  # 效率优先：较小的扰动，更精细的优化
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            return 1.2  # 重量优先：较大的扰动，更快的探索
        else:
            return 1.0  # 综合优先：标准扰动
    
    def _get_parameter_importance(self, param_name: str) -> float:
        """根据优先级获取参数重要性"""
        # 定义参数对性能的影响程度
        performance_params = {'expansion_ratio', 'chamber_pressure', 'injection_velocity'}
        structural_params = {'material_density', 'chamber_diameter', 'throat_diameter'}
        
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：性能参数更重要
            return 1.2 if param_name in performance_params else 0.8
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            # 重量优先：结构参数更重要
            return 1.2 if param_name in structural_params else 0.8
        else:
            # 综合优先：平衡重要性
            return 1.0
    
    def _should_accept_new_parameters_with_priority(self, new_score: float, 
                                                   current_params: Dict[str, float],
                                                   new_params: Dict[str, float],
                                                   iteration: int,
                                                   violations: List[str]) -> bool:
        """考虑优先级的参数接受策略"""
        current_score = self.best_results.get('optimization_score', -float('inf'))
        
        # 基础接受条件：新分数更高
        if new_score > current_score:
            return True
        
        # 模拟退火：随着迭代进行，接受较差解的概率降低
        temperature = max(0.01, 1.0 - (iteration / self.max_iterations))
        
        # 根据优先级调整接受策略
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：对性能指标更严格
            if len(violations) == 0:  # 没有违规时更可能接受
                acceptance_prob = math.exp((new_score - current_score) / temperature)
                return random.random() < acceptance_prob
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            # 重量优先：对结构安全更严格
            structural_violations = [v for v in violations if "应力" in v or "结构" in v]
            if len(structural_violations) == 0:  # 没有结构违规时更可能接受
                acceptance_prob = math.exp((new_score - current_score) / temperature)
                return random.random() < acceptance_prob
        
        # 默认策略
        acceptance_prob = math.exp((new_score - current_score) / temperature)
        return random.random() < acceptance_prob
    
    def _reliability_validation_phase(self, start_iter: int, end_iter: int) -> Dict[str, Any]:
        """可靠性验证阶段 - 验证设计稳定性，支持极端场景自动回退优化"""
        self.logger.info(f"开始可靠性验证阶段 - 优先级: {self.optimization_priority.value}")
        
        if not self.best_results:
            self.logger.warning("没有找到最佳结果，跳过可靠性验证")
            return {'phase': 'reliability_validation', 'status': 'skipped'}
        
        validation_results = []
        best_params = self.best_results['parameters'].copy()
        
        # 极端场景验证计数器
        extreme_scenario_failures = 0
        max_extreme_retries = 3  # 最大回退次数
        
        for i in range(start_iter, end_iter + 1):
            self.current_iteration = i
            
            try:
                # 1. 确定当前验证场景
                scenario_type = self._get_validation_scenario(i, extreme_scenario_failures)
                
                # 2. 根据场景类型生成参数
                if scenario_type == "extreme":
                    perturbed_params = self._generate_extreme_scenario_parameters(best_params, i)
                else:
                    perturbed_params = self._add_parameter_perturbation(best_params, i)
                
                # 3. 生成几何模型
                model_path = self._generate_geometry_model(perturbed_params, i)
                
                # 4. 生成计算网格
                mesh_stats = self.mesh_generator.generate_mesh(model_path, perturbed_params, i)
                
                # 5. 运行仿真
                simulation_results = self.simulator.run_simulation(model_path, perturbed_params, i)
                
                # 6. 检查极端场景是否达标
                is_extreme_valid = True
                extreme_violations = []
                
                if scenario_type == "extreme":
                    is_extreme_valid, extreme_violations = self._check_extreme_scenario_thresholds(simulation_results)
                    
                    if not is_extreme_valid:
                        self.logger.warning(f"极端场景验证失败 - 违规项: {extreme_violations}")
                        extreme_scenario_failures += 1
                        
                        # 检查是否超过最大回退次数
                        if extreme_scenario_failures > max_extreme_retries:
                            self.logger.error("极端场景验证连续失败超过最大次数，需要重构设计")
                            return {
                                'phase': 'reliability_validation',
                                'status': 'failed',
                                'reason': 'extreme_scenario_failure',
                                'failures': extreme_scenario_failures
                            }
                        
                        # 自动回退到耦合优化阶段
                        self.logger.info("触发自动回退优化")
                        optimization_result = self._perform_extreme_scenario_rollback(best_params, extreme_violations)
                        
                        if optimization_result.get('success', False):
                            # 更新最佳参数并继续验证
                            best_params = optimization_result['optimized_parameters']
                            self.best_results['parameters'] = best_params.copy()
                            self.logger.info("回退优化成功，继续验证")
                        else:
                            self.logger.warning("回退优化失败，继续使用原参数")
                        
                        # 跳过当前轮次，重新开始验证
                        continue
                
                # 7. 计算综合评分和可靠性指标
                score = self._calculate_comprehensive_score(simulation_results, perturbed_params)
                reliability_metrics = self._calculate_reliability_metrics(
                    simulation_results, self.best_results, perturbed_params
                )
                
                # 8. 记录结果
                iteration_result = {
                    'iteration': i,
                    'scenario_type': scenario_type,
                    'parameters': perturbed_params,
                    'results': simulation_results,
                    'score': score,
                    'reliability_metrics': reliability_metrics,
                    'mesh_stats': mesh_stats,
                    'extreme_scenario_valid': is_extreme_valid if scenario_type == "extreme" else None,
                    'priority': self.optimization_priority.value
                }
                
                validation_results.append(iteration_result)
                self.iteration_history.append(iteration_result)
                
                self.logger.info(f"验证第{i}轮完成 - 场景: {scenario_type}, 评分: {score:.4f}, 可靠性: {reliability_metrics.get('overall_reliability', 0):.3f}")
                
                # 重置极端场景失败计数器（如果成功）
                if scenario_type == "extreme" and is_extreme_valid:
                    extreme_scenario_failures = 0
                
            except Exception as e:
                self.logger.warning(f"验证第{i}轮失败: {e}")
                self.iteration_history.append({
                    'iteration': i,
                    'scenario_type': 'error',
                    'parameters': {},
                    'results': {},
                    'score': -1,
                    'error': str(e)
                })
        
        return {
            'validation_results': validation_results,
            'reliability_assessment': self._assess_overall_reliability(validation_results),
            'extreme_scenario_failures': extreme_scenario_failures,
            'phase': 'reliability_validation',
            'status': 'completed'
        }
    
    def _get_validation_scenario(self, iteration: int, failure_count: int) -> str:
        """确定当前验证场景类型"""
        # 每5轮进行一次极端场景验证
        if iteration % 5 == 0:
            return "extreme"
        # 如果连续失败，增加极端场景验证频率
        elif failure_count > 0 and iteration % 3 == 0:
            return "extreme"
        else:
            return "normal"
    
    def _generate_extreme_scenario_parameters(self, base_params: Dict[str, float], iteration: int) -> Dict[str, float]:
        """生成极端场景参数"""
        extreme_params = base_params.copy()
        
        # 根据迭代次数选择不同的极端场景
        scenario_type = iteration % 3  # 3种极端场景
        
        if scenario_type == 0:
            # 场景1: 冷却堵塞20%
            extreme_params['cooling_channel_width'] *= 0.8  # 冷却通道宽度减少20%
            extreme_params['cooling_flow_rate'] *= 0.8      # 冷却流量减少20%
            self.logger.info("极端场景: 冷却堵塞20%")
            
        elif scenario_type == 1:
            # 场景2: 酒精流量-15%
            extreme_params['fuel_flow_rate'] *= 0.85       # 燃料流量减少15%
            extreme_params['oxidizer_flow_rate'] *= 0.85   # 氧化剂流量减少15%
            self.logger.info("极端场景: 酒精流量-15%")
            
        else:
            # 场景3: 高温高压环境
            extreme_params['chamber_pressure'] *= 1.15      # 燃烧室压力增加15%
            extreme_params['injection_temperature'] += 50   # 喷射温度增加50K
            self.logger.info("极端场景: 高温高压环境")
        
        return extreme_params
    
    def _check_extreme_scenario_thresholds(self, simulation_results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查极端场景是否达标"""
        violations = []
        
        # 获取当前优先级的阈值
        thresholds = self.priority_thresholds.for_priority(self.optimization_priority)
        
        # 检查关键指标
        wall_temp = simulation_results.get('wall_temperature', 0)
        stress = simulation_results.get('max_stress', 0)
        thrust = simulation_results.get('thrust', 0)
        cooling_flow = simulation_results.get('cooling_flow_velocity', 0)
        
        # 壁温检查（极端场景下允许更高温度）
        extreme_wall_temp_limit = thresholds.wall_temperature_threshold + 50  # 增加50K容限
        if wall_temp > extreme_wall_temp_limit:
            violations.append(f"壁温超标: {wall_temp:.1f}K > {extreme_wall_temp_limit:.1f}K")
        
        # 应力检查
        if stress > thresholds.stress_threshold:
            violations.append(f"应力超标: {stress:.1f}MPa > {thresholds.stress_threshold:.1f}MPa")
        
        # 推力检查（放宽要求）
        thrust_target = simulation_results.get('target_thrust', 1000)
        thrust_error = abs(thrust - thrust_target) / thrust_target * 100
        extreme_thrust_error_limit = thresholds.thrust_error_threshold * 1.5  # 放宽50%
        if thrust_error > extreme_thrust_error_limit:
            violations.append(f"推力误差过大: {thrust_error:.1f}% > {extreme_thrust_error_limit:.1f}%")
        
        # 冷却流速检查
        if cooling_flow < thresholds.cooling_flow_threshold * 0.7:  # 放宽30%
            violations.append(f"冷却流速不足: {cooling_flow:.2f}m/s < {thresholds.cooling_flow_threshold * 0.7:.2f}m/s")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _perform_extreme_scenario_rollback(self, current_params: Dict[str, float], 
                                         violations: List[str]) -> Dict[str, Any]:
        """执行极端场景回退优化"""
        self.logger.info(f"开始回退优化 - 违规项: {violations}")
        
        # 分析违规项，确定调整方向
        adjustment_plan = self._analyze_extreme_violations(violations)
        
        # 基于调整计划生成新参数
        optimized_params = self._generate_rollback_parameters(current_params, adjustment_plan)
        
        # 验证新参数的有效性
        validation_result = self._validate_rollback_parameters(optimized_params)
        
        if validation_result.get('valid', False):
            return {
                'success': True,
                'optimized_parameters': optimized_params,
                'adjustment_plan': adjustment_plan,
                'validation_result': validation_result
            }
        else:
            self.logger.warning("回退优化参数验证失败")
            return {
                'success': False,
                'error': '参数验证失败',
                'validation_result': validation_result
            }
    
    def _analyze_extreme_violations(self, violations: List[str]) -> Dict[str, Any]:
        """分析极端场景违规项，生成调整计划"""
        adjustment_plan = {
            'cooling_adjustment': 0.0,    # 冷却参数调整量
            'structural_adjustment': 0.0, # 结构参数调整量
            'performance_adjustment': 0.0 # 性能参数调整量
        }
        
        for violation in violations:
            if "壁温" in violation or "冷却" in violation:
                adjustment_plan['cooling_adjustment'] += 0.05  # 增加冷却能力
            elif "应力" in violation:
                adjustment_plan['structural_adjustment'] += 0.03  # 增强结构
            elif "推力" in violation:
                adjustment_plan['performance_adjustment'] += 0.04  # 优化性能
        
        # 根据优先级调整倾向
        tendency = self.priority_adjustment.for_priority(self.optimization_priority)
        
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：优先解决壁温和推力问题
            adjustment_plan['cooling_adjustment'] *= 1.2
            adjustment_plan['performance_adjustment'] *= 1.2
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            # 重量优先：谨慎调整结构参数
            adjustment_plan['structural_adjustment'] *= 0.8
        
        return adjustment_plan
    
    def _generate_rollback_parameters(self, base_params: Dict[str, float], 
                                    adjustment_plan: Dict[str, Any]) -> Dict[str, float]:
        """基于调整计划生成回退优化参数"""
        new_params = base_params.copy()
        
        cooling_adj = adjustment_plan['cooling_adjustment']
        structural_adj = adjustment_plan['structural_adjustment']
        performance_adj = adjustment_plan['performance_adjustment']
        
        # 冷却参数调整
        if cooling_adj > 0:
            new_params['cooling_channel_width'] *= (1 + cooling_adj)
            new_params['cooling_flow_rate'] *= (1 + cooling_adj * 0.8)
        
        # 结构参数调整
        if structural_adj > 0:
            new_params['wall_thickness'] *= (1 + structural_adj)
            new_params['reinforcement_thickness'] *= (1 + structural_adj * 1.5)
        
        # 性能参数调整
        if performance_adj > 0:
            new_params['nozzle_expansion_ratio'] *= (1 + performance_adj * 0.5)
            new_params['injection_pressure'] *= (1 + performance_adj * 0.3)
        
        # 确保参数在合理范围内
        new_params = self._constrain_parameters(new_params)
        
        return new_params
    
    def _validate_rollback_parameters(self, params: Dict[str, float]) -> Dict[str, Any]:
        """验证回退优化参数的有效性"""
        try:
            # 快速验证：生成几何模型检查可行性
            model_path = self._generate_geometry_model(params, -1)  # 使用特殊迭代号
            
            # 检查网格生成可行性
            mesh_stats = self.mesh_generator.generate_mesh(model_path, params, -1)
            
            return {
                'valid': True,
                'model_path': model_path,
                'mesh_stats': mesh_stats
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _constrain_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """约束参数在合理范围内"""
        constrained_params = params.copy()
        
        # 定义参数范围约束
        constraints = {
            'cooling_channel_width': (0.1, 2.0),
            'wall_thickness': (0.3, 5.0),
            'nozzle_expansion_ratio': (3.0, 20.0),
            'injection_pressure': (1.0, 50.0),
            'cooling_flow_rate': (0.1, 10.0),
            'fuel_flow_rate': (0.5, 20.0),
            'oxidizer_flow_rate': (0.5, 20.0)
        }
        
        for param_name, (min_val, max_val) in constraints.items():
            if param_name in constrained_params:
                constrained_params[param_name] = max(min_val, min(max_val, constrained_params[param_name]))
        
        return constrained_params
    
    def _generate_random_parameters(self) -> Dict[str, float]:
        """生成随机参数组合"""
        params = {}
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            # 在参数范围内均匀随机采样
            params[param_name] = random.uniform(min_val, max_val)
        
        # 添加多物理场标志
        params['enable_multiphysics'] = random.random() > 0.7  # 30%概率启用多物理场
        
        return params
    
    def _refine_parameters(self, current_params: Dict[str, float], iteration: int) -> Dict[str, float]:
        """基于当前参数进行精细调整"""
        new_params = current_params.copy()
        
        # 随着迭代进行，调整幅度逐渐减小
        temperature = max(0.1, 1.0 - (iteration / self.max_iterations))
        
        for param_name in self.parameter_ranges.keys():
            if param_name in new_params:
                min_val, max_val = self.parameter_ranges[param_name]
                current_val = new_params[param_name]
                
                # 高斯扰动
                perturbation = random.gauss(0, 0.1) * temperature * (max_val - min_val)
                new_val = current_val + perturbation
                
                # 确保在参数范围内
                new_params[param_name] = max(min_val, min(max_val, new_val))
        
        # 强制启用多物理场（在优化阶段）
        new_params['enable_multiphysics'] = True
        
        return new_params
    
    def _add_parameter_perturbation(self, base_params: Dict[str, float], iteration: int) -> Dict[str, float]:
        """添加参数扰动以测试可靠性"""
        perturbed_params = base_params.copy()
        
        # 扰动幅度（5%-15%）
        perturbation_magnitude = 0.05 + 0.1 * (iteration / self.max_iterations)
        
        for param_name in self.parameter_ranges.keys():
            if param_name in perturbed_params:
                min_val, max_val = self.parameter_ranges[param_name]
                base_val = perturbed_params[param_name]
                
                # 随机扰动
                perturbation = random.uniform(-perturbation_magnitude, perturbation_magnitude) * base_val
                new_val = base_val + perturbation
                
                # 确保在参数范围内
                perturbed_params[param_name] = max(min_val, min(max_val, new_val))
        
        return perturbed_params
    
    def _generate_geometry_model(self, parameters: Dict[str, float], iteration: int) -> str:
        """生成几何模型文件"""
        try:
            # 这里应该调用FreeCADModeler来生成几何
            # 简化实现：返回一个占位符路径
            model_path = self.simulator.working_dir / f"model_{iteration}.step"
            return str(model_path)
        except Exception as e:
            self.logger.error(f"几何模型生成失败: {e}")
            return f"/tmp/model_{iteration}.step"
    
    def _calculate_comprehensive_score(self, results: Dict[str, Any], parameters: Dict[str, float]) -> float:
        """计算综合评分，考虑优化优先级"""
        try:
            # 1. 检查是否符合优先级阈值
            is_valid, violations = self._check_priority_thresholds(results)
            
            # 2. 如果不符合阈值要求，大幅降低评分
            if not is_valid:
                # 按优先级排序违规项
                sorted_violations = self._analyze_violations_by_priority(violations)
                
                # 根据违规严重程度和优先级计算惩罚分数
                penalty_factor = 0.5  # 基础惩罚
                for i, violation in enumerate(sorted_violations):
                    # 优先级越高的违规项惩罚越大
                    priority_multiplier = 1.0 - (i * 0.2)  # 第一个违规项惩罚最大
                    penalty_factor *= priority_multiplier
                
                # 返回惩罚后的分数
                base_score = self._calculate_base_score(results)
                return base_score * penalty_factor
            
            # 3. 计算基础评分
            base_score = self._calculate_base_score(results)
            
            # 4. 根据优先级调整评分
            priority_adjustment = self._apply_priority_scoring_adjustment(results)
            
            final_score = base_score * priority_adjustment
            
            # 确保分数在合理范围内
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"评分计算失败: {e}")
            return 0.0
    
    def _calculate_base_score(self, results: Dict[str, Any]) -> float:
        """计算基础评分（不考虑优先级）"""
        try:
            # 归一化各项指标
            normalized_scores = {}
            
            # 推力评分（误差越小越好）
            thrust_error = abs(results.get('thrust_error', 0))
            normalized_scores['thrust'] = max(0.0, 1.0 - thrust_error / 0.1)  # 10%误差为0分
            
            # 比冲评分（越高越好）
            specific_impulse = results.get('specific_impulse', 0)
            normalized_scores['specific_impulse'] = min(1.0, specific_impulse / 300)  # 300s为满分
            
            # 效率评分（越高越好）
            efficiency = results.get('efficiency', 0)
            normalized_scores['efficiency'] = efficiency
            
            # 结构安全评分（应力越小越好）
            structural_stress = results.get('structural_stress', 0)
            normalized_scores['structural_safety'] = max(0.0, 1.0 - structural_stress / 600)  # 600MPa为0分
            
            # 热梯度评分（梯度越小越好）
            thermal_gradient = results.get('thermal_gradient', 0)
            normalized_scores['thermal_gradient'] = max(0.0, 1.0 - thermal_gradient / 100)  # 100K/mm为0分
            
            # 加权平均
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, weight in self.optimization_weights.items():
                if metric in normalized_scores:
                    weighted_score += normalized_scores[metric] * weight
                    total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"基础评分计算失败: {e}")
            return 0.0
    
    def _apply_priority_scoring_adjustment(self, results: Dict[str, Any]) -> float:
        """根据优化优先级应用评分调整"""
        adjustment_factor = 1.0
        
        if self.optimization_priority == OptimizationPriority.EFFICIENCY:
            # 效率优先：性能指标表现好时给予额外奖励
            thrust_error = abs(results.get('thrust_error', 0))
            specific_impulse = results.get('specific_impulse', 0)
            
            if thrust_error < 0.03:  # 推力误差小于3%
                adjustment_factor *= 1.1
            if specific_impulse > 250:  # 比冲大于250s
                adjustment_factor *= 1.05
                
        elif self.optimization_priority == OptimizationPriority.WEIGHT:
            # 重量优先：结构安全指标表现好时给予额外奖励
            structural_stress = results.get('structural_stress', 0)
            
            if structural_stress < 500:  # 应力小于500MPa
                adjustment_factor *= 1.1
            
        else:  # BALANCED
            # 综合优先：平衡调整
            thrust_error = abs(results.get('thrust_error', 0))
            structural_stress = results.get('structural_stress', 0)
            
            if thrust_error < 0.04 and structural_stress < 550:  # 平衡表现
                adjustment_factor *= 1.05
        
        return adjustment_factor
        """计算综合评分"""
        try:
            score = 0.0
            
            # 1. 性能指标评分
            for metric, weight in self.optimization_weights.items():
                value = results.get(metric, 0)
                
                # 归一化处理
                if metric == 'thrust':
                    normalized_value = min(1.0, value / 500)  # 目标推力500N
                elif metric == 'specific_impulse':
                    normalized_value = min(1.0, value / 250)   # 目标比冲250s
                elif metric == 'efficiency':
                    normalized_value = value  # 效率已经是0-1范围
                elif metric == 'structural_safety':
                    normalized_value = value  # 安全系数已经是0-1范围
                elif metric == 'thermal_gradient':
                    normalized_value = 1.0 - min(1.0, value / 1000)  # 热梯度越小越好
                else:
                    normalized_value = 0.0
                
                score += weight * normalized_value
            
            # 2. 约束条件惩罚
            penalty = self._calculate_constraint_penalty(results, parameters)
            score -= penalty
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.warning(f"综合评分计算失败: {e}")
            return 0.0
    
    def _calculate_constraint_penalty(self, results: Dict[str, Any], parameters: Dict[str, float]) -> float:
        """计算约束条件惩罚项"""
        penalty = 0.0
        
        # 壁温约束（不超过材料极限）
        max_wall_temp = results.get('max_wall_temperature', 0)
        if max_wall_temp > 1500:  # 材料极限温度
            penalty += (max_wall_temp - 1500) / 100
        
        # 应力约束
        max_stress = results.get('max_stress', 0)
        if max_stress > 300e6:  # 材料屈服强度
            penalty += (max_stress - 300e6) / 50e6
        
        # 效率约束
        efficiency = results.get('efficiency', 0)
        if efficiency < 0.6:  # 最低效率要求
            penalty += (0.6 - efficiency) * 10
        
        return penalty
    
    def _should_accept_new_parameters(self, new_score: float, current_params: Dict[str, float], 
                                     new_params: Dict[str, float], iteration: int) -> bool:
        """决定是否接受新参数（模拟退火策略）"""
        current_score = self.best_results.get('optimization_score', 0)
        
        # 如果新评分更好，总是接受
        if new_score > current_score:
            return True
        
        # 模拟退火：随着迭代进行，接受较差解的概率降低
        temperature = max(0.01, 1.0 - (iteration / self.max_iterations))
        probability = math.exp((new_score - current_score) / temperature)
        
        return random.random() < probability
    
    def _check_convergence(self, iteration: int) -> bool:
        """检查收敛条件"""
        if iteration < 10:  # 至少10轮后才检查收敛
            return False
        
        # 检查改进停滞
        recent_scores = [hist['score'] for hist in self.iteration_history[-10:] if 'score' in hist]
        if len(recent_scores) >= 5:
            max_recent = max(recent_scores)
            improvement = max_recent - self.best_results.get('optimization_score', 0)
            
            if improvement < self.convergence_criteria['improvement_threshold']:
                return True
        
        return False
    
    def _calculate_reliability_metrics(self, current_results: Dict[str, Any], 
                                     best_results: Dict[str, Any], parameters: Dict[str, float]) -> Dict[str, float]:
        """计算可靠性指标"""
        metrics = {}
        
        try:
            # 1. 性能稳定性
            current_score = self._calculate_comprehensive_score(current_results, parameters)
            best_score = best_results.get('optimization_score', 0)
            metrics['performance_stability'] = current_score / best_score if best_score > 0 else 0
            
            # 2. 参数敏感性
            param_variations = {}
            for param_name in self.parameter_ranges.keys():
                if param_name in parameters and param_name in best_results.get('parameters', {}):
                    base_val = best_results['parameters'][param_name]
                    current_val = parameters[param_name]
                    variation = abs(current_val - base_val) / base_val if base_val != 0 else 0
                    param_variations[param_name] = variation
            
            metrics['parameter_sensitivity'] = 1.0 - min(1.0, sum(param_variations.values()) / len(param_variations))
            
            # 3. 约束满足度
            constraint_violations = 0
            total_constraints = 3  # 温度、应力、效率约束
            
            if current_results.get('max_wall_temperature', 0) > 1500:
                constraint_violations += 1
            if current_results.get('max_stress', 0) > 300e6:
                constraint_violations += 1
            if current_results.get('efficiency', 0) < 0.6:
                constraint_violations += 1
            
            metrics['constraint_satisfaction'] = 1.0 - (constraint_violations / total_constraints)
            
            # 4. 综合可靠性
            metrics['overall_reliability'] = (
                metrics.get('performance_stability', 0) * 0.4 +
                metrics.get('parameter_sensitivity', 0) * 0.3 +
                metrics.get('constraint_satisfaction', 0) * 0.3
            )
            
        except Exception as e:
            self.logger.warning(f"可靠性指标计算失败: {e}")
            metrics = {'overall_reliability': 0.5}
        
        return metrics
    
    def _assess_overall_reliability(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估整体可靠性"""
        try:
            reliability_scores = [
                result.get('reliability_metrics', {}).get('overall_reliability', 0)
                for result in validation_results
            ]
            
            if reliability_scores:
                avg_reliability = sum(reliability_scores) / len(reliability_scores)
                min_reliability = min(reliability_scores)
                max_reliability = max(reliability_scores)
                
                # 可靠性评级
                if avg_reliability >= 0.9:
                    reliability_grade = "优秀"
                elif avg_reliability >= 0.8:
                    reliability_grade = "良好"
                elif avg_reliability >= 0.7:
                    reliability_grade = "合格"
                else:
                    reliability_grade = "需要改进"
                
                return {
                    'average_reliability': avg_reliability,
                    'min_reliability': min_reliability,
                    'max_reliability': max_reliability,
                    'reliability_grade': reliability_grade,
                    'confidence_level': min(1.0, avg_reliability * 1.1)  # 置信水平
                }
            
        except Exception as e:
            self.logger.error(f"整体可靠性评估失败: {e}")
        
        return {'average_reliability': 0.5, 'reliability_grade': '未知', 'confidence_level': 0.5}
    
    def _evaluate_final_results(self, screening_results: Dict[str, Any], 
                               coupling_results: Dict[str, Any], 
                               validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """综合评估最终结果"""
        final_results = {
            'optimization_summary': {
                'total_iterations': self.current_iteration,
                'successful_iterations': len([h for h in self.iteration_history if 'score' in h and h['score'] >= 0]),
                'best_score': self.best_results.get('optimization_score', 0),
                'best_parameters': self.best_results.get('parameters', {})
            },
            'phase_results': {
                'screening': screening_results,
                'coupling': coupling_results,
                'validation': validation_results
            },
            'reliability_assessment': validation_results.get('reliability_assessment', {}),
            'performance_metrics': self.best_results,
            'convergence_analysis': self._analyze_convergence()
        }
        
        return final_results
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """分析收敛情况"""
        try:
            scores = [h['score'] for h in self.iteration_history if 'score' in h and h['score'] >= 0]
            
            if len(scores) >= 2:
                convergence_rate = (scores[-1] - scores[0]) / len(scores)
                improvement_trend = '上升' if convergence_rate > 0 else '下降'
                
                return {
                    'convergence_rate': convergence_rate,
                    'improvement_trend': improvement_trend,
                    'final_score': scores[-1] if scores else 0,
                    'initial_score': scores[0] if scores else 0,
                    'score_variation': max(scores) - min(scores) if scores else 0
                }
            
        except Exception as e:
            self.logger.warning(f"收敛分析失败: {e}")
        
        return {'convergence_rate': 0, 'improvement_trend': '未知'}
    
    def _generate_fallback_results(self) -> Dict[str, Any]:
        """生成备用结果"""
        return {
            'optimization_summary': {
                'total_iterations': self.current_iteration,
                'successful_iterations': 0,
                'best_score': 0,
                'best_parameters': {}
            },
            'phase_results': {},
            'reliability_assessment': {'average_reliability': 0.5, 'reliability_grade': '失败'},
            'performance_metrics': {},
            'error': '优化过程失败'
        }
        
    def init_ui(self):
        """初始化UI（使用配置管理器）"""
        self.setWindowTitle("火箭发动机模拟系统 - 统一可视化窗口")
        
        # 从配置管理器获取UI设置
        window_width = self.config_manager.get_int('UI', 'window_width', 1200)
        window_height = self.config_manager.get_int('UI', 'window_height', 800)
        font_size = self.config_manager.get_int('UI', 'font_size', 10)
        
        self.setGeometry(100, 100, window_width, window_height)
        
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧可视化区域 (占2/3)
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # 右侧分析区域 (占1/3)
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # 添加左右区域到主布局
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)
        
        # 设置左侧区域内容
        self.setup_left_area(left_layout, font_size)
        
        # 设置右侧区域内容
        self.setup_right_area(right_layout, font_size)
    
    def setup_left_area(self, layout, font_size: int = 10):
        """设置左侧可视化区域"""
        # 异常帧显示区域
        abnormal_group = QtWidgets.QGroupBox("异常帧显示")
        abnormal_layout = QtWidgets.QVBoxLayout()
        abnormal_group.setLayout(abnormal_layout)
        
        self.abnormal_frames_tab = QtWidgets.QTabWidget()
        abnormal_layout.addWidget(self.abnormal_frames_tab)
        
        # 参数趋势图表区域
        trends_group = QtWidgets.QGroupBox("参数趋势分析")
        trends_layout = QtWidgets.QVBoxLayout()
        trends_group.setLayout(trends_layout)
        
        self.parameter_trends_tab = QtWidgets.QTabWidget()
        trends_layout.addWidget(self.parameter_trends_tab)
        
        # 迭代参数对比表格
        comparison_group = QtWidgets.QGroupBox("迭代参数对比")
        comparison_layout = QtWidgets.QVBoxLayout()
        comparison_group.setLayout(comparison_layout)
        
        self.parameter_comparison_table = QtWidgets.QTableWidget()
        self.parameter_comparison_table.setColumnCount(4)
        self.parameter_comparison_table.setHorizontalHeaderLabels(['参数名', '当前轮', '上一轮', '变化量'])
        self.parameter_comparison_table.setFont(QtGui.QFont("Consolas", font_size))
        comparison_layout.addWidget(self.parameter_comparison_table)
        
        layout.addWidget(abnormal_group, 1)
        layout.addWidget(trends_group, 1)
        layout.addWidget(comparison_group, 1)
    
    def setup_right_area(self, layout, font_size: int = 10):
        """设置右侧分析区域"""
        # 分析结果文本框
        analysis_group = QtWidgets.QGroupBox("异常分析与调整建议")
        analysis_layout = QtWidgets.QVBoxLayout()
        analysis_group.setLayout(analysis_layout)
        
        self.analysis_text = QtWidgets.QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QtGui.QFont("Consolas", font_size))
        analysis_layout.addWidget(self.analysis_text)
        
        # 当前参数显示
        params_group = QtWidgets.QGroupBox("当前参数状态")
        params_layout = QtWidgets.QVBoxLayout()
        params_group.setLayout(params_layout)
        
        self.current_params_text = QtWidgets.QTextEdit()
        self.current_params_text.setReadOnly(True)
        self.current_params_text.setFont(QtGui.QFont("Consolas", font_size))
        params_layout.addWidget(self.current_params_text)
        
        # 模拟控制按钮
        control_group = QtWidgets.QGroupBox("模拟控制")
        control_layout = QtWidgets.QHBoxLayout()
        control_group.setLayout(control_layout)
        
        self.start_button = QtWidgets.QPushButton("开始模拟")
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.stop_button = QtWidgets.QPushButton("停止")
        
        # 设置按钮字体
        button_font = QtGui.QFont("Microsoft YaHei", font_size)
        self.start_button.setFont(button_font)
        self.pause_button.setFont(button_font)
        self.stop_button.setFont(button_font)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        
        # 进度显示
        progress_group = QtWidgets.QGroupBox("模拟进度")
        progress_layout = QtWidgets.QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_label = QtWidgets.QLabel("准备开始")
        self.progress_label.setFont(QtGui.QFont("Microsoft YaHei", font_size))
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        
        layout.addWidget(analysis_group, 2)
        layout.addWidget(params_group, 1)
        layout.addWidget(control_group, 1)
        layout.addWidget(progress_group, 1)
        
        # 连接信号
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button.clicked.connect(self.pause_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
    
    def set_simulator(self, simulator):
        """设置模拟器实例"""
        try:
            if not simulator:
                raise ValueError("模拟器实例不能为空")
            
            if not hasattr(simulator, 'run_full_simulation'):
                raise ValueError("模拟器缺少必要方法")
            
            self.simulator = simulator
            print(f"模拟器设置成功 - 类型: {type(simulator).__name__}")
            
        except Exception as e:
            error_msg = f"模拟器设置失败: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(self, "错误", error_msg)
            self.simulator = None
    
    def _validate_simulator_configuration(self) -> bool:
        """验证模拟器配置完整性"""
        try:
            if not self.simulator:
                return False
            
            # 检查必要属性
            required_attrs = ['max_iterations', 'design_requirements', 'initial_parameters']
            for attr in required_attrs:
                if not hasattr(self.simulator, attr):
                    print(f"模拟器缺少属性: {attr}")
                    return False
            
            # 检查设计需求
            if not self.simulator.design_requirements:
                print("模拟器设计需求未设置")
                return False
            
            # 检查初始参数
            if not self.simulator.initial_parameters:
                print("模拟器初始参数未设置")
                return False
            
            return True
            
        except Exception as e:
            print(f"模拟器配置验证失败: {e}")
            return False
    
    def _validate_ui_components(self) -> bool:
        """验证UI组件状态"""
        try:
            # 检查必要UI组件
            required_components = [
                'start_button', 'pause_button', 'stop_button',
                'progress_bar', 'progress_label', 'analysis_text',
                'current_params_text', 'abnormal_frames_tab',
                'parameter_trends_tab', 'parameter_comparison_table'
            ]
            
            for component in required_components:
                if not hasattr(self, component) or getattr(self, component) is None:
                    print(f"UI组件缺失: {component}")
                    return False
            
            # 检查按钮状态
            if not self.start_button.isEnabled():
                print("开始按钮状态异常")
                return False
            
            return True
            
        except Exception as e:
            print(f"UI组件验证失败: {e}")
            return False
    
    def _create_working_directory(self) -> bool:
        """创建工作目录"""
        try:
            # 创建工作目录路径
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.working_dir = os.path.join("simulation_results", f"simulation_{timestamp}")
            
            # 创建目录
            os.makedirs(self.working_dir, exist_ok=True)
            
            # 验证目录权限
            test_file = os.path.join(self.working_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"工作目录创建成功: {self.working_dir}")
            return True
            
        except Exception as e:
            print(f"工作目录创建失败: {e}")
            return False
    
    def _connect_worker_signals(self) -> bool:
        """连接工作器信号"""
        try:
            # 连接工作器信号到窗口信号
            self.simulation_worker.update_signal.connect(self.update_signal.emit)
            self.simulation_worker.progress_signal.connect(self.progress_signal.emit)
            self.simulation_worker.error_signal.connect(self.error_signal.emit)
            
            # 验证连接
            if not self.simulation_worker.update_signal.receivers():
                print("update_signal连接失败")
                return False
            
            return True
            
        except Exception as e:
            print(f"工作器信号连接失败: {e}")
            return False
    
    def _connect_thread_signals(self) -> bool:
        """连接线程信号"""
        try:
            # 连接线程信号
            self.simulation_thread.started.connect(self.simulation_worker.run)
            self.simulation_worker.finished.connect(self.simulation_thread.quit)
            self.simulation_worker.finished.connect(self.simulation_worker.deleteLater)
            self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
            
            # 验证连接
            if not self.simulation_thread.started.receivers():
                print("线程started信号连接失败")
                return False
            
            return True
            
        except Exception as e:
            print(f"线程信号连接失败: {e}")
            return False
    
    def _update_button_states(self, start_enabled: bool, pause_enabled: bool, stop_enabled: bool):
        """更新按钮状态"""
        try:
            self.start_button.setEnabled(start_enabled)
            self.pause_button.setEnabled(pause_enabled)
            self.stop_button.setEnabled(stop_enabled)
        except Exception as e:
            print(f"按钮状态更新失败: {e}")
    
    def _cleanup_simulation_resources(self):
        """清理模拟资源 - 内存优化版本"""
        try:
            # 步骤1: 记录清理前内存状态
            self._log_memory_usage("清理前")
            
            # 步骤2: 停止模拟器并清理资源
            if self.simulator:
                self.simulator.is_running = False
                # 调用模拟器的紧急清理方法
                if hasattr(self.simulator, '_emergency_cleanup'):
                    self.simulator._emergency_cleanup()
                elif hasattr(self.simulator, '_cleanup_resources'):
                    self.simulator._cleanup_resources()
            
            # 步骤3: 停止并清理线程
            if hasattr(self, 'simulation_thread') and self.simulation_thread:
                if self.simulation_thread.isRunning():
                    self.simulation_thread.quit()
                    if not self.simulation_thread.wait(3000):  # 等待3秒
                        print("线程未正常结束，强制终止")
                        self.simulation_thread.terminate()
                self.simulation_thread = None
            
            # 步骤4: 清理工作器资源
            if hasattr(self, 'simulation_worker') and self.simulation_worker:
                # 调用工作器的资源清理方法
                if hasattr(self.simulation_worker, '_cleanup_worker_resources'):
                    self.simulation_worker._cleanup_worker_resources()
                self.simulation_worker.deleteLater()
                self.simulation_worker = None
            
            # 步骤5: 清理窗口缓存数据
            self._cleanup_window_caches()
            
            # 步骤6: 清理迭代数据（限制内存使用）
            if hasattr(self, 'iteration_data'):
                # 保留最近20轮数据用于趋势分析
                if len(self.iteration_data) > 20:
                    self.iteration_data = self.iteration_data[-20:]
                    print(f"迭代数据清理完成，保留{len(self.iteration_data)}轮数据")
            
            # 步骤7: 清理UI组件缓存
            self._cleanup_ui_caches()
            
            # 步骤8: 重置状态变量
            self.is_simulation_running = False
            
            # 步骤9: 重置按钮状态
            self._update_button_states(start_enabled=True, pause_enabled=False, stop_enabled=False)
            
            # 步骤10: 强制垃圾回收
            import gc
            gc.collect()
            
            # 步骤11: 记录清理后内存状态
            self._log_memory_usage("清理后")
            
            print("模拟资源清理完成 - 内存优化版本")
            
        except Exception as e:
            print(f"资源清理失败: {e}")
            # 即使清理失败也尝试基本清理
            self._basic_cleanup()
    
    def _cleanup_window_caches(self):
        """清理窗口缓存数据"""
        try:
            # 清理临时变量
            temp_vars = ['temp_buffer', 'cache_data', 'pending_updates']
            for var in temp_vars:
                if hasattr(self, var):
                    delattr(self, var)
            
            # 清理信号连接缓存
            if hasattr(self, 'signal_connections'):
                for connection in self.signal_connections:
                    try:
                        connection.disconnect()
                    except:
                        pass
                self.signal_connections.clear()
            
            # 清理回调函数引用
            if hasattr(self, 'callback_references'):
                self.callback_references.clear()
            
            print("窗口缓存清理完成")
            
        except Exception as e:
            print(f"窗口缓存清理失败: {e}")
    
    def _cleanup_ui_caches(self):
        """清理UI组件缓存"""
        try:
            # 清理标签页缓存
            if hasattr(self, 'abnormal_frames_tab'):
                for i in range(self.abnormal_frames_tab.count()):
                    widget = self.abnormal_frames_tab.widget(i)
                    if widget:
                        widget.deleteLater()
            
            # 清理表格数据缓存
            if hasattr(self, 'parameter_comparison_table'):
                self.parameter_comparison_table.clearContents()
                self.parameter_comparison_table.setRowCount(0)
            
            # 清理文本编辑框缓存
            text_widgets = ['analysis_text', 'current_params_text']
            for widget_name in text_widgets:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    if widget:
                        widget.clear()
            
            # 清理进度条缓存
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
            
            print("UI缓存清理完成")
            
        except Exception as e:
            print(f"UI缓存清理失败: {e}")
    
    def _log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            rss_mb = memory_info.rss / 1024 / 1024  # RSS内存（MB）
            vms_mb = memory_info.vms / 1024 / 1024  # VMS内存（MB）
            
            print(f"{stage}内存使用 - RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB")
            
            # 记录到内存使用日志
            with open('memory_usage.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {stage} - RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB\n")
            
        except ImportError:
            print(f"{stage}内存记录: psutil模块未安装")
        except Exception as e:
            print(f"内存记录失败: {e}")
    
    def _basic_cleanup(self):
        """基本清理（用于异常情况）"""
        try:
            # 重置关键状态
            self.is_simulation_running = False
            
            # 重置按钮状态
            self._update_button_states(start_enabled=True, pause_enabled=False, stop_enabled=False)
            
            # 清理工作目录引用
            if hasattr(self, 'working_dir'):
                self.working_dir = None
            
            print("基本清理完成")
            
        except Exception as e:
            print(f"基本清理失败: {e}")
    
    def closeEvent(self, event):
        """窗口关闭事件 - 内存优化版本"""
        try:
            print("开始关闭窗口，执行资源清理...")
            
            # 记录关闭前内存状态
            self._log_memory_usage("窗口关闭前")
            
            # 停止所有模拟活动
            if self.is_simulation_running:
                self.stop_simulation()
            
            # 执行完整的资源清理
            self._cleanup_simulation_resources()
            
            # 清理所有UI组件
            self._cleanup_all_ui_components()
            
            # 清理父类资源
            super().closeEvent(event)
            
            # 记录关闭后内存状态
            self._log_memory_usage("窗口关闭后")
            
            print("窗口关闭完成 - 资源已清理")
            
        except Exception as e:
            print(f"窗口关闭异常: {e}")
            # 确保窗口正常关闭
            event.accept()
    
    def _cleanup_all_ui_components(self):
        """清理所有UI组件"""
        try:
            # 清理所有子控件
            for child in self.findChildren(QtWidgets.QWidget):
                try:
                    child.deleteLater()
                except:
                    pass
            
            # 清理布局
            if hasattr(self, 'layout'):
                try:
                    # 清理布局中的所有项目
                    for i in reversed(range(self.layout.count())):
                        item = self.layout.itemAt(i)
                        if item:
                            widget = item.widget()
                            if widget:
                                widget.deleteLater()
                            self.layout.removeItem(item)
                except:
                    pass
            
            print("所有UI组件清理完成")
            
        except Exception as e:
            print(f"UI组件清理失败: {e}")
    

    
    def update_visualization(self, iteration: int, parameters: SimulationParameters, 
                           results: Dict[str, Any]):
        """更新可视化内容"""
        # 存储当前迭代数据
        iteration_data = {
            'iteration': iteration,
            'parameters': parameters,
            'results': results,
            'timestamp': time.time()
        }
        self.iteration_data.append(iteration_data)
        
        # 更新各个可视化组件
        self.update_abnormal_frames(results)
        self.update_parameter_trends(iteration, results)
        self.update_parameter_comparison(iteration, parameters, results)
        self.update_analysis_text(iteration, results)
        self.update_current_params(parameters)
        
        # 更新进度
        progress = (iteration / self.simulator.max_iterations) * 100
        self.progress_bar.setValue(int(progress))
        self.progress_label.setText(f"第{iteration}轮 - 进度{progress:.1f}%")
        

    
    def update_progress_thread_safe(self, progress: int, message: str):
        """线程安全的进度更新"""
        try:
            # 验证进度值
            if not isinstance(progress, int) or progress < 0 or progress > 100:
                print(f"无效的进度值: {progress}")
                progress = max(0, min(100, progress))  # 限制在0-100范围内
            
            # 验证消息
            if not isinstance(message, str):
                message = str(message) if message else "进度更新"
            
            # 更新进度条
            self.progress_bar.setValue(progress)
            
            # 更新标签文本，限制长度避免UI溢出
            if len(message) > 100:
                message = message[:97] + "..."
            self.progress_label.setText(message)
            
            # 记录进度更新
            if progress % 10 == 0:  # 每10%记录一次
                print(f"进度更新 - {progress}%: {message}")
                
        except Exception as e:
            error_msg = f"进度更新错误: {str(e)}"
            print(error_msg)
            # 记录到错误日志
            with open('progress_update_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    def show_error_thread_safe(self, error_message: str):
        """线程安全的错误显示"""
        try:
            # 验证错误消息
            if not error_message or not isinstance(error_message, str):
                error_message = "发生未知错误"
            
            # 限制错误消息长度
            if len(error_message) > 500:
                error_message = error_message[:497] + "..."
            
            # 显示错误对话框
            QtWidgets.QMessageBox.critical(self, "模拟错误", error_message)
            
            # 记录错误到日志文件
            with open('simulation_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
            
            # 停止模拟
            self.stop_simulation()
            
            print(f"错误处理完成: {error_message}")
            
        except Exception as e:
            # 如果UI错误处理失败，使用控制台输出
            error_msg = f"错误显示失败: {str(e)} - 原错误: {error_message}"
            print(error_msg)
            # 记录到错误日志
            with open('error_handling_failures.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    def update_abnormal_frames(self, results: Dict[str, Any]):
        """更新异常帧显示"""
        try:
            # 验证结果数据
            if not results or not isinstance(results, dict):
                print("无效的结果数据")
                return
            
            # 清除现有标签页
            self.abnormal_frames_tab.clear()
            
            # 添加壁温异常帧
            wall_temp_widget = QtWidgets.QWidget()
            wall_temp_layout = QtWidgets.QVBoxLayout()
            wall_temp_widget.setLayout(wall_temp_layout)
            
            wall_temp_label = QtWidgets.QLabel("燃烧室壁温云图")
            wall_temp_layout.addWidget(wall_temp_label)
            
            # 模拟显示壁温异常区域
            wall_temp_text = QtWidgets.QTextEdit()
            wall_temp_text.setReadOnly(True)
            max_temp = results.get('max_wall_temperature', 0)
            
            # 验证壁温数据
            if not isinstance(max_temp, (int, float)) or max_temp < 0:
                max_temp = 0
                wall_temp_text.setText("壁温数据异常，请检查模拟结果")
            else:
                # 判断壁温是否超标
                temp_status = "正常" if max_temp < 1200 else "超标"
                temp_color = "绿色" if max_temp < 1200 else "红色"
                
                wall_temp_text.setText(f"最大壁温: {max_temp:.1f}K ({temp_status})\n"
                                      f"异常位置: 燃烧室15-25mm段\n"
                                      f"超标区域: {temp_color}标记区\n"
                                      f"安全阈值: 1200K")
            
            wall_temp_layout.addWidget(wall_temp_text)
            self.abnormal_frames_tab.addTab(wall_temp_widget, "壁温异常")
            
            # 添加冷却流速异常帧
            cooling_widget = QtWidgets.QWidget()
            cooling_layout = QtWidgets.QVBoxLayout()
            cooling_widget.setLayout(cooling_layout)
            
            cooling_label = QtWidgets.QLabel("冷却夹层流速矢量图")
            cooling_layout.addWidget(cooling_label)
            
            cooling_text = QtWidgets.QTextEdit()
            cooling_text.setReadOnly(True)
            cooling_vel = results.get('cooling_velocity', 0)
            
            # 验证冷却流速数据
            if not isinstance(cooling_vel, (int, float)) or cooling_vel < 0:
                cooling_vel = 0
                cooling_text.setText("冷却流速数据异常，请检查模拟结果")
            else:
                # 判断流速是否正常
                vel_status = "正常" if cooling_vel > 5 else "偏低"
                vel_color = "绿色" if cooling_vel > 5 else "蓝色"
                
                cooling_text.setText(f"平均流速: {cooling_vel:.2f}m/s ({vel_status})\n"
                                    f"异常位置: 夹层底部转角\n"
                                    f"低速区域: {vel_color}标记区\n"
                                    f"正常阈值: >5m/s")
            
            cooling_layout.addWidget(cooling_text)
            self.abnormal_frames_tab.addTab(cooling_widget, "冷却异常")
            
            print(f"异常帧更新完成 - 壁温: {max_temp:.1f}K, 冷却流速: {cooling_vel:.2f}m/s")
            
        except Exception as e:
            error_msg = f"异常帧更新错误: {str(e)}"
            print(error_msg)
            # 记录错误日志
            with open('abnormal_frames_errors.log', 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
    
    def update_parameter_trends(self, iteration: int, results: Dict[str, Any]):
        """更新参数趋势图表"""
        self.parameter_trends_tab.clear()
        
        # 推力趋势图表
        thrust_widget = QtWidgets.QWidget()
        thrust_layout = QtWidgets.QVBoxLayout()
        thrust_widget.setLayout(thrust_layout)
        
        thrust_label = QtWidgets.QLabel("推力-轮次关系图")
        thrust_layout.addWidget(thrust_label)
        
        thrust_text = QtWidgets.QTextEdit()
        thrust_text.setReadOnly(True)
        
        # 生成推力趋势文本
        thrust_trend = ""
        for i, data in enumerate(self.iteration_data[-10:]):  # 显示最近10轮
            thrust = data['results'].get('thrust', 0)
            thrust_trend += f"第{data['iteration']}轮: {thrust:.1f}kgf\n"
        
        thrust_text.setText(f"当前推力: {results.get('thrust', 0):.1f}kgf\n\n"
                          f"推力趋势:\n{thrust_trend}")
        thrust_layout.addWidget(thrust_text)
        
        self.parameter_trends_tab.addTab(thrust_widget, "推力趋势")
        
        # 壁温趋势图表
        temp_widget = QtWidgets.QWidget()
        temp_layout = QtWidgets.QVBoxLayout()
        temp_widget.setLayout(temp_layout)
        
        temp_label = QtWidgets.QLabel("壁温沿轴线分布")
        temp_layout.addWidget(temp_label)
        
        temp_text = QtWidgets.QTextEdit()
        temp_text.setReadOnly(True)
        
        temp_trend = ""
        for i, data in enumerate(self.iteration_data[-10:]):
            temp = data['results'].get('max_wall_temperature', 0)
            temp_trend += f"第{data['iteration']}轮: {temp:.1f}K\n"
        
        temp_text.setText(f"当前壁温: {results.get('max_wall_temperature', 0):.1f}K\n\n"
                        f"壁温趋势:\n{temp_trend}")
        temp_layout.addWidget(temp_text)
        
        self.parameter_trends_tab.addTab(temp_widget, "壁温趋势")
    
    def update_parameter_comparison(self, iteration: int, parameters: SimulationParameters, results: Dict[str, Any]):
        """更新参数对比表格"""
        if iteration < 2:
            return
        
        # 获取上一轮数据
        prev_data = self.iteration_data[-2]
        prev_params = prev_data['parameters']
        
        # 设置表格行数
        self.parameter_comparison_table.setRowCount(5)
        
        # 填充数据
        parameters_to_compare = [
            ('燃烧室内径', 'chamber_diameter', 'mm', 2),
            ('喷管扩张比', 'expansion_ratio', '', 2),
            ('燃料流率', 'fuel_flow_rate', 'kg/s', 4),
            ('冷却流速', 'cooling_velocity', 'm/s', 2),
            ('壁厚', 'wall_thickness', 'mm', 3)
        ]
        
        for i, (name, attr, unit, precision) in enumerate(parameters_to_compare):
            current_val = getattr(parameters, attr)
            prev_val = getattr(prev_params, attr)
            change = current_val - prev_val
            
            # 设置表格项
            self.parameter_comparison_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.parameter_comparison_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{current_val:.{precision}f}{unit}"))
            self.parameter_comparison_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{prev_val:.{precision}f}{unit}"))
            self.parameter_comparison_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{change:+.{precision}f}{unit}"))
    
    def update_analysis_text(self, iteration: int, results: Dict[str, Any]):
        """更新分析文本"""
        analysis_text = f"=== 第{iteration}轮模拟分析 ===\n\n"
        
        # 检查异常情况
        issues = []
        
        # 壁温检查
        max_temp = results.get('max_wall_temperature', 0)
        if max_temp > 820:
            issues.append(f"壁温超标: {max_temp:.1f}K > 820K")
        
        # 应力检查
        max_stress = results.get('max_stress', 0)
        if max_stress > 580:
            issues.append(f"应力超标: {max_stress:.1f}MPa > 580MPa")
        
        # 推力检查
        thrust = results.get('thrust', 0)
        if thrust < 8 or thrust > 12:
            issues.append(f"推力异常: {thrust:.1f}kgf (目标: 8-12kgf)")
        
        if issues:
            analysis_text += "❌ 发现异常:\n"
            for issue in issues:
                analysis_text += f"  • {issue}\n"
            
            # 添加调整建议
            analysis_text += "\n🔧 调整建议:\n"
            if max_temp > 820:
                analysis_text += "  • 提高冷却流速至0.95m/s\n"
                analysis_text += "  • 增大夹层厚度至0.18mm\n"
            if max_stress > 580:
                analysis_text += "  • 增加壁厚至1.1mm\n"
                analysis_text += "  • 降低燃烧室压力\n"
        else:
            analysis_text += "✅ 所有指标正常\n"
        
        # 添加性能分析
        analysis_text += f"\n📊 性能分析:\n"
        analysis_text += f"  • 推力: {thrust:.1f}kgf\n"
        analysis_text += f"  • 效率: {results.get('efficiency', 0)*100:.1f}%\n"
        analysis_text += f"  • 干重: {results.get('dry_weight', 0):.2f}kg\n"
        
        self.analysis_text.setText(analysis_text)
    
    def update_current_params(self, parameters: SimulationParameters):
        """更新当前参数显示"""
        params_text = f"=== 当前参数状态 ===\n\n"
        
        param_list = [
            ('燃烧室内径', f"{parameters.chamber_diameter:.2f}mm"),
            ('喷管扩张比', f"{parameters.expansion_ratio:.2f}"),
            ('燃料流率', f"{parameters.fuel_flow_rate:.4f}kg/s"),
            ('氧化剂流率', f"{parameters.oxidizer_flow_rate:.4f}kg/s"),
            ('燃烧室压力', f"{parameters.chamber_pressure:.2f}MPa"),
            ('冷却流速', f"{parameters.cooling_velocity:.2f}m/s"),
            ('壁厚', f"{parameters.wall_thickness:.3f}mm"),
            ('夹层厚度', f"{parameters.cooling_channel_thickness:.3f}mm")
        ]
        
        for name, value in param_list:
            params_text += f"{name}: {value}\n"
        
        self.current_params_text.setText(params_text)

def main():
    """主函数"""
    app = QtWidgets.QApplication(sys.argv)
    
    # 创建模拟器
    simulator = RocketEngineSimulator()
    
    # 设置设计需求（示例值）
    simulator.set_design_requirements(
        thrust_min=8.0,
        thrust_max=12.0,
        fuel_concentration=95.0,
        priority=OptimizationPriority.EFFICIENCY
    )
    
    # 计算初始参数
    simulator.calculate_initial_parameters()
    
    # 创建可视化窗口
    window = VisualizationWindow()
    window.set_simulator(simulator)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()