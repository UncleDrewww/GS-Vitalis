# --- START OF FILE mcp_server copy.py ---

from mcp.server.fastmcp import FastMCP
import requests
import pandas as pd
import io
import os
from typing import List, Any, Tuple, Optional, Dict
import time
import base64
from datetime import datetime, timedelta
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import webbrowser
import yaml # 需要 pip install pyyaml
from scipy import signal
import logging
import sys
import time

# 设置日志，确保在 Cherry Studio 中可见
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("AOCS_FullFidelity_Eval")

# 设置 Matplotlib 后端为非交互式，防止在服务器端弹出窗口报错
plt.switch_backend('Agg')
# 【新增修改】设置中文字体，解决图表乱码 (□□A) 问题
# 优先使用黑体或微软雅黑，兼容 Windows/Linux
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

mcp = FastMCP(port=8001)

mcp = FastMCP(port=8001)

# ==============================================================================
# 第一层：底层实现 (Implementation Layer)
# ==============================================================================

# 全局缓存配置，避免频繁IO
_SAT_CONFIG_CACHE = None

def _get_codes_impl(satellite_name: str, query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    [重构版] 读取 satellites.yaml 获取代号。
    支持中文模糊查询自动映射到 YAML Key。
    """
    global _SAT_CONFIG_CACHE
    
    # 1. 加载配置 (带缓存)
    if _SAT_CONFIG_CACHE is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "doc", "satellites.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    _SAT_CONFIG_CACHE = yaml.safe_load(f)
            except Exception as e:
                print(f"YAML 加载失败: {e}")
                return None, None
        else:
            print(f"配置文件未找到: {yaml_path}")
            return None, None

    config = _SAT_CONFIG_CACHE.get('satellites', {})
    satellite_name = (satellite_name or "").strip().upper()
    query = (query or "").strip()

    # 2. 查找卫星 (匹配 ID, Name 或 Aliases)
    target_sat_config = None
    target_db_table = None

    for sat_id, sat_data in config.items():
        # 检查 ID
        match = (sat_id.upper() == satellite_name)
        # 检查 name
        if not match and sat_data.get('name') and sat_data['name'].upper() == satellite_name:
            match = True
        # 检查 aliases
        if not match and 'aliases' in sat_data:
            if any(alias.upper() == satellite_name for alias in sat_data['aliases']):
                match = True
        
        if match:
            target_sat_config = sat_data
            target_db_table = sat_data.get('db_table')
            break
    
    if not target_sat_config:
        print(f"警告: 未找到卫星 '{satellite_name}' 的配置")
        return None, None

    # 3. 查找遥测代号 (Query 映射逻辑)
    telemetry_map = target_sat_config.get('telemetry', {})
    
    # 定义中文查询词到 YAML Key 的映射关系
    # 格式: "查询关键词": ["优先匹配的YAML Key", "备选Key"...]
    KEYWORD_MAP = {
        # --- 姿态敏感器 ---
        "星敏A": ["star_sensor_a"],
        "星敏B": ["star_sensor_b"],
        "星敏":   ["star_sensor_a"], # 默认查A
        "陀螺A": ["gyro_a"],
        "陀螺B": ["gyro_b"],
        "陀螺":   ["gyro_a"],
        
        # --- 执行机构 ---
        "飞轮A": ["wheel_a"],
        "飞轮B": ["wheel_b"],
        "飞轮C": ["wheel_c"],
        "飞轮D": ["wheel_d"],
        "电推":   ["propulsion"],
        
        # --- 综合 ---
        "姿态":   ["attitude_control"],
        "控制":   ["attitude_control"],
        "热变形": ["thermal_deformation"],
        "位置":   ["orbit_position"],
        "半长轴": ["orbit_semimajor_axis"],
        "LTDN":  ["orbit_ltdn"],
        "降交点": ["orbit_ltdn"],
        "纬度":   ["latitude"],
        "星数":   ["gnss_stars"],

        # --- 故障 ---
        "敏感器错误": ["error_sensors"],
        "执行器错误": ["error_actuators"],
        "GNSS错误":  ["error_gnss"], # 用于检测故障段
        "故障置出":   ["fault_gnss_count"], # 用于统计总数

        "故障计数": ["fault_exclusions"],
        "单机故障": ["fault_exclusions"],
        "零偏": ["gyro_a_bias", "gyro_b_bias"],
        "控制模式": ["control_mode"],
        "错误日志": ["error_log_count"],
    }

    found_key = None
    
    # 逻辑 A: 直接匹配 YAML Key (如果调用方传的是标准 Key)
    if query in telemetry_map:
        found_key = query
        
    # 逻辑 B: 关键词模糊匹配
    if not found_key:
        for keyword, candidate_keys in KEYWORD_MAP.items():
            if keyword in query:
                for key in candidate_keys:
                    if key in telemetry_map:
                        found_key = key
                        break
            if found_key: break
            
    # 逻辑 C: 兜底匹配 (如果 query 包含 YAML Key 的一部分)
    if not found_key:
        for tm_key in telemetry_map.keys():
            if tm_key in query: # 比如 query="check_wheel_a"
                found_key = tm_key
                break

    if found_key:
        tm_entry = telemetry_map[found_key]
        # YAML 里可以存 string 也可以存 object
        code_str = tm_entry.get('code') if isinstance(tm_entry, dict) else tm_entry
        return target_db_table, code_str

    print(f"警告: 在卫星 '{satellite_name}' 中未找到匹配 '{query}' 的遥测项。")
    return target_db_table, None

def _get_data_impl(satellite_code: str, telemetry_code: str, start_time_str: str = None, end_time_str: str = None) -> pd.DataFrame:
    """
    内部逻辑：请求 Grafana API。
    """
    GRAFANA_URL = "http://grafana10.prod.yhroot.com"
    DATASOURCE_UID = "f1771c95-2940-4f40-a814-65fdfb1838c0" 
    GRAFANA_API_KEY = "eyJrIjoi...YOUR_VERY_LONG_API_KEY...IjozfQ==" # 替换为真实 Key
    
    api_path = f"/api/datasources/proxy/uid/{DATASOURCE_UID}/query"
    full_url = GRAFANA_URL + api_path
    headers = {'Authorization': f'Bearer {GRAFANA_API_KEY}', 'Content-Type': 'application/json'}

    now = datetime.now()
    try:
        end_dt = datetime.strptime(end_time_str.strip(), '%Y-%m-%d %H:%M:%S') if end_time_str and end_time_str.strip() else now
        start_dt = datetime.strptime(start_time_str.strip(), '%Y-%m-%d %H:%M:%S') if start_time_str and start_time_str.strip() else end_dt - timedelta(days=30)
    except Exception:
        end_dt = now
        start_dt = end_dt - timedelta(days=30)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    codes = [c.strip().strip('"').strip("'") for c in str(telemetry_code).split(',') if c.strip()]
    if not codes: return pd.DataFrame()

    select_fields = ', '.join(f'"{c}"' for c in codes)
    query_string = f'SELECT {select_fields} FROM "{satellite_code}" WHERE time >= {start_ms}ms AND time <= {end_ms}ms ORDER BY time ASC'
    
    try:
        response = requests.get(full_url, headers=headers, params={'db': 'measure', 'q': query_string, 'pretty': 'true'}, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        result0 = data['results'][0]
        if not result0 or 'series' not in result0: return pd.DataFrame()
        
        dfs = []
        for s in result0.get('series', []):
            df_s = pd.DataFrame(s.get('values', []), columns=s.get('columns', []))
            dfs.append(df_s)
            
        df_raw = pd.concat(dfs, ignore_index=True, sort=False)
        final_cols = [col for col in codes if col in df_raw.columns]
        if not final_cols: return pd.DataFrame()
        
        return df_raw[final_cols]

    except Exception as e:
        print(f"Grafana 请求失败: {e}")
        return pd.DataFrame()

def _analyze_star_sensor_impl(data: pd.DataFrame, sensor_name: str = "星敏") -> Dict:
    if data.shape[1] < 5:
        return {"is_abnormal": True, "summary": f"{sensor_name} 数据列数不足", "html": f"<div class='error'>{sensor_name} 数据列数不足。</div>"}

    try:
        time_stamps_values = data.iloc[:, 0]
        quaternions = data.iloc[:, 1:5].values
        time_stamps_numeric = pd.to_numeric(time_stamps_values, errors='raise')
        T = time_stamps_numeric - time_stamps_numeric.iloc[0]
        num_points = len(T)
        if num_points < 4:
            return {"is_abnormal": True, "summary": "数据点过少", "html": f"<div class='error'>{sensor_name} 数据点过少。</div>"}
        
        degree = min(8, num_points - 2)
        q_fitted = np.zeros_like(quaternions)
        for i in range(4):
            q_component = quaternions[:, i]
            coeffs = np.polyfit(T, q_component, degree)
            q_fitted[:, i] = np.polyval(coeffs, T)
        
        quaternions_scipy_order = quaternions[:, [1, 2, 3, 0]]
        q_fitted_scipy_order = q_fitted[:, [1, 2, 3, 0]]
        R_raw = Rotation.from_quat(quaternions_scipy_order)
        R_fitted = Rotation.from_quat(q_fitted_scipy_order)
        delta_R = R_raw.inv() * R_fitted
        euler_angles_rad = delta_R.as_euler('zyx', degrees=False)
        s = euler_angles_rad[:, ::-1]
        
        RAD_TO_ARCSEC = np.rad2deg(1) * 3600
        noises = [3 * np.std(s[:, 0]) * RAD_TO_ARCSEC, 3 * np.std(s[:, 1]) * RAD_TO_ARCSEC, 3 * np.std(s[:, 2]) * RAD_TO_ARCSEC]
        
        limits = [3.0, 3.0, 30.0]
        axes = ['X', 'Y', 'Z']
        table_rows = ""
        has_anomaly = False
        anomaly_details = []
        
        for i in range(3):
            val = noises[i]
            lim = limits[i]
            if val > lim:
                status = "<span style='color:#dc3545; font-weight:bold;'>异常</span>"
                val_style = "color:#dc3545; font-weight:bold;"
                has_anomaly = True
                anomaly_details.append(f"{axes[i]}轴({val:.2f}\")")
            else:
                status = "<span style='color:#28a745; font-weight:bold;'>合格</span>"
                val_style = "color:#333;"
            table_rows += f"<tr><td style='font-weight:bold;'>{axes[i]} 轴</td><td style='{val_style}'>{val:.4f}</td><td style='color:#666;'>&le; {lim}</td><td>{status}</td></tr>"
            
        summary_style = "background:#fff5f5; border-left:4px solid #dc3545;" if has_anomaly else "background:#f0fff4; border-left:4px solid #28a745;"
        summary_text = f"{sensor_name} 存在指标超差。" if has_anomaly else f"{sensor_name} 状态良好。"
        
        html_fragment = f"""
        <div class="section">
            <h2>{sensor_name} 测量噪声分析</h2>
            <div style="padding:10px; margin-bottom:15px; {summary_style} font-size:13px; color:#333;">
                <strong>诊断结论：</strong> {summary_text}
            </div>
            <p style="font-size:12px; color:#666;">分析配置：{degree} 阶拟合，数据点数 {num_points}。</p>
            <table style="width:100%; text-align:center; font-size:13px;">
                <thead><tr style="background:#f8f9fa;"><th style="padding:8px;">分析轴</th><th>实测噪声 (3σ, ″)</th><th>指标要求 (″)</th><th>判定结果</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """
        return {"is_abnormal": has_anomaly, "summary": f"{sensor_name}: " + (", ".join(anomaly_details) if has_anomaly else "合格"), "html": html_fragment}

    except Exception as e:
        return {"is_abnormal": True, "summary": f"{sensor_name} 分析出错", "html": f"<div class='error'>分析错误: {e}</div>"}

def _analyze_gyro_impl(data: pd.DataFrame, gyro_name: str, limit_val: float) -> Dict:
    if data.shape[1] < 4:
        return {"is_abnormal": True, "summary": f"{gyro_name} 数据不足", "html": f"<div class='error'>{gyro_name} 数据列数不足。</div>"}

    try:
        raw_data = data.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce').dropna()
        if raw_data.empty:
            return {"is_abnormal": True, "summary": f"{gyro_name} 数据为空", "html": f"<div class='error'>{gyro_name} 有效数据为空。</div>"}

        axes = ['X', 'Y', 'Z']
        centered_data = raw_data - raw_data.mean()
        noise_3sigma = centered_data.std() * 3
        
        table_rows = ""
        has_anomaly = False
        anomaly_details = []
        
        for i, axis in enumerate(axes):
            val = noise_3sigma.iloc[i]
            if val > limit_val:
                status = "<span style='color:#dc3545; font-weight:bold;'>异常</span>"
                val_style = "color:#dc3545; font-weight:bold;"
                has_anomaly = True
                anomaly_details.append(f"{axis}轴({val:.4f})")
            else:
                status = "<span style='color:#28a745; font-weight:bold;'>合格</span>"
                val_style = "color:#333;"
            table_rows += f"<tr><td style='font-weight:bold;'>{axis} 轴</td><td style='{val_style}'>{val:.6f}</td><td style='color:#666;'>&le; {limit_val:.6f}</td><td>{status}</td></tr>"

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 4))
        x_axis = range(len(centered_data))
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        for i, col_name in enumerate(centered_data.columns):
            ax.plot(x_axis, centered_data[col_name], label=f'{axes[i]}轴噪声', color=colors[i], linewidth=0.5, alpha=0.8)
        ax.set_title(f'{gyro_name} 输出噪声')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        summary_style = "background:#fff5f5; border-left:4px solid #dc3545;" if has_anomaly else "background:#f0fff4; border-left:4px solid #28a745;"
        summary_text = f"{gyro_name} 存在噪声超标。" if has_anomaly else f"{gyro_name} 状态良好。"

        html = f"""
        <div class="section">
            <h2>{gyro_name} 噪声水平评估</h2>
            <div style="padding:10px; margin-bottom:15px; {summary_style} font-size:13px; color:#333;">
                <strong>诊断结论：</strong> {summary_text}
            </div>
            <table style="width:100%; text-align:center; font-size:13px;">
                <thead><tr style="background:#f8f9fa;"><th>轴系</th><th>实测噪声 (3σ)</th><th>指标要求</th><th>判定结果</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            <div style="text-align:center; margin-top:15px;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd;"></div>
        </div>
        """
        return {"is_abnormal": has_anomaly, "summary": f"{gyro_name}: " + (", ".join(anomaly_details) if has_anomaly else "合格"), "html": html}
    except Exception as e:
        return {"is_abnormal": True, "summary": f"{gyro_name} 内部错误", "html": f"<div class='error'>{gyro_name} 分析出错: {e}</div>"}

def _analyze_gyro_bias_impl(data: pd.DataFrame, gyro_name: str) -> Dict:
    """
    [高保真] 陀螺零偏月度稳定性分析。
    统计全月均值、3-Sigma稳定性及漂移范围。
    """
    if data.empty or data.shape[1] < 3:
        return {"is_abnormal": False, "summary": "无零偏数据", "html": ""}

    try:
        # 1. 数据统计
        # 假设数据列顺序为 X, Y, Z
        bias_values = data.apply(pd.to_numeric, errors='coerce').dropna()
        if bias_values.empty: return {"is_abnormal": False, "summary": "数据无效", "html": ""}

        stats = []
        axes = ['X轴', 'Y轴', 'Z轴']
        for i in range(3):
            col_data = bias_values.iloc[:, i]
            stats.append({
                "axis": axes[i],
                "mean": col_data.mean(),
                "std3": col_data.std() * 3,
                "p2p": col_data.max() - col_data.min()
            })

        # 2. 绘图 (三轴趋势图)
        fig, axes_plt = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for i in range(3):
            ax = axes_plt[i]
            y_data = bias_values.iloc[:, i].values
            ax.plot(y_data, color=colors[i], linewidth=0.8, rasterized=True)
            ax.set_ylabel(f'{axes[i]} (deg/s)') # 单位根据实际情况调整，通常星上估计为deg/s或rad/s
            ax.grid(True, alpha=0.3)
            if i == 0: ax.set_title(f'{gyro_name} 零偏全月演变趋势')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 3. 构造 HTML 表格
        table_rows = ""
        for s in stats:
            table_rows += f"""
            <tr>
                <td style="padding:8px; border:1px solid #eee;">{s['axis']}</td>
                <td style="padding:8px; border:1px solid #eee;">{s['mean']:.8f}</td>
                <td style="padding:8px; border:1px solid #eee;">{s['std3']:.8f}</td>
                <td style="padding:8px; border:1px solid #eee;">{s['p2p']:.8f}</td>
            </tr>
            """

        html = f"""
        <div class="section">
            <h2>{gyro_name} 零偏稳定性评估 (全月)</h2>
            <table style="width:100%; text-align:center; border-collapse:collapse; font-size:12px; margin-bottom:15px;">
                <thead style="background:#f8f9fa;">
                    <tr>
                        <th style="padding:10px; border:1px solid #eee;">轴系</th>
                        <th style="border:1px solid #eee;">月均值 (deg/s)</th>
                        <th style="border:1px solid #eee;">稳定性 (3σ)</th>
                        <th style="border:1px solid #eee;">全月峰峰值</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
            <div style="text-align:center;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd;"></div>
        </div>
        """
        return {"is_abnormal": False, "summary": "已分析", "html": html}

    except Exception as e:
        logger.error(f"{gyro_name} 零偏分析失败: {e}")
        return {"is_abnormal": False, "summary": "分析出错", "html": ""}
    
def _analyze_wheel_impl(data: pd.DataFrame, wheel_name: str, limit_val: float = 0.05) -> Dict:
    if data.shape[1] < 3:
        return {"is_abnormal": True, "summary": f"{wheel_name} 数据不足", "html": f"<div class='error'>{wheel_name} 数据列数不足。</div>"}

    try:
        cmd_data = pd.to_numeric(data.iloc[:, 1], errors='coerce')
        fbk_data = pd.to_numeric(data.iloc[:, 2], errors='coerce')
        df_calc = pd.DataFrame({'cmd': cmd_data, 'fbk': fbk_data}).dropna()
        if df_calc.empty:
            return {"is_abnormal": True, "summary": f"{wheel_name} 无数据", "html": f"<div class='error'>{wheel_name} 无有效数据。</div>"}

        raw_error = df_calc['cmd'] - df_calc['fbk']
        OUTLIER_THRESHOLD = 10.0
        valid_mask = raw_error.abs() <= OUTLIER_THRESHOLD
        df_clean = df_calc[valid_mask]
        error_clean = raw_error[valid_mask]
        
        if error_clean.empty:
            return {"is_abnormal": True, "summary": f"{wheel_name} 全异常", "html": f"<div class='error'>{wheel_name} 数据均被剔除。</div>"}

        control_accuracy_3sigma = error_clean.std() * 3
        has_anomaly = control_accuracy_3sigma > limit_val
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        x_axis = range(len(error_clean))
        ax1.plot(x_axis, df_clean['cmd'], color='#3498db', label='指令')
        ax1.plot(x_axis, df_clean['fbk'], color='#e67e22', linestyle='--', label='反馈')
        ax1.legend()
        ax1.set_title(f'{wheel_name} 转速跟踪')
        ax2.plot(x_axis, error_clean, color='#e74c3c')
        ax2.axhline(y=limit_val, color='red', linestyle=':')
        ax2.axhline(y=-limit_val, color='red', linestyle=':')
        ax2.set_title(f'控制误差 (3σ: {control_accuracy_3sigma:.4f} rpm)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        status = "<span style='color:#dc3545;'>异常</span>" if has_anomaly else "<span style='color:#28a745;'>合格</span>"
        val_style = "color:#dc3545;" if has_anomaly else "color:#333;"
        
        html = f"""
        <div class="section">
            <h2>{wheel_name} 性能评估</h2>
            <table style="width:100%; text-align:center; font-size:13px;">
                <thead><tr style="background:#f8f9fa;"><th>评估项</th><th>计算结果 (3σ)</th><th>指标要求</th><th>判定结果</th></tr></thead>
                <tbody>
                    <tr><td>转速控制精度</td><td style="{val_style}">{control_accuracy_3sigma:.4f} rpm</td><td>&le; {limit_val}</td><td>{status}</td></tr>
                </tbody>
            </table>
            <div style="text-align:center; margin-top:15px;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd;"></div>
        </div>
        """
        return {"is_abnormal": has_anomaly, "summary": f"{wheel_name}: " + (f"精度超差({control_accuracy_3sigma:.3f})" if has_anomaly else "合格"), "html": html}
    except Exception as e:
        return {"is_abnormal": True, "summary": f"{wheel_name} 出错", "html": f"<div class='error'>{wheel_name} 分析出错: {e}</div>"}

# ==========================================
# 增强版姿态分析逻辑 (支持全量统计与超差计数)
# ==========================================
def _analyze_attitude_monthly_impl(data: pd.DataFrame) -> Dict:
    """
    内部逻辑：针对月度全量数据进行姿态分析。
    增加了超差计数逻辑。
    """
    if data.empty or data.shape[1] < 7:
        return {"is_abnormal": True, "summary": "姿态数据不足", "html": "<div class='error'>数据列数不足</div>"}
    
    try:
        # 指标阈值定义
        LIMIT_AGL = 0.02    # 姿态角超差门限 (deg)
        LIMIT_W = 0.003     # 姿态稳定度门限 (deg/s)
        
        # 1. 数据转换
        raw_values = data.values
        # 姿态角 [Time, Roll, Pitch, Yaw] -> index 1,2,3
        agl_data = raw_values[:, 1:4].astype(float)
        # 角速度 [Time, ..., Wx, Wy, Wz] -> index 4,5,6
        w_data = raw_values[:, 4:7].astype(float)
        
        # 2. 核心统计计算 (全量数据)
        # 计算 3-Sigma
        agl_std = np.nanstd(agl_data, axis=0, ddof=1)
        w_std = np.nanstd(w_data, axis=0, ddof=1)
        agl_3sigma = 3 * agl_std
        w_3sigma = 3 * w_std
        
        # --- [新增] 超差统计逻辑 ---
        # 只要任意一轴超标，即计为一次超差样本
        agl_excess_mask = np.any(np.abs(agl_data) > LIMIT_AGL, axis=1)
        w_excess_mask = np.any(np.abs(w_data) > LIMIT_W, axis=1)
        
        count_agl_excess = int(np.sum(agl_excess_mask))
        count_w_excess = int(np.sum(w_excess_mask))
        total_samples = len(data)
        agl_excess_rate = (count_agl_excess / total_samples) * 100
        w_excess_rate = (count_w_excess / total_samples) * 100
        
        # 3. 判定与报告生成
        axes_name = ['Roll', 'Pitch', 'Yaw']
        table_rows = ""
        has_anomaly = False
        
        # 姿态角行
        for i in range(3):
            val = agl_3sigma[i]
            res_html = "<span style='color:#dc3545;'>超标</span>" if val > LIMIT_AGL else "<span style='color:#28a745;'>合格</span>"
            if val > LIMIT_AGL: has_anomaly = True
            table_rows += f"<tr><td>姿态精度(3σ)</td><td>{axes_name[i]}</td><td>{val:.5f}</td><td>&le; {LIMIT_AGL}</td><td>{res_html}</td></tr>"
        
        # 角速度行
        for i in range(3):
            val = w_3sigma[i]
            res_html = "<span style='color:#dc3545;'>超标</span>" if val > LIMIT_W else "<span style='color:#28a745;'>合格</span>"
            if val > LIMIT_W: has_anomaly = True
            table_rows += f"<tr><td>姿态稳定度(3σ)</td><td>{axes_name[i]}</td><td>{val:.6f}</td><td>&le; {LIMIT_W}</td><td>{res_html}</td></tr>"

        # 4. 绘图 (全月全量绘图，使用 rasterized=True 优化性能)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        x_axis = range(total_samples)
        
        # 姿态角曲线
        ax1.plot(x_axis, agl_data, linewidth=0.5, alpha=0.8, rasterized=True)
        ax1.axhline(y=LIMIT_AGL, color='r', linestyle='--', alpha=0.3)
        ax1.axhline(y=-LIMIT_AGL, color='r', linestyle='--', alpha=0.3)
        ax1.set_title(f'全月姿态角演变 (超差样本数: {count_agl_excess})')
        
        # 角速度曲线
        ax2.plot(x_axis, w_data, linewidth=0.5, alpha=0.8, rasterized=True)
        ax2.axhline(y=LIMIT_W, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=-LIMIT_W, color='r', linestyle='--', alpha=0.3)
        ax2.set_title(f'全月角速度演变 (超差样本数: {count_w_excess})')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 5. HTML 结构
        summary_color = "#dc3545" if has_anomaly else "#28a745"
        html = f"""
        <div class="section">
            <h2>姿态控制性能月度统计 (全量高保真)</h2>
            <div style="display:flex; gap:10px; margin-bottom:15px;">
                <div style="flex:1; padding:15px; background:#f8f9fa; border-radius:8px; border-top:4px solid {summary_color};">
                    <div style="font-size:12px; color:#666;">姿态角超差总数</div>
                    <div style="font-size:24px; font-weight:bold; color:{summary_color};">{count_agl_excess} <small style="font-size:12px; font-weight:normal;">({agl_excess_rate:.4f}%)</small></div>
                </div>
                <div style="flex:1; padding:15px; background:#f8f9fa; border-radius:8px; border-top:4px solid {summary_color};">
                    <div style="font-size:12px; color:#666;">角速度超差总数</div>
                    <div style="font-size:24px; font-weight:bold; color:{summary_color};">{count_w_excess} <small style="font-size:12px; font-weight:normal;">({w_excess_rate:.4f}%)</small></div>
                </div>
            </div>
            <table>
                <thead><tr style="background:#f1f3f5;"><th>分析项目</th><th>轴</th><th>实测(3σ)</th><th>指标门限</th><th>判定</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            <div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%; border:1px solid #ddd;"></div>
        </div>
        """
        return {
            "is_abnormal": has_anomaly or (count_agl_excess > 0), 
            "summary": f"姿态: 超差{count_agl_excess}点", 
            "html": html
        }
    except Exception as e:
        logger.error(f"姿态月度分析计算失败: {e}")
        return {"is_abnormal": True, "summary": "计算出错", "html": f"<div>分析异常: {e}</div>"}

def _analyze_device_errors_impl(sat_code: str, start_str: str, end_str: str) -> Tuple[List[Dict], str]:
    """
    修改后：返回 (11个单机的独立结果列表, 合并后的HTML表格)
    """
    groups = [
        {"query": "敏感器错误", "map": {"TMKA601": "陀螺A", "TMKA617": "陀螺B", "TMKA633": "磁强计A", "TMKA646": "磁强计B", "TMKA659": "星敏A", "TMKA666": "星敏B"}},
        {"query": "执行器错误", "map": {"TMKA673": "飞轮A", "TMKA679": "飞轮B", "TMKA685": "飞轮C", "TMKA691": "飞轮D", "TMKA697": "GNSS"}}
    ]
    table_rows = ""
    has_any_error = False
    
    individual_results = [] # 用于仪表盘计数的独立结果

    for group in groups:
        _, tm_codes_str = _get_codes_impl(sat_code, group["query"])
        if not tm_codes_str: 
            # 如果没查到代号，也要把这些单机标记为“未配置”，以免总数对不上
            for _, name in group["map"].items():
                individual_results.append({
                    "name": name, 
                    "is_abnormal": False, 
                    "summary": "未配置/跳过", 
                    "html": "" # 不需要单独的HTML
                })
            continue

        df = _get_data_impl(sat_code, tm_codes_str, start_str, end_str)
        
        for code, name in group["map"].items():
            increase_count = 0
            # 默认为 False，只有真的检测到错误才置 True
            is_abnormal = False
            status_text = "正常"

            if not df.empty and code in df.columns:
                series = pd.to_numeric(df[code], errors='coerce').dropna()
                if not series.empty:
                    vals = series.astype(int) % 256
                    diffs = vals.diff().fillna(0)
                    adjusted = np.where(diffs < -200, diffs + 256, diffs)
                    incs = np.where(adjusted > 0, adjusted, 0)
                    increase_count = int(np.sum(incs))
                    
                    if increase_count > 0:
                        is_abnormal = True
                        has_any_error = True
                        status_text = f"通信错误 +{increase_count}"
            
            # 添加到独立结果列表
            individual_results.append({
                "name": name,
                "is_abnormal": is_abnormal,
                "summary": status_text,
                "html": "" # 占位，避免单独打印
            })

            # 构造大表格行
            if increase_count > 0:
                name_style = "color:#dc3545; font-weight:bold;"
                val_style = "background:#fff5f5; color:#dc3545; font-weight:bold;"
            else:
                name_style, val_style = "color:#333;", "color:#28a745;"
            
            table_rows += f"<tr><td style='{name_style}'>{name}</td><td style='{val_style}'>+{increase_count}</td></tr>"

    summary_text = "发现单机通信异常" if has_any_error else "单机通信状态良好"
    color = "#dc3545" if has_any_error else "#28a745"
    
    # 构造合并后的 HTML
    full_html_table = f"""
    <div class="section">
        <h2>全星单机通信错误统计</h2>
        <div style="padding:10px; border-left:4px solid {color}; background:{color}1a; color:#333;">
            <strong>统计结论：</strong> {summary_text}
        </div>
        <table><thead><tr><th>单机</th><th>错误增量</th></tr></thead><tbody>{table_rows}</tbody></table>
    </div>
    """
    
    return individual_results, full_html_table

def _analyze_all_unit_faults_impl(satellite_name: str, start_str: str, end_str: str) -> Tuple[List[Dict], str]:
    global _SAT_CONFIG_CACHE
    if _SAT_CONFIG_CACHE is None: _get_codes_impl(satellite_name, "任意")
    
    target_sat_config = None
    search_q = satellite_name.upper().strip()
    for sid, sdata in _SAT_CONFIG_CACHE.get('satellites', {}).items():
        if sid.upper() == search_q or sdata.get('name','').upper() == search_q or search_q in [a.upper() for a in sdata.get('aliases', [])]:
            target_sat_config = sdata
            break
    if not target_sat_config: return [], ""

    entry = target_sat_config.get('telemetry', {}).get('fault_exclusions', {})
    if not entry: return [], ""

    codes = [c.strip() for c in entry.get('code', '').split(',') if c.strip()]
    names = [n.strip() for n in entry.get('desc', '').split(',') if n.strip()]
    fault_map = dict(zip(codes, names))
    df = _get_data_impl(target_sat_config.get('db_table'), entry.get('code'), start_str, end_str)
    
    if df.empty: return [], ""
    
    res_list, table_rows, has_fault = [], "", False
    for code in codes:
        unit_name = fault_map.get(code, "未知")
        inc = 0
        if code in df.columns:
            series = pd.to_numeric(df[code], errors='coerce').dropna()
            if not series.empty:
                vals = series.astype(int) % 256
                diffs = vals.diff().fillna(0)
                adjusted = np.where(diffs < 0, diffs + 256, diffs)
                inc = int(np.sum(adjusted))
                if inc > 0: has_fault = True
        
        res_list.append({"name": f"{unit_name}故障", "is_abnormal": inc > 0, "summary": f"+{inc}" if inc > 0 else "正常"})
        row_style = "background:#fff5f5; color:#dc3545; font-weight:bold;" if inc > 0 else ""
        table_rows += f"<tr style='{row_style}'><td>{unit_name}</td><td>{inc}</td></tr>"

    html = f"""
    <div class="section">
        <h4>全星单机故障置出统计 (1month)</h4>
        <table style="width:100%; border-collapse:collapse; text-align:center; font-size:12px;">
            <thead style="background:#f8f9fa;"><tr><th>单机名称</th><th>故障增量</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
    </div>"""
    return res_list, html

def _analyze_orbit_impl(data: pd.DataFrame) -> Dict:
    """
    轨道半长轴分析。
    逻辑：高度 = 半长轴 - 地球半径 (6378140)
    指标：平均高度 [500, 530] km
    """
    # 增加列数检查，防止空数据导致后续报错
    if data.empty or data.shape[1] < 1:
        return {"is_abnormal": True, "summary": "轨道数据为空", "html": "<div class='error'>轨道数据为空或格式错误</div>"}

    try:
        # 1. 数据提取与转换
        # 【修正点】这里改为 iloc[:, 0]，因为返回的 Dataframe 只有一列数据
        raw_values = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
        
        if raw_values.empty:
            return {"is_abnormal": True, "summary": "轨道数据无效", "html": "<div class='error'>轨道数据无效</div>"}

        R_EARTH = 6378140.0
        
        # 计算高度 (km)
        # 假设遥测值单位为米(m)
        altitude_km = (raw_values - R_EARTH) / 1000.0
        
        # 2. 统计指标
        mean_h = altitude_km.mean()
        std_h = altitude_km.std()
        sigma3_h = 3 * std_h
        
        # 3. 判定 (500 ~ 530 km)
        LIMIT_MIN = 500.0
        LIMIT_MAX = 530.0
        
        is_abnormal = False
        if not (LIMIT_MIN <= mean_h <= LIMIT_MAX):
            is_abnormal = True
            summary_text = f"轨道高度异常 (均值 {mean_h:.2f} km)"
            status_html = "<span style='color:#dc3545; font-weight:bold;'>异常 (超标)</span>"
            val_style = "color:#dc3545; font-weight:bold;"
        else:
            summary_text = "轨道高度正常"
            status_html = "<span style='color:#28a745; font-weight:bold;'>合格</span>"
            val_style = "color:#333;"

        # 4. 绘图
        fig, ax = plt.subplots(figsize=(10, 4))
        x_axis = range(len(altitude_km))
        ax.plot(x_axis, altitude_km, color='#9b59b6', linewidth=1.5, label='轨道高度')
        
        # 画限制线
        ax.axhline(y=LIMIT_MAX, color='red', linestyle='--', alpha=0.3, label='上限 530km')
        ax.axhline(y=LIMIT_MIN, color='red', linestyle='--', alpha=0.3, label='下限 500km')
        
        ax.set_title(f'轨道高度变化趋势 (均值: {mean_h:.2f} km)')
        ax.set_ylabel('高度 (km)')
        ax.set_xlabel('采样点')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 5. 生成 HTML
        summary_bg = "#fff5f5" if is_abnormal else "#f0fff4"
        summary_border = "#dc3545" if is_abnormal else "#28a745"
        
        html = f"""
        <div class="section">
            <h2>轨道维持分析</h2>
            <div style="padding:10px; margin-bottom:15px; background:{summary_bg}; border-left:4px solid {summary_border}; font-size:13px; color:#333;">
                <strong>诊断结论：</strong> {summary_text}
            </div>
            <table style="width:100%; text-align:center; font-size:13px;">
                <thead><tr style="background:#f8f9fa;"><th>评估项</th><th>计算结果</th><th>指标要求</th><th>判定</th></tr></thead>
                <tbody>
                    <tr>
                        <td>平均轨道高度</td>
                        <td style="{val_style}">{mean_h:.4f} km</td>
                        <td>500 ~ 530 km</td>
                        <td>{status_html}</td>
                    </tr>
                    <tr>
                        <td>高度波动 (3σ)</td>
                        <td>{sigma3_h:.4f} km</td>
                        <td>-</td>
                        <td>参考</td>
                    </tr>
                </tbody>
            </table>
            <div style="text-align:center; margin-top:15px;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd;">
            </div>
        </div>
        """
        
        return {
            "is_abnormal": is_abnormal,
            "summary": f"轨道高度: {mean_h:.1f}km ({'异常' if is_abnormal else '正常'})",
            "html": html
        }

    except Exception as e:
        return {"is_abnormal": True, "summary": "轨道分析出错", "html": f"<div class='error'>轨道分析出错: {e}</div>"}

def _analyze_ltdn_impl(data: pd.DataFrame) -> Dict:
    """
    降交点地方时 (LTDN) 分析。
    指标：平均值 [10.0, 11.0] 小时
    """
    # 同样注意：只有一列数据，使用 iloc[:, 0]
    if data.empty or data.shape[1] < 1:
        return {"is_abnormal": True, "summary": "LTDN数据为空", "html": "<div class='error'>LTDN数据为空</div>"}

    try:
        raw_values = pd.to_numeric(data.iloc[:, 0], errors='coerce').dropna()
        if raw_values.empty:
            return {"is_abnormal": True, "summary": "LTDN数据无效", "html": "<div class='error'>LTDN数据无效</div>"}

        # 统计指标
        mean_val = raw_values.mean()
        std_val = raw_values.std()
        sigma3_val = 3 * std_val
        
        # 判定 (10 ~ 11 h)
        LIMIT_MIN = 10.0
        LIMIT_MAX = 11.0
        
        is_abnormal = False
        if not (LIMIT_MIN <= mean_val <= LIMIT_MAX):
            is_abnormal = True
            summary_text = f"降交点地方时异常 (均值 {mean_val:.2f}h)"
            status_html = "<span style='color:#dc3545; font-weight:bold;'>异常 (超标)</span>"
            val_style = "color:#dc3545; font-weight:bold;"
        else:
            summary_text = "降交点地方时正常"
            status_html = "<span style='color:#28a745; font-weight:bold;'>合格</span>"
            val_style = "color:#333;"

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 4))
        x_axis = range(len(raw_values))
        ax.plot(x_axis, raw_values, color='#e67e22', linewidth=1.5, label='LTDN')
        
        # 画限制线
        ax.axhline(y=LIMIT_MAX, color='red', linestyle='--', alpha=0.3, label='上限 11h')
        ax.axhline(y=LIMIT_MIN, color='red', linestyle='--', alpha=0.3, label='下限 10h')
        
        ax.set_title(f'降交点地方时演变 (均值: {mean_val:.4f} h)')
        ax.set_ylabel('地方时 (Hour)')
        ax.set_xlabel('采样点')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 生成 HTML
        summary_bg = "#fff5f5" if is_abnormal else "#f0fff4"
        summary_border = "#dc3545" if is_abnormal else "#28a745"
        
        html = f"""
        <div class="section">
            <h2>降交点地方时 (LTDN) 分析</h2>
            <div style="padding:10px; margin-bottom:15px; background:{summary_bg}; border-left:4px solid {summary_border}; font-size:13px; color:#333;">
                <strong>诊断结论：</strong> {summary_text}
            </div>
            <table style="width:100%; text-align:center; font-size:13px;">
                <thead><tr style="background:#f8f9fa;"><th>评估项</th><th>计算结果</th><th>指标要求</th><th>判定</th></tr></thead>
                <tbody>
                    <tr>
                        <td>平均地方时</td>
                        <td style="{val_style}">{mean_val:.4f} h</td>
                        <td>10.0 ~ 11.0 h</td>
                        <td>{status_html}</td>
                    </tr>
                    <tr>
                        <td>稳定性 (3σ)</td>
                        <td>{sigma3_val:.4f} h</td>
                        <td>-</td>
                        <td>参考</td>
                    </tr>
                </tbody>
            </table>
            <div style="text-align:center; margin-top:15px;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd;">
            </div>
        </div>
        """
        
        return {
            "is_abnormal": is_abnormal,
            "summary": f"LTDN: {mean_val:.2f}h ({'异常' if is_abnormal else '正常'})",
            "html": html
        }

    except Exception as e:
        return {"is_abnormal": True, "summary": "LTDN分析出错", "html": f"<div class='error'>LTDN分析出错: {e}</div>"}

def _analyze_propulsion_impl(data: pd.DataFrame) -> Dict:
    if data.empty or data.shape[1] < 2:
        return {"is_abnormal": False, "summary": "无电推数据", "html": "<div class='error'>无电推数据</div>"}
    try:
        valid_data = data.dropna().apply(pd.to_numeric, errors='coerce')
        if valid_data.empty: return {"is_abnormal": False, "summary": "数据无效", "html": ""}

        # 【核心修正】使用 .iloc 获取元素
        latest_row = valid_data.iloc[-1]
        raw_ticks = latest_row.iloc[0] 
        work_cycles = int(latest_row.iloc[1])
        
        duration_sec = raw_ticks / 4.0
        thrust_N, rated_impulse = 0.015, 72480.0
        current_impulse = duration_sec * thrust_N
        used_p = (current_impulse / rated_impulse) * 100.0
        
        # 绘图逻辑 (简化展示，确保不崩)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(['已使用', '剩余'], [used_p, 100-used_p], color=['#bdc3c7', '#27ae60'])
        ax.set_title(f"电推寿命占比 (已用 {used_p:.1f}%)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "is_abnormal": used_p > 90.0,
            "summary": f"已用 {used_p:.1f}%",
            "html": f"""
            <div style='text-align:center;'>
                <p>累计工作: {duration_sec/3600:.2f}h | 次数: {work_cycles}次 | 总冲: {current_impulse:.1f}Ns</p>
                <img src="data:image/png;base64,{img_b64}" style="max-width:400px;">
            </div>"""
        }
    except Exception as e:
        return {"is_abnormal": False, "summary": "分析异常", "html": f"<div>分析错误: {e}</div>"}

def _analyze_thermal_impl(sat_code: str, start_str: str, end_str: str) -> Tuple[Dict, str]:
    _, tm_code = _get_codes_impl(sat_code, "热变形")
    if not tm_code: return {"error": "未配置"}, "<div class='error'>未配置热变形遥测</div>"
    
    df = _get_data_impl(sat_code, tm_code, start_str, end_str)
    if df.empty or df.shape[1] < 9: return {"error": "数据不足"}, "<div class='error'>数据不足</div>"

    try:
        t_vals = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df_valid = df[~t_vals.isna()]
        QA_raw = df_valid.iloc[:, 1:5].apply(pd.to_numeric, errors='coerce').values
        QB_raw = df_valid.iloc[:, 5:9].apply(pd.to_numeric, errors='coerce').values
        
        if len(QA_raw) < 10: return {"error": "数据太少"}, "<div class='error'>数据太少</div>"

        QA_scipy = QA_raw[:, [1, 2, 3, 0]]
        QB_scipy = QB_raw[:, [1, 2, 3, 0]]
        z_axis = np.array([0, 0, 1])
        rot_a = Rotation.from_quat(QA_scipy)
        rot_b = Rotation.from_quat(QB_scipy)
        vec_a = rot_a.apply(z_axis)
        vec_b = rot_b.apply(z_axis)
        
        dots = np.clip(np.sum(vec_a * vec_b, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        median_angle = np.median(angles)
        angles_clean = angles[np.abs(angles - median_angle) < 0.5]
        
        if len(angles_clean) < 10: return {"error": "清洗后无数据"}, "<div class='error'>清洗后无数据</div>"

        Ts = 8.0 
        fs = 1 / Ts 
        fc = 1 / 216.0
        wn = fc / (fs / 2)
        if wn >= 1: wn = 0.99
        b, a = signal.butter(N=2, Wn=wn, btype='low')
        y_thermal = signal.filtfilt(b, a, angles_clean)
        y_noise = angles_clean - y_thermal
        thermal_variation = y_thermal - np.mean(y_thermal)
        
        thermal_3sigma = np.std(thermal_variation) * 3
        noise_3sigma = np.std(y_noise) * 3
        mean_angle = np.mean(angles_clean)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        x_axis = range(len(angles_clean))
        ax1.plot(x_axis, angles_clean, color='#333', alpha=0.5)
        ax1.plot(x_axis, y_thermal, color='red')
        ax1.set_title(f'光轴夹角 (均值: {mean_angle:.4f}°)')
        ax2.plot(x_axis, thermal_variation, color='#e67e22')
        ax2.set_title(f'热变形 (3σ: {thermal_3sigma:.6f}°)')
        ax3.plot(x_axis, y_noise, color='#3498db', alpha=0.6)
        ax3.set_title(f'噪声 (3σ: {noise_3sigma:.6f}°)')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        html = f"""
        <div class="section">
            <h2>星敏热变形分析</h2>
            <table style="width: 60%; margin: auto;">
                <tr><td style="text-align:left">光轴平均夹角</td><td>{mean_angle:.4f}°</td></tr>
                <tr><td style="text-align:left">热变形稳定性 (3σ)</td><td style="color:#e67e22; font-weight:bold;">{thermal_3sigma:.6f}°</td></tr>
                <tr><td style="text-align:left">高频测量噪声 (3σ)</td><td style="color:#3498db; font-weight:bold;">{noise_3sigma:.6f}°</td></tr>
            </table>
            <div style="text-align:center; margin-top:20px;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%;"></div>
        </div>
        """
        stats = {"mean": mean_angle, "thermal_3sigma": thermal_3sigma, "noise_3sigma": noise_3sigma}
        return stats, html

    except Exception as e:
        return {"error": str(e)}, f"<div class='error'>计算过程出错: {e}</div>"

def _analyze_system_faults_impl(satellite_name: str, start_str: str, end_str: str) -> Tuple[List[Dict], str]:
    """
    [系统级故障统计] 
    1. 安全模式: 统计 TMKP040 跳变为 5 的次数。
    2. 错误日志: 统计 TMKR012 的累计增量。
    """
    global _SAT_CONFIG_CACHE
    if _SAT_CONFIG_CACHE is None: _get_codes_impl(satellite_name, "任意")

    # 1. 定位卫星配置
    target_sat_config = None
    query = satellite_name.upper().strip()
    for sid, sdata in _SAT_CONFIG_CACHE.get('satellites', {}).items():
        if sid.upper() == query or sdata.get('name','').upper() == query or query in [a.upper() for a in sdata.get('aliases', [])]:
            target_sat_config = sdata
            break
    
    if not target_sat_config: return [], ""

    # 2. 定义要检查的项目
    # 格式: (配置Key, 显示名称, 逻辑类型)
    check_items = [
        ("control_mode", "控制模式监视", "safety_mode"),
        ("error_log_count", "错误日志计数", "counter_inc")
    ]
    
    # 3. 获取所有相关代号
    telemetry_map = target_sat_config.get('telemetry', {})
    codes_to_fetch = []
    item_configs = {} # 存储 key -> code 映射

    for key, label, logic in check_items:
        entry = telemetry_map.get(key)
        if entry:
            code = entry.get('code') if isinstance(entry, dict) else entry
            codes_to_fetch.append(code)
            item_configs[key] = {"code": code, "label": label, "logic": logic}

    if not codes_to_fetch:
        return [], "<div style='color:#ccc; text-align:center;'>未配置系统级故障遥测</div>"

    # 4. 拉取数据
    codes_str = ",".join(codes_to_fetch)
    db_table = target_sat_config.get('db_table')
    df = _get_data_impl(db_table, codes_str, start_str, end_str)

    # 5. 分析计算
    results = []
    table_rows = ""
    
    for key, label, logic in check_items:
        if key not in item_configs: continue
        
        cfg = item_configs[key]
        code = cfg["code"]
        val_res = 0
        is_abnormal = False
        summary = "正常"
        criterion_text = "-"

        if df.empty or code not in df.columns:
            val_text = "<span style='color:#ccc'>无数据</span>"
        else:
            series = pd.to_numeric(df[code], errors='coerce').dropna()
            
            if series.empty:
                val_text = "<span style='color:#ccc'>无有效值</span>"
            else:
                # --- 逻辑 A: 安全模式检测 ---
                if logic == "safety_mode":
                    criterion_text = "进入安全模式(5)"
                    # 检测上升沿: 当前=5 且 前一刻!=5
                    is_safe = (series == 5)
                    count = int((is_safe & (~is_safe.shift(1, fill_value=False))).sum())
                    val_res = count
                    if count > 0:
                        is_abnormal = True
                        summary = f"进入安全模式 {count} 次"
                        val_text = f"<span style='color:#dc3545; font-weight:bold;'>{count} 次</span>"
                    else:
                        val_text = f"<span style='color:#28a745;'>0 次</span>"

                # --- 逻辑 B: 计数器增量检测 ---
                elif logic == "counter_inc":
                    criterion_text = "计数增量"
                    # 计算累计增量 (处理重置情况)
                    diffs = series.diff().fillna(0)
                    # 只统计正向增长 (忽略因复位导致的负值，或者假设复位后从0开始计数)
                    # 如果需要处理循环计数(如uint16)，需要知道最大值。这里简化为统计所有正增量。
                    increases = diffs[diffs > 0]
                    total_inc = int(increases.sum())
                    
                    val_res = total_inc
                    if total_inc > 0:
                        is_abnormal = True
                        summary = f"错误日志新增 {total_inc} 条"
                        val_text = f"<span style='color:#dc3545; font-weight:bold;'>+{total_inc}</span>"
                    else:
                        val_text = f"<span style='color:#28a745;'>0</span>"

        # 存入结果
        if is_abnormal:
            results.append({"name": label, "is_abnormal": True, "summary": summary})
        
        # 表格行
        bg_style = "background:#fff5f5;" if is_abnormal else ""
        table_rows += f"""
        <tr style="{bg_style}">
            <td style="padding:10px; border:1px solid #eee;">{cfg['label']}</td>
            <td style="border:1px solid #eee;">{code}</td>
            <td style="border:1px solid #eee; color:#666; font-size:12px;">{criterion_text}</td>
            <td style="border:1px solid #eee;">{val_text}</td>
        </tr>
        """

    # 6. 生成 HTML
    html = f"""
    <table style="width:100%; border-collapse:collapse; text-align:center; font-size:13px; margin-bottom:15px;">
        <thead style="background:#f8f9fa;">
            <tr>
                <th style="padding:10px; border:1px solid #eee;">监测项目</th>
                <th style="border:1px solid #eee;">遥测代号</th>
                <th style="border:1px solid #eee;">统计判据</th>
                <th style="border:1px solid #eee;">统计结果</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
    """
    return results, html

def _analyze_fault_count_impl(sat_code: str, start_str: str, end_str: str) -> Tuple[Dict, str]:
    _, tm_code = _get_codes_impl(sat_code, "故障置出")
    if not tm_code: return {"error": "未配置"}, "<div class='error'>未配置代号</div>"
    
    df = _get_data_impl(sat_code, tm_code, start_str, end_str)
    if df.empty: return {"error": "无数据"}, "<div class='error'>无数据</div>"

    try:
        raw_col = df.iloc[:, 0]
        values = pd.to_numeric(raw_col, errors='coerce').dropna().values
        if len(values) == 0: return {"error": "无有效数据"}, "<div class='error'>无有效数据</div>"

        total_count = 0
        for i in range(1, len(values)):
            if values[i] < values[i-1]:
                total_count += values[i-1]
        total_count += values[-1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(range(len(values)), values, where='post')
        ax.set_title(f'GNSS 故障计数 (累计: {int(total_count)})')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        html = f"""
        <div class="section">
            <h2>GNSS 故障置出计数统计</h2>
            <div style="font-size:24px; font-weight:bold; color:green;">累计故障: {int(total_count)} 次</div>
            <div style="text-align:center;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%;"></div>
        </div>
        """
        return {"total": int(total_count)}, html
    except Exception as e:
        return {"error": str(e)}, f"<div class='error'>统计出错: {e}</div>"

def _wrap_html_report(body_content: str, title: str) -> str:
    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: 'PingFang SC', sans-serif; background:#f0f2f5; padding:40px 20px; }}
            .container {{ max-width:1100px; margin:0 auto; background:white; padding:50px; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,0.1); }}
            h1 {{ text-align:center; color:#1a202c; font-size:28px; }}
            table {{ width:100%; border-collapse:collapse; margin:15px 0; }}
            th, td {{ border:1px solid #edf2f7; padding:10px; text-align:center; }}
            th {{ background:#f7fafc; }}
            img {{ max-width:100%; height:auto; border:1px solid #eee; border-radius:4px; margin-top:10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <p style="text-align:center; color:#718096; font-size:12px;">内部资料 · 严密保存</p>
            <h1>{title}</h1>
            <p style="text-align:center; color:#a0aec0; font-size:13px; margin-bottom:40px;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {body_content}
        </div>
    </body>
    </html>"""
    
    filename = f"Satellite_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{os.path.abspath(filename)}')
        # --- 核心修复：100% 返回字符串 ---
        return f"✅ 报告已成功生成并打开: {filename}"
    except Exception as e:
        return f"❌ 报告保存失败: {str(e)}"

def _generate_satellite_health_viz(check_results: List[Dict]) -> str:
    """
    [实拍图版] 在 LZ04 真实卫星图像上叠加健康状态。
    """
    # 1. 统计各分系统异常
    stats = {"ADCS": 0, "EPS": 0, "OBDH": 0, "PAYLOAD": 0, "TTC": 0, "THERMAL": 0}
    for r in check_results:
        if r.get('is_abnormal'):
            name = r.get('name', '')
            if any(k in name for k in ['热', '变形']): stats["THERMAL"] += 1
            elif any(k in name for k in ['轨道', '电推', '姿态', '星敏', '陀螺', '飞轮', '故障', '通信']): stats["ADCS"] += 1
            elif '电' in name: stats["EPS"] += 1
            else: stats["OBDH"] += 1 # 默认归入星务/综电

    # 2. 读取图片并转 Base64
    img_base64 = ""
    # 路径：当前脚本所在目录/doc/lz04_real.png
    img_path = os.path.join(os.path.dirname(__file__), "doc", "lz04_real.png") 
    
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
    else:
        # 兜底显示
        return f"<div style='color:red; padding:20px; border:1px dashed red;'>❌ 图片未找到: {img_path}</div>"

    # 3. 定义打点坐标 (基于你提供的图片透视关系)
    # label_pos: 控制文字是在点的左边还是右边显示
    spots = {
        "PAYLOAD": {
            "top": "22%", "left": "58%", 
            "label": "有效载荷", "label_pos": "left: 140%;" 
        },
        "ADCS": {
            "top": "40%", "left": "48%", 
            "label": "姿轨控分系统", "label_pos": "right: 140%; text-align:right;" 
        },
        "EPS": {
            "top": "58%", "left": "32%", 
            "label": "能源分系统", "label_pos": "right: 140%; text-align:right;" 
        },
        "THERMAL": {
            "top": "65%", "left": "58%", 
            "label": "热控分系统", "label_pos": "left: 140%;" 
        },
        "OBDH": {
            "top": "50%", "left": "65%", 
            "label": "星务/综电", "label_pos": "left: 140%;" 
        }
    }

    # 4. 生成打点 HTML
    spots_html = ""
    for key, pos in spots.items():
        count = stats.get(key, 0)
        
        # 颜色逻辑：正常=绿色，异常=橙红+闪烁
        dot_color = "#f6ad55" if count > 0 else "#48bb78" # 橙/绿
        ring_color = "rgba(246, 173, 85, 0.4)" if count > 0 else "rgba(72, 187, 120, 0.4)"
        text_color = "#c05621" if count > 0 else "#2f855a"
        
        # 异常时的脉动动画
        animation = """
            <div class="pulse-ring" style="border-color: {c};"></div>
            <div class="pulse-ring delay" style="border-color: {c};"></div>
        """.format(c=dot_color) if count > 0 else ""

        spots_html += f"""
        <div class="spot-wrapper" style="top: {pos['top']}; left: {pos['left']};">
            <!-- 状态点 -->
            <div class="spot-dot" style="background: {dot_color};"></div>
            {animation}
            
            <!-- 文字标签 + 连线 -->
            <div class="spot-label" style="{pos['label_pos']} color: {text_color}; border-bottom: 2px solid {dot_color};">
                <div style="font-weight: bold; white-space: nowrap;">{pos['label']}</div>
                <div style="font-size: 11px; opacity: 0.9;">异常项: {count}</div>
            </div>
            
            <!-- 简单的连接线 -->
            <div class="connector-line" style="{ 'left: 10px; width: 30px;' if 'left' in pos['label_pos'] else 'right: 10px; width: 30px;' } background: {dot_color};"></div>
        </div>
        """

    # 5. 组装完整组件
    viz_html = f"""
    <div style="background: #fff; padding: 10px; border-radius: 15px; margin-bottom: 30px; border: 1px solid #eee;">
        <h3 style="margin: 0 0 20px 20px; color: #2d3748; font-size: 16px;">🛰️ 卫星健康状态可视化</h3>
        
        <div style="position: relative; max-width: 800px; margin: 0 auto; overflow: hidden;">
            <style>
                .spot-wrapper {{ position: absolute; width: 0; height: 0; }}
                
                /* 中心实心点 */
                .spot-dot {{ 
                    position: absolute; top: -6px; left: -6px; width: 12px; height: 12px; 
                    border-radius: 50%; border: 2px solid #fff; z-index: 10; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }}
                
                /* 脉动光环动画 */
                .pulse-ring {{
                    position: absolute; top: -15px; left: -15px; width: 30px; height: 30px;
                    border-radius: 50%; border: 2px solid; opacity: 0;
                    animation: pulse-animation 2s infinite; pointer-events: none;
                }}
                .pulse-ring.delay {{ animation-delay: 1s; }}
                @keyframes pulse-animation {{ 
                    0% {{ transform: scale(0.5); opacity: 1; }} 
                    100% {{ transform: scale(2); opacity: 0; }} 
                }}

                /* 标签文字 */
                .spot-label {{ 
                    position: absolute; top: -18px; padding: 2px 5px; 
                    background: rgba(255,255,255,0.85); border-radius: 4px; z-index: 5;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                
                /* 连接线 */
                .connector-line {{
                    position: absolute; top: 0; height: 1px; z-index: 4;
                }}
            </style>
            
            <!-- 卫星底图 -->
            <img src="data:image/png;base64,{img_base64}" style="width: 100%; display: block; object-fit: contain;">
            
            <!-- 叠加层 -->
            {spots_html}
        </div>
        
        <div style="text-align:center; margin-top:15px; font-size:12px; color:#718096;">
            <span style="margin-right:20px"><span style="color:#48bb78; font-size:14px;">●</span> 状态正常</span>
            <span><span style="color:#f6ad55; font-size:14px;">●</span> 存在异常/关注项</span>
        </div>
    </div>
    """
    return viz_html

def _generate_final_report_content(check_results: List[Dict], adcs_subsections: Dict, thermal_html: str) -> str:
    """
    [整星体检完整版] 生成标准化月度体检报告内容。
    集成：健康可视化图 + 异常看板 + 分系统详情 + 预测模块。
    """
    # --- 1. 数据安全校验 ---
    # 确保输入数据非空，防止拼接时报错
    thermal_html = thermal_html or ""
    for k in adcs_subsections:
        if adcs_subsections[k] is None: adcs_subsections[k] = ""
    
    # 过滤无效结果
    safe_results = [r for r in check_results if isinstance(r, dict) and 'name' in r]
    
    # 统计数据
    total_checks = len(safe_results)
    anomalies = [r for r in safe_results if r.get('is_abnormal')]
    count_abnormal = len(anomalies)

    # --- 2. 生成第一部分：重要异常展示 (Dashboard) ---
    
    # 2.1 调用卫星可视化生成函数 (需确保 _generate_satellite_health_viz 已定义)
    satellite_viz_html = _generate_satellite_health_viz(safe_results)

    # 2.2 生成异常文字列表
    status_color = "#e53e3e" if count_abnormal > 0 else "#2f855a"
    if count_abnormal > 0:
        items = "".join([f"<li style='margin-bottom:6px;'><b>[{r.get('name','未知')}]</b> {r.get('summary','异常')}</li>" for r in anomalies])
        anomaly_text_html = f"""
        <div style="background:#fff5f5; padding:15px; border-radius:8px; border:1px solid #fed7d7; color:#c53030;">
            <ul style="margin:0; padding-left:20px;">{items}</ul>
        </div>
        """
    else:
        anomaly_text_html = """
        <div style="background:#f0fff4; padding:15px; border-radius:8px; border:1px solid #c6f6d5; color:#2f855a; text-align:center;">
            ✅ 本月全星关键指标均在门限范围内，未发现重大异常。
        </div>
        """

    # 2.3 组装 Part 1
    part1_dashboard = f"""
    <div style="background:white; padding:30px; border-radius:15px; box-shadow:0 4px 20px rgba(0,0,0,0.08); margin-bottom:50px;">
        <h2 style="margin-top:0; color:#1a202c; border-bottom:2px solid #edf2f7; padding-bottom:15px;">一、 重要异常展示</h2>
        
        <!-- 顶部数字统计 -->
        <div style="display: flex; gap: 20px; margin-bottom: 30px;">
            <div style="flex: 1; text-align: center; border-right: 1px solid #eee;">
                <div style="font-size: 32px; font-weight: bold; color: #4a5568;">{total_checks}</div>
                <div style="font-size: 12px; color: #a0aec0; text-transform: uppercase;">总监测指标</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 32px; font-weight: bold; color: {status_color};">{count_abnormal}</div>
                <div style="font-size: 12px; color: #a0aec0; text-transform: uppercase;">异常/预警项</div>
            </div>
        </div>

        <!-- 核心：卫星健康可视化图 -->
        {satellite_viz_html}

        <!-- 底部：异常文字详情 -->
        <div style="margin-top:25px;">
            <h4 style="margin-bottom:10px; color:#4a5568;">异常/预警项列表：</h4>
            {anomaly_text_html}
        </div>
    </div>
    """

    # --- 3. 生成第二部分：所有指标评估结果 (Detailed Results) ---
    
    # 辅助函数：生成标准化的分系统容器
    def make_subsystem_box(title, content, is_empty=False):
        tag = " <small style='color:#999; font-weight:normal;'>(本月未评估)</small>" if is_empty else ""
        inner_content = content if content and content.strip() else '<div style="text-align:center; color:#ccc; padding:20px;">暂无相关数据</div>'
        
        return f"""
        <div style="margin-bottom:30px; border:1px solid #edf2f7; border-radius:8px; overflow:hidden;">
            <div style="background:#f8f9fa; padding:12px 20px; font-weight:bold; border-bottom:1px solid #edf2f7; color:#2d3748; font-size:16px;">
                ■ {title}{tag}
            </div>
            <div style="padding:20px;">{inner_content}</div>
        </div>
        """

    # 构造姿轨控内部的细分目录
    adcs_body = f"""
        <div style="margin-left:10px;">
            <h3 style="font-size:14px; color:#4a5568; border-bottom:1px dashed #eee; padding-bottom:5px;">1. 单机故障统计 (通信/故障置出)</h3>
            {adcs_subsections.get('fault_stats','')}
            
            <h3 style="font-size:14px; color:#4a5568; border-bottom:1px dashed #eee; padding-bottom:5px; margin-top:25px;">2. 单机性能评估</h3>
            {adcs_subsections.get('unit_perf','')}
            
            <!-- 【更新】系统故障统计 -->
            <h3 style="font-size:14px; color:#4a5568; border-bottom:1px dashed #eee; padding-bottom:5px; margin-top:25px;">3. 系统故障统计</h3>
            {adcs_subsections.get('sys_faults', '<p style="color:#bbb; font-style:italic;">(无相关数据)</p>')}
            
            <h3 style="font-size:14px; color:#4a5568; border-bottom:1px dashed #eee; padding-bottom:5px; margin-top:25px;">4. 系统性能评估 (姿态/轨道/电推)</h3>
            {adcs_subsections.get('sys_perf','')}
        </div>
    """

    part2_details = f"""
    <div style="margin-top: 50px;">
        <h2 style="color:#2d3748; border-bottom: 2px solid #eee; padding-bottom: 10px;">二、 所有指标评估结果</h2>
        {make_subsystem_box("姿轨控分系统", adcs_body)}
        {make_subsystem_box("星务分系统", "", True)}
        {make_subsystem_box("综电分系统", "", True)}
        {make_subsystem_box("能源分系统", "", True)}
        {make_subsystem_box("载荷分系统", "", True)}
        {make_subsystem_box("数传分系统", "", True)}
        {make_subsystem_box("热控分系统 (热变形)", thermal_html)}
    </div>
    """

    # --- 4. 生成第三部分：指标对比和健康预测 (Predictions) ---
    part3_predictions = f"""
    <div style="margin-top: 50px; background: #fdfaf5; border: 1px solid #faead1; padding: 25px; border-radius: 12px;">
        <h2 style="margin-top:0; color:#856404;">三、 指标对比和健康预测</h2>
        <div style="text-align:center; padding: 30px 0;">
            <div style="font-size:40px; margin-bottom:10px;">🛠️</div>
            <p style="color:#856404; font-size:14px;">
                关键指标历史趋势对比及剩余寿命预测模型模块正在开发中。<br>
                预计下个版本上线。
            </p>
        </div>
    </div>
    """

    # --- 5. 拼接返回 ---
    return part1_dashboard + part2_details + part3_predictions


# ==============================================================================
# 第二层：原子工具 (Atomic Tools)
# ==============================================================================

@mcp.tool(description="查找卫星和遥测代号。")
def get_satellite_codes(satellite_name: str, query: str) -> Any:
    sat, tm = _get_codes_impl(satellite_name, query)
    if sat and tm:
        return pd.DataFrame([{"satellite_code": sat, "telemetry_code": tm}])
    return pd.DataFrame(columns=["satellite_code", "telemetry_code"])

@mcp.tool(description="获取卫星遥测数据。")
def get_satellite_data(satellite_code: str, telemetry_code: str, start_time_str: str = None, end_time_str: str = None) -> str:
    df = _get_data_impl(satellite_code, telemetry_code, start_time_str, end_time_str)
    return df.to_json(orient='split', date_format='iso')

@mcp.tool(description="[单项] 星敏噪声分析。")
def calculate_star_sensor_noise(satellite_name: str = None, data_json: str = None, start_time_str: str = None, end_time_str: str = None) -> str:
    try:
        df = pd.DataFrame()
        if data_json and len(data_json) > 10:
            try: df = pd.read_json(io.StringIO(data_json), orient='split')
            except: pass
        elif satellite_name:
            sat_code, tm_code = _get_codes_impl(satellite_name, "星敏")
            if sat_code: df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        
        if df.empty: return "错误: 无数据。"
        result_dict = _analyze_star_sensor_impl(df)
        return _wrap_html_report(result_dict['html'], "星敏噪声分析报告")
    except Exception as e:
        return f"运行错误: {e}"

@mcp.tool(description="[单项] 星敏热变形分析工具。")
def analyze_thermal_deformation(satellite_name: str, start_time_str: str = None, end_time_str: str = None) -> str:
    import json
    sat_code, _ = _get_codes_impl(satellite_name, "热变形")
    if not sat_code: return json.dumps({"error": f"未找到卫星 {satellite_name}"})
    summary_dict, html_fragment = _analyze_thermal_impl(sat_code, start_time_str, end_time_str)
    if "error" in summary_dict: return json.dumps(summary_dict)
    _wrap_html_report(html_fragment, f"{satellite_name} 热变形分析报告")
    return json.dumps(summary_dict, ensure_ascii=False)

@mcp.tool(description="[单项] GNSS故障置出计数统计。")
def calculate_gnss_fault_count(satellite_name: str, start_time_str: str = None, end_time_str: str = None) -> str:
    import json
    sat_code, _ = _get_codes_impl(satellite_name, "故障置出")
    if not sat_code: return json.dumps({"error": f"未找到卫星 {satellite_name}"})
    summary_dict, html_fragment = _analyze_fault_count_impl(sat_code, start_time_str, end_time_str)
    if "error" in summary_dict: return json.dumps(summary_dict)
    _wrap_html_report(html_fragment, f"{satellite_name} 故障置出统计")
    return json.dumps(summary_dict, ensure_ascii=False)

@mcp.tool(description="[侦察] 检测 GNSS 通信故障时间段。")
def detect_gnss_fault_segments(satellite_name: str, start_time_str: str = None, end_time_str: str = None) -> str:
    import json
    sat_code, err_tm_code = _get_codes_impl(satellite_name, "GNSS错误")
    if not sat_code: return json.dumps({"error": "未找到卫星"})
    
    df = _get_data_impl(sat_code, err_tm_code, start_time_str, end_time_str)
    if df.empty: return json.dumps({"status": "normal", "message": "无数据", "segments": []})
    
    try:
        raw_diff = pd.to_numeric(df.iloc[:, 1], errors='coerce').diff().fillna(0)
        real_diff = np.where(raw_diff < 0, raw_diff + 256, raw_diff)
        error_indices = df[real_diff > 0].index.tolist()
        
        valid_segments = []
        if error_indices:
            timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce')
            curr_start = error_indices[0]
            curr_last = error_indices[0]
            
            def save_segment(start, end):
                segment_mask = df.index.isin(range(start, end+1))
                segment_err = int(np.sum(real_diff[segment_mask]))
                if segment_err >= 5:
                    t_start = timestamps.iloc[start]
                    t_end = timestamps.iloc[end]
                    valid_segments.append({
                        "start_time": datetime.fromtimestamp(t_start).strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": datetime.fromtimestamp(t_end).strftime('%Y-%m-%d %H:%M:%S'),
                        "error_count": segment_err
                    })

            for idx in error_indices[1:]:
                if (timestamps.iloc[idx] - timestamps.iloc[curr_last]) > 60:
                    save_segment(curr_start, curr_last)
                    curr_start = idx
                curr_last = idx
            save_segment(curr_start, curr_last)
            
        return json.dumps({"status": "fault_found" if valid_segments else "normal", "segments": valid_segments}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

# 增加一个全局缓存用于知识库
_KNOWLEDGE_CACHE = None

@mcp.tool(description="[知识库] 查询卫星故障处置预案和专家经验。输入关键词（如'星敏噪声'、'安全模式'）。")
def query_knowledge_base(query: str) -> str:
    """
    检索故障处置知识库。
    """
    global _KNOWLEDGE_CACHE
    import yaml
    
    # 1. 加载知识库
    if _KNOWLEDGE_CACHE is None:
        kb_path = os.path.join(os.path.dirname(__file__), "doc", "knowledge.yaml")
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                _KNOWLEDGE_CACHE = yaml.safe_load(f)
        else:
            return "错误：未找到知识库文件 doc/knowledge.yaml"

    kb = _KNOWLEDGE_CACHE.get('knowledge_base', {})
    
    # 2. 模糊搜索
    results = []
    query = query.strip()
    
    for key, info in kb.items():
        # 如果 key 包含 query 或者 query 包含 key
        if query in key or key in query:
            results.append(f"【{key}】\n现象: {info['symptom']}\n原因: {info['cause']}\n处置建议: {info['action']}\n")
    
    if not results:
        return f"知识库中未找到关于'{query}'的相关条目。建议人工查阅详细手册。"
    
    return "\n".join(results)

@mcp.tool(description="""[侦察] 检测 GNSS 通信故障时间段。
**用途**：当用户询问 GNSS 状态时首先调用此工具。
**返回**：JSON 格式的故障时间段列表 (segments)。
大模型收到结果后，应结合知识库判断需要进一步查询哪些关联数据（如纬度、星数、姿态等）。
""")
def detect_gnss_fault_segments(satellite_name: str, start_time_str: str = None, end_time_str: str = None) -> str:
    """
    检测 GNSS 错误计数，返回故障时间段。
    """
    import json
    
    # 1. 获取错误计数代号
    sat_code, err_tm_code = _get_codes_impl(satellite_name, "GNSS错误")
    if not sat_code or not err_tm_code:
        return json.dumps({"error": f"未找到卫星 {satellite_name} 的GNSS错误计数代号"})
    
    # 2. 获取数据
    df = _get_data_impl(sat_code, err_tm_code, start_time_str, end_time_str)
    if df.empty:
        return json.dumps({"status": "normal", "message": "无遥测数据", "segments": []})
        
    # 3. 核心算法 (Diff -> Uint8处理 -> 聚类)
    # 假设数据列: [Time, ErrCount]
    time_col = df.iloc[:, 0]
    err_col = df.iloc[:, 1]
    
    try:
        # 处理 uint8 翻转 (0-255)
        raw_diff = pd.to_numeric(err_col, errors='coerce').diff().fillna(0)
        real_diff = np.where(raw_diff < 0, raw_diff + 256, raw_diff)
        
        # 找出错误点
        error_indices = df[real_diff > 0].index.tolist()
        
        if not error_indices:
            return json.dumps({"status": "normal", "message": "计数器未增加", "segments": []})
            
        timestamps = pd.to_numeric(time_col, errors='coerce')
        TIME_GAP_THRESHOLD = 60 # 60秒内的错误视为同一段
        ERROR_THRESHOLD = 5     # 忽略小于5次的微小波动
        
        valid_segments = []
        
        if error_indices:
            curr_start, curr_last = error_indices[0], error_indices[0]
            
            def save_segment(start, end):
                # 计算该段内的总错误增量
                segment_mask = df.index.isin(range(start, end+1))
                segment_err = int(np.sum(real_diff[segment_mask]))
                
                if segment_err >= ERROR_THRESHOLD:
                    t_start = timestamps.iloc[start]
                    t_end = timestamps.iloc[end]
                    valid_segments.append({
                        "start_time": datetime.fromtimestamp(t_start).strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": datetime.fromtimestamp(t_end).strftime('%Y-%m-%d %H:%M:%S'),
                        "duration_sec": int(t_end - t_start),
                        "total_error_increase": segment_err
                    })

            for idx in error_indices[1:]:
                if (timestamps.iloc[idx] - timestamps.iloc[curr_last]) > TIME_GAP_THRESHOLD:
                    save_segment(curr_start, curr_last)
                    curr_start = idx
                curr_last = idx
            save_segment(curr_start, curr_last) # 保存最后一段
                
        return json.dumps({
            "status": "fault_found" if valid_segments else "normal",
            "satellite_name": satellite_name,
            "fault_segments": valid_segments
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool(description="""[探针] 关联趋势分析工具。
**用途**：用于验证故障原因。查询指定参数的统计数据并绘图。
**输入**：请使用 **精确的故障时间段** (fault_start / fault_end) 调用此工具。
**输出**：
1. JSON: 包含 mean (平均), max_abs (最大绝对值), start_val (起始值) 等统计指标。
2. HTML: 包含趋势图 (姿态精度和稳定度分栏绘制，X轴为采样点)。
""")
def investigate_telemetry_trends(satellite_name: str, start_time_str: str, end_time_str: str, queries: str) -> str:
    import json
    
    # 字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial'] 
    plt.rcParams['axes.unicode_minus'] = False

    query_list = [q.strip() for q in queries.split(',') if q.strip()]
    sat_code, _ = _get_codes_impl(satellite_name, "任意")
    if not sat_code: return json.dumps({"error": f"未找到卫星 {satellite_name}"})

    # 1. 准备时间窗口
    try:
        dt_start = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        dt_end = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
        ctx_start_str = (dt_start - timedelta(seconds=120)).strftime('%Y-%m-%d %H:%M:%S')
        ctx_end_str = (dt_end + timedelta(seconds=120)).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return json.dumps({"error": "时间格式解析失败"})

    ai_stats_summary = {}
    plot_queue = []

    for query_item in query_list:
        _, tm_code = _get_codes_impl(satellite_name, query_item)
        if not tm_code: continue
            
        # A. 获取数据
        df_plot = _get_data_impl(sat_code, tm_code, ctx_start_str, ctx_end_str)
        if df_plot.empty: 
            ai_stats_summary[query_item] = "no_data_fetched"
            continue

        # B. 时间列处理 & 数据列识别
        t_col_name = None
        # 尝试寻找显式的时间列名
        for col in ["TMKP808", "Time", "time"]:
            if col in df_plot.columns: 
                t_col_name = col
                break
        
        # --- 【核心修复 1】 智能识别数据列 ---
        if t_col_name:
            # 如果找到了时间列，数据列就是除了它之外的所有列
            all_data_cols = [c for c in df_plot.columns if c != t_col_name]
            # 解析时间用于切片
            t_vals = pd.to_numeric(df_plot[t_col_name], errors='coerce')
        else:
            # 如果没找到时间列 (说明返回的纯数据)，那么所有列都是数据列
            all_data_cols = df_plot.columns.tolist()
            # 这种情况下，我们暂时用索引作为假时间，或者尝试用第0列强行解析(风险较大，这里选择放弃时间切片，用全量)
            # 但为了保持逻辑一致，还是尝试用第0列作为时间参考，如果它看起来像时间戳的话
            t_vals = pd.to_numeric(df_plot.iloc[:, 0], errors='coerce')
            # 检查一下第0列是不是时间戳 (比如 > 1980年)
            is_timestamp = False
            if not t_vals.dropna().empty:
                check_val = t_vals.dropna().iloc[0]
                if check_val > 1e9: # 粗略判断
                    is_timestamp = True
            
            if is_timestamp:
                # 如果第0列看起来像时间，那它就是时间，不作为数据
                t_col_name = df_plot.columns[0]
                all_data_cols = df_plot.columns[1:].tolist()
            else:
                # 否则，第0列也是数据，不要丢弃！
                t_vals = pd.Series(df_plot.index) # 用索引代替时间

        # --- 时间对齐逻辑 ---
        try:
            t_series = pd.to_datetime(t_vals, unit='ms', errors='coerce')
            if not t_series.dropna().empty and t_series.dropna().iloc[0].year < 1980:
                 t_series = pd.to_datetime(t_vals, unit='s', errors='coerce')
            
            if t_series.dt.tz is not None: t_series = t_series.dt.tz_localize(None)
            
            if not t_series.dropna().empty and t_col_name: # 只有真的是时间列才做时区修正
                diff_hours = (dt_start - t_series.dropna().iloc[0]).total_seconds() / 3600
                if 7 < diff_hours < 9: t_series = t_series + timedelta(hours=8)
        except:
            t_series = pd.Series(df_plot.index)

        # 制作精确切片
        mask_exact = (t_series >= dt_start) & (t_series <= dt_end)
        df_stats = df_plot[mask_exact]
        if df_stats.empty: df_stats = df_plot

        # C. 核心处理：统计与绘图
        
        # --- 情况 1: 姿态控制 (聚合统计) ---
        if "姿态" in query_item or "Attitude" in query_item:
            # 假设: 如果有6列，前3角度后3角速度；如果有3列，全是角度
            angle_cols = []
            omega_cols = []
            
            if len(all_data_cols) >= 6:
                angle_cols = all_data_cols[:3]
                omega_cols = all_data_cols[3:6]
            else:
                angle_cols = all_data_cols # 默认全是角度
            
            stats_obj = {}
            
            # 1.1 角度 (最大绝对值)
            if angle_cols:
                angle_data_stats = df_stats[angle_cols].apply(pd.to_numeric, errors='coerce').dropna()
                if not angle_data_stats.empty:
                    max_err = float(angle_data_stats.abs().max().max())
                    stats_obj["max_abs_error"] = round(max_err, 5)
                else:
                    stats_obj["max_abs_error"] = "no_data"

                # 绘图
                angle_series_list = []
                for col in angle_cols:
                    s_plot = pd.to_numeric(df_plot[col], errors='coerce').dropna()
                    if not s_plot.empty:
                        angle_series_list.append({'label': col, 'values': s_plot.values})
                if angle_series_list:
                    plot_queue.append({'title': '姿态控制精度 (角度)', 'series': angle_series_list})

            # 1.2 角速度 (最大3σ)
            if omega_cols:
                omega_data_stats = df_stats[omega_cols].apply(pd.to_numeric, errors='coerce').dropna()
                if not omega_data_stats.empty:
                    max_stab = float((omega_data_stats.std() * 3).max())
                    stats_obj["max_stability_3sigma"] = round(max_stab, 6)
                else:
                    stats_obj["max_stability_3sigma"] = "no_data"

                # 绘图
                omega_series_list = []
                for col in omega_cols:
                    s_plot = pd.to_numeric(df_plot[col], errors='coerce').dropna()
                    if not s_plot.empty:
                        omega_series_list.append({'label': col, 'values': s_plot.values})
                if omega_series_list:
                    plot_queue.append({'title': '姿态稳定度 (角速度)', 'series': omega_series_list})
            
            ai_stats_summary[query_item] = stats_obj

        # --- 情况 2: 普通遥测 ---
        else:
            stats_obj = {}
            common_series_list = []
            
            for col in all_data_cols:
                # 统计
                s_stat = pd.to_numeric(df_stats[col], errors='coerce').dropna()
                if not s_stat.empty:
                    val_mean = float(s_stat.mean())
                    val_start = float(s_stat.iloc[0])
                    
                    if "纬度" in query_item or "Lat" in col:
                        stats_obj[col] = {"start_val": round(val_start, 4)}
                    elif "星数" in query_item:
                        stats_obj[col] = {"mean_val": round(val_mean, 2)}
                    elif "错误" in query_item:
                         stats_obj[col] = {"increase": int(s_stat.iloc[-1] - s_stat.iloc[0])}
                    else:
                        stats_obj[col] = {"mean": round(val_mean, 4)}
                
                # 绘图
                s_plot = pd.to_numeric(df_plot[col], errors='coerce').dropna()
                if not s_plot.empty:
                    common_series_list.append({'label': col, 'values': s_plot.values})
            
            if common_series_list:
                plot_queue.append({'title': query_item, 'series': common_series_list})
                
            ai_stats_summary[query_item] = stats_obj

    # 3. 绘图渲染
    img_base64 = ""
    if plot_queue:
        try:
            num_plots = len(plot_queue)
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3.5 * num_plots), sharex=False)
            if num_plots == 1: axes = [axes]
            
            for i, pdata in enumerate(plot_queue):
                ax = axes[i]
                ax.set_title(pdata['title'], fontsize=11, pad=10)
                ax.grid(True, alpha=0.3)
                
                for series in pdata['series']:
                    y_data = series['values']
                    x_data = range(len(y_data)) 
                    ax.plot(x_data, y_data, label=series['label'], linewidth=1)
                
                ax.legend(loc='upper right', fontsize=9)
                if i == num_plots - 1:
                    ax.set_xlabel("Sampling Points")
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            print(f"绘图失败: {e}")

    # 4. 生成报告
    html_report = f"""
    <div class="section">
        <h2>关联趋势分析</h2>
        <p><strong>分析窗口:</strong> {start_time_str} ~ {end_time_str}</p>
        <p><strong>统计结果 (JSON):</strong></p>
        <pre style="background:#f4f4f4; padding:10px; font-size:12px;">{json.dumps(ai_stats_summary, indent=2, ensure_ascii=False)}</pre>
        <div style="text-align:center; margin-top:20px;">
            {'<img src="data:image/png;base64,' + img_base64 + '" style="max-width:100%; border:1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">' if img_base64 else '<p>无绘图数据</p>'}
        </div>
    </div>
    """
    
    _wrap_html_report(html_report, f"{satellite_name} 关联趋势详情")
    return json.dumps(ai_stats_summary, ensure_ascii=False)
# ==============================================================================
# 第三层：聚合工具 (Composite Tool)
# ==============================================================================

@mcp.tool(description="""[整星月度体检] 生成卫星全系统月度运行评估报告。
包含：1.重要异常展示；2.分系统评估结果（姿轨控、热控等）；3.趋势预测（占位）。
分析尺度：星敏(3min)、单机性能/姿态/热控(1day调试模式)、全月统计项(1month)。
""")
def assess_monthly_performance(satellite_name: str, year_month: str = None) -> str:
    logger.info(f"🚀 [整星月报] 任务启动: {satellite_name}")
    try:
        # 1. 时间窗口计算
        if year_month: target_dt = datetime.strptime(year_month, '%Y-%m')
        else: target_dt = (datetime.now().replace(day=1) - timedelta(days=1)).replace(day=1)
        
        m_start = target_dt.strftime('%Y-%m-01 00:00:00')
        if target_dt.month == 12: next_m = target_dt.replace(year=target_dt.year + 1, month=1)
        else: next_m = target_dt.replace(month=target_dt.month + 1)
        m_end = (next_m - timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')

        d_start_dt = target_dt.replace(day=15)
        d_start, d_end = d_start_dt.strftime('%Y-%m-%d 00:00:00'), d_start_dt.strftime('%Y-%m-%d 23:59:59')
        s3_start, s3_end = d_start, (d_start_dt + timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S')

        # 2. 卫星配置定位
        base_sat_code, _ = _get_codes_impl(satellite_name, "任意")
        if not base_sat_code: return f"❌ 未找到卫星 {satellite_name} 配置"

        check_results, adcs_subs = [], {"fault_stats": "", "unit_perf": "","sys_faults": "", "sys_perf": ""}
        
        # --- 3. 姿轨控分析 ---
        # 3.1 故障统计 (1month)
        logger.info("📡 [ADCS] 分析故障计数与通信...")
        c_res, c_html = _analyze_device_errors_impl(base_sat_code, m_start, m_end)
        f_res, f_html = _analyze_all_unit_faults_impl(satellite_name, m_start, m_end)
        check_results.extend(c_res + f_res)
        adcs_subs["fault_stats"] = (c_html or "") + (f_html or "")

        # 3.2 单机性能 (3min/1day)
        logger.info("📡 [ADCS] 分析单机噪声与零偏...")
        for label in ["星敏A", "星敏B"]:
            _, tm = _get_codes_impl(satellite_name, label)
            if tm:
                df = _get_data_impl(base_sat_code, tm, s3_start, s3_end)
                res = _analyze_star_sensor_impl(df, label)
                check_results.append({"name": label, **res}); adcs_subs["unit_perf"] += res['html']

        for g_cfg in [{"key": "gyro_a_bias", "name": "陀螺A"}, {"key": "gyro_b_bias", "name": "陀螺B"}]:
            _, tm_b = _get_codes_impl(satellite_name, g_cfg["key"])
            if tm_b:
                df_b = _get_data_impl(base_sat_code, tm_b, d_start, d_end)
                res_b = _analyze_gyro_bias_impl(df_b, g_cfg["name"])
                adcs_subs["unit_perf"] += res_b["html"]

        # 3.4 系统性能 (1day姿态, 1month轨道)
        logger.info("🛰️ [ADCS] 分析系统精度与轨道维持...")
        _, tm_att = _get_codes_impl(satellite_name, "姿态")
        if tm_att:
            df_att = _get_data_impl(base_sat_code, tm_att, d_start, d_end) # 调试模式1day
            res_att = _analyze_attitude_monthly_impl(df_att)
            check_results.append({"name": "姿态控制", **res_att}); adcs_subs["sys_perf"] += res_att['html']

        for item in ["平根半长轴", "降交点", "电推"]:
            _, tm_item = _get_codes_impl(satellite_name, item)
            if tm_item:
                df_item = _get_data_impl(base_sat_code, tm_item, m_start, m_end)
                if "半长轴" in item: res_s = _analyze_orbit_impl(df_item)
                elif "降交点" in item: res_s = _analyze_ltdn_impl(df_item)
                else: res_s = _analyze_propulsion_impl(df_item)
                check_results.append({"name": item, **res_s}); adcs_subs["sys_perf"] += res_s['html']

        # --- [插入位置] 3.5 系统故障统计 (1month) ---
        logger.info("📡 [ADCS] 分析系统级故障(安全模式)...")
        sys_res, sys_html = _analyze_system_faults_impl(satellite_name, m_start, m_end)
        check_results.extend(sys_res)
        adcs_subs["sys_faults"] = sys_html

        # --- 4. 热控分析 (1day) ---
        logger.info("🌡️ [Thermal] 分析热变形...")
        _, thermal_html = _analyze_thermal_impl(base_sat_code, d_start, d_end)

        # 5. 汇总生成
        logger.info("📝 渲染最终整星月度报告...")
        full_body = _generate_final_report_content(check_results, adcs_subs, thermal_html)
        title = f"{satellite_name} 卫星月度体检报告"
        final_msg = _wrap_html_report(full_body, title)
        # ================= 核心修改点 =================
        # 提取异常项，构造一段给 AI 看的文本摘要
        anomalies = [r for r in check_results if isinstance(r, dict) and r.get('is_abnormal')]
        
        if anomalies:
            summary_text = f"报告已生成。监测到 {len(anomalies)} 项异常，请立即分析：\n"
            for i, r in enumerate(anomalies, 1):
                summary_text += f"{i}. [{r.get('name')}] : {r.get('summary')}\n"
            
            summary_text += "\n请根据上述异常，调用 knowledge_base 工具获取处置建议，并给出分析结论。"
            return str(final_msg) + "\n\n" + summary_text
        else:
            return str(final_msg) + "\n\n报告已生成。本月全星状态良好，无异常项，无需额外处置。"

    except Exception as e:
        logger.error(f"严重错误: {e}", exc_info=True)
        return f"运行评估时发生错误: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="sse")