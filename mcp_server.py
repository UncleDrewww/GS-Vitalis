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

def _analyze_attitude_impl(data: pd.DataFrame) -> Dict:
    if data.empty or data.shape[1] < 7:
        return {"is_abnormal": True, "summary": "姿态数据不足", "html": "<div class='error'>[姿态] 数据列数不足。</div>"}
    try:
        origin_data = data.values
        LIMIT_AGL = 0.02
        LIMIT_W = 0.003
        
        valid_agl = origin_data[:, [0, 1, 2, 3]]
        valid_w = origin_data[:, [0, 4, 5, 6]]
        agl_std = np.std(valid_agl[:, 1:4].astype(float), axis=0, ddof=1)
        w_std = np.std(valid_w[:, 1:4].astype(float), axis=0, ddof=1)
        agl_3sigma = 3 * agl_std
        w_3sigma = 3 * w_std
        
        axes_name = ['Roll', 'Pitch', 'Yaw']
        table_rows = ""
        has_anomaly = False
        details = []

        for i in range(3):
            val = agl_3sigma[i]
            if val > LIMIT_AGL:
                res_html = "<span style='color:#dc3545;'>超标</span>"
                has_anomaly = True
                details.append(f"{axes_name[i]}角")
            else:
                res_html = "<span style='color:#28a745;'>合格</span>"
            table_rows += f"<tr><td>姿态精度</td><td>{axes_name[i]}</td><td>{val:.4f}</td><td>&le; {LIMIT_AGL}</td><td>{res_html}</td></tr>"

        table_rows += "<tr style='background:#eee;'><td colspan='5' style='height:1px;'></td></tr>"

        for i in range(3):
            val = w_3sigma[i]
            if val > LIMIT_W:
                res_html = "<span style='color:#dc3545;'>超标</span>"
                has_anomaly = True
                details.append(f"{axes_name[i]}速")
            else:
                res_html = "<span style='color:#28a745;'>合格</span>"
            table_rows += f"<tr><td>姿态稳定度</td><td>{axes_name[i]}</td><td>{val:.5f}</td><td>&le; {LIMIT_W}</td><td>{res_html}</td></tr>"

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        x_axis = range(len(valid_agl))
        ax1.plot(x_axis, valid_agl[:, 1:4])
        ax1.set_title('姿态角')
        ax2.plot(x_axis, valid_w[:, 1:4])
        ax2.set_title('角速度')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        summary_color = "#dc3545" if has_anomaly else "#28a745"
        summary_text = "存在指标超差" if has_anomaly else "姿态控制指标全部合格"

        html = f"""
        <div class="section">
            <h2>姿态控制性能分析</h2>
            <div style="padding:12px; margin-bottom:20px; border-left:5px solid {summary_color}; background:{summary_color}1a;">
                <strong style="color:{summary_color}; font-size:14px;">诊断结论：{summary_text}</strong>
            </div>
            <table>
                <thead><tr style="background:#f1f3f5;"><th>项目</th><th>轴</th><th>实测(3σ)</th><th>指标</th><th>判定</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            <div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"></div>
        </div>
        """
        return {"is_abnormal": has_anomaly, "summary": "姿态: " + (",".join(details) + "超标" if has_anomaly else "合格"), "html": html}
    except Exception as e:
        return {"is_abnormal": True, "summary": "姿态分析异常", "html": f"<div class='error'>{e}</div>"}

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
    """
    电推寿命/燃料消耗分析
    TMKR322: 总工作时长节拍数 (需 /4 换算为秒)
    TMKR323: 总工作次数
    推力: 15mN (0.015 N)
    额定总冲: 72480 Ns
    """
    if data.empty or data.shape[1] < 2:
        return {"is_abnormal": False, "summary": "无电推数据", "html": "<div class='error'>无电推数据</div>"}

    try:
        # 1. 数据清洗与提取
        # 假设列顺序与查询一致：[TMKR322(时长), TMKR323(次数)]
        # 取最后一行有效数据作为当前状态
        valid_data = data.dropna().apply(pd.to_numeric, errors='coerce')
        if valid_data.empty:
            return {"is_abnormal": False, "summary": "数据无效", "html": "<div class='error'>电推数据无效</div>"}

        latest_row = valid_data.iloc[-1]
        
        raw_ticks = latest_row[0] # TMKR322
        work_cycles = int(latest_row[1]) # TMKR323
        
        # 2. 核心计算
        duration_sec = raw_ticks / 4.0
        thrust_N = 0.015 # 15 mN
        rated_impulse = 72480.0 # Ns
        
        current_impulse = duration_sec * thrust_N
        used_percentage = (current_impulse / rated_impulse) * 100.0
        remaining_percentage = 100.0 - used_percentage
        
        # 边界处理
        if used_percentage > 100: used_percentage = 100.0
        if remaining_percentage < 0: remaining_percentage = 0.0

        # 3. 判定逻辑 (例如：寿命使用超过 90% 预警)
        is_abnormal = False
        summary_text = f"剩余寿命 {remaining_percentage:.2f}%"
        status_color = "#28a745" # Green
        
        if used_percentage > 90.0:
            is_abnormal = True # 标记为关注项（虽然不是故障，但属于重要状态）
            summary_text = f"燃料告急 (剩余 {remaining_percentage:.1f}%)"
            status_color = "#e67e22" # Orange
        if used_percentage > 98.0:
            status_color = "#dc3545" # Red

        # 4. 绘图 (左侧饼图展示寿命，右侧折线图展示消耗趋势)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        
        # 4.1 饼图：寿命占比
        sizes = [used_percentage, remaining_percentage]
        labels = ['已使用', '剩余']
        colors = ['#bdc3c7', status_color] # 灰色已用，彩色剩余
        explode = (0, 0.1) 
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontsize': 11})
        ax1.set_title(f'电推工质消耗占比\n(额定: {rated_impulse} Ns)')

        # 4.2 趋势图：总冲积累
        # 计算历史序列的总冲
        hist_ticks = valid_data.iloc[:, 0]
        hist_impulse = (hist_ticks / 4.0) * thrust_N
        x_axis = range(len(hist_impulse))
        
        ax2.plot(x_axis, hist_impulse, color=status_color, linewidth=2)
        ax2.fill_between(x_axis, hist_impulse, color=status_color, alpha=0.1)
        ax2.set_title('总冲积累趋势 (Ns)')
        ax2.set_ylabel('Total Impulse (Ns)')
        ax2.set_xlabel('Sampling Points')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # 5. 生成 HTML
        html = f"""
        <div class="section">
            <h2>电推系统健康评估</h2>
            <div style="padding:15px; margin-bottom:15px; background:#f8f9fa; border-left:4px solid {status_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:14px; color:#666;">累计工作时长</div>
                        <div style="font-size:20px; font-weight:bold; color:#333;">{duration_sec/3600:.2f} h</div>
                    </div>
                    <div>
                        <div style="font-size:14px; color:#666;">累计点火次数</div>
                        <div style="font-size:20px; font-weight:bold; color:#333;">{work_cycles} 次</div>
                    </div>
                    <div>
                        <div style="font-size:14px; color:#666;">当前消耗总冲</div>
                        <div style="font-size:20px; font-weight:bold; color:{status_color};">{current_impulse:.2f} Ns</div>
                    </div>
                    <div>
                        <div style="font-size:14px; color:#666;">消耗进度</div>
                        <div style="font-size:20px; font-weight:bold; color:{status_color};">{used_percentage:.2f}%</div>
                    </div>
                </div>
            </div>
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; border:1px solid #ddd; border-radius:5px;">
            </div>
        </div>
        """

        return {
            "is_abnormal": is_abnormal,
            "summary": f"电推寿命: 已用 {used_percentage:.1f}% ({work_cycles}次)",
            "html": html
        }

    except Exception as e:
        return {"is_abnormal": True, "summary": "电推分析出错", "html": f"<div class='error'>电推分析出错: {e}</div>"}

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
            body {{ font-family: 'Segoe UI', sans-serif; background: #f4f7f9; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            h2 {{ color: #3498db; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #f8f9fa; }}
            .error {{ color: red; background: #fee; padding: 15px; border-radius: 4px; }}
            .section {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <p style="text-align:right; color:#777;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {body_content}
        </div>
    </body>
    </html>
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"Report_{title}_{timestamp}.html"
    try:
        abs_path = os.path.abspath(filename)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        webbrowser.open(f'file://{abs_path}')
        return f"报告已生成并打开: {filename}"
    except Exception as e:
        return f"报告生成失败: {e}"

def _generate_final_report_content(check_results: List[Dict], part1_html: str, part2_html: str, part3_html: str) -> str:
    """
    [通用功能] 生成卫星体检报告的内部 HTML 内容，包含仪表盘和分章节正文。
    """
    # --- 1. 数据统计 ---
    total_checks = len(check_results)
    anomalies = [r for r in check_results if r.get('is_abnormal')]
    count_abnormal = len(anomalies)
    
    # --- 2. 仪表盘样式与状态判定 ---
    if count_abnormal > 0:
        status_color = "#e53e3e"  # 红色
        status_bg = "#fff5f5"
        status_icon = "⚠️"
        status_text = "存在风险"
        
        # 构造异常列表 HTML
        anomaly_items = ""
        for item in anomalies:
            anomaly_items += f"""
            <li style="margin-bottom: 8px; padding: 10px; background: white; border-left: 4px solid {status_color}; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                <strong style="color: {status_color};">[{item.get('name', '未知项')}]</strong> 
                <span style="color: #4a5568; margin-left: 10px;">{item.get('summary', '未提供摘要')}</span>
            </li>"""
        anomaly_list_html = f"""
        <div style="margin-top: 20px;">
            <div style="font-size: 13px; color: #718096; font-weight: bold; text-transform: uppercase; margin-bottom: 10px;">异常详情</div>
            <ul style="list-style: none; padding: 0; margin: 0;">{anomaly_items}</ul>
        </div>"""
    else:
        status_color = "#2f855a"  # 绿色
        status_bg = "#f0fff4"
        status_icon = "✅"
        status_text = "状态良好"
        anomaly_list_html = f"""
        <div style="margin-top: 20px; padding: 15px; background: white; color: {status_color}; text-align: center; border-radius: 6px; border: 1px dashed {status_color}80;">
            🎉 所有检测项均符合设计指标要求
        </div>"""

    # --- 3. 构造仪表盘 HTML ---
    dashboard_html = f"""
    <div style="background: #ffffff; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; margin-bottom: 40px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h2 style="margin-top:0; color: #2d3748; border-bottom: 2px solid #edf2f7; padding-bottom: 15px; display: flex; align-items: center;">
            <span style="margin-right: 10px;">🩺</span> 卫星在轨状态健康摘要
        </h2>
        <div style="display: flex; gap: 20px; margin-top: 20px;">
            <div style="flex: 1; background: #f7fafc; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #cbd5e0;">
                <div style="font-size: 36px; font-weight: bold; color: #4a5568;">{total_checks}</div>
                <div style="color: #718096; font-size: 13px; font-weight: bold; margin-top: 5px;">检测总数</div>
            </div>
            <div style="flex: 1; background: {status_bg}; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid {status_color}40;">
                <div style="font-size: 36px; font-weight: bold; color: {status_color};">{count_abnormal}</div>
                <div style="color: {status_color}; font-size: 13px; font-weight: bold; margin-top: 5px;">异常数量 {status_icon}</div>
            </div>
        </div>
        {anomaly_list_html}
    </div>
    """

    # --- 4. 辅助函数：分节标题 ---
    def make_header(title, icon):
        return f"""
        <div style="margin-top: 60px; margin-bottom: 25px; border-left: 6px solid #3498db; padding-left: 15px; background: linear-gradient(to right, #eef2f7, transparent); padding-top: 10px; padding-bottom: 10px;">
            <h1 style="margin: 0; color: #2c3e50; font-size: 22px; display: flex; align-items: center;">
                <span style="margin-right: 10px;">{icon}</span> {title}
            </h1>
        </div>"""

    # --- 5. 拼装总正文 ---
    full_body = dashboard_html
    
    if part1_html and len(part1_html.strip()) > 0:
        full_body += make_header("第一部分：单机性能评估", "⚙️")
        full_body += part1_html
        
    if part2_html and len(part2_html.strip()) > 0:
        full_body += make_header("第二部分：系统性能评估", "🛰️")
        full_body += part2_html
        
    if part3_html and len(part3_html.strip()) > 0:
        full_body += make_header("第三部分：结构热变形分析", "🌡️")
        full_body += part3_html

    return full_body
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

@mcp.tool(description="[单项] 姿态控制精度分析。")
def calculate_attitude_control_accuracy(satellite_name: str = None, data_json: str = None, start_time_str: str = None, end_time_str: str = None) -> str:
    try:
        df = pd.DataFrame()
        if data_json and len(data_json) > 10:
            try: df = pd.read_json(io.StringIO(data_json), orient='split')
            except: pass
        elif satellite_name:
            sat_code, tm_code = _get_codes_impl(satellite_name, "姿态")
            if sat_code: df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        
        if df.empty: return "错误: 无数据。"
        result_dict = _analyze_attitude_impl(df)
        return _wrap_html_report(result_dict['html'], "姿态控制精度报告")
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

@mcp.tool(description="""[一键评估] 卫星状态全面评估。
自动执行：星敏噪声、姿态精度、飞轮性能、单机通信错误、热变形等所有检查项。
报告顶部包含异常统计仪表盘。
""")
@mcp.tool(description="""[一键评估] 卫星状态全面评估。
自动执行：星敏噪声、姿态精度、飞轮性能、单机通信错误、热变形等所有检查项。
报告顶部包含异常统计仪表盘。
""")
@mcp.tool(description="""[一键评估] 卫星状态全面评估。
自动执行：星敏噪声、姿态精度、飞轮性能、单机通信错误、热变形等所有检查项。
报告顶部包含异常统计仪表盘，正文分为单机性能、系统性能、热变形分析三部分。
""")
def assess_satellite_status(satellite_name: str, start_time_str: str = None, end_time_str: str = None) -> str:
    """
    综合评估入口：串行调用各个底层 _impl 函数，按章节拼装总报告。
    """
    check_results = [] # 用于仪表盘统计
    
    # 定义三个部分的 HTML 缓冲区
    html_part1_components = ""  # 第一部分：单机 (星敏/陀螺/飞轮/通信)
    html_part2_system = ""      # 第二部分：系统 (姿态)
    html_part3_thermal = ""     # 第三部分：热变形

    base_sat_code, _ = _get_codes_impl(satellite_name, "任意")
    if not base_sat_code:
        return f"错误：未找到卫星 {satellite_name} 的定义。"

    # ==========================================
    # 第一部分：单机性能评估
    # ==========================================
    
    # 1. 星敏分析
    star_analysis_end_time = end_time_str
    if start_time_str:
        try:
            s_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
            star_analysis_end_time = (s_dt + timedelta(minutes=3)).strftime('%Y-%m-%d %H:%M:%S')
        except: pass

    for label in ["星敏A", "星敏B"]:
        sat_code, tm_code = _get_codes_impl(satellite_name, label)
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, start_time_str, star_analysis_end_time)
            res = _analyze_star_sensor_impl(df, sensor_name=label)
            check_results.append({"name": label, **res})
            if res.get('html'): html_part1_components += res['html'] + "<br>"
        else:
            check_results.append({"name": label, "is_abnormal": False, "summary": "未配置", "html": ""})

    # 2. 陀螺分析
    for cfg in [{"name": "陀螺A", "limit": 0.0004}, {"name": "陀螺B", "limit": 0.0020}]:
        sat_code, tm_code = _get_codes_impl(satellite_name, cfg["name"])
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
            res = _analyze_gyro_impl(df, cfg["name"], cfg["limit"])
            check_results.append({"name": cfg["name"], **res})
            if res.get('html'): html_part1_components += res['html'] + "<br>"

    # 3. 飞轮分析
    for fw in ["飞轮A", "飞轮B", "飞轮C", "飞轮D"]:
        sat_code, tm_code = _get_codes_impl(satellite_name, fw)
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
            res = _analyze_wheel_impl(df, fw, 0.5)
            check_results.append({"name": fw, **res})
            if res.get('html'): html_part1_components += res['html'] + "<br>"

    # 4. 单机通信错误统计 (归入第一部分)
    dev_results_list, dev_html_table = _analyze_device_errors_impl(base_sat_code, start_time_str, end_time_str)
    check_results.extend(dev_results_list) # 计入仪表盘总数
    html_part1_components += dev_html_table + "<br>" # 表格放入第一部分

    # ==========================================
    # 第二部分：系统性能评估
    # ==========================================

    # 5. 姿态分析
    sat_code, tm_code = _get_codes_impl(satellite_name, "姿态")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        res = _analyze_attitude_impl(df)
        check_results.append({"name": "姿态控制", **res})
        if res.get('html'): html_part2_system += res['html']

    # 6. 【新增】轨道高度分析
    sat_code, tm_code = _get_codes_impl(satellite_name, "平根半长轴")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        res = _analyze_orbit_impl(df)
        check_results.append({"name": "轨道维持", **res})
        if res.get('html'): html_part2_system += res['html']
    else:
        check_results.append({"name": "轨道维持", "is_abnormal": False, "summary": "未配置", "html": ""})

    # 7. 【新增】降交点地方时 (LTDN) 分析
    sat_code, tm_code = _get_codes_impl(satellite_name, "降交点")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        res = _analyze_ltdn_impl(df)
        check_results.append({"name": "LTDN维持", **res})
        if res.get('html'): html_part2_system += res['html']
    else:
        check_results.append({"name": "LTDN维持", "is_abnormal": False, "summary": "未配置", "html": ""})
    # ==========================================
    # 第三部分：结构热变形分析
    # ==========================================

    # ================= 添加在这里 =================
    # 7.5 电推寿命分析 【新增】
    sat_code, tm_code = _get_codes_impl(satellite_name, "电推")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, start_time_str, end_time_str)
        res = _analyze_propulsion_impl(df)
        check_results.append({"name": "电推系统", **res}) # 加入仪表盘统计
        if res.get('html'): html_part2_system += res['html'] # 加入正文
    else:
        # 如果不是所有卫星都有电推，可以根据需要决定是否显示“未配置”
        # check_results.append({"name": "电推系统", "is_abnormal": False, "summary": "未配置", "html": ""})
        pass
    # ============================================

    # 8. 热变形
    _, html_thermal = _analyze_thermal_impl(base_sat_code, start_time_str, end_time_str)
    check_results.append({"name": "结构热变形", "is_abnormal": False, "summary": "已分析", "html": html_thermal})
    html_part3_thermal += html_thermal

    # ==========================================
    # 生成仪表盘 (Dashboard)
    # ==========================================
    total_checks = len(check_results)
    anomalies = [r for r in check_results if r.get('is_abnormal')]
    count_abnormal = len(anomalies)
    
    if count_abnormal > 0:
        anomaly_items = ""
        for item in anomalies:
            anomaly_items += f"""
            <li style="margin-bottom: 5px; padding: 8px; background: #fff5f5; border-left: 3px solid #e53e3e; border-radius: 4px;">
                <span style="font-weight:bold; color: #c53030;">[{item['name']}]</span> 
                <span style="color: #4a5568;">{item['summary']}</span>
            </li>
            """
        anomaly_list_html = f"<ul style='list-style: none; padding: 0; margin-top: 15px;'>{anomaly_items}</ul>"
        status_card_color = "#fff5f5"
        status_icon = "⚠️"
        status_text_color = "#c53030"
    else:
        anomaly_list_html = "<div style='margin-top:15px; padding:10px; background:#f0fff4; color:#2f855a; text-align:center; border-radius:4px;'>🎉 所有检测项均符合指标要求</div>"
        status_card_color = "#f0fff4"
        status_icon = "✅"
        status_text_color = "#2f855a"

    dashboard_html = f"""
    <div style="background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; margin-bottom: 30px; font-family: sans-serif;">
        <h2 style="margin-top:0; color: #2d3748; border-bottom: 2px solid #edf2f7; padding-bottom: 15px;">
            🩺 卫星健康体检摘要
        </h2>
        <div style="display: flex; gap: 20px; margin-top: 20px;">
            <div style="flex: 1; background: #f7fafc; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #cbd5e0;">
                <div style="font-size: 42px; font-weight: bold; color: #4a5568; line-height: 1;">{total_checks}</div>
                <div style="color: #718096; font-weight: bold; margin-top: 5px; font-size: 14px;">已执行测试项</div>
                <div style="font-size: 24px; margin-top: 5px;">📋</div>
            </div>
            <div style="flex: 1; background: {status_card_color}; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid {status_text_color}40;">
                <div style="font-size: 42px; font-weight: bold; color: {status_text_color}; line-height: 1;">{count_abnormal}</div>
                <div style="color: {status_text_color}; font-weight: bold; margin-top: 5px; font-size: 14px;">异常项数量</div>
                <div style="font-size: 24px; margin-top: 5px;">{status_icon}</div>
            </div>
        </div>
        <div style="margin-top: 20px;">
            <div style="font-size: 14px; color: #718096; font-weight: bold; text-transform: uppercase; letter-spacing: 0.5px;">
                诊断详情 ({'存在风险' if count_abnormal > 0 else '正常'})
            </div>
            {anomaly_list_html}
        </div>
    </div>
    """

    # ==========================================
    # 拼装最终报告
    # ==========================================
    
    # 辅助函数：生成带样式的分节标题
    def make_header(title):
        return f"""
        <div style="margin-top: 50px; margin-bottom: 20px; border-left: 6px solid #3498db; padding-left: 15px;">
            <h1 style="margin: 0; color: #2c3e50; font-size: 24px;">{title}</h1>
        </div>
        """

    full_body = dashboard_html
    full_body += "<hr style='border: 0; border-top: 1px solid #eee; margin: 30px 0;'>"
    
    # 第一部分：单机
    if html_part1_components:
        full_body += make_header("第一部分：单机性能评估")
        full_body += html_part1_components
        
    # 第二部分：系统
    if html_part2_system:
        full_body += make_header("第二部分：系统性能评估")
        full_body += html_part2_system
        
    # 第三部分：热变形
    if html_part3_thermal:
        full_body += make_header("第三部分：结构热变形分析")
        full_body += html_part3_thermal

    return _wrap_html_report(full_body, f"{satellite_name} 状态全检报告")

@mcp.tool(description="""[月度评估] 自动化执行在轨卫星月度性能体检。
1. 性能项(星敏/陀螺/飞轮): 采样每月15日0点的3分钟高频数据。
2. 热变形: 采样每月15日开始的24小时数据，分析轨道周期性漂移。
3. 系统项(姿态/轨道/通信/电推): 统计全月完整数据。
""")
def assess_monthly_performance(satellite_name: str, year_month: str = None) -> str:
    """
    year_month 格式: '2023-10'。如果不传，默认评估当前月。
    """
    # --- 1. 时间窗口计算 ---
    if year_month:
        try:
            target_dt = datetime.strptime(year_month, '%Y-%m')
        except:
            return "错误：月份格式应为 YYYY-MM，例如 2023-10"
    else:
        target_dt = datetime.now().replace(day=1)

    # A. 全月窗口 (用于：姿态控精、通信错误、轨道高度、LTDN、电推)
    month_start_str = target_dt.strftime('%Y-%m-01 00:00:00')
    if target_dt.month == 12:
        next_month = target_dt.replace(year=target_dt.year + 1, month=1)
    else:
        next_month = target_dt.replace(month=target_dt.month + 1)
    month_end_str = (next_month - timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')

    # B. 3分钟采样窗口 (用于：星敏噪声、陀螺噪声、飞轮精度)
    sample_3m_start = target_dt.replace(day=15, hour=0, minute=0, second=0)
    sample_3m_end = sample_3m_start + timedelta(minutes=3)
    s3m_start_str = sample_3m_start.strftime('%Y-%m-%d %H:%M:%S')
    s3m_end_str = sample_3m_end.strftime('%Y-%m-%d %H:%M:%S')

    # C. 24小时热变形窗口 (用于：星敏光轴夹角热稳定性)
    sample_1d_start = sample_3m_start # 同样从15号0点开始
    sample_1d_end = sample_1d_start + timedelta(days=1)
    s1d_start_str = sample_1d_start.strftime('%Y-%m-%d %H:%M:%S')
    s1d_end_str = sample_1d_end.strftime('%Y-%m-%d %H:%M:%S')

    print(f">>> 启动 {satellite_name} 月度评估报告生成")
    print(f">>> [性能采样] {s3m_start_str} (3min)")
    print(f">>> [热变形窗] {s1d_start_str} (24h)")
    print(f">>> [全月统计] {month_start_str} 至 {month_end_str}")

    base_sat_code, _ = _get_codes_impl(satellite_name, "任意")
    if not base_sat_code: return f"错误：未找到卫星 {satellite_name}"

    check_results = []
    html_part1_components = "" 
    html_part2_system = ""      
    html_part3_thermal = ""     

    # --- 2. 执行分析 (分窗口调用) ---

    # [3分钟采样] 星敏、陀螺、飞轮
    for label in ["星敏A", "星敏B"]:
        sat_code, tm_code = _get_codes_impl(satellite_name, label)
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, s3m_start_str, s3m_end_str)
            res = _analyze_star_sensor_impl(df, sensor_name=label)
            check_results.append({"name": label, **res})
            html_part1_components += res['html']

    for cfg in [{"name": "陀螺A", "limit": 0.0004}, {"name": "陀螺B", "limit": 0.0020}]:
        sat_code, tm_code = _get_codes_impl(satellite_name, cfg["name"])
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, s3m_start_str, s3m_end_str)
            res = _analyze_gyro_impl(df, cfg["name"], cfg["limit"])
            check_results.append({"name": cfg["name"], **res})
            html_part1_components += res['html']

    for fw in ["飞轮A", "飞轮B", "飞轮C", "飞轮D"]:
        sat_code, tm_code = _get_codes_impl(satellite_name, fw)
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, s3m_start_str, s3m_end_str)
            res = _analyze_wheel_impl(df, fw, 0.5)
            check_results.append({"name": fw, **res})
            html_part1_components += res['html']

    # [全月统计] 通信错误
    dev_results, dev_html = _analyze_device_errors_impl(base_sat_code, month_start_str, month_end_str)
    check_results.extend(dev_results)
    html_part1_components += dev_html

    # [全月统计] 姿态控制性能 (重点：这里改用全月窗口)
    sat_code, tm_code = _get_codes_impl(satellite_name, "姿态")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, month_start_str, month_end_str)
        res = _analyze_attitude_impl(df)
        check_results.append({"name": "月度姿态控制", **res})
        html_part2_system += res['html']

    # [全月统计] 轨道与电推
    for item in ["平根半长轴", "降交点"]:
        sat_code, tm_code = _get_codes_impl(satellite_name, item)
        if sat_code and tm_code:
            df = _get_data_impl(sat_code, tm_code, month_start_str, month_end_str)
            if "半长轴" in item: res = _analyze_orbit_impl(df)
            else: res = _analyze_ltdn_impl(df)
            check_results.append({"name": f"月度{item}", **res})
            html_part2_system += res['html']

    sat_code, tm_code = _get_codes_impl(satellite_name, "电推")
    if sat_code and tm_code:
        df = _get_data_impl(sat_code, tm_code, month_start_str, month_end_str)
        res = _analyze_propulsion_impl(df)
        check_results.append({"name": "月度电推系统", **res})
        html_part2_system += res['html']

    # [24小时采样] 热变形分析
    _, html_thermal = _analyze_thermal_impl(base_sat_code, s1d_start_str, s1d_end_str)
    check_results.append({"name": "结构热稳定性(24h)", "is_abnormal": False, "summary": "已分析", "html": html_thermal})
    html_part3_thermal = html_thermal

    # --- 3. 生成最终报告 (调用共用的 HTML 拼装逻辑) ---
    title = f"{satellite_name} {target_dt.strftime('%Y年%m月')} 在轨运行月度分析报告"
    
    # 这里的 _generate_final_report_content 是一个建议抽离的函数，见下文
    full_content = _generate_final_report_content(check_results, html_part1_components, html_part2_system, html_part3_thermal)
    
    return _wrap_html_report(full_content, title)

if __name__ == "__main__":
    mcp.run(transport="sse")