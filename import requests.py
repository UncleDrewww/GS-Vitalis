import requests
import pandas as pd
from datetime import datetime, timedelta

# ================= 配置 =================
GRAFANA_API_KEY = "eyJrIjoi...YOUR_REAL_API_KEY..." # 🔴 请填入真实 Key
GRAFANA_URL = "http://grafana10.prod.yhroot.com"
DATASOURCE_UID = "f1771c95-2940-4f40-a814-65fdfb1838c0"
SATELLITE_CODE = "tm_all_LZ04"
TARGET_CODE = "TMKP040"

# 搜索时间：2025年1月20日 ~ 1月22日 (覆盖发射后几天)
START_TIME = "2025-01-20 00:00:00"
END_TIME   = "2025-01-22 00:00:00"
# =======================================

def debug_raw_mode():
    print(f"=== 纯净模式查询: {TARGET_CODE} ===")
    print(f"时间范围: {START_TIME} ~ {END_TIME}")

    # 1. 构造请求
    try:
        dt_start = datetime.strptime(START_TIME, '%Y-%m-%d %H:%M:%S')
        dt_end = datetime.strptime(END_TIME, '%Y-%m-%d %H:%M:%S')
        start_ms = int(dt_start.timestamp() * 1000)
        end_ms = int(dt_end.timestamp() * 1000)
    except Exception as e:
        print(f"时间解析错误: {e}")
        return

    api_path = f"/api/datasources/proxy/uid/{DATASOURCE_UID}/query"
    full_url = GRAFANA_URL + api_path
    headers = {'Authorization': f'Bearer {GRAFANA_API_KEY}', 'Content-Type': 'application/json'}
    
    # 简单的 InfluxQL
    query_string = f'SELECT "{TARGET_CODE}" FROM "{SATELLITE_CODE}" WHERE time >= {start_ms}ms AND time <= {end_ms}ms'
    
    print(f"Query: {query_string}")

    # 2. 发送请求
    try:
        response = requests.get(full_url, headers=headers, params={'db': 'measure', 'q': query_string}, timeout=30)
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print("Response Text:", response.text)
            return

        data = response.json()
        
        # 3. 检查原始 JSON 结构
        if 'results' not in data:
            print("❌ JSON 中没有 results 字段")
            print(data)
            return
            
        series_list = data['results'][0].get('series', [])
        if not series_list:
            print("❌ results[0] 中没有 series (数据为空)")
            return

        print(f"✅ 收到 {len(series_list)} 个 Series")
        
        # 4. 打印数据详情
        for i, series in enumerate(series_list):
            columns = series.get('columns', [])
            values = series.get('values', [])
            name = series.get('name', 'unknown')
            
            print(f"\n--- Series {i} ({name}) ---")
            print(f"列名: {columns}")
            print(f"行数: {len(values)}")
            
            # 转 DataFrame 方便看
            df = pd.DataFrame(values, columns=columns)
            print("前 5 行数据:")
            print(df.head(5))
            
            print("\n数值统计:")
            # 尝试找到非 time 列
            val_col = [c for c in columns if "time" not in c.lower()][0]
            print(df[val_col].value_counts())

    except Exception as e:
        print(f"请求异常: {e}")

if __name__ == "__main__":
    debug_raw_mode()