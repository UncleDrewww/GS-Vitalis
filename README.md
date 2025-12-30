# 🛰️ 在轨数据智能挖掘系统
**On-orbit Data Intelligent Mining System**

本文档指导用户完成系统的环境搭建、VSCode MCP 配置文件生成、服务启动及 Cherry Studio 客户端连接。

## 📋 1. 环境准备 (Prerequisites)

*   **IDE 编辑器**: VSCode (Visual Studio Code)
*   **编程语言**: Python **3.13.2**

## 🛠️ 2. Python 环境安装 (Installation)

在 VSCode 终端中依次执行以下命令：

### 2.1 创建与激活虚拟环境
```powershell
# 创建名为 env 的虚拟环境
python -m venv env

# 激活虚拟环境 (Windows PowerShell)
.\env\Scripts\Activate.ps1
```

### 2.2 安装依赖
```bash
pip install -r requirements.txt
```

## ⚙️ 3. 配置 MCP 服务器 (生成 JSON 配置文件)

本步骤通过 VSCode 聊天窗口的配置工具，生成连接所需的 JSON 配置。

1.  **打开聊天窗口**：
    在 VSCode 中打开 AI 助手的聊天界面。
2.  **打开配置工具**：
    点击聊天输入框附近的 **配置工具 (Configuration Tool/图标)**。
3.  **添加服务器**：
    选择 **添加 MCP 服务器 (Add MCP Server)** 选项。
4.  **选择连接类型**：
    选择 **HTTP** 模式。
5.  **输入连接信息**：
    *   输入地址：`http://127.0.0.1:8001/sse`
    *   （可选）输入名称：`WorkSpace`
6.  **生成配置**：
    确认后，工具会自动将配置写入项目的 JSON 配置文件中。

## 🚀 4. 启动服务 (Start Server)

### 4.1 运行 Python 服务脚本
在终端（已激活虚拟环境）中运行 MCP 服务器入口程序：
```powershell
# 假设入口文件名为 mcp_server.py 或作为模块运行
python mcp_server.py 
# 或者根据您的实际启动命令
mcp_server
```

### 4.2 通过 JSON 启动连接
1.  确保 4.1 中的 Python 服务已成功运行。
2.  在 VSCode 中利用步骤 3 生成的 JSON 配置启动/连接 MCP 服务。

## 🍒 5. 客户端连接：Cherry Studio

使用 Cherry Studio 作为独立客户端连接本系统。

1.  **打开设置**：进入 Cherry Studio 的 MCP 服务器管理界面。
2.  **添加服务器 (Add Server)**：
    *   **类型 (Type)**: `SSE` (Server-Sent Events)
    *   **URL 地址**: `http://127.0.0.1:8001/sse`
3.  **启动连接**：点击启动，确认状态显示为已连接。