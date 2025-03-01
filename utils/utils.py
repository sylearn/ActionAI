import json
import os
import sys
import tiktoken
from typing import List,Dict,Any
from dotenv import load_dotenv
import time

# 环境变量加载
def load_env_files(seconds: int = 3):
    """加载环境变量文件
    优先加载通用.env文件，然后根据系统加载特定的环境变量文件
    特定系统的环境变量会覆盖通用环境变量
    """
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本位置: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"入口脚本: {sys.argv[0]}")
    
    # 获取入口脚本所在的目录作为基础路径
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        # 使用入口脚本的目录作为基础路径
        base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    print(f"最终使用的基础路径: {base_path}")
    
    loaded_files = []  # 记录加载的文件信息
    
    # 首先尝试加载通用.env文件
    common_env = os.path.join(base_path, '.env')
    if os.path.exists(common_env):
        before_vars = set(os.environ.keys())
        load_dotenv(common_env)
        after_vars = set(os.environ.keys())
        new_vars = after_vars - before_vars
        loaded_files.append({
            "type": "通用配置",
            "path": common_env,
            "vars_count": len(new_vars),
            "status": "已加载"
        })
    
    # 然后根据系统加载特定的环境变量文件
    system_type = "Windows" if sys.platform == "win32" else "MacOS/Linux"
    system_env = os.path.join(
        base_path,
        '.env_win' if sys.platform == "win32" else '.env_mac'
    )
    if os.path.exists(system_env):
        before_vars = set(os.environ.keys())
        load_dotenv(system_env, override=True)
        after_vars = set(os.environ.keys())
        new_vars = after_vars - before_vars
        loaded_files.append({
            "type": f"{system_type}配置",
            "path": system_env,
            "vars_count": len(new_vars),
            "status": "已加载"
        })
    
    # 必需配置检查
    required_vars = {
        "MODEL": "模型选择",
        "OPENAI_API_KEY": "API密钥"
    }
    
    # 可选配置
    optional_vars = {
        "OPENAI_BASE_URL": ("API基础URL", "https://api.openai.com/v1"),
        "MAX_TOKENS": ("最大令牌数", "4096"),
        "TEMPERATURE": ("温度系数", "0.7"),
        "STREAM": ("流式输出", "True"),
        "DEBUG": ("调试模式", "False"),
        "IS_FUNCTION_CALLING": ("工具调用", "True"),
        "TIMEOUT": ("超时时间", "300"),
        "MAX_MESSAGES": ("最大消息数", "100"),
        "MAX_MESSAGES_TOKENS": ("最大令牌限制", "60000"),
        "MESSAGES_PATH": ("消息存储路径", "./messages"),
        "MCP_SERVER_CONFIG_PATH": ("服务器配置", ""),
        "SYSTEM_PROMPT": ("系统提示词", ""),
        "PYTHON_PATH": ("Python路径", "python"),
        "NODE_PATH": ("Node路径", "node")
    }
    
    # 显示必需配置状态
    print("\n❗ 必需配置检查:")
    all_required_set = True
    for var, desc in required_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ✅ {desc:<10} ({var}): 已设置")
        else:
            all_required_set = False
            print(f"  ❌ {desc:<10} ({var}): 未设置")
    
    # 显示可选配置状态
    print("\n⚙️  可选配置状态:")
    for var, (desc, default) in optional_vars.items():
        value = os.getenv(var)
        if value:
            print(f"  ℹ️ {desc:<12} ({var}): {value}")
        else:
            print(f"  💭 {desc:<12} ({var}): 使用默认值 [{default}]")
    
    # 配置检查结果
    print("\n📊 配置检查结果:")
    if all_required_set:
        print("  ✅ 所有必需配置已设置，系统可以正常启动")
    else:
        print("  ❌ 缺少必需配置，系统可能无法正常工作")
        print("  ⚠️ 请检查上述标记为未设置的必需配置项")
        
    if seconds > 0:
        # 添加一个空行
        print("\n⚙️  正在初始化系统...")
        for i in range(seconds, 0, -1):
            print(f"\r⏳ {i} 秒后继续...", end="", flush=True)
            time.sleep(1)
        print("\n")  # 最后打印一个换行

def parse_mcp_servers(file_path):
    """
    解析MCP服务器配置JSON文件，提取命令和参数
    
    Args:
        file_path (str): JSON配置文件的路径
        
    Returns:
        dict: 包含每个服务器的命令和参数的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
            
        # 读取并解析JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
            
        # 检查是否包含mcpServers键
        if 'mcpServers' not in config:
            raise KeyError("JSON文件中没有找到'mcpServers'键")
            
        # 提取每个服务器的命令和参数
        servers_info = {}
        for server_name, server_config in config['mcpServers'].items():
            # 提取命令
            command = server_config.get('command', '')
            
            # 提取参数
            args = server_config.get('args', [])
            
            # 提取环境变量
            env = server_config.get('env', {})
            # 存储服务器信息
            servers_info[server_name] = {
                'command': command,
                'args': args,
                'env': env
            }
            
        return servers_info
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {str(e)}")
    except Exception as e:
        raise Exception(f"解析MCP服务器配置时出错: {str(e)}")

def get_server_command(servers_info, server_name):
    """
    获取指定服务器的命令
    
    Args:
        servers_info (dict): 服务器信息字典
        server_name (str): 服务器名称
        
    Returns:
        str: 服务器命令
    """
    if server_name not in servers_info:
        raise KeyError(f"未找到服务器: {server_name}")
    
    return servers_info[server_name]['command']

def get_server_args(servers_info, server_name):
    """
    获取指定服务器的参数
    
    Args:
        servers_info (dict): 服务器信息字典
        server_name (str): 服务器名称
        
    Returns:
        list: 服务器参数列表
    """
    if server_name not in servers_info:
        raise KeyError(f"未找到服务器: {server_name}")
    
    return servers_info[server_name]['args']

def get_server_env(servers_info, server_name):
    """
    获取指定服务器的环境变量
    
    Args:
        servers_info (dict): 服务器信息字典
        server_name (str): 服务器名称
        
    Returns:
        dict: 服务器环境变量字典
    """
    if server_name not in servers_info:
        raise KeyError(f"未找到服务器: {server_name}")
    
    return servers_info[server_name]['env']

def get_all_server_names(servers_info):
    """
    获取所有服务器名称
    
    Args:
        servers_info (dict): 服务器信息字典
        
    Returns:
        list: 服务器名称列表
    """
    return list(servers_info.keys())

def get_token_count(message: List[Dict[str, Any]], model: str) -> tuple[int, int, int, int]:
    """
    计算多轮对话中的token消耗
    
    Args:
        message: 消息列表，每个消息包含role和content
        model: 模型名称
        
    Returns:
        tuple: (total_input_tokens, total_output_tokens, current_input_tokens)
        - total_input_tokens: 所有轮次的输入token总和
        - total_output_tokens: 所有轮次的输出token总和
        - current_input_tokens: 如果开始新的对话轮次，当前所有消息将产生的输入token数
    """
    # 尝试导入tiktoken扩展模块，解决打包后找不到编码的问题
    import tiktoken_ext
    import tiktoken_ext.openai_public
        
    # 始终使用cl100k_base编码，不使用model参数
    encoding = tiktoken.get_encoding("cl100k_base")
    
    total_input_tokens = 0
    total_output_tokens = 0
    current_input_tokens = 0
    
    # 找到所有user消息的索引，用于划分对话轮次
    user_indices = [i for i, msg in enumerate(message) if msg["role"] == "user"]
    # 计算已经对话的轮次
    round_count = len(user_indices)
    
    for round_idx, user_idx in enumerate(user_indices):
        # 计算当前轮次的范围
        round_start = user_idx
        round_end = user_indices[round_idx + 1] if round_idx + 1 < len(user_indices) else len(message)

        # 将user之前的所有消息计入input tokens
        for msg in message[:round_start]:
            tokens = encoding.encode(msg["content"])
            total_input_tokens += len(tokens)
            if msg["role"] == "assistant" and "function_call" in msg:
                function_tokens = encoding.encode(json.dumps(msg["function_call"]))
                total_input_tokens += len(function_tokens)
    
        # 当前轮次的user消息计入input tokens
        user_tokens = encoding.encode(message[user_idx]["content"])
        total_input_tokens += len(user_tokens)
    
        # 当前轮次user之后的assistant消息计入output tokens
        for msg in message[round_start + 1:round_end]:
            if msg["role"] == "assistant":
                content_tokens = encoding.encode(msg["content"])
                total_output_tokens += len(content_tokens)
                # 如果包含function call，要计入output tokens
                if "function_call" in msg:
                    function_tokens = encoding.encode(json.dumps(msg["function_call"]))
                    total_output_tokens += len(function_tokens)

    # 计算current_input_tokens（所有当前消息的token数）
    for msg in message:
        tokens = encoding.encode(msg["content"])
        current_input_tokens += len(tokens)
        if msg["role"] == "assistant" and "function_call" in msg:
            function_tokens = encoding.encode(json.dumps(msg["function_call"]))
            current_input_tokens += len(function_tokens)

    return total_input_tokens, total_output_tokens, current_input_tokens, round_count

if __name__ == "__main__":
    # 测试用例
    message = [
        {"role": "system", "content": "你是一个AI助手"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮回答", 
         "function_call": {"name": "test", "arguments": "{}"}}
    ]
    input_tokens, output_tokens, current_tokens = get_token_count(message, "gpt-4")
    print(f"Total input tokens: {input_tokens}")
    print(f"Total output tokens: {output_tokens}")
    print(f"Current input tokens (for next round): {current_tokens}")