import asyncio
import json
import logging
import os
import tiktoken
import readline  # 添加readline导入
from contextlib import AsyncExitStack
from typing import List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from dataclasses import dataclass
import sys

def setup_base_paths():
    """
    设置应用程序的基础路径并添加到系统路径
    
    Returns:
        str: 应用程序的基础路径
    """
    # 获取应用程序的基础路径
    if getattr(sys, 'frozen', False):
        # 如果是打包后的应用
        base_path = os.path.dirname(sys.executable)  # 获取当前应用的绝对路径
    else:
        # 如果是开发环境
        base_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的绝对路径

    # 将基础路径添加到系统路径
    # 无论程序是作为脚本运行还是作为打包应用运行，都能正确导入所需的模块。
    sys.path.insert(0, base_path)
    if os.path.exists(os.path.join(base_path, '..')):
        sys.path.insert(0, os.path.abspath(os.path.join(base_path, '..')))
    
    return base_path

def preload_encodings():
    """
    预加载常用编码并为所有模型注册相同的编码
    """
    try:
        # 预加载常用编码
        logging.info("开始预加载tiktoken编码...")
        # 预先导入tiktoken扩展模块，解决打包后找不到编码的问题
        import tiktoken_ext
        import tiktoken_ext.openai_public
        # 尝试加载编码
        tiktoken.get_encoding("cl100k_base")
        logging.info("成功加载cl100k_base编码")
        
        # 为所有模型注册相同的编码
        # 从环境变量获取模型列表
        model_list_str = os.getenv("MODEL", "")
        logging.info(f"环境变量MODEL值: {model_list_str}")
        if model_list_str:
            # 分割模型列表
            model_list = [model.strip() for model in model_list_str.split(";") if model.strip()]
            logging.info(f"解析出的模型列表: {model_list}")
            # 为每个模型注册相同的编码
            for model in model_list:
                if model:
                    try:
                        logging.info(f"正在为模型 {model} 注册cl100k_base编码")
                        tiktoken.model.MODEL_TO_ENCODING[model] = "cl100k_base"
                        logging.info(f"模型 {model} 注册成功")
                    except Exception as inner_e:
                        logging.error(f"为模型 {model} 注册编码时出错: {inner_e}")  
                        import traceback
                        logging.error(traceback.format_exc())
    except Exception as e:
        logging.error(f"预加载 tiktoken 编码失败: {e}")
        import traceback
        logging.error(traceback.format_exc())

# 初始化基础路径
base_path = setup_base_paths()

from utils.utils import parse_mcp_servers, get_all_server_names, get_server_command, get_server_args, get_server_env
from utils.utils import get_token_count
from utils.utils import load_env_files

# 加载环境变量
load_env_files(seconds=1)

# 预加载编码 - 移到加载环境变量之后
preload_encodings()

def setup_logging():
    """
    配置日志系统
    
    根据环境变量DEBUG设置日志级别和输出方式
    """
    # 配置日志级别
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "False").lower() == "true" else logging.ERROR)

    # 配置统一的日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 仅在debug模式下添加文件处理器
    if os.getenv("DEBUG", "False").lower() == "true":
        # 配置文件处理器
        file_handler = logging.FileHandler('client_dev_debug.log', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# 设置日志
logger = setup_logging()

@dataclass
class LLMConfig:
    """LLM配置类"""
    # 流式输出设置
    stream: bool = True if os.getenv("STREAM", "True").lower() == "true" else False
    # 模型参数设置
    model: str = os.getenv("MODEL", "")
    max_tokens: int = int(os.getenv("MAX_TOKENS", 4096))
    temperature: float = float(os.getenv("TEMPERATURE", 0.7))
    # 获取系统提示文件路径，优先使用环境变量，否则使用基础路径下的默认文件
    system_prompt: str = os.getenv("SYSTEM_PROMPT", os.path.join(base_path, "prompt/system.md"))
    # 调试和功能设置
    debug: bool = True if os.getenv("DEBUG", "False").lower() == "true" else False
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    is_function_calling: bool = True if os.getenv("IS_FUNCTION_CALLING", "True").lower() == "true" else False
    is_human_control: bool = True if os.getenv("IS_HUMAN_CONTROL", "True").lower() == "true" else False
    # 超时和限制设置
    timeout: int = int(os.getenv("TIMEOUT", 300))
    max_messages: int = int(os.getenv("MAX_MESSAGES", 100))
    max_messages_tokens: int = int(os.getenv("MAX_MESSAGES_TOKENS", 60000))
    # 路径设置
    messages_path: str = os.getenv("MESSAGES_PATH", os.path.join(base_path, "messages"))
    mcp_server_config: str = os.getenv("MCP_SERVER_CONFIG_PATH", os.path.join(base_path, "mcp_server_config.json"))
    python_path: str = os.getenv("PYTHON_PATH", "python")
    node_path: str = os.getenv("NODE_PATH", "node")
    npx_path: str = os.getenv("NPX_PATH", "npx")

def load_config_from_env(base_path: str) -> LLMConfig:
    """
    从环境变量加载配置
    
    Args:
        base_path: 应用程序基础路径
        
    Returns:
        LLMConfig: 配置对象
    """
    return LLMConfig(
        stream=True if os.getenv("STREAM", "True").lower() == "true" else False,
        model=os.getenv("MODEL", ""),
        max_tokens=int(os.getenv("MAX_TOKENS", 4096)),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        system_prompt=os.getenv("SYSTEM_PROMPT", os.path.join(base_path, "prompt/system.md")),
        debug=True if os.getenv("DEBUG", "False").lower() == "true" else False,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        is_function_calling=True if os.getenv("IS_FUNCTION_CALLING", "True").lower() == "true" else False,
        is_human_control=True if os.getenv("IS_HUMAN_CONTROL", "True").lower() == "true" else False,
        timeout=int(os.getenv("TIMEOUT", 300)),
        max_messages=int(os.getenv("MAX_MESSAGES", 100)),
        max_messages_tokens=int(os.getenv("MAX_MESSAGES_TOKENS", 60000)),
        messages_path=os.getenv("MESSAGES_PATH", os.path.join(base_path, "messages")),
        mcp_server_config=os.getenv("MCP_SERVER_CONFIG_PATH", os.path.join(base_path, "mcp_server_config.json")),
        python_path=os.getenv("PYTHON_PATH", "python"),
        node_path=os.getenv("NODE_PATH", "node"),
        npx_path=os.getenv("NPX_PATH", "npx")
    )

class LLM_Client:
    def __init__(self, config: LLMConfig = None):
        """
        初始化LLM客户端
        Args:
            config: LLM配置对象，如果为None则使用默认配置
        """
        # 使用提供的配置或创建默认配置
        self.config = config or load_config_from_env(base_path)
        # MCP相关属性
        self.server_names: List[str] = []
        self.session_list: List[ClientSession] = []
        self.available_tools: List = []
        self.exit_stack = AsyncExitStack()
        
        # OpenAI相关属性
        self.openai = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.current_input_tokens = 0
        self.thinking_tokens = 0
        self.round_count = 0 # 当前轮次
        self.tool_tokens = []
        # 消息相关属性
        self.messages: List = []
        self._console = Console()
        
        # 输入提示符
        self.input_prompt = "\n[bold yellow]User:[/bold yellow] (按Enter换行，输入'\\q'结束，'\\c'清除)\n"

    def _models_ifc(self):
        """判断模型是否支持工具调用"""
        if self.model in ['deepseek-r1']:
            self.config.is_function_calling = False
        else:
            self.config.is_function_calling = True
            
    def choose_model(self):
        """选择要使用的AI模型"""
        
        # 直接读取环境变量
        try:
            # 从环境变量中获取模型列表并处理
            model_list = [model.strip() for model in self.config.model.split(";") if model.strip() and len(model.strip()) > 0]
        except:
            return ValueError("环境变量MODEL格式错误")
        # 渲染状态
        def render_status(message="", style="cyan"):
            # 创建一个更美观的表格
            model_table = Table(
                show_header=True,
                header_style="bold cyan",
                box=ROUNDED,
                expand=True,
                padding=(0, 1)  # 添加一些内边距
            )
            
            # 添加列，设置合适的宽度和对齐方式
            model_table.add_column("序号", justify="center", width=6)
            model_table.add_column("模型名称", justify="left")
            
            # 添加行，使用更好的格式
            for i, model in enumerate(model_list, 1):
                model_table.add_row(
                    f"{i}",
                    model,
                    style="white"
                )

            # 创建面板
            status_panel = Panel(
                Group(
                    model_table,
                    Text("\n" + message, style=style) if message else "",
                ),
                title=f"[bold]🤖 模型选择 [/bold]",
                border_style=style,
                padding=(1, 2)  # 添加面板内边距
            )
            return status_panel
        console=self._console
        with Live(render_status(), console=console, auto_refresh=False) as live:
            render_status_message=""
            while True:
                live.stop()  # 暂停Live显示，光标在Live下方
                try:
                    #防止信息输入到Console上方
                    choice_input = console.input("🔢 请输入您想使用的模型编号: ")
                    live.start()  # 恢复Live显示，光标在Live上方
                    #直接回车则选择默认第一个模型
                    if not choice_input.strip():
                        choice = 1
                    else:
                        choice = int(choice_input)
                    #直接回车则选择默认第一个模型
                    if not choice:
                        choice = 1
                    
                    if 1 <= choice <= len(model_list):
                        self.model = model_list[choice - 1]
                        render_status_message = f"✅ 您已选择模型: {self.model}"
                        live.update(render_status(render_status_message, "green"), refresh=True)
                        console.clear()  #清空屏幕。但保留最后一次的输出在屏幕上
                        #判断模型是否支持工具调用
                        self._models_ifc()
                        #返回选择的模型
                        return None
                    
                    else:
                        render_status_message = f"❌ 请输入1到{len(model_list)}之间的数字"
                        #更新状态消息
                        #console.clear()
                        live.update(render_status(render_status_message, "yellow"), refresh=True)
                        console.clear()
                except ValueError:
                    live.start()  # 确保在显示错误时Live是开启的
                    render_status_message = "❗ 请输入有效的数字"
                    live.update(render_status(render_status_message, "red"), refresh=True)
                    console.clear()
                    

    def get_response(self) -> tuple[list, list]:
        """从OpenAI获取响应并处理工具调用

        Args:
            messages: 对话历史消息列表

        Returns:
            tuple[list, list]: (工具调用列表, 响应消息列表)
        """
        [logging.debug(message) for message in self.messages]
        if self.config.is_function_calling:
            # 计算tool tokens
            encoding = tiktoken.encoding_for_model(self.model)
            name = [json.dumps(tool["function"]["name"]) for available_tool in self.available_tools for tool in available_tool]
            parameters = [json.dumps(tool["function"]["parameters"]) for available_tool in self.available_tools for tool in available_tool]
            description = [json.dumps(tool["function"]["description"]) for available_tool in self.available_tools for tool in available_tool]

            # 每个字符串的token数之和为tool_tokens
            self.tool_tokens.append(sum([len(encoding.encode(n)) for n in name]) + sum([len(encoding.encode(p)) for p in parameters]))
            completion = self.openai.chat.completions.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                messages=self.messages,
                tools=[tool for available_tool in self.available_tools for tool in available_tool],
                stream=self.config.stream,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
        else:
            self.tool_tokens.append(0)
            completion = self.openai.chat.completions.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                messages=self.messages,
                stream=self.config.stream,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
        
        # 处理流式响应
        if self.config.stream:
            return self._handle_stream_response(completion)
        # 处理普通响应
        else:
            return self._handle_normal_response(completion)
    #定义一个函数，负责处理终端的输出，包含思考和响应，并且刷新终端
    def _print_thinking_and_response(self, chunk, live: Live,thinking_content: str="", full_content: str="",spinner: Spinner=None) -> tuple[str,str]:
        """打印思考和响应，并刷新终端
        Args:
            chunk: 响应块/completion
            live: 终端对象
        """
        output_content = ""
        if self.config.stream:
            # 更新内容
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                thinking_content += delta.reasoning_content
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content
            
            title_content = f"{self.model}"
            
            # 分别处理思考内容和响应内容
            if thinking_content:
                if spinner:
                    spinner.text = "🤔 思考中... | Thinking..."
                output_content += f"# Thinking\n\n```markdown\n{thinking_content}\n```\n\n"
            
            if full_content:
                if spinner:
                    spinner.text = "💬 正在输出... | Outputting..."
                output_content += f"# Response\n\n{full_content}"
            
            # 只有在有内容时才更新显示
            if output_content:
                if spinner:
                    panel = Panel(
                        Group(spinner, Markdown(output_content)),
                        title=title_content
                    )
                else:
                    panel = Panel(
                        Markdown(output_content),
                        title=title_content
                    )
                live.update(panel, refresh=True)
            return thinking_content, full_content

        else:
            #获取思考和响应内容
            thinking_content =chunk.choices[0].message.reasoning_content if hasattr(chunk.choices[0].message, 'reasoning_content') and chunk.choices[0].message.reasoning_content else ""
            full_content = chunk.choices[0].message.content if hasattr(chunk.choices[0].message, 'content') and chunk.choices[0].message.content else ""
            #打印token数量
            title_content = f"{self.model}"
            if thinking_content != "":
                output_content +=  f"# Thinking...\n\n```markdown\n\n"+thinking_content+ "\n```\n"
            if full_content != "":
                output_content += f"# Response\n\n"+full_content
            # 使用Panel显示内容，添加token信息作为subtitle
            live.update(Panel(
                Markdown(output_content),
                #subtitle=f"[dim]{token_info}[/dim]",
                title=title_content
            ), refresh=True)

        return thinking_content, full_content

    def _handle_stream_response(self, completion) -> tuple[list, list]:
        """处理流式响应

        Args:
            completion: OpenAI的流式响应对象

        Returns:
            tuple[list, list]: (工具调用列表, 响应消息列表)
        """
        tool_functions_index = {}
        thinking_content=""
        full_content =  ""
        current_index=0
        #遍历响应块
        # 创建spinner
        spinner = Spinner('dots', text='💬 正在输出中 | Outputting...')
        with Live(console=self._console, auto_refresh=False,vertical_overflow="ellipsis") as live:
            for chunk in completion:
                logging.debug(chunk)
                #处理工具调用
                if tool_calls := chunk.choices[0].delta.tool_calls:
                    #流式响应tool_calls列表只包含一个对象，获取工具名称和参数
                    full_content,current_index = self._process_stream_tool_call(chunk, tool_calls[0], tool_functions_index, full_content, current_index, live,spinner)
                    
                #没有工具调用，直接打印内容
                elif chunk.choices[0].delta.content or (hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content):
                    thinking_content,full_content = self._process_stream_content(chunk,thinking_content,full_content, live,spinner)
            #最后更新spinner
            spinner.text = "✨ 输出完成 | Output completed ✅"
            tool_functions = [v for k, v in tool_functions_index.items()]
            #如果有思维，则计算思维token
            if thinking_content:
                encoding = tiktoken.encoding_for_model(self.model)
                self.thinking_tokens += len(encoding.encode(thinking_content))
        return tool_functions, self._create_response(full_content, tool_functions)

    def _process_stream_tool_call(self, chunk, tool_call, tool_functions_index: dict, full_content: str, current_index: int, live: Live,spinner: Spinner) -> str:
        """处理OpenAI流式响应中的工具调用。OpenAI的工具调用响应是分块发送的，主要有两种类型：

        1. 新工具调用的开始：
           - 包含完整的调用信息（id, name, type等）
           - id不为空，表示这是一个新的工具调用
           示例：
           {
               "index": 0,
               "id": "call_xxx",
               "function": {"arguments": "", "name": "get_weather"},
               "type": "function"
           }

        2. 工具参数的持续传输：
           - id为空，表示这是当前工具调用的参数内容
           - 参数内容会分多次发送，需要拼接
           示例：
           {"index": 0, "id": null, "function": {"arguments": "{\\"city\\": \\"", "name": null}}
           {"index": 0, "id": null, "function": {"arguments": "Beijing\\"}", "name": null}}

        实现逻辑：
        1. 使用current_index记录当前正在处理的工具调用
        2. 当收到id不为空的chunk时，创建新的工具调用记录
        3. 当收到id为空的chunk时，将内容追加到current_index对应的工具调用中
        4. 不用index来区分不同的工具调用是因为claude等其他模型的index始终为空
        Args:
            chunk: 响应块
            tool_call: 工具调用对象
            tool_functions_index: 工具调用索引字典
            content: 当前累积的内容
            current_index: 当前索引

        Returns:
            str: 更新后的内容
        """
        #如果调用工具时内容不为空，打印并累加
        if chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
            
            output_content=f"# Response\n\n"+full_content
            if spinner:
                spinner.text = "💬 正在输出... | Outputting..."
                display_content = Panel(
                    Group(spinner, Markdown(output_content)),
                    #subtitle=f"[dim]{token_info}[/dim]"
                )
                live.update(display_content, refresh=True)
            else:
                live.update(Panel(
                    Markdown(output_content),
                    #subtitle=f"[dim]{token_info}[/dim]"
                ), refresh=True)
            logging.debug("既有有tool_calls字段，又有content字段。")
        else:
            if spinner:
                spinner.text = "🔧 正在准备工具调用中... | Preparing tool calls..."
                output_content=f"# Response\n\n"+full_content
                display_content = Panel(
                    Group(spinner, Markdown(output_content)),
                    #subtitle=f"[dim]{token_info}[/dim]"
                )
                live.update(display_content, refresh=True)
            else:
                live.update(Panel(
                    Markdown(full_content+"\n\n # 正在准备工具调用，请稍后..."),
                    #subtitle=f"[dim]{token_info}[/dim]"
                ), refresh=True)
            logging.debug("只有tool_calls字段。")
           
    
        #获取工具名称和参数
        tool_name = tool_call.function.name if tool_call.function.name else ""
        tool_args = tool_call.function.arguments if tool_call.function.arguments else ""
        #如果工具调用ID为空，说明是旧的调用
        if not tool_call.id:
            tool_functions_index[current_index]["function"]["name"] += tool_name
            tool_functions_index[current_index]["function"]["arguments"] += tool_args
            
        #如果工具调用ID不为空，说明是新的调用
        else:
            #更新索引
            current_index += 1
            tool_functions_index[current_index] = {
                "id": tool_call.id or "",
                "type": tool_call.type or "",
                "function": {"name": tool_name, 
                             "arguments": tool_args}
            }
        return full_content,current_index
    

    def _process_stream_content(self, chunk, thinking_content: str, full_content: str, live: Live,spinner: Spinner) -> str:
        """处理流式响应中的普通内容
            响应块的格式如下
            ChatCompletionChunk(id='chatcmpl-BiDxhaqknhcZCUs1A22bV0U9972719HG', 
                                choices=[Choice(delta=ChoiceDelta(content='***', function_call=None, refusal=None, role='assistant', tool_calls=None), 
                                         finish_reason=None, index=0, logprobs=None)], 
                                created=1740117256, model='gpt-4o-2024-08-06', 
                                object='chat.completion.chunk', 
                                service_tier=None, 
                                system_fingerprint='', 
                                usage=None)
        Args:
            chunk: 响应块
            content: 当前累积的内容

        Returns:
            str: 更新后的内容
        """
        logging.debug("没有tool_calls字段，有content字段:")
        logging.debug(chunk)
        # 更新内容
        thinking_content, full_content = self._print_thinking_and_response(chunk, live, thinking_content, full_content,spinner)
        
        return thinking_content,full_content

    def _handle_normal_response(self, completion) -> tuple[list, list]:
        """处理普通（非流式）响应
        Args:
            completion: OpenAI的响应对象

        Returns:
            tuple[list, list]: (工具调用列表, 响应消息列表)
        """
        full_content = ""
        tool_functions = []
        logging.debug(completion)
        # 如果有工具调用
        if tool_calls := completion.choices[0].message.tool_calls:
            #如果调用工具时候还有文本内容，则打印
            if completion.choices[0].message.content:
                with Live(console=self._console, auto_refresh=False) as live:
                    thinking_content, full_content = self._print_thinking_and_response(completion, live)
            #遍历工具调用
            for tool_call in tool_calls:
                tool_functions.append({
                    #"index": tool_call.index,
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
                
        # 没有工具调用
        else:
            with Live(console=self._console, auto_refresh=False) as live:
                thinking_content, full_content = self._print_thinking_and_response(completion, live)
        
        return tool_functions, self._create_response(full_content, tool_functions)

    def _create_response(self, full_content: str, tool_functions: list) -> list:
        """创建响应消息列表

        Args:
            content: 响应内容
            tool_functions: 工具调用列表

        Returns:
            list: 响应消息列表
        """
        if not tool_functions:
            return [{
                "role": "assistant",
                "content": full_content,
            }]
        
        response = []
        for tool_function in tool_functions:
            response.append({
                "role": "assistant",
                "content": full_content or f"Tool Function Called: {tool_function['function']['name']}\nArguments: {tool_function['function']['arguments'][:300]}{'...' if len(tool_function['function']['arguments']) > 200 else ''}", # Claude requires non-empty content
                "function_call": {
                    "name": tool_function["function"]["name"],
                    "arguments": tool_function["function"]["arguments"]
                }
            })
        return response

    async def get_tools(self):
        for session in self.session_list:
            # 列出可用工具
            available_tools = [] 
            response = await session.list_tools()
            for tool in response.tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            self.available_tools.append(available_tools)
        
        if not self.available_tools:
            self._console.print(Panel("[yellow]警告: 没有找到任何可用工具。[/yellow]"))
        else:
            total_tools = sum(len(tools) for tools in self.available_tools)
            self._console.print(Panel(f"[green]成功获取到 {len(self.session_list)} 个服务器，总计 {total_tools} 个工具。[/green]"))
    async def connect_to_server(self):
        """连接到 MCP 服务器
    
        参数:
            server_script_path: 服务器脚本路径 (.py 或 .js)
        """
        if self.config.mcp_server_config:
            try:
                # 解析配置文件
                servers_info = parse_mcp_servers(self.config.mcp_server_config)
                # 获取所有服务器名称
                server_names = get_all_server_names(servers_info)
                #print(f"服务器列表: {server_names}")
                
                # 遍历每个服务器，为每个服务器创建单独的连接
                for server_name in server_names:
                    command = get_server_command(servers_info, server_name)
                    args = get_server_args(servers_info, server_name)
                    env = get_server_env(servers_info, server_name)  
                    
                    # 检查命令是否为 npx
                    if command.lower() == "npx":
                        command = self.config.npx_path
                        # 如果是 npx 命令，检查是否需要连接到网络
                        if any("@modelcontextprotocol" in arg for arg in args):
                            # 这是一个需要从网络下载的包，可能会遇到网络问题
                            self._console.print(Panel(
                                Markdown(f"🌐 **正在从网络下载并安装服务: {server_name}**\n\n请耐心等待，这可能需要一些时间..."),
                                title="[bold blue]远程安装进行中[/bold blue]",
                                border_style="blue",
                                expand=False
                            ))
                            self._console.print(f"[dim]命令: {command} {' '.join(args)}[/dim]")
                    elif command.lower() == "python":
                        command = self.config.python_path
                        self._console.print(f"[dim]命令: {command} {' '.join(args)}[/dim]")
                    elif command.lower() == "node":
                        command = self.config.node_path
                        self._console.print(f"[dim]命令: {command} {' '.join(args)}[/dim]")
                    else:
                        # 对于其他命令，也使用原始字符串
                        command = rf"{command}"
                        self._console.print(f"[dim]命令: {command} {' '.join(args)}[/dim]")

                    # 创建 StdioServerParameters 对象
                    server_params = StdioServerParameters(
                        command=command,
                        args=args,
                        env={
                            **os.environ.copy(),  # 复制当前环境变量
                            # 确保关键路径配置被正确传递
                            "PYTHON_PATH": self.config.python_path,
                            "NODE_PATH": self.config.node_path,
                            "NPX_PATH": self.config.npx_path
                        }
                    )
                
                    try:
                        # 使用 stdio_client 创建与服务器的 stdio 传输
                        studiocilent=stdio_client(server_params)
                        stdio_transport = await self.exit_stack.enter_async_context(studiocilent)
                        # 解包 stdio_transport，获取读取和写入句柄
                        stdio, write = stdio_transport
                        # 创建 ClientSession 对象，用于与服务器通信
                        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                        # 初始化会话
                        await session.initialize()
                        self.session_list.append(session)
                        # 列出可用工具
                        response = await session.list_tools()
                        tools = response.tools
                        # 构建工具列表字符串
                        tools_list = []
                        for tool in tools:
                            tool_str = f"- **{tool.name}**:\n  ```\n  {tool.description}\n  ```"
                            tools_list.append(tool_str)
                        tools_text = "\n".join(tools_list)

                        self._console.print(Panel(
                            Markdown(f"""
# 服务器连接成功 ✅

## 服务器信息
- **名称**: {server_name}
- **状态**: 在线 🟢

## 可用工具 🛠️
{tools_text}
                            """),
                            title="[green]服务器连接状态[/green]",
                            border_style="green"
                        ))
                        self.server_names.append(server_name)
                    except FileNotFoundError as e:
                        self._console.print(Panel(
                            Markdown(f"❌ 找不到文件: `{str(e)}`\n\n请检查命令和参数是否正确。"),
                            title=f"[bold red]服务器 {server_name} 启动失败[/bold red]",
                            border_style="red",
                            expand=False
                        ))
                    except Exception as e:
                        self._console.print(Panel(
                            Markdown(f"❌ 连接失败: `{str(e)}`"),
                            title=f"[bold red]服务器 {server_name} 启动失败[/bold red]",
                            border_style="red",
                            expand=False
                        ))
            except Exception as e:
                self._console.print(Panel(
                    Markdown(f"❌ MCP配置解析失败: `{str(e)}`\n\n将以无服务器模式运行。"),
                    title="[bold red]配置错误[/bold red]",
                    border_style="red",
                    padding=(1, 2)
                ))

    async def process_query(self, query: str) -> str:
        """使用 OpenAI 和可用工具处理查询"""
        
        # 创建消息列表
        self.messages.append({"role": "user", "content": query})

        # 处理消息
        tool_functions,response = self.get_response()
        # 如果有工具调用
        while tool_functions:
            self.messages.extend(response)
            #迭代调用工具·
            for tool_function in tool_functions:
                tool_call_id = tool_function["id"]
                tool_name = tool_function["function"]["name"]
                logging.debug(f"工具函数原始数据: {tool_function}")
                logging.debug(f"工具函数参数: {tool_function['function'].get('arguments', '')}")
                
                try:
                    tool_args = json.loads(tool_function["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    print(f"原始参数字符串: '{tool_function['function'].get('arguments', '')}'")
                    tool_args = {}
                # 处理参数显示
                args_str = json.dumps(tool_args, ensure_ascii=False, indent=2)
                if len(args_str) > 1000:
                    args_str = args_str[:1000] + "..."

                self._console.print(Panel(
                    Markdown(f"**Tool Call**\n\n调用工具: {tool_name}\n\n参数:\n\n```json\n{args_str}\n```"),
                    title="工具调用",
                    expand=False
                ))
                    
                # 执行工具调用结果
                tool_to_session = {tool["function"]["name"]: idx for idx, tools in enumerate(self.available_tools) for tool in tools}
                session_idx = tool_to_session.get(tool_name)
                if session_idx is not None:
                    session = self.session_list[session_idx]

                    if self.config.is_human_control:
                        # 如果有人工控制，则需要人工确认(回车默认确认，否则输入的是拒绝理由)
                        #用户确认洁面
                        is_confirm = self._console.input("是否确认调用工具? (回车确认|拒绝理由): ")
                    
                        if is_confirm == "":
                            result = await session.call_tool(tool_name, tool_args)
                        else:
                            
                            self._console.print(Panel(
                                Markdown(f"❌ 工具调用已取消\n\n**拒绝理由**:\n> {is_confirm}"),
                                title="[red]工具调用取消[/red]",
                                border_style="red",
                                expand=False
                            ))
                            # 递归调用process_query，重新获取工具调用
                            query = f"我拒绝你申请的调用工具:{tool_name},参数:{args_str[:200]} ；理由是：{is_confirm}"
                            return await self.process_query(query)

                    else:
                        result = await session.call_tool(tool_name, tool_args)
                    logging.debug(f"工具调用完整结果: {result}")
                else:
                    raise ValueError(f"Tool '{tool_name}' not found in available tools")
                self._console.print(Panel(
                    Markdown("**Result**\n\n" + "\n\n".join(content.text for content in result.content) + f"\n\nMeta: {str(result.meta)}\nIsError: {str(result.isError)}"),
                    title="工具调用结果",
                    expand=False
                ))
                
                #将工具调用结果添加到消息
                if not self.model.startswith("gpt"):
                    self.messages.append(
                    {
                        "role": "user", # gpt可以用function
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": "Tool call result:\n"+str(
                            {
                                "type": "tool_result",
                                "tool_name": tool_name,
                                "tool_use_id": tool_call_id,
                                "result": [content.text for content in result.content],
                                "meta": str(result.meta),
                                "isError": str(result.isError)
                            })
                        }
                    )
                    
                    
                else:
                    self.messages.append({
                        "role": "user",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": str(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "result": [content.text for content in result.content],
                                "meta": str(result.meta),
                                "isError": str(result.isError)
                            })
                        }
                    )
            # 将调用结果返回给LLM，获取回复
            tool_functions,response = self.get_response()
            
        # 最终响应    
        self.messages.extend(response)
        return self.messages
        
    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose() 

    def save_messages(self):
        """保存对话消息到文件"""
        import time
        timestamp = int(time.time())
        filename = f"messages_{timestamp}.json"
        save_path = self.config.messages_path
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=4)

        return filename

    def manage_message_history(self):
        if self.config.max_messages_tokens > 0:
            # Trim message history if it exceeds the maximum length
            if len(self.messages) > self.config.max_messages:
                self.messages = self.messages[-self.config.max_messages:]
            
            # Ensure system prompt is at the beginning of the message history
            if self.config.system_prompt and (not self.messages or self.messages[0]["role"] != "system"):
                # 检查系统提示文件是否存在
                if os.path.exists(self.config.system_prompt):
                    try:
                        with open(self.config.system_prompt, "r", encoding="utf-8") as f:
                            system_prompt = f.read()
                        logging.debug(f"成功从文件加载系统提示: {self.config.system_prompt}")
                    except Exception as e:
                        logging.error(f"读取系统提示文件出错: {e}")
                        system_prompt = "You are a helpful assistant."
                else:
                    # 如果文件不存在，使用默认提示
                    logging.warning(f"系统提示文件不存在: {self.config.system_prompt}")
                    system_prompt = "You are a helpful assistant."
                self.messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            # 如果设置小于等于0，则不限制消息长度
            pass
    
    async def handle_command(self, command: str):
        if command == 'quit' or command == 'exit':
            return 'exit'
        elif command == 'fc':
            self.config.is_function_calling = not self.config.is_function_calling
            status = "启动" if self.config.is_function_calling else "取消"
            # 使用主控制台显示状态更新
            self._console.print(Panel(
                Markdown(f"🛠️ 工具调用功能已{status}"),
                title="[bold green]功能状态更新[/bold green]",
                border_style="green"
            ))
            return 'fc'
        elif command == 'model':
            self.choose_model()
            return 'model'
        elif command == 'save':
            filename = self.save_messages()
            self._console.print(Panel(
                Markdown(f"📁 消息历史已成功保存\n\n📄 文件名: `{filename}`"),
                title="[bold blue]保存成功[/bold blue]",
                border_style="cyan"
            ))
            return 'save'
        elif command == 'human':
            self.config.is_human_control = not self.config.is_human_control
            status = "启动" if self.config.is_human_control else "取消"
            self._console.print(Panel(
                Markdown(f"👤 人类干预已{status}"),
                title="[bold green]功能状态更新[/bold green]",
                border_style="green"
            ))
            return 'human'
        elif command.startswith("compact "):
            # 获取用户指定的字符数限制,默认200
            parts = command[8:].strip().split(" ")
            char_limit = 200  # 默认值
            if len(parts) > 0:
                try:
                    char_limit = int(parts[0])
                    if char_limit <= 0:
                        raise ValueError("字符数限制必须大于0")
                except ValueError as e:
                    self._console.print(Panel(
                        f"[red]错误: {str(e)}\n使用默认值200[/red]",
                        title="[bold red]参数错误[/bold red]",
                        border_style="red"
                    ))
                    char_limit = 200
            
            # 统计压缩信息
            compressed_count = 0
            total_saved = 0
            
            # 压缩消息历史
            for message in self.messages:
                # 检查是否有content字段
                if "content" not in message:
                    continue
                    
                content_length = len(message["content"])
                if content_length > char_limit:
                    original_length = len(message["content"])
                    message["content"] = message["content"][:char_limit] + "..."
                    compressed_count += 1
                    total_saved += original_length - len(message["content"])
            
            # 显示详细的压缩结果
            self._console.print(Panel(
                Markdown(f"""
# 📝 消息历史压缩完成

- 字符数限制: `{char_limit}`
- 压缩消息数: `{compressed_count}`
- 节省字符数: `{total_saved}`
- 压缩比例: `{compressed_count/len(self.messages)*100:.1f}%`
                """),
                title="[bold green]压缩成功[/bold green]",
                border_style="green"
            ))
            return 'compact'
        
        elif command.startswith('cost'):
            self.total_input_tokens,self.total_output_tokens,self.current_input_tokens,self.round_count=get_token_count(self.messages,self.model)
            self.total_input_tokens += sum(self.tool_tokens)
            if self.config.is_function_calling :
                # 计算tool tokens
                encoding = tiktoken.encoding_for_model(self.model)
                name=[json.dumps(tool["function"]["name"]) for available_tool in self.available_tools for tool in available_tool]
                parameters=[json.dumps(tool["function"]["parameters"]) for available_tool in self.available_tools for tool in available_tool]
                description=[json.dumps(tool["function"]["description"]) for available_tool in self.available_tools for tool in available_tool]
                #每个字符串的token数之和为tool_tokens
                tool_tokens=sum([len(encoding.encode(n)) for n in name])+sum([len(encoding.encode(p)) for p in parameters])
                self.current_input_tokens += tool_tokens
            self._console.print(Panel(
                Markdown(f"""
# 📊 Token 使用统计报告

- 🔹 输入消耗总计: `{self.total_input_tokens}` Tokens
- 🔸 输出消耗总计: `{self.total_output_tokens+self.thinking_tokens}` Tokens
- 💭 思维链消耗: `{self.thinking_tokens}` Tokens  
- ⏳ 下轮预估消耗: `{self.current_input_tokens}` Tokens
- 🔁 当前对话轮次: `{self.round_count}`
"""),
                title="[bold green]Token消耗统计[/bold green]",
                border_style="green"
            ))
            return 'cost'
        elif command.startswith('load '):
            file_path = command[5:].strip()
            try:
                file_path = os.path.expanduser(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_messages = json.load(f)
                self.messages = loaded_messages
                self._console.print(Panel(
                    Markdown(f"📂 消息历史已成功加载\n\n📄 文件路径: `{file_path}`"),
                    title="[bold blue]加载成功[/bold blue]",
                    border_style="cyan"
                ))
        
            except FileNotFoundError:
                self._console.print(f"[bold red]错误：文件 '{file_path}' 不存在[/bold red]")
            except json.JSONDecodeError:
                self._console.print(f"[bold red]错误：文件 '{file_path}' 不是有效的 JSON 格式[/bold red]")
            except Exception as e:
                self._console.print(f"[bold red]加载文件时出错：{str(e)}[/bold red]")
            return 'load'
        elif command.startswith('mcp '):
            try:
                parts = command[4:].strip().split(" ")
                if not parts or not parts[0]:
                    self._console.print(Panel(
                        Markdown("请提供MCP配置文件路径"),
                        title="[bold yellow]提示[/bold yellow]",
                        border_style="yellow"
                    ))
                    return 'mcp_error'
                
                new_mcp_config_path = parts[0]
                if not os.path.exists(new_mcp_config_path):
                    self._console.print(Panel(
                        Markdown(f"配置文件 '{new_mcp_config_path}' 不存在"),
                        title="[bold red]错误[/bold red]",
                        border_style="red"
                    ))
                    return 'mcp_error'
                
                self._console.print(Panel(
                    Markdown(f"正在切换新的MCP配置文件: {new_mcp_config_path}"),
                    title="[bold green]切换配置文件[/bold green]",
                    border_style="green"
                ))
                
                # 清理旧的资源
                await self.cleanup()
                
                # 重置会话相关的属性
                self.server_names = []
                self.session_list = []
                self.available_tools = []
                self.exit_stack = AsyncExitStack()  # 创建新的 exit_stack
                
                # 更新配置
                self.config.mcp_server_config = new_mcp_config_path
                
                try:
                    # 重新连接服务器
                    await self.connect_to_server()
                    # 获取新的工具列表
                    await self.get_tools()
                    
                    self._console.print(Panel(
                        Markdown("✅ MCP配置切换完成"),
                        title="[bold green]成功[/bold green]",
                        border_style="green"
                    ))
                    return 'mcp'
                except Exception as e:
                    self._console.print(Panel(
                        Markdown(f"服务器连接失败：\n\n```\n{str(e)}\n```"),
                        title="[bold red]错误[/bold red]",
                        border_style="red"
                    ))
                    return 'mcp_error'
                    
            except Exception as e:
                import traceback
                self._console.print(Panel(
                    Markdown(f"切换MCP配置文件时发生错误：\n\n```\n{str(e)}\n{traceback.format_exc()}\n```"),
                    title="[bold red]错误[/bold red]",
                    border_style="red"
                ))
                return 'mcp_error'
        

        
        elif command == "help":
            self._console.print(Panel(
                Markdown(self.get_help_text()),
                title="[bold cyan]🔍 命令帮助中心[/bold cyan]",
                subtitle="[dim]输入 \\ 可查看简洁命令列表[/dim]",
                border_style="cyan",
                padding=(1, 2)
            ))
            return 'help'
        elif command == "clear":
            # 清空历史消息
            self.messages = []
            self._console.print(Panel(
                "🧹 对话历史已清空",
                title="[bold green]操作成功[/bold green]",
                border_style="green"
            ))
            return 'clear'
        elif command == "debug":
            self.config.debug = not self.config.debug
            logging.basicConfig(level=logging.DEBUG if self.config.debug else logging.ERROR)
            self._console.print(Panel(
                f"调试模式{'开启' if self.config.debug else '关闭'}",
                title="[bold blue]操作成功[/bold blue]",
                border_style="cyan"
            ))
            return 'debug'
        else:
            self._console.print(Panel(
                f"未知命令：\\{command}\n使用 \\help 查看可用命令",
                title="[bold yellow]提示[/bold yellow]",
                border_style="yellow"
            ))
            return 'help'
            
    def get_multiline_input(self) -> str:
        """获取多行输入
        
        支持以下特性：
        1. 多行输入，按Enter继续输入
        2. 输入 \\q 单独一行来结束输入
        3. 输入 \\c 单独一行来清除当前输入
        4. 空行会被保留
        5. 命令行（以\\开头）会直接执行，不需要\\q
        6. 输入\\时会显示命令提示
        
        Returns:
            str: 用户输入的文本
        """
        # 初始化readline
        try:
            # 定义一个空的补全函数
            def empty_completer(text, state):
                return None
                
            # 完全禁用readline的补全功能
            readline.set_completer(empty_completer)
            readline.parse_and_bind('bind ^I rl_complete')  # 禁用tab补全
            readline.parse_and_bind('set disable-completion on')  # 禁用所有补全
            readline.parse_and_bind('set show-all-if-ambiguous off')  # 禁用模糊匹配显示
            readline.parse_and_bind('set show-all-if-unmodified off')  # 禁用未修改时的显示
            readline.parse_and_bind('set completion-ignore-case off')  # 禁用大小写忽略
            readline.parse_and_bind('set completion-query-items 0')  # 禁用补全项查询
        except Exception as e:
            logging.warning(f"readline初始化失败: {e}")
            
        self._console.print(self.input_prompt, end="")
        lines = []
        while True:
            try:
                # 使用readline获取输入
                line = input().strip()
                # 处理命令（以\开头）
                if line.startswith('\\'):
                    if len(lines) == 0:  # 只有在第一行时才处理命令
                        # 如果只输入了\，显示命令提示
                        if line == '\\':
                            self._show_command_suggestions()
                            self._console.print(self.input_prompt, end="")
                            continue
                        return line
                    elif line == r'\q':  # 使用原始字符串来判断
                        # 显示输入完成的提示
                        self._console.print(Panel(
                            Group(
                                Spinner('dots', text='正在处理您的输入...', style="cyan"),
                                Text("\n输入已完成，共 {} 行".format(len(lines)), style="dim")
                            ),
                            title="[bold green]✨ 输入完成[/bold green]",
                            border_style="green",
                            padding=(1, 2)
                        ))
                        break
                    elif line == r'\c':  # 使用原始字符串来判断
                        lines = []
                        self._console.print("[yellow]已清除当前输入[/yellow]")
                        self._console.print(self.input_prompt, end="")
                        continue
                    else:  # 其他\开头的内容作为普通文本处理
                        lines.append(line)
                else:
                    lines.append(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                self._console.print("\n[yellow]已取消当前输入[/yellow]")
                return ""
            
        # 合并所有行
        return '\n'.join(lines)

    def _show_command_suggestions(self):
        """显示命令提示"""
        # 定义命令分类
        command_categories = {
            "基础命令": [
                ("", "显示此命令列表"),
                ("quit/exit", "退出系统"),
                ("clear", "清空当前会话历史"),
                ("help", "显示详细帮助信息")
            ],
            "模型与工具": [
                ("model", "切换语言模型"),
                ("fc", "开启/关闭工具调用"),
                ("human", "开启/关闭人类干预"),
                ("mcp <路径>", "切换MCP配置文件")
            ],
            "会话管理": [
                ("save", "保存当前会话"),
                ("compact <字符数>", "压缩消息历史"),
                ("load <路径>", "加载历史会话")
            ],
            "统计与调试": [
                ("cost", "显示Token统计"),
                ("debug", "切换调试模式")
            ]
        }
        
        # 创建表格
        table = Table(
            title="快速命令参考",
            box=ROUNDED,
            expand=False,
            padding=(0, 1),
            border_style="cyan",
            highlight=True
        )
        
        # 添加列
        table.add_column("分类", style="bold cyan", justify="left", no_wrap=True)
        table.add_column("命令", style="green", justify="left")
        table.add_column("描述", style="white", justify="left")
        
        # 添加行
        for category, commands in command_categories.items():
            for i, (cmd, desc) in enumerate(commands):
                if i == 0:
                    # 第一行显示分类
                    table.add_row(
                        f"[bold]{category}[/bold]", 
                        f"[cyan]\\{cmd}[/cyan]" if cmd else "[cyan]\\[/cyan]", 
                        desc
                    )
                else:
                    # 后续行不显示分类
                    table.add_row(
                        "", 
                        f"[cyan]\\{cmd}[/cyan]", 
                        desc
                    )
        
        # 显示表格
        self._console.print(Panel(
            Group(
                table,
                Text("\n💡 提示: 输入 [cyan]\\help[/cyan] 获取更详细的命令说明", style="dim")
            ),
            title="[bold cyan]🔍 可用命令[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))

    async def chat_loop(self):
        """运行交互式聊天循环"""
        server_message = "## 🖥️ 已连接服务器\n" + "\n".join(f"- `{path}`" for path in self.server_names)
        
        # ASCII艺术标志
        ascii_logo = """
   █████╗  ██████╗████████╗██╗ ██████╗ ███╗   ██╗ █████╗ ██╗
  ██╔══██╗██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔══██╗██║
  ███████║██║        ██║   ██║██║   ██║██╔██╗ ██║███████║██║
  ██╔══██║██║        ██║   ██║██║   ██║██║╚██╗██║██╔══██║██║
  ██║  ██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║██║  ██║██║
  ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝
"""
        
        # 创建欢迎界面组件
        welcome_components = [
            Text(ascii_logo, style="cyan bold"),
            Text(""),
            Text("         智能AI助手 | 强大工具链 | 多模型支持", style="green bold"),
            Text(""),
            Text("✨ 输入 \\ 查看所有可用命令", style="yellow"),
            Text("📚 输入 \\help 获取详细帮助", style="yellow"),
            Text("🔄 输入 \\model 切换AI模型", style="yellow")
        ]
        
        # 如果有服务器连接，添加服务器信息
        if self.server_names:
            welcome_components.extend([
                Text(""),
                Markdown(server_message)
            ])
        
        # 显示欢迎界面
        self._console.print(Panel(
            Group(*welcome_components),
            title="[bold cyan]ActionAI 智能助手[/bold cyan]",
            subtitle="[dim]输入问题开始对话，输入 \\ 查看命令[/dim]",
            border_style="cyan",
            padding=(1, 2)
        ))
        
        while True:
            try:
                
                # 先获取用户输入
                query = self.get_multiline_input().strip()
                
                # 如果直接回车，则跳过
                if not query:
                    continue
                    
                # 处理所有\开头的命令
                if query.startswith('\\'):
                    command = query[1:].strip().lower()
                    result = await self.handle_command(command)
                    if result == 'exit':
                        break
                    # 其余命令跳过执行问答
                    continue
                
                # 在处理实际对话之前检查token限制
                self.manage_message_history()
                self.total_input_tokens, self.total_output_tokens, self.current_input_tokens, self.round_count = get_token_count(self.messages, self.model)
                # 如果此次对话是工具调用，则需要计算tool tokens
                if self.config.is_function_calling:
                    encoding = tiktoken.encoding_for_model(self.model)
                    name=[json.dumps(tool["function"]["name"]) for available_tool in self.available_tools for tool in available_tool]
                    parameters=[json.dumps(tool["function"]["parameters"]) for available_tool in self.available_tools for tool in available_tool]
                    description=[json.dumps(tool["function"]["description"]) for available_tool in self.available_tools for tool in available_tool]
                    #每个字符串的token数之和为tool_tokens
                    tool_tokens=sum([len(encoding.encode(n)) for n in name])+sum([len(encoding.encode(p)) for p in parameters])
                    self.current_input_tokens += tool_tokens
                    
                #计算query的token数
                encoding = tiktoken.encoding_for_model(self.model)
                query_tokens = encoding.encode(query)
                self.current_input_tokens += len(query_tokens)
                
                if self.current_input_tokens > self.config.max_messages_tokens:
                    self._console.print(Panel(
                        Markdown("""
                        # ⚠️ 对话长度超出限制
                        
                        当前Token数：{current}
                        最大限制：{max}
                        
                        建议操作：
                        1. 使用 `\\clear` 清空对话历史
                        2. 使用 `\\compact` 压缩历史消息
                        3. 使用 `\\save` 保存当前对话后再清空
                        """.format(
                            current=self.current_input_tokens,
                            max=self.config.max_messages_tokens
                        )),
                        title="[bold red]警告[/bold red]",
                        border_style="red"
                    ))
                    continue
                    
                # 处理正常的对话
                await self.process_query(query)
                    
            except Exception as e:
                self._console.print(f"[bold red]错误: {str(e)}[/bold red]")
    
    def get_help_text(self):
        """返回帮助信息文本"""
        return """
# 🚀 ActionAI 命令指南

## 📋 基础命令

| 命令 | 描述 | 示例 |
|------|------|------|
| `\\` | 显示简洁命令列表 | `\\` |
| `\\quit` 或 `\\exit` | 退出系统 | `\\quit` |
| `\\clear` | 清空当前会话历史 | `\\clear` |
| `\\help` | 显示此帮助信息 | `\\help` |

## 🤖 模型与工具控制

| 命令 | 描述 | 示例 |
|------|------|------|
| `\\model` | 切换语言模型 | `\\model` |
| `\\fc` | 开启/关闭工具调用 | `\\fc` |
| `\\human` | 开启/关闭人类干预模式 | `\\human` |
| `\\mcp <配置文件路径>` | 切换MCP配置文件 | `\\mcp ./config.json` |

## 💾 会话管理

| 命令 | 描述 | 示例 |
|------|------|------|
| `\\save` | 保存当前会话到文件 | `\\save` |
| `\\load <文件路径>` | 加载历史会话 | `\\load ./messages_1234567890.json` |
| `\\compact <字符数>` | 压缩消息历史 | `\\compact 200` |

## 📊 统计与调试

| 命令 | 描述 | 示例 |
|------|------|------|
| `\\cost` | 显示Token使用统计 | `\\cost` |
| `\\debug` | 切换调试模式 | `\\debug` |

## 💡 输入技巧

- 多行输入：按Enter继续输入
- 结束输入：输入 `\\q` 单独一行
- 清除输入：输入 `\\c` 单独一行
"""

async def main():
    """
    主函数
    """
    # 创建客户端实例
    config = load_config_from_env(base_path)
    client = LLM_Client(config)
    #选择模型
    client.choose_model()
    try:
        #连接服务器
        await client.connect_to_server()
        # 列出可用工具
        await client.get_tools()
        # 运行聊天循环
        await client.chat_loop()
    finally:
        # 确保在任何情况下都清理资源
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
