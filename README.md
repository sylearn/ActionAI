# <div align="center">🚀 ActionAI</div>

<div align="center">

![版本](https://img.shields.io/badge/版本-0.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![许可证](https://img.shields.io/badge/许可证-MIT-blue)

**强大的大语言模型交互框架，支持多模型实时切换和MCP协议**

</div>

## 📖 简介

ActionAI 是一个基于语言大模型的命令行交互框架，支持多种模型切换、MCP协议，提供流畅的交互式聊天体验。通过终端界面，您可以与AI进行自然对话，并通过强大丰富MCP服务端，让AI具备文件读写、网络访问、代码执行等能力。

**🔥 自动化能力：** 只需简单配置MCP协议，ActionAI即可赋予AI强大的系统操作能力，包括文件管理、应用控制、文档编辑等。AI可以在几乎不需要人类干预的情况下，自动完成复杂任务流程，大幅提升工作效率。

## ✨ 核心特性

| 特性 | 描述 |
|:------:|:------|
| 🔌 | **MCP协议** - 无缝集成外部工具 |
| 🎭 | **多模型** - 动态切换不同大语言模型 |
| 💾 | **会话管理** - 保存、加载和管理对话历史 |
| 📊 | **Token计数** - 实时监控使用情况 |
| ⚙️ | **高度可配置** - 灵活调整系统行为 |
| 🤖 | **自动化执行** - 基于MCP服务器，AI可自主完成复杂任务流程 |

## 🔧 快速开始

### 方法一：一键安装（推荐）

下载并运行适合您系统的安装包：

| 操作系统 | 下载链接 |
|---------|---------|
| Windows | [ActionAI](https://github.com/sylearn/ActionAI/releases) |
| macOS | [ActionAI](https://github.com/sylearn/ActionAI/releases) |

安装完成后，直接从应用列表启动ActionAI即可开始使用。

<details>
<summary>Mac 用户额外步骤</summary>
打开终端，进入应用所在目录
执行以下命令使文件可执行：

```shell
chmod +x ./ActionAI
```

如果遇到无法打开提示，[点击这里](https://sysin.org/blog/macos-if-crashes-when-opening/)查看解决方法

</details>

### 方法二：源码安装

如果您希望自定义安装或参与开发，可以通过源码安装：

```bash
# 克隆仓库
git clone https://github.com/sylearn/ActionAI.git
cd ActionAI

# 安装依赖
pip install -r requirements.txt

# 配置环境
cp .env.template .env

# 启动
python ActionAI.py
```

## 🚀 AI自动化示例

通过简单的MCP配置，ActionAI可以实现以下自动化任务：

- **文档处理**：自动整理、分析和生成各类文档
- **代码开发**：从需求分析到编码实现，全流程AI辅助
- **数据分析**：自动收集、清洗、分析数据并生成报告
- **系统管理**：执行系统维护、文件管理等操作
- **内容创作**：自动撰写文章、生成图表、制作演示文稿

只需一句简单的指令，AI就能自主完成一系列复杂操作，无需人工干预每一步骤。

### 自动化工作流程示例

以下是一个完整的自动化工作流程示例，展示ActionAI如何通过简单指令完成复杂任务：

```
用户: 帮我分析最近一周的销售数据，生成报告并发送给团队

ActionAI: 好的，我将为您完成这个任务。

[执行以下步骤]
1. 连接到销售数据库，提取最近一周数据
2. 数据清洗与分析，计算关键指标
3. 生成可视化图表
4. 创建Word报告文档
5. 编写分析总结和建议
6. 通过邮件系统发送给团队成员
7. 将报告保存到指定文件夹并创建备份
...
[接着会通过MCP服务器完成上述操作]
ActionAI: 任务已完成！销售报告已生成并发送至团队所有成员。
报告显示销售额较上周增长12.5%，主要增长来自电子产品类别。
报告副本已保存至"销售报告/2023/周报"文件夹。
```

通过配置相应的MCP服务，ActionAI可以无缝连接各种系统和应用，实现真正的端到端自动化。

## 📋 使用指南

### 内置命令

| 命令 | 描述 |
|------|------|
| `\quit` | 退出程序 |
| `\fc` | 切换工具调用功能 |
| `\model` | 切换对话模型 |
| `\clear` | 清除聊天历史 |
| `\save` | 保存当前对话历史 |
| `\load <文件路径>` | 加载对话历史 |
| `\help` | 显示帮助信息 |
| `\debug` | 切换调试模式 |
| `\compact <字符数>` | 压缩历史消息 |
| `\cost` | 显示Token使用统计 |
| `\mcp <配置文件>` | 切换MCP配置文件 |

### 多行输入

- 按Enter继续输入
- 输入`\q`结束输入
- 输入`\c`清除当前输入

## 🔌 MCP 服务器配置

MCP（Model Context Protocol）是ActionAI的核心功能，它通过简单的配置即可让AI获得强大的系统操作能力。

以下资源提供了丰富的MCP工具和服务器:

- [OpenTools](https://opentools.com/) - 提供丰富的AI应用工具库和MCP服务器
- [Glamama](https://glama.ai/mcp/servers) - 提供多种开源MCP服务器实现


### 配置示例

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/Users/username/Desktop",
        "/path/to/other/allowed/dir"
      ]
    },
    "custom_tool": {
      "command": "python",
      "args": ["path/to/your/tool_server.py"]
    }
  }
}
```

您也可以轻松创建自己的MCP服务，扩展AI的能力。


## 📝 许可证

[MIT License](https://opensource.org/licenses/MIT)


