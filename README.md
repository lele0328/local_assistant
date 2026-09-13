# Local AI Assistant - 本地隐私智能助手

## 项目简介
一个本地运行的智能助手，集成RAG检索增强生成和ReAct Agent，所有用户数据存储在本地，保护隐私不泄露。

## 功能特性
- **RAG文档问答**：加载本地文档，基于内容回答问题
- **Agent工具调用**：支持计算器、笔记、文件搜索、天气查询等工具
- **对话记忆**：支持多轮对话，滑动窗口管理历史
- **隐私保护**：所有数据本地存储，支持数据加密
- **上下文工程**：GSSC流水线管理上下文

## 技术栈
- Python 3.10+
- LangChain (LCEL管道)
- ChromaDB (向量数据库)
- OpenAI API (LLM调用，仅API调用，不上传数据)
- Pydantic (数据验证)

## 项目结构
```
local_assistant/
├── config.py              # 全局配置
├── main.py                # 主程序入口
├── core/
│   ├── agent.py           # ReAct Agent
│   ├── memory.py          # 对话记忆管理
│   └── context.py         # 上下文工程(GSSC)
├── rag/
│   ├── loader.py          # 文档加载
│   ├── splitter.py        # 文本切分
│   ├── vectorstore.py     # 向量数据库
│   └── chain.py           # LCEL检索链
├── tools/
│   ├── base.py            # 工具基类+注册器
│   ├── calculator.py      # 计算器
│   ├── notes.py           # 笔记管理
│   ├── file_search.py     # 文件搜索
│   └── weather.py         # 天气查询
├── storage/
│   ├── database.py         # 本地数据库
│   └── encryption.py      # 数据加密
├── utils/
│   ├── logger.py           # 日志系统
│   └── helpers.py          # 工具函数
├── docs/                  # 知识库文档
└── requirements.txt       # 依赖列表
```

## 安装与运行
```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
# 在项目根目录创建 .env 文件：
# LLM_API_KEY=你的API密钥
# LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
# LLM_MODEL_ID=glm-4-flash-250414

# 运行
python main.py
```

## 使用说明
- 输入 `rag` 切换到RAG文档问答模式
- 输入 `agent` 切换到Agent工具调用模式
- 输入 `quit` 退出程序
