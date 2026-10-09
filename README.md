# Local Assistant — 本地 RAG + ReAct Agent 助手

一个在本地运行的智能助手，集成 **RAG 文档问答**、**ReAct Agent 工具调用** 和 **多轮对话记忆**。
面向「AI Agent 应用开发」方向的学习与实践项目，重点在于**工程可靠性**而非模型训练。

---

## 效果数据

> 评测脚本：`tests_eval.py`（四类用例：正常问答 / 多跳 / 无答案 / 格式约束）

| 指标 | 数值 |
|------|------|
| 评测集规模 | 13 条（示例集，建议按自己的知识库扩充至 30-50 条） |
| 检索层召回率 | 待填（`python tests_eval.py --layer retrieval`） |
| 正常问答准确率 | 待填（`python tests_eval.py --layer e2e`） |
| 拒答准确率 | 待填 |
| 幻觉率 | 待填 |
| 平均延迟 / P95 延迟 | 待填 |

**为什么统计「拒答准确率」和「幻觉率」**：这两项衡量的是系统在知识库中没有答案时，
能否正确地说「不知道」，而不是编造内容。它是幻觉的负向指标，也是 RAG 系统最容易被忽视的部分。

> 跑完评测后把真实数字填进上表 —— 用数据说明效果，比描述「效果良好」有说服力得多。

---

## 架构

```
┌─────────────┐
│  用户输入   │
└──────┬──────┘
       │
   ┌───▼────────────────────────────┐
   │  MemoryManager                 │  ← 按「轮次」管理，保证 tool_calls 与
   │  (短期记忆 / 滑动窗口 / 摘要)  │     tool 响应成组不被截断
   └───┬────────────────────────────┘
       │ messages
   ┌───▼────────────────────────────┐
   │  Agent (ReAct Loop)            │
   │  Thought → Action → Observation│
   │                                │
   │  · 重复调用检测                │
   │  · 工具参数校验与错误回喂      │
   │  · 结果裁剪（防止撑爆上下文）  │
   │  · 最大轮数限制                │
   └───┬────────────────────────┬───┘
       │ 工具调用               │ 检索
   ┌───▼──────────┐      ┌──────▼─────────────┐
   │ Tools        │      │ RAG Pipeline       │
   │ · calculator │      │ loader → splitter  │
   │ · notes      │      │ → vectorstore      │
   │ · file_search│      │ → LCEL chain       │
   │ · weather    │      └────────────────────┘
   │ · web_search │
   └───┬──────────┘
       │ 每次运行
   ┌───▼────────────────────────────┐
   │  TraceRecorder                 │  ← 完整轨迹落盘：每轮 think 的上下文
   │  data/traces/*.json            │     规模、每次工具调用的参数与结果
   └────────────────────────────────┘
```

---

## 技术难点与解决过程

这一节是项目的核心。以下每个问题都是实际遇到并处理过的。

### 1. 小模型（GLM-4-Flash）工具调用不稳定

**现象**：模型在该调用 `web_search` 的时候不调用，直接凭已有知识编造答案。

**当时的处理**：在 Agent 入口加了一层关键词意图识别，命中关键词就先替模型把工具调掉。

**问题所在**：这个解法**绕过了 ReAct 循环的自治性**，使系统从「Agent」退化成了
「关键词路由的 Workflow」—— 看起来能用，但 Agent 的核心价值（自主决策）已经不存在了。
问题的本质是模型能力不足被掩盖成了控制流的硬编码。

**本次重构后的处理**：移除绕过逻辑，改为在正确的层面上解决：

1. **工具描述工程** —— 在 `description` 中写清「什么时候该调用、什么时候不该调用」并给出示例
2. **系统提示词** —— 明确列出「需要实时信息时用 web_search」等判断原则，以及连续失败时如何降级
3. **保留自主性** —— 让模型自己决定是否调用工具，而不是外部替它决定

**结论**：Agent 的不确定性应该在「提示 + 工具设计」层修复，
而不是在控制流外面打补丁。如果换模型后仍不稳定，正确的方向是加 few-shot 示例或
换用工具调用能力更强的模型，而不是继续加关键词规则。

### 2. 滑动窗口截断导致 API 报错

**现象**：多轮对话到达一定长度后，请求直接失败。

**原因**：原来的滑动窗口按「消息条数」截断（`history[-max_history:]`）。
一轮工具调用会产生 3–4 条消息（`user` → `assistant(tool_calls)` → `tool` → `assistant`），
截断点可能恰好落在 `assistant(tool_calls)` 和它的 `tool` 响应之间。
而 API 要求 `tool` 消息必须紧跟对应的 `tool_calls`，否则直接返回 400。

**修复**：把截断单位从「消息条数」改为「**轮次**」——
以 `user` 消息为界切分，保证工具调用链永远成组保留。
截断后再做一次配对自检，剥掉开头的孤儿 `tool` 消息和孤立的 `tool_calls`。

### 3. 记忆模块存在但从未生效

**现象**：`MemoryManager` 写好了，但多轮对话实际上不记得上一轮。

**原因**：`Agent.run()` 每次调用都重新构造 `messages`（只有 system + 当前 user），
`MemoryManager` 在 `main.py` 里创建后从未被使用。

**修复**：`Agent` 内部持有 `MemoryManager` 实例，
`run()` 通过 `build_messages()` 构建「system + 历史 + 当前输入」，并在循环中同步写入
工具调用链和最终答案。

### 4. 文件编码问题（GBK / UTF-8 混杂）

**现象**：加载本地文档时抛出 `UnicodeDecodeError`，RAG 初始化失败。

**修复**：加载器和上传接口都改为多编码降级尝试（`utf-8` → `gbk` → `gb18030` → `latin-1`），
上传接口额外用 `chardet` 做编码探测（置信度 > 0.7 时采用探测结果）。

### 5. 工具层安全问题

**现象**：`calculator` 最初用 `eval(expression)` 实现，存在**任意代码执行**漏洞 ——
`__import__('os').system(...)` 这类输入会被真实执行。

**修复**：改为 **AST 白名单求值**，只允许数字字面量、四则运算、幂、取模和一元正负号，
显式拒绝函数调用、属性访问、下标、推导式等其他一切语法，并加上指数上限与结果范围保护。

`storage/encryption.py` 同样做了修正：原实现用 `key + text` 再 base64 编码，
**这是编码不是加密**（去掉前缀即可还原）。现改为基于口令派生的 **PBKDF2 + Fernet（AES + HMAC）**
认证加密，支持篡改检测，口令哈希用带盐 PBKDF2 + 常数时间比较。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`（**该文件已被 .gitignore 忽略，不会上传**）：

```bash
# LLM 配置（示例使用智谱 GLM，任何 OpenAI 兼容接口均可）
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL_ID=glm-4-flash-250414

# 天气工具（高德开放平台）
AMAP_MAPS_API_KEY=你的高德Key

# 本地数据加密口令（storage/encryption.py 使用）
LOCAL_ENC_PASSWORD=设置一个强口令
```

### 3. 运行

**命令行模式**：

```bash
python main.py
```

```
[agent] 你: 北京天气怎么样          # Agent 模式（带多轮记忆，自主调用工具）
[agent] 你: rag                     # 切换到 RAG 文档问答模式
[rag]   你: 什么是 RAG？            # 基于 docs/ 下的文档回答
[agent] 你: clear                   # 清空对话记忆
[agent] 你: quit                    # 退出
```

**API 服务模式**：

```bash
uvicorn demo_api:app --reload
# 交互式接口文档：http://localhost:8000/docs
```

| 接口 | 说明 |
|------|------|
| `POST /chat` | 聊天（`mode` 可选 `agent` / `rag`） |
| `POST /upload` | 上传 txt/md 到知识库 |
| `POST /memory/clear` | 清空对话记忆 |
| `GET /search?query=` | 直接调用网络搜索 |
| `GET /weather/{city}` | 查询城市天气 |
| `GET /conversations?limit=` | 查看历史对话 |
| `GET /health` | 健康检查 |

---

## 评测

```bash
# 检索层评测（不消耗 token，先看检索质量）
python tests_eval.py --layer retrieval --report

# 端到端评测（需要配置好 API Key）
python tests_eval.py --layer e2e --mode rag --report

# 两个都跑
python tests_eval.py --layer all --report
```

**用例分四类**：

| 类型 | 测什么 |
|------|--------|
| `normal` | 知识库中有答案，能否答对 |
| `multi_hop` | 需要综合多段内容 |
| `no_answer` | **知识库中无答案，必须拒答** —— 测幻觉率 |
| `format` | 格式与行为约束（如句子数限制） |

**失败自动归因**，输出按原因分组，这正是「Bad Case 分析」的落地：

```
【幻觉】    无答案却编造内容（最严重）
【漏召】    有答案却拒答
【答案错误】召回了但回答不对
【格式】    违反格式约束
```

结果以 JSON 落盘到 `eval_results/`，便于追踪每次改动带来的指标变化。

---

## 运行轨迹（可观测性）

每次 Agent 运行都会把完整轨迹写入 `data/traces/*.json`：

- 每轮 `think` 的上下文规模（`prompt_tokens_est` / `message_count`）与请求的工具
- 每次工具调用的名称、参数、结果、耗时、成功与否
- 结束原因（`answer` / `max_steps` / `error`）与总耗时

查看某次运行的摘要：

```bash
python utils/trace.py data/traces/xxxxx.json
```

```
==========================================================
Run dae922651f0a   [agent]
问题: 北京天气怎么样
==========================================================
  第1轮 think  上下文≈6tok (2 msgs)  请求工具: weather
  第2轮 think  上下文≈52tok (4 msgs)  请求工具: (直接回答)
      [OK ] weather({'city': '北京'})  0.81s
----------------------------------------------------------
总轮数: 2   工具调用: 1 (失败 0)
总耗时: 1.63s   结束原因: answer
==========================================================
```

**为什么记录上下文规模**：出现「对话越长效果越差」时，
可以直接定位到是哪一步把上下文撑爆的（通常是某个工具返回了超长内容）。
超长字段会标注 `_original_len` 并保留预览，作为上下文污染的直接证据。

---

## 项目结构

```
local_assistant/
├── config.py              # 全局配置（Pydantic 集中管理 + 环境变量）
├── main.py                # 命令行入口
├── demo_api.py            # FastAPI 服务
├── tests_eval.py          # 评测脚手架
├── core/
│   ├── agent.py           # ReAct 循环 + 工具调度 + 记忆 + 轨迹记录
│   ├── memory.py          # 对话记忆（按轮次管理，保证 tool 配对完整）
│   └── context.py         # 上下文构建（GSSC 流水线）
├── rag/
│   ├── loader.py          # 文档加载（多编码降级）
│   ├── splitter.py        # 文本切分
│   ├── vectorstore.py     # 向量库（ChromaDB）
│   └── chain.py           # LCEL 检索链
├── tools/
│   ├── base.py            # 工具抽象基类 + 注册器
│   ├── calculator.py      # 计算器（AST 白名单求值，非 eval）
│   ├── notes.py           # 笔记管理
│   ├── file_search.py     # 文件搜索
│   ├── weather.py         # 天气查询
│   └── web_search.py      # 网络搜索
├── storage/
│   ├── database.py        # 对话持久化（JSON）
│   └── encryption.py      # 数据加密（PBKDF2 + Fernet）
├── utils/
│   ├── logger.py          # 日志
│   ├── trace.py           # 运行轨迹记录与查看
│   └── helpers.py         # 工具函数
├── docs/                  # 知识库文档
└── requirements.txt       # 依赖（已锁版本）
```

---

## 已知局限

诚实列出当前不足，也是后续迭代方向：

- **文档格式单一**：仅支持 `.txt` / `.md`，未处理 PDF、Word、表格
- **无 Rerank**：只有单路向量召回，缺少重排精排
- **无混合检索**：未引入 BM25，精确词（错误码、人名）召回不稳定
- **无引用溯源**：RAG 回答未标注来源文档编号
- **上下文压缩较简单**：`context.py` 仍是截断策略，未接入 LLM 摘要
- **单 Agent**：未涉及多 Agent 编排（在多数场景下单 Agent + 好工具已足够）
- **无持久化向量库**：每次启动重新构建索引（`config.db_path` 已预留但未启用）
- **无测试与容器化**：缺少单元测试和 Dockerfile
- **知识库**：`docs/knowledge.txt` 为 AI 相关概念的示例文档，便于演示评测流程

---

## 技术栈

Python 3.10+ · LangChain (LCEL) · ChromaDB · FastAPI · Pydantic · 智谱 GLM（OpenAI 兼容接口）· cryptography
