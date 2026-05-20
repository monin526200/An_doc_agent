# RAG Agent System

基于LangGraph的RAG智能问答系统，支持文档解析、向量检索、MCP工具调用和多轮对话记忆。

## 主要功能

- **文档处理**：PDF/DOCX/TXT解析，语义分块，多模态内容提取
- **向量检索**：Milvus向量数据库，稠密+稀疏混合检索，RRF融合与重排序
- **Agent工作流**：基于LangGraph的查询路由、知识检索、答案生成
- **MCP工具**：集成arXiv论文搜索与下载
- **记忆管理**：Redis短期对话记忆 + 长期滚动摘要

## 环境配置

### 1. 创建Conda环境

```bash
conda create -n doc_agent python=3.10
conda activate doc_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载Embedding和Rerank模型

```bash
# 创建模型目录
mkdir -p llm/model

# 下载BGE中文Embedding模型
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5 llm/model/bge-small-zh-v1.5

# 下载BGE英文Embedding模型
git clone https://huggingface.co/BAAI/bge-small-en-v1.5 llm/model/bge-small-en-v1.5

# 下载BGE Reranker模型
git clone https://huggingface.co/BAAI/bge-reranker-base llm/model/bge-reranker-base
```

### 4. 配置模型信息

编辑 `config.yaml`：

**使用云端API（OpenAI兼容）：**

```yaml
llm:
  mode: api
  api:
    base_url: "https://api.openai.com/v1"
    key: "sk-your-api-key"
    model: "gpt-3.5-turbo"
```

**使用本地模型（vLLM）：**

```yaml
llm:
  mode: local
  local:
    path: "./models/llama-2-7b-chat-hf"
```

## 文档入库

将文档放入 `rag/data/source_file/` 目录，然后构建向量数据库：

```bash
# 启动Redis和MCP服务（如未启动）
python run_mcp_redis.py

# 构建向量数据库（新终端）
python build_database.py
```

支持的格式：PDF、DOCX、TXT。重复运行将自动跳过已处理文件。

## Quickstart

### 方式1：命令行直接对话

```bash
# 启动Redis和MCP服务（新终端）
python run_mcp_redis.py

# 运行对话（另一终端）
python main.py
```

命令：`<end>`退出，`<new>`新建会话，`<memory>`查看记忆状态。

### 方式2：前后端分离（Web界面）

```bash
# 步骤1：启动Redis和MCP服务
python run_mcp_redis.py

# 步骤2：启动后端API（新终端）
uvicorn server:app --host 0.0.0.0 --port 8080

# 步骤3：启动前端UI（新终端）
cd ui && chainlit run app.py --port 7860
```

访问 [http://localhost:7860](http://localhost:7860) 使用Web界面。

## 项目结构

```
.
├── config.yaml              # 统一配置文件
├── server.py                # FastAPI后端
├── main.py                  # 命令行对话入口
├── build_database.py        # 文档入库工具
├── run_mcp_redis.py         # 启动Redis和MCP服务
├── ui/
│   └── app.py               # Chainlit前端
├── llm/
│   └── llm_server.py        # LLM封装
├── workflow/
│   ├── main_workflow.py     # 主工作流
│   ├── rag_workflow.py      # RAG子工作流
│   └── mcp_workflow.py      # MCP工具工作流
├── rag/
│   ├── file2chunk2.py       # 文档解析分块
│   ├── rag_query.py         # 检索查询
│   └── database_building.py # 向量数据库
└── history_redis/
    └── store.py             # 对话记忆存储
```

## 许可证

MIT
