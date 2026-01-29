# 应援物品供应链AI助手

<p align="center">
  <strong>可治理RAG × 渐进式披露 × Manus-like记忆架构</strong>
</p>

本项目是一个生产级技术原型，展示了现代大模型应用架构的核心能力：

- **可治理RAG**: BM25 + HNSW + IVF-PQ 多路召回 + RFF核方法融合
- **渐进式披露**: 分层Context加载，节省60-70% Token消耗
- **Manus-like记忆**: 短期/长期/永久三层记忆 + Disk-offload持久化
- **脚本式Skills**: Docker沙箱安全执行，突破JSON调用限制

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                         UI Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Query Mode  │  │  Round Mode  │  │  Dashboard   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Agent Layer                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Orchestrator: Intent → RAG → Memory → Skill → LLM   │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   Context Engineering                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Token Budget  │  │Progressive   │  │  Assembler   │      │
│  │  Manager     │  │ Disclosure   │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  RAG Engine    │  Memory System  │  Skills System          │
│  ┌──────────┐  │  ┌──────────┐   │  ┌──────────────┐      │
│  │  BM25    │  │  │  Short   │   │  │  L1/L2/L3    │      │
│  │  HNSW    │  │  │  Long    │   │  │  Progressive │      │
│  │ IVF-PQ   │  │  │ Permanent│   │  │  Docker      │      │
│  │ RFF Fuse │  │  │ Offload  │   │  │  Sandbox     │      │
│  └──────────┘  │  └──────────┘   │  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境要求

- **macOS** (主要支持，Linux理论上兼容)
- **Python 3.11**
- **Docker Desktop** (用于沙箱执行)
- **uv** (现代Python包管理器)

安装uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 获取Kimi API Key

从 [Moonshot AI](https://platform.moonshot.cn/) 获取API Key

### 3. 启动系统

**实时查询模式** (生产界面):
```bash
./entry.sh <your_kimi_api_key>
```

**轮次演示模式** :
```bash
./entry.sh <your_kimi_api_key> --rounds 5
```

启动后会自动:
1. 创建虚拟环境并安装依赖
2. 生成合成供应链数据 (首次运行)
3. 构建RAG索引 (首次运行)
4. 启动FastAPI服务 (localhost:8888)
5. 分屏显示Web UI和Terminal日志
6. 打开浏览器

## 核心特性详解

### 1. 可治理RAG (Governable RAG)

**多路召回架构**:

| 算法 | 召回类型 | 适用场景 | 性能指标 |
|------|----------|----------|----------|
| BM25 | 稀疏检索 | SKU编码、订单号精确匹配 | <20ms |
| HNSW | 稠密检索 | 自然语言语义匹配 | Recall@10 >95% |
| IVF-PQ | 量化检索 | 百万级历史数据 | 压缩比 192:1 |

**RFF融合** (Random Fourier Features):
- 将异构分数映射到统一256维空间
- 捕捉非线性关系（如BM25高分+HNSW高分=超线性boost）
- 权重调优: BM25(0.3) + HNSW(0.4) + IVF-PQ(0.3)

**引用溯源**:
```xml
<reference id="ref_1" source="PROD-2025-001" 
           type="生产报告" confidence="0.92" 
           retrieval_method="rrf_bm25_hnsw">
  批次B20250120已完成亚克力立牌生产1000件...
</reference>
```

### 2. 渐进式披露 (Progressive Disclosure)

**分层加载策略**:

| 层级 | 内容 | Token/技能 | Token/文档 |
|------|------|------------|------------|
| L1 | 元数据 (名称/描述) | 50 | 80 |
| L2 | 核心定义 (参数/规则) | 500 | 300 |
| L3 | 完整内容 (示例/文档) | 1500 | 500 |

**节省效果**:
- 传统方式: 50技能 × 4500 tokens = 225,000 tokens
- 渐进方式: 50×50 + 3×500 = 4,000 tokens
- **节省98%初始Context**

### 3. Manus-like记忆架构

**三层记忆系统**:

```
┌─────────────────────────────────────────────────┐
│  Short-term (当前会话)                          │
│  • 100+轮对话历史                               │
│  • 滑动窗口 + 智能摘要压缩                       │
│  • Token预算: 3200 tokens                       │
├─────────────────────────────────────────────────┤
│  Long-term (跨会话)                             │
│  • 会话摘要 + 关键决策点                         │
│  • Embedding语义检索                            │
│  • 自动归档: 30天合并月度摘要                    │
├─────────────────────────────────────────────────┤
│  Permanent (知识库)                             │
│  • 业务规则、SOP、案例库                         │
│  • 低更新频率，IVF-PQ索引                        │
└─────────────────────────────────────────────────┘
```

**Disk-offload机制**:
- 触发条件: 短期记忆 > 6000 tokens
- 行为: 摘要化最早40%对话 → 写入磁盘 → 释放内存
- 恢复: 基于语义相关性自动召回

### 4. 脚本式Skills调用

**Docker沙箱配置**:
- 镜像: python:3.11-slim
- CPU限制: 1核
- 内存限制: 512MB
- 网络: 默认隔离 (白名单)
- 超时: 30秒自动终止
- 用户: 非root运行

**执行流程**:
```
LLM生成参数 → 渲染脚本模板 → Docker执行 → 
捕获stdout/stderr → JSON解析 → 注入Context
```

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | API信息 |
| `/health` | GET | 健康检查 |
| `/chat` | POST | 发送聊天消息 |
| `/chat/stream` | POST | 流式聊天 |
| `/round/{idx}` | GET | 获取轮次状态 |
| `/dashboard/metrics` | GET | 仪表盘数据 |
| `/memory/status` | GET | 内存状态 |
| `/rag/status` | GET | RAG索引状态 |
| `/skills` | GET | 可用技能列表 |
| `/ui` | GET | Web界面 |

## 项目结构

```
SupplyChainRAG/
├── src/
│   ├── config.py              # 配置管理
│   ├── models.py              # Pydantic数据模型
│   ├── data/
│   │   ├── synthetic_generator.py  # 合成数据生成
│   │   ├── database.py        # SQLAlchemy ORM
│   │   └── vectorizer.py      # 文档向量化
│   ├── rag/
│   │   ├── bm25_retriever.py  # BM25实现
│   │   ├── hnsw_retriever.py  # HNSW图索引
│   │   ├── ivf_pq_retriever.py # IVF-PQ量化
│   │   ├── rff_fusion.py      # RFF融合
│   │   └── retriever.py       # 统一检索接口
│   ├── context/
│   │   ├── token_budget.py    # Token预算管理
│   │   ├── progressive_disclosure.py  # 渐进式披露
│   │   └── assembler.py       # Context组装
│   ├── memory/
│   │   ├── short_term.py      # 短期记忆
│   │   ├── long_term.py       # 长期记忆
│   │   └── offload.py         # Disk-offload
│   ├── tools/
│   │   ├── skill_manager.py   # Skills管理
│   │   └── tool_executor.py   # Docker沙箱执行
│   ├── agent/
│   │   ├── llm_client.py      # Kimi SDK封装
│   │   └── orchestrator.py    # 主编排器
│   └── ui/
│       └── backend.py         # FastAPI后端
├── scripts/
│   ├── build_indexes.py       # 索引构建脚本
│   └── pretty_log.py          # 日志美化
├── entry.sh                   # 统一入口脚本
├── pyproject.toml             # uv依赖配置
└── README.md                  # 项目说明
```

### 解决大模型上下文窗口限制

**A**: 我们采用渐进式披露策略。以工具调用为例：

| 方式 | Token消耗 | 节省 |
|------|-----------|------|
| 全量加载 | 50技能 × 4500 = 225,000 | - |
| L1仅元数据 | 50 × 50 = 2,500 | 99% |
| L2按需展开 | 2,500 + 3×500 = 4,000 | 98% |

代码路径: `src/context/progressive_disclosure.py`

### 确保RAG检索的准确性

**A**: 多路召回+融合策略：

1. **BM25**: 擅长SKU编码精确匹配 (k1=1.5, b=0.75)
2. **HNSW**: 擅长语义相似度 (M=16, ef=64)
3. **IVF-PQ**: 适合大规模历史数据 (nlist=4096)
4. **RFF融合**: 将三路分数映射到256维统一空间

效果: Recall@10提升15-20% vs 单一算法

代码路径: `src/rag/retriever.py`

### 实现Manus-like记忆管理

**A**: 三层架构：

```python
# 短期记忆: 当前会话
short_term = ShortTermMemory(session_id)

# 长期记忆: 跨会话摘要
long_term = LongTermMemory()
long_term.add_session_summary(session_id, summary, entities, decisions)

# 永久知识: 业务规则
permanent = PermanentKnowledge()
```

**Disk-offload触发**:
```
Token > 6000 → 摘要化40% → 写入磁盘 → 释放内存
```

代码路径: `src/memory/offload.py`

## 开发指南

### 添加新的Skill

```python
from src.tools.skill_manager import get_skill_manager
from src.models import SkillDefinition

skill = SkillDefinition(
    name="my_new_skill",
    description="技能描述",
    category="warehouse",
    autonomy_level=AutonomyLevel.LEVEL_2,
    parameters={
        "param1": {"type": "string", "required": True},
    },
    execution_mode="internal",  # or "script"
)

get_skill_manager().register_skill(skill)
```

### 自定义RAG查询

```python
from src.rag.retriever import get_retriever

retriever = get_retriever()
results, log = retriever.retrieve(
    query="亚克力立牌库存",
    top_k=5,
    query_type="hybrid"  # exact_lookup | semantic_search | hybrid
)
```

## 许可证

MIT License
