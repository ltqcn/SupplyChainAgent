#!/bin/zsh
# entry.sh - SupplyChainRAG统一入口脚本
# 
# 使用方式:
#   ./entry.sh <kimi_api_key> [--rounds <number>]
# 
# 示例:
#   ./entry.sh sk-xxx --rounds 5    # 轮次演示模式
#   ./entry.sh sk-xxx               # 实时查询模式

set -e  # 遇到错误立即退出

# =============================================================================
# 参数解析
# =============================================================================

KIMI_API_KEY=$1
ROUNDS=$3  # 第三个参数是rounds数量 (--rounds是$2)

# 参数校验
if [[ -z "$KIMI_API_KEY" ]]; then
    echo "❌ 错误：请提供Kimi API Key作为第一个参数"
    echo ""
    echo "用法："
    echo "  ./entry.sh <api_key> [--rounds <number>]"
    echo ""
    echo "示例："
    echo "  ./entry.sh sk-xxx --rounds 5    # 启动轮次演示模式"
    echo "  ./entry.sh sk-xxx               # 启动实时查询模式"
    exit 1
fi

# 解析--rounds参数
if [[ "$2" == "--rounds" && -n "$ROUNDS" ]]; then
    echo "🎬 检测到轮次模式参数: $ROUNDS 轮"
elif [[ -n "$2" && "$2" != "--rounds" ]]; then
    echo "❌ 未知参数: $2"
    echo "提示: 使用 --rounds <number> 启动轮次模式"
    exit 1
fi

# =============================================================================
# 环境变量设置
# =============================================================================

export KIMI_API_KEY=$KIMI_API_KEY
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export UV_PYTHON=3.11

# HuggingFace 镜像配置（中国大陆网络优化）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
echo "🌐 HuggingFace镜像: $HF_ENDPOINT"

echo "🔧 SupplyChainRAG 启动中..."
echo ""

# =============================================================================
# 依赖检查
# =============================================================================

echo "📋 检查依赖环境..."

# 检查uv安装
if ! command -v uv &> /dev/null; then
    echo "❌ 未检测到uv，请先安装：https://github.com/astral-sh/uv"
    echo "   推荐安装方式: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "  ✓ uv 已安装"

# 检查Docker
if ! docker info &> /dev/null; then
    echo "❌ Docker守护进程未运行，请启动Docker Desktop"
    exit 1
fi
echo "  ✓ Docker 运行中"

# 检查Python 3.11
if ! uv python find 3.11 &> /dev/null; then
    echo "⚠️  未找到Python 3.11，尝试安装..."
    uv python install 3.11
fi
echo "  ✓ Python 3.11 可用"

echo ""

# =============================================================================
# 虚拟环境与依赖
# =============================================================================

if [[ ! -d ".venv" ]]; then
    echo "📦 创建虚拟环境..."
    uv venv --python 3.11
fi

source .venv/bin/activate

# 安装依赖（若pyproject.toml变化）
if [[ ! -f ".venv/installed" ]] || [[ "pyproject.toml" -nt ".venv/installed" ]]; then
    echo "📥 安装Python依赖..."
    uv pip install -e .
    touch .venv/installed
    echo "  ✓ 依赖安装完成"
else
    echo "  ✓ 依赖已安装"
fi

echo ""

# =============================================================================
# 模型预下载
# =============================================================================

echo "📥 检查Embedding模型..."
if ! python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}', cache_folder='./models/cache')" 2>/dev/null; then
    echo "  预下载Embedding模型（使用镜像加速）..."
    python scripts/download_model.py || echo "  ⚠️ 模型下载可能失败，将在使用时重试"
fi
echo ""

# =============================================================================
# 数据初始化（懒加载）
# =============================================================================

if [[ ! -f "data/supply_chain.db" ]]; then
    echo "🗄️  生成合成供应链数据..."
    python -c "
import sys
sys.path.insert(0, '.')
from src.data.synthetic_generator import SupplyChainDataGenerator
from src.data.database import db_manager

# 生成数据
generator = SupplyChainDataGenerator(seed=42)
data = generator.generate_all(scale='medium')

# 保存到JSON
generator.save_to_json('data')

# 导入数据库
db_manager.create_tables()
db_manager.load_from_synthetic_data('data')
print('✓ 数据生成完成')
"
    echo ""
else
    echo "  ✓ 数据已存在"
fi

# 构建RAG索引（若不存在）
if [[ ! -f "data/indices/bm25.pkl" ]]; then
    echo "🔍 构建RAG索引..."
    
    # 创建索引目录
    mkdir -p data/indices
    
    # 运行构建脚本（使用脚本文件而不是内联代码，避免tmux问题）
    python scripts/build_indexes.py --data-dir data --output-dir data/indices 2>&1 | tee logs/index_build.log
    
    if [[ $? -eq 0 ]]; then
        echo "✓ 索引构建完成"
    else
        echo "⚠️ 索引构建可能有警告，但继续启动..."
    fi
    echo ""
else
    echo "  ✓ RAG索引已存在"
fi

echo ""

# =============================================================================
# 模式设置
# =============================================================================

if [[ -n "$ROUNDS" ]]; then
    export MODE="round"
    export MAX_ROUNDS=$ROUNDS
    echo "🎬 启动轮次演示模式（最大轮次: $ROUNDS）"
else
    export MODE="query"
    echo "💬 启动实时查询模式"
fi

# 创建日志目录（在索引构建前创建）
mkdir -p logs

# 确保 logs 目录有写入权限
if [[ ! -w "logs" ]]; then
    echo "❌ logs 目录没有写入权限"
    exit 1
fi

# =============================================================================
# 启动服务（tmux分屏）
# =============================================================================

echo ""
echo "🚀 启动服务..."

# 清理已有的tmux会话
if tmux has-session -t supplychainrag 2>/dev/null; then
    tmux kill-session -t supplychainrag
fi

# 创建新的tmux会话（单窗格模式，避免同步问题）
tmux new-session -d -s supplychainrag -n "SupplyChainRAG"

# 等待会话创建完成
sleep 0.5

# 垂直分割窗口（创建右窗格）
tmux split-window -h -t supplychainrag:0

# 等待窗格创建
sleep 0.5

# 左屏：后端服务
tmux send-keys -t supplychainrag:0.left "cd ${PWD} && source .venv/bin/activate && uvicorn src.ui.backend_main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee logs/backend.log" C-m

# 右屏：日志监控
tmux send-keys -t supplychainrag:0.right "cd ${PWD} && echo '等待服务启动...' && sleep 5 && tail -f logs/backend.log 2>/dev/null || echo '日志监控就绪'" C-m

# 设置窗格标题
tmux select-pane -t supplychainrag:0.left -T "Backend API"
tmux select-pane -t supplychainrag:0.right -T "Logs"

echo ""
echo "✅ 服务已启动！"
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  访问地址: http://localhost:8888                            │"
echo "│  模式: $MODE                                                  │"
if [[ -n "$ROUNDS" ]]; then
    echo "│  轮次: $ROUNDS                                                │"
fi
echo "│                                                             │"
echo "│  快捷键:                                                    │"
echo "│    Ctrl+B, 方向键  - 切换窗格                              │"
echo "│    Ctrl+B, D        - 分离会话（后台运行）                 │"
echo "│    Ctrl+C           - 停止服务                             │"
echo "│                                                             │"
echo "│  查看日志: tail -f logs/backend.log                         │"
echo "│  停止服务: tmux kill-session -t supplychainrag              │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

# 延迟打开浏览器
(
    sleep 4
    if command -v open &> /dev/null; then
        open "http://localhost:8888/ui?mode=${MODE}&rounds=${ROUNDS:-}"
    fi
) &

# 附加到tmux会话
tmux attach-session -t supplychainrag
