#!/bin/zsh
# Simple start script without tmux for testing

set -e

KIMI_API_KEY=$1
ROUNDS=$3

if [[ -z "$KIMI_API_KEY" ]]; then
    echo "用法: ./start_simple.sh <api_key> [--rounds <number>]"
    exit 1
fi

export KIMI_API_KEY=$KIMI_API_KEY
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export HF_ENDPOINT=https://hf-mirror.com

if [[ "$2" == "--rounds" && -n "$ROUNDS" ]]; then
    export MODE="round"
    export MAX_ROUNDS=$ROUNDS
    echo "🎬 轮次模式: $ROUNDS 轮"
else
    export MODE="query"
    echo "💬 查询模式"
fi

source .venv/bin/activate

# Check data
if [[ ! -f "data/supply_chain.db" ]]; then
    echo "📊 生成数据..."
    python -c "
import sys
sys.path.insert(0, '.')
from src.data.synthetic_generator import SupplyChainDataGenerator
from src.data.database import db_manager
gen = SupplyChainDataGenerator(seed=42)
gen.generate_all(scale='small')
gen.save_to_json('data')
db_manager.create_tables()
db_manager.load_from_synthetic_data('data')
"
fi

# Check indices
if [[ ! -f "data/indices/bm25.pkl" ]]; then
    echo "🔍 构建索引..."
    python scripts/build_indexes.py --data-dir data --output-dir data/indices
fi

echo ""
echo "🚀 启动服务..."
echo "   URL: http://localhost:8888/ui"
echo ""

# Open browser in background
(
    sleep 3
    open "http://localhost:8888/ui?mode=${MODE}&rounds=${ROUNDS:-}"
) &

# Start server
uvicorn src.ui.backend_main:app --host 0.0.0.0 --port 8888 --reload
