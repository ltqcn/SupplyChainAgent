#!/bin/zsh
# Fully offline start script - no FAISS, no model download

set -e

KIMI_API_KEY=$1
ROUNDS=$3

if [[ -z "$KIMI_API_KEY" ]]; then
    echo "用法: ./start_offline_safe.sh <api_key> [--rounds <number>]"
    echo ""
    echo "此脚本完全离线运行:"
    echo "  - 使用简单向量器（无需下载模型）"
    echo "  - 使用纯 BM25 检索（无 FAISS）"
    exit 1
fi

export KIMI_API_KEY=$KIMI_API_KEY
export PYTHONPATH="${PWD}:${PYTHONPATH}"

if [[ "$2" == "--rounds" && -n "$ROUNDS" ]]; then
    export MODE="round"
    export MAX_ROUNDS=$ROUNDS
    echo "🎬 轮次模式: $ROUNDS 轮"
else
    export MODE="query"
    echo "💬 查询模式"
fi

echo ""
echo "⚡ 完全离线模式"
echo "   - 向量器: SimpleVectorizer (384d hash-based)"
echo "   - 检索: BM25 only (no FAISS)"
echo ""

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
print('✓ 数据准备完成')
"
fi

# Build indices (safe version)
if [[ ! -f "data/indices/bm25.pkl" ]]; then
    echo "🔍 构建 BM25 索引..."
    
    export USE_SIMPLE_RETRIEVER=1
    python -c "
import os
os.environ['USE_SIMPLE_RETRIEVER'] = '1'

import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from pathlib import Path
import pickle

from src.data.vectorizer import SupplyChainDocumentBuilder
from src.rag.simple_retriever import SimpleRetriever

# Build documents
data_dir = Path('data')
builder = SupplyChainDocumentBuilder(data_dir)
doc_ids, texts, metadata = builder.build_all_documents()
print(f'  文档数: {len(texts)}')

# Use simple retriever
retriever = SimpleRetriever()
retriever.build_indices(doc_ids, texts, None, metadata)

# Save
indices_dir = Path('data/indices')
indices_dir.mkdir(parents=True, exist_ok=True)
retriever.save_indices(indices_dir)

# Save documents
with open(indices_dir / 'documents.pkl', 'wb') as f:
    pickle.dump({'doc_ids': doc_ids, 'texts': texts, 'metadata': metadata}, f)

print('✓ 索引构建完成')
"
fi

echo ""
echo "🚀 启动服务..."
echo "   URL: http://localhost:8888/ui"
echo "   模式: 完全离线（BM25 检索）"
echo ""

# Open browser
(
    sleep 2
    open "http://localhost:8888/ui?mode=${MODE}&rounds=${ROUNDS:-}"
) &

# Start server with USE_SIMPLE_RETRIEVER
USE_SIMPLE_RETRIEVER=1 uvicorn src.ui.backend_main:app --host 0.0.0.0 --port 8000 --reload
