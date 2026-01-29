#!/bin/zsh
# Offline start script - no model download required

set -e

KIMI_API_KEY=$1
ROUNDS=$3

if [[ -z "$KIMI_API_KEY" ]]; then
    echo "用法: ./start_offline.sh <api_key> [--rounds <number>]"
    echo ""
    echo "此脚本使用离线模式，无需下载Embedding模型"
    exit 1
fi

export KIMI_API_KEY=$KIMI_API_KEY
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Use offline mode
export USE_SIMPLE_VECTORIZER=1

if [[ "$2" == "--rounds" && -n "$ROUNDS" ]]; then
    export MODE="round"
    export MAX_ROUNDS=$ROUNDS
    echo "🎬 轮次模式: $ROUNDS 轮"
else
    export MODE="query"
    echo "💬 查询模式"
fi

echo ""
echo "⚡ 离线模式 - 使用简单向量器（无需下载模型）"
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
"
fi

# Check indices
if [[ ! -f "data/indices/bm25.pkl" ]]; then
    echo "🔍 构建索引（离线模式）..."
    
    # Use simple vectorizer for index building
    python -c "
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pickle
import numpy as np
from pathlib import Path

from src.data.simple_vectorizer import SimpleVectorizer
from src.data.vectorizer import SupplyChainDocumentBuilder
from src.rag.retriever import UnifiedRetriever

# Build documents
data_dir = Path('data')
builder = SupplyChainDocumentBuilder(data_dir)
doc_ids, texts, metadata = builder.build_all_documents()
print(f'Built {len(texts)} documents')

# Use simple vectorizer
vectorizer = SimpleVectorizer(384)
embeddings = vectorizer.encode(texts, batch_size=32)
print(f'Encoded {len(embeddings)} vectors')

# Save
indices_dir = Path('data/indices')
indices_dir.mkdir(parents=True, exist_ok=True)
with open(indices_dir / 'documents.pkl', 'wb') as f:
    pickle.dump({'doc_ids': doc_ids, 'texts': texts, 'metadata': metadata}, f)
np.save(indices_dir / 'embeddings.npy', embeddings)

# Build indices
retriever = UnifiedRetriever()
retriever.build_indices(doc_ids, texts, embeddings, metadata)
retriever.save_indices(indices_dir)
print('✓ 索引构建完成')
"
fi

echo ""
echo "🚀 启动服务..."
echo "   URL: http://localhost:8888/ui"
echo "   模式: 离线（使用简单向量器）"
echo ""

# Open browser in background
(
    sleep 2
    open "http://localhost:8888/ui?mode=${MODE}&rounds=${ROUNDS:-}"
) &

# Start server
uvicorn src.ui.backend_main:app --host 0.0.0.0 --port 8000 --reload
