#!/bin/zsh
# Quick test script for SupplyChainRAG

set -e

export KIMI_API_KEY="test_key"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export HF_ENDPOINT=https://hf-mirror.com

echo "🔧 Testing SupplyChainRAG setup..."

source .venv/bin/activate

# Test 1: Generate small dataset
echo ""
echo "1. Testing data generation..."
python -c "
import sys
sys.path.insert(0, '.')
from src.data.synthetic_generator import SupplyChainDataGenerator
from src.data.database import db_manager

generator = SupplyChainDataGenerator(seed=42)
data = generator.generate_all(scale='small')  # Small scale for testing
generator.save_to_json('data')
db_manager.create_tables()
db_manager.load_from_synthetic_data('data')
print(f'✓ Generated {len(data[\"batches\"])} batches')
"

# Test 2: Build indices
echo ""
echo "2. Testing index building..."
python scripts/build_indexes.py --data-dir data --output-dir data/indices

echo ""
echo "✅ All tests passed!"
