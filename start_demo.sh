#!/bin/bash
# Start SupplyChainRAG in demo mode with real-time reasoning visualization

cd "$(dirname "$0")"

echo "=================================================="
echo "  SupplyChainRAG Demo Mode"
echo "  Real-time reasoning visualization"
echo "=================================================="
echo ""

# Set HuggingFace mirror for China mainland
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export HF_HUB_DOWNLOAD_TIMEOUT="60"
echo "🌐 Using HuggingFace mirror: $HF_ENDPOINT"
echo ""

# Check and download embedding model if needed
MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_SHORT_NAME=$(basename "$MODEL_NAME")
CACHE_DIR="$HOME/.cache/huggingface/hub"

echo "🔍 Checking embedding model: $MODEL_SHORT_NAME"

# Check if model exists in cache
if [ -d "$CACHE_DIR" ] && ls "$CACHE_DIR" | grep -q "$MODEL_SHORT_NAME"; then
    echo "✅ Model already cached"
else
    echo "📥 Model not found in cache, pre-downloading..."
    
    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Download model using the script
    python scripts/download_model.py "$MODEL_NAME"
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Model download failed, will try fallback during runtime"
    fi
fi

echo ""

# Check if backend is running
PORT=8765
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "🚀 Starting backend server..."
    
    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Start backend in background
    python -c "import uvicorn; uvicorn.run('src.ui.backend:app', host='0.0.0.0', port=$PORT, reload=False)" &
    BACKEND_PID=$!
    
    # Wait for backend to be ready
    echo "⏳ Waiting for backend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo "✅ Backend is ready!"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "❌ Backend failed to start"
        exit 1
    fi
else
    echo "✅ Backend is already running"
    BACKEND_PID=""
fi

echo ""
echo "🌐 Opening demo frontend..."
echo ""
echo "  📊 Demo UI: file://$(pwd)/demo_frontend.html"
echo "  🔌 API:    http://localhost:$PORT"
echo "  📡 Stream: http://localhost:$PORT/demo/stream"
echo ""
echo "  ⚠️ 注意: 如需使用真实 Kimi API，请设置环境变量:"
echo "     export KIMI_API_KEY='your-api-key'"
echo ""

# Open browser (macOS)
if command -v open >/dev/null 2>&1; then
    open "demo_frontend.html"
# Open browser (Linux)
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "demo_frontend.html"
fi

echo "Press Ctrl+C to stop"

# Keep script running if we started the backend
if [ -n "$BACKEND_PID" ]; then
    wait $BACKEND_PID
fi
