#!/bin/bash
# Start SupplyChainRAG in demo mode with real-time reasoning visualization

cd "$(dirname "$0")"

echo "=================================================="
echo "  SupplyChainRAG Demo Mode"
echo "  Real-time reasoning visualization"
echo "=================================================="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8888/health > /dev/null 2>&1; then
    echo "🚀 Starting backend server..."
    
    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    
    # Start backend in background
    python -c "import uvicorn; uvicorn.run('src.ui.backend:app', host='0.0.0.0', port=8000, reload=False)" &
    BACKEND_PID=$!
    
    # Wait for backend to be ready
    echo "⏳ Waiting for backend to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8888/health > /dev/null 2>&1; then
            echo "✅ Backend is ready!"
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:8888/health > /dev/null 2>&1; then
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
echo "  🔌 API:    http://localhost:8888"
echo "  📡 Stream: http://localhost:8888/demo/stream"
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
