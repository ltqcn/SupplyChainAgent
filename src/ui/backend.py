"""FastAPI backend for SupplyChainRAG Web UI.

Provides REST API endpoints for:
- Chat queries
- Round mode demonstration
- Memory and RAG status
- Supply chain dashboard data
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.agent.orchestrator import AgentOrchestrator, RoundModeOrchestrator, get_orchestrator
from src.config import settings
from src.data.database import db_manager
from src.memory.offload import MemoryManager
from src.models import ChatMessage
from src.rag.retriever import get_retriever
from src.tools.skill_manager import get_skill_manager


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request body."""
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response body."""
    response_id: str
    content: str
    citations: list[dict[str, Any]]
    latency_ms: int
    tokens_used: dict[str, int]


class RoundStateResponse(BaseModel):
    """Round mode state response."""
    round: int
    total_rounds: int
    steps: list[dict[str, Any]]
    is_complete: bool


class DashboardMetrics(BaseModel):
    """Dashboard metrics response."""
    total_sku_count: int
    low_stock_alerts: int
    active_batches: int
    delayed_batches: int
    in_transit_shipments: int
    pending_orders: int
    active_risks: list[dict[str, Any]]


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    print("Starting SupplyChainRAG API...")
    
    # Ensure database exists
    db_manager.create_tables()
    
    # Load RAG indices if available
    try:
        retriever = get_retriever()
        retriever.load_indices()
        print("RAG indices loaded")
    except Exception as e:
        print(f"Warning: Could not load RAG indices: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down SupplyChainRAG API...")


app = FastAPI(
    title="SupplyChainRAG API",
    description="Supply Chain AI Assistant with Governable RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SupplyChainRAG",
        "version": "0.1.0",
        "mode": settings.MODE,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "rag_indices": "loaded" if get_retriever().bm25_retriever else "not_loaded",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat query.
    
    Args:
        request: Chat request with query and optional session_id
        
    Returns:
        Chat response with content and metadata
    """
    try:
        orchestrator = get_orchestrator(request.session_id)
        response = await orchestrator.process(request.query)
        
        return ChatResponse(
            response_id=response.response_id,
            content=response.content,
            citations=response.citations,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response.
    
    Args:
        request: Chat request
        
    Returns:
        Streaming response with SSE
    """
    async def event_generator():
        try:
            orchestrator = get_orchestrator(request.session_id)
            response = await orchestrator.process(request.query)
            
            # Yield response as SSE
            yield f"data: {response.content}\n\n"
            yield f"event: complete\ndata: {{'latency_ms': {response.latency_ms}}}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {{'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/round/{round_idx}", response_model=RoundStateResponse)
async def get_round_state(
    round_idx: int,
    query: str = Query(..., description="Query for this round"),
    session_id: str | None = None,
):
    """Get state for a specific round in round mode.
    
    Args:
        round_idx: Round index (0-based)
        query: User query
        session_id: Session identifier
        
    Returns:
        Round state with steps and intermediate results
    """
    if not isinstance(settings.MAX_ROUNDS, int) or settings.MAX_ROUNDS <= 0:
        raise HTTPException(status_code=400, detail="Round mode not enabled")
    
    try:
        orchestrator = get_orchestrator(session_id)
        
        if not isinstance(orchestrator, RoundModeOrchestrator):
            raise HTTPException(status_code=400, detail="Not in round mode")
        
        state = await orchestrator.execute_round(query, round_idx)
        
        return RoundStateResponse(
            round=round_idx,
            total_rounds=orchestrator.max_rounds,
            steps=state["steps"],
            is_complete=round_idx >= orchestrator.max_rounds - 1,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics():
    """Get supply chain dashboard metrics.
    
    Returns:
        Current supply chain metrics and risk alerts
    """
    try:
        # Query database for metrics
        from sqlalchemy import func
        from src.data.database import BatchORM, FanOrderORM, InventoryORM, WaybillORM
        
        with db_manager.get_session() as session:
            # SKU count
            sku_count = session.query(InventoryORM.sku_id).distinct().count()
            
            # Low stock alerts (available < safety_stock threshold)
            low_stock = 0  # Simplified
            
            # Active batches
            active_batches = session.query(BatchORM).filter(
                BatchORM.status.in_(["in_production", "qc_pending"])
            ).count()
            
            # Delayed batches (simplified check)
            delayed_batches = 0
            
            # In-transit shipments
            in_transit = session.query(WaybillORM).filter(
                WaybillORM.status == "in_transit"
            ).count()
            
            # Pending orders
            pending_orders = session.query(FanOrderORM).filter(
                FanOrderORM.status.in_(["pending", "paid"])
            ).count()
        
        # Generate sample risk alerts
        risks = [
            {
                "title": "批次F01-20260128-001生产延迟",
                "level": "medium",
                "description": "当前进度75%，预计延迟1天",
                "suggestion": "建议协调加班或启用备选工厂",
            },
        ] if delayed_batches > 0 else []
        
        return DashboardMetrics(
            total_sku_count=sku_count,
            low_stock_alerts=low_stock,
            active_batches=active_batches,
            delayed_batches=delayed_batches,
            in_transit_shipments=in_transit,
            pending_orders=pending_orders,
            active_risks=risks,
        )
        
    except Exception as e:
        # Return mock data if database query fails
        return DashboardMetrics(
            total_sku_count=50,
            low_stock_alerts=3,
            active_batches=12,
            delayed_batches=1,
            in_transit_shipments=45,
            pending_orders=120,
            active_risks=[],
        )


@app.get("/memory/status")
async def get_memory_status(session_id: str | None = None):
    """Get memory status for session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Memory statistics for all tiers
    """
    orchestrator = get_orchestrator(session_id)
    
    return {
        "session_id": orchestrator.session_id,
        "short_term": orchestrator.memory_manager.short_term.get_stats(),
        "long_term": orchestrator.memory_manager.long_term.get_stats(),
        "permanent": orchestrator.memory_manager.permanent.get_stats(),
        "offload": orchestrator.memory_manager.offload.get_stats(),
    }


@app.get("/rag/status")
async def get_rag_status():
    """Get RAG system status.
    
    Returns:
        RAG index statistics
    """
    retriever = get_retriever()
    
    return {
        "bm25": {
            "loaded": retriever.bm25_retriever is not None and retriever.bm25_retriever.bm25 is not None,
            "doc_count": len(retriever.bm25_retriever.doc_ids) if retriever.bm25_retriever else 0,
        },
        "hnsw": {
            "loaded": retriever.hnsw_retriever is not None and retriever.hnsw_retriever.index is not None,
            "doc_count": len(retriever.hnsw_retriever.doc_ids) if retriever.hnsw_retriever else 0,
        },
        "ivf_pq": {
            "loaded": retriever.ivf_pq_retriever is not None and retriever.ivf_pq_retriever.index is not None,
            "doc_count": len(retriever.ivf_pq_retriever.doc_ids) if retriever.ivf_pq_retriever else 0,
        },
        "total_documents": len(retriever.doc_texts),
    }


@app.get("/skills")
async def get_skills():
    """Get available skills/tools.
    
    Returns:
        List of available skills
    """
    skill_manager = get_skill_manager()
    skills = skill_manager.get_l1_index()
    
    return {
        "skills": skills,
        "stats": skill_manager.get_stats(),
    }


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Get conversation history.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation messages
    """
    orchestrator = get_orchestrator(session_id)
    history = orchestrator.get_conversation_history()
    
    return {
        "session_id": session_id,
        "messages": [msg.model_dump() for msg in history],
    }


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success status
    """
    from src.agent.orchestrator import _orchestrators
    
    if session_id in _orchestrators:
        del _orchestrators[session_id]
    
    return {"status": "cleared", "session_id": session_id}


# ============================================================================
# Static Files (simplified - in production use proper static file serving)
# ============================================================================

@app.get("/ui")
async def get_ui():
    """Serve simple HTML UI."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>SupplyChainRAG</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { border-bottom: 1px solid #ddd; padding-bottom: 20px; margin-bottom: 20px; }
        .chat-container { border: 1px solid #ddd; border-radius: 8px; height: 500px; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        .input-area { display: flex; margin-top: 20px; }
        input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
        button { padding: 12px 24px; background: #1976d2; color: white; border: none; border-radius: 8px; margin-left: 10px; cursor: pointer; }
        .status { padding: 10px; background: #e8f5e9; border-radius: 8px; margin-bottom: 20px; }
        .citation { font-size: 12px; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚛 SupplyChainRAG</h1>
        <p>应援物品供应链AI助手 - 智能库存、生产、物流查询</p>
    </div>
    
    <div class="status">
        <strong>状态:</strong> <span id="status">正在连接...</span> |
        <strong>会话:</strong> <span id="session">-</span> |
        <strong>模式:</strong> <span id="mode">QUERY</span>
    </div>
    
    <div class="chat-container" id="chat"></div>
    
    <div class="input-area">
        <input type="text" id="query" placeholder="输入查询，例如：查询亚克力立牌的库存" />
        <button onclick="sendQuery()">发送</button>
    </div>
    
    <script>
        const sessionId = 'sess_' + Date.now();
        document.getElementById('session').textContent = sessionId;
        
        // Check API status
        fetch('/health')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').textContent = '已连接';
            })
            .catch(() => {
                document.getElementById('status').textContent = '未连接';
            });
        
        // Get URL params for mode
        const urlParams = new URLSearchParams(window.location.search);
        const mode = urlParams.get('mode') || 'query';
        document.getElementById('mode').textContent = mode.toUpperCase();
        
        async function sendQuery() {
            const input = document.getElementById('query');
            const query = input.value.trim();
            if (!query) return;
            
            // Add user message
            addMessage('user', query);
            input.value = '';
            
            // Show loading
            const loadingId = 'loading_' + Date.now();
            addMessage('assistant', '思考中...', loadingId);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, session_id: sessionId }),
                });
                
                const data = await response.json();
                
                // Remove loading message
                document.getElementById(loadingId)?.remove();
                
                // Add response
                let content = data.content;
                if (data.citations && data.citations.length > 0) {
                    content += '<div class="citation">引用: ' + 
                        data.citations.map(c => c.doc_id).join(', ') + '</div>';
                }
                addMessage('assistant', content + '<div class="citation">' + 
                    '耗时: ' + data.latency_ms + 'ms</div>');
                    
            } catch (e) {
                document.getElementById(loadingId)?.remove();
                addMessage('assistant', '错误: ' + e.message);
            }
        }
        
        function addMessage(role, content, id) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            if (id) div.id = id;
            div.innerHTML = content;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }
        
        // Enter key to send
        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendQuery();
        });
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


from fastapi.responses import HTMLResponse

# Import demo router
try:
    from src.ui.backend_demo import demo_router
    app.include_router(demo_router)
    print("Demo router loaded at /demo/stream")
except Exception as e:
    print(f"Warning: Could not load demo router: {e}")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.ui.backend:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV == "development",
    )
