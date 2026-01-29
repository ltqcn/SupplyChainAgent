"""Enhanced demo mode with real Kimi SDK integration.

This demo mode streams real-time reasoning events including:
1. Real intent classification via Kimi SDK
2. RAG retrieval from actual indices
3. Context assembly with token budgeting
4. Real LLM generation via Kimi SDK with output parsing
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import AsyncIterator, Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.config import settings
from src.agent.llm_client import get_kimi_client
from src.rag.retriever import get_retriever
from src.context.assembler import ContextAssembler
from src.memory.offload import MemoryManager

# Demo queries
DEMO_QUERIES = [
    "查询灯牌SKU-LP-001的库存情况",
    "批次F01-20260128-001的生产进度如何？",
    "查看上海仓库的所有库存",
    "演唱会应援棒还有多久到货？",
    "供应商SUP001最近的表现怎么样？",
    "物流单WB123456789现在在哪里？",
    "是否有延迟的生产批次？",
    "推荐一个靠谱的亚克力供应商",
    "查询订单ORD-20260128-00001的状态",
    "下个月的演唱会物料准备情况",
    "哪种应援物品库存最充足？",
    "所有在途物流的状态汇总",
    "哪些SKU低于安全库存？",
    "上海到北京的物流时效如何？",
    "最近的质检报告有什么异常？",
]


def generate_demo_query() -> str:
    return random.choice(DEMO_QUERIES)


def parse_llm_output(content: str) -> dict:
    """Parse LLM output to extract actions, risks, and recommendations.
    
    Args:
        content: Raw LLM response content
        
    Returns:
        Dictionary with parsed actions, risks, and recommendations
    """
    actions = []
    risks = []
    recommendations = []
    
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section markers
        if '【行动】' in line or '行动：' in line or '行动:' in line:
            current_section = 'action'
            # Extract content after marker
            text = line.split('】', 1)[-1].split('：', 1)[-1].split(':', 1)[-1].strip('- ')
            if text:
                actions.append(text)
        elif '【风险】' in line or '风险：' in line or '风险:' in line:
            current_section = 'risk'
            text = line.split('】', 1)[-1].split('：', 1)[-1].split(':', 1)[-1].strip('- ')
            if text:
                risks.append(text)
        elif '【建议】' in line or '建议：' in line or '建议:' in line or '【推荐】' in line:
            current_section = 'recommendation'
            text = line.split('】', 1)[-1].split('：', 1)[-1].split(':', 1)[-1].strip('- ')
            if text:
                recommendations.append(text)
        elif line.startswith('- ') or line.startswith('• '):
            text = line[2:].strip()
            if current_section == 'action':
                actions.append(text)
            elif current_section == 'risk':
                risks.append(text)
            elif current_section == 'recommendation':
                recommendations.append(text)
        elif '建议' in line and '：' in line:
            # Inline recommendation
            text = line.split('：', 1)[-1].strip()
            if text and len(text) > 5:
                recommendations.append(text)
    
    # Fallback: if no structured sections found, try keyword extraction
    if not actions and not risks and not recommendations:
        for line in lines:
            line = line.strip()
            if any(kw in line for kw in ['建议', '推荐', '应当', '需要']):
                if len(line) > 10 and len(line) < 200:
                    recommendations.append(line)
    
    return {
        "actions": actions if actions else ["（无特定行动）"],
        "risks": risks if risks else ["（未发现明显风险）"],
        "recommendations": recommendations if recommendations else ["（按正常流程处理）"],
    }


# Pre-built system prompt
SYSTEM_PROMPT = """你是应援物品供应链智能助手，专注于以下业务领域：

【业务范围】
- 物料采购：供应商评估、采购订单管理、入库质检
- 生产制造：工厂排期、生产进度跟踪、质量验收
- 仓储管理：库存查询、库位管理、出入库记录
- 物流配送：承运商对接、在途跟踪、异常预警
- 粉丝订单：订单履约、发货策略、售后处理

【决策权限】
- Level 3（全自动）：库存查询、物流跟踪、订单状态查询
- Level 2（辅助决策）：供应商选择建议、生产排期优化
- Level 1（人工主导）：新供应商准入、大额采购审批、质量事故处理

【回答规范】
1. 所有结论必须引用提供的数据来源
2. 不确定的信息明确标注"信息不足"
3. 高风险建议提示"需人工确认"
4. 主动识别供应链风险并预警

【输出格式】
请在回答中包含以下部分：
【行动】- 需要执行的具体操作
【风险】- 识别的潜在风险
【建议】- 给用户的建议
"""


async def demo_stream() -> AsyncIterator[str]:
    """Stream demo events with REAL Kimi SDK integration.
    
    This function:
    1. Uses real KimiClient for intent classification
    2. Performs actual RAG retrieval from indices
    3. Assembles context with proper token budgeting
    4. Calls real Kimi API for LLM generation
    5. Parses output for actions/risks/recommendations
    """
    query_id = 0
    
    # Initialize components
    kimi_client = get_kimi_client()
    retriever = get_retriever()
    memory_manager = MemoryManager("demo_session")
    
    # Check if Kimi API is available
    has_kimi_api = bool(settings.KIMI_API_KEY)
    if not has_kimi_api:
        print("⚠️ Warning: KIMI_API_KEY not set, will use fallback mode")
    
    while True:
        query_id += 1
        query = generate_demo_query()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Start event
        start_event = {
            "type": "start",
            "query_id": query_id,
            "query": query,
            "timestamp": timestamp,
            "message": f"[{timestamp}] 新查询: {query}"
        }
        yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        
        try:
            # Step 1: Intent Analysis (REAL Kimi SDK)
            intent_event = {
                "type": "step",
                "step": "intent",
                "message": "步骤1: 意图分析 (Kimi SDK)...",
                "detail": "使用Kimi模型进行意图分类" if has_kimi_api else "模拟模式: 规则匹配意图"
            }
            yield f"data: {json.dumps(intent_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)
            
            if has_kimi_api:
                try:
                    intent = await kimi_client.classify_intent(query)
                    intent_source = "kimi"
                except Exception as e:
                    print(f"Intent classification failed: {e}, using fallback")
                    intent = generate_mock_intent(query)
                    intent_source = "fallback"
            else:
                intent = generate_mock_intent(query)
                intent_source = "mock"
            
            intent_result = {
                "type": "intent_result",
                "intent": intent.get("primary_intent", "general"),
                "confidence": intent.get("confidence", 0.5),
                "query_type": intent.get("query_type", "hybrid"),
                "entities": intent.get("entities", []),
                "source": intent_source,
                "message": f"意图: {intent.get('primary_intent', 'general')} (置信度: {intent.get('confidence', 0.5):.2f}) [{intent_source}]"
            }
            yield f"data: {json.dumps(intent_result, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Step 2: RAG Retrieval (REAL)
            rag_event = {
                "type": "step", 
                "step": "rag",
                "message": "步骤2: RAG检索 (BM25 + HNSW + IVF-PQ)...",
                "detail": "多路召回 + RFF融合排序..."
            }
            yield f"data: {json.dumps(rag_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)
            
            query_type = intent.get("query_type", "hybrid")
            
            # Try real RAG retrieval
            try:
                rag_results, rag_log = retriever.retrieve(query, top_k=5, query_type=query_type)
                rag_source = "real"
            except Exception as e:
                print(f"RAG retrieval failed: {e}, using fallback")
                rag_results = generate_mock_rag_results(query)
                rag_source = "mock"
            
            rag_result_event = {
                "type": "rag_result",
                "result_count": len(rag_results),
                "bm25_count": len([r for r in rag_results if "BM25" in str(r.retrieval_algo)]),
                "hnsw_count": len([r for r in rag_results if "HNSW" in str(r.retrieval_algo)]),
                "ivf_pq_count": len([r for r in rag_results if "IVF" in str(r.retrieval_algo)]),
                "top_results": [
                    {"doc_id": r.doc_id, "type": r.doc_type, "confidence": round(r.confidence, 3)}
                    for r in rag_results[:3]
                ],
                "source": rag_source,
                "message": f"RAG检索完成: {len(rag_results)}条结果 [{rag_source}]"
            }
            yield f"data: {json.dumps(rag_result_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Step 3: Memory Retrieval
            memory_event = {
                "type": "step",
                "step": "memory",
                "message": "步骤3: 记忆检索...",
                "detail": "检索短期记忆和长期记忆..."
            }
            yield f"data: {json.dumps(memory_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            try:
                memories = memory_manager.retrieve_relevant_memories(query)
                stm_count = len(memories.get("short_term", []))
                ltm_count = len(memories.get("long_term", []))
            except Exception as e:
                print(f"Memory retrieval failed: {e}")
                stm_count, ltm_count = 0, 0
            
            memory_result = {
                "type": "memory_result",
                "short_term_count": stm_count,
                "long_term_count": ltm_count,
                "message": f"记忆检索: {stm_count}条短期, {ltm_count}条长期"
            }
            yield f"data: {json.dumps(memory_result, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Step 4: Context Assembly (Detailed)
            context_event = {
                "type": "step",
                "step": "context",
                "message": "步骤4: Context组装 (详细)...",
                "detail": "Token预算管理 + 渐进式披露..."
            }
            yield f"data: {json.dumps(context_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Detailed context assembly
            assembler = ContextAssembler()
            assembler.start_assembly(query_type=query_type)
            
            # Add system prompt
            assembler.add_system_prompt(SYSTEM_PROMPT, version="2.3")
            step_event = {
                "type": "assembly_step",
                "component": "system_prompt",
                "tokens": assembler.assembly_log.steps[-1]["tokens"],
                "priority": 0,
                "message": f"  ├─ 系统提示词: {assembler.assembly_log.steps[-1]['tokens']} tokens (P0-固定)"
            }
            yield f"data: {json.dumps(step_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)
            
            # Add RAG results to context
            if rag_results:
                assembler.add_rag_results(rag_results, level="L2")
                rag_tokens = assembler.assembly_log.steps[-1]["tokens"]
                step_event = {
                    "type": "assembly_step",
                    "component": "rag_results",
                    "tokens": rag_tokens,
                    "priority": 1,
                    "message": f"  ├─ RAG结果: {rag_tokens} tokens, {len(rag_results)}条引用 (P1-高优先级)"
                }
                yield f"data: {json.dumps(step_event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)
            
            # Add user query
            assembler.add_user_query(query)
            step_event = {
                "type": "assembly_step",
                "component": "user_query",
                "tokens": assembler.assembly_log.steps[-1]["tokens"],
                "priority": 0,
                "message": f"  ├─ 用户查询: {assembler.assembly_log.steps[-1]['tokens']} tokens (P0-固定)"
            }
            yield f"data: {json.dumps(step_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)
            
            # Finalize context
            context, assembly_log = assembler.finalize()
            
            context_summary = {
                "type": "context_summary",
                "total_tokens": assembly_log.used_tokens,
                "max_tokens": assembly_log.max_tokens,
                "chunks_count": len(assembly_log.chunks),
                "assembly_steps": [
                    {"component": s["step"].replace("add_", ""), "tokens": s["tokens"], "priority": s["priority"]}
                    for s in assembly_log.steps if s["step"].startswith("add_")
                ],
                "message": f"Context组装完成: {assembly_log.used_tokens}/{assembly_log.max_tokens} tokens, {len(assembly_log.chunks)}个chunks"
            }
            yield f"data: {json.dumps(context_summary, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Step 5: LLM Generation with REAL Kimi SDK
            llm_event = {
                "type": "step",
                "step": "llm",
                "message": "步骤5: LLM生成与输出解析 (Kimi SDK)...",
                "detail": "调用Kimi API生成回复 + 解析Action/风险/建议..." if has_kimi_api else "模拟LLM响应..."
            }
            yield f"data: {json.dumps(llm_event, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.5)
            
            if has_kimi_api:
                try:
                    # REAL Kimi API call
                    llm_response = await kimi_client.generate_with_rag(
                        query=query,
                        context=context,
                        system_prompt=SYSTEM_PROMPT,
                    )
                    content = llm_response.get("content", "")
                    usage = llm_response.get("usage", {})
                    latency_ms = llm_response.get("latency_ms", 0)
                    llm_source = "kimi"
                    
                    # Parse output for actions/risks/recommendations
                    parsed = parse_llm_output(content)
                    actions = parsed["actions"]
                    risks = parsed["risks"]
                    recommendations = parsed["recommendations"]
                    
                except Exception as e:
                    print(f"LLM generation failed: {e}, using fallback")
                    mock_response = generate_mock_llm_response(query)
                    content = mock_response["content"]
                    usage = mock_response["usage"]
                    latency_ms = mock_response["latency_ms"]
                    actions = mock_response["actions"]
                    risks = mock_response["risks"]
                    recommendations = mock_response["recommendations"]
                    llm_source = "fallback"
            else:
                # Mock mode
                mock_response = generate_mock_llm_response(query)
                content = mock_response["content"]
                usage = mock_response["usage"]
                latency_ms = mock_response["latency_ms"]
                actions = mock_response["actions"]
                risks = mock_response["risks"]
                recommendations = mock_response["recommendations"]
                llm_source = "mock"
            
            llm_result = {
                "type": "llm_result",
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "tokens_used": usage,
                "latency_ms": latency_ms,
                "actions": actions,
                "risks": risks,
                "recommendations": recommendations,
                "source": llm_source,
                "message": f"LLM生成完成: {latency_ms}ms, 解析出 {len(actions)} actions, {len(risks)} risks, {len(recommendations)} recommendations [{llm_source}]"
            }
            yield f"data: {json.dumps(llm_result, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
            
            # Final result with full details
            result_event = {
                "type": "result",
                "query_id": query_id,
                "query": query,
                "response": content[:300] + "..." if len(content) > 300 else content,
                "citations": len(rag_results),
                "latency_ms": latency_ms,
                "context_tokens": assembly_log.used_tokens,
                "actions": actions,
                "risks": risks,
                "recommendations": recommendations,
                "llm_source": llm_source,
                "message": f"✅ 完成: {content[:50]}..."
            }
            yield f"data: {json.dumps(result_event, ensure_ascii=False)}\n\n"
            
            # Persist to memory
            try:
                memory_manager.add_to_short_term(
                    content=query,
                    source="user",
                    metadata={"intent": intent.get("primary_intent")},
                )
                memory_manager.add_to_short_term(
                    content=content,
                    source="llm",
                    metadata={"rag_count": len(rag_results)},
                )
            except Exception as e:
                print(f"Memory persistence failed: {e}")
            
        except Exception as e:
            import traceback
            error_event = {
                "type": "error",
                "query_id": query_id,
                "message": f"错误: {str(e)}",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            print(f"Demo error: {e}\n{traceback.format_exc()}")
        
        # Wait 3 seconds
        wait_event = {
            "type": "wait",
            "message": "等待3秒后开始下一个查询..."
        }
        yield f"data: {json.dumps(wait_event, ensure_ascii=False)}\n\n"
        await asyncio.sleep(3)


# Fallback functions for when components fail
def generate_mock_intent(query: str) -> dict:
    """Generate mock intent classification as fallback."""
    if "库存" in query or "到货" in query:
        return {"primary_intent": "inventory_query", "confidence": 0.92, "query_type": "exact_lookup", "entities": [{"type": "sku", "value": "SKU-LP-001"}]}
    elif "批次" in query or "生产" in query:
        return {"primary_intent": "production_query", "confidence": 0.88, "query_type": "exact_lookup", "entities": [{"type": "batch", "value": "F01-20260128-001"}]}
    elif "物流" in query or "单" in query:
        return {"primary_intent": "logistics_query", "confidence": 0.90, "query_type": "exact_lookup", "entities": [{"type": "logistics", "value": "WB123456789"}]}
    elif "供应商" in query:
        return {"primary_intent": "supplier_query", "confidence": 0.85, "query_type": "semantic_search", "entities": [{"type": "supplier", "value": "SUP001"}]}
    elif "订单" in query:
        return {"primary_intent": "order_query", "confidence": 0.87, "query_type": "exact_lookup", "entities": [{"type": "order", "value": "ORD-20260128-00001"}]}
    else:
        return {"primary_intent": "general", "confidence": 0.75, "query_type": "hybrid", "entities": []}


def generate_mock_rag_results(query: str) -> list:
    """Generate mock RAG results as fallback."""
    from src.models import RAGResult, RetrievalAlgorithm
    
    if "库存" in query:
        return [
            RAGResult(doc_id="INV-001", content="灯牌SKU-LP-001库存: 850件", doc_type="inventory", confidence=0.95, retrieval_algo=RetrievalAlgorithm.BM25, raw_score=2.5, normalized_score=0.95),
            RAGResult(doc_id="INV-002", content="上海仓库总库存: 5000件", doc_type="inventory", confidence=0.88, retrieval_algo=RetrievalAlgorithm.HNSW, raw_score=0.88, normalized_score=0.88),
        ]
    elif "批次" in query or "生产" in query:
        return [
            RAGResult(doc_id="BATCH-001", content="批次F01-20260128-001: 生产进度78%", doc_type="batch", confidence=0.96, retrieval_algo=RetrievalAlgorithm.BM25, raw_score=2.8, normalized_score=0.96),
        ]
    elif "物流" in query:
        return [
            RAGResult(doc_id="LOG-001", content="运单WB123456789: 运输中，上海分拨中心", doc_type="logistics", confidence=0.97, retrieval_algo=RetrievalAlgorithm.BM25, raw_score=3.0, normalized_score=0.97),
        ]
    else:
        return [
            RAGResult(doc_id="DOC-001", content="供应链整体状况正常", doc_type="general", confidence=0.75, retrieval_algo=RetrievalAlgorithm.HNSW, raw_score=0.75, normalized_score=0.75),
        ]


def generate_mock_llm_response(query: str) -> dict:
    """Generate mock LLM response as fallback."""
    
    if "库存" in query:
        content = """【库存状态】
- 灯牌SKU-LP-001: 当前库存 850件，安全库存500件，状态正常

【行动】建议联系仓库确认实际可发货数量
【风险】大促期间可能出现临时缺货
【建议】保持与供应商的密切沟通，提前备货"""
        actions = ["联系仓库确认实际可发货数量", "更新库存预警阈值"]
        risks = ["大促期间可能出现临时缺货", "物流延迟风险"]
        recommendations = ["建议提前备货至1200件", "设置库存预警自动通知"]
    
    elif "批次" in query or "生产" in query:
        content = """【生产状态】
- 批次F01-20260128-001: 生产中，当前完成78%
- 预计完工: 2026-01-30

【行动】建议联系工厂确认具体排期
【风险】原料供应可能有2天延迟
【建议】提前准备替代供应商方案"""
        actions = ["联系工厂确认具体排期", "准备替代供应商方案"]
        risks = ["原料供应可能有2天延迟", "质检环节时间不可控"]
        recommendations = ["建议提前2天下单备料", "与工厂协商加班赶工"]
    
    elif "物流" in query:
        content = """【物流状态】
- 运单WB123456789: 运输中
- 当前位置: 上海分拨中心
- 预计到达: 2026-01-30 14:00

【行动】建议联系承运商确认实时位置
【风险】天气原因可能导致延误
【建议】提前通知收货方准备接货"""
        actions = ["联系承运商确认实时位置", "通知收货方准备接货"]
        risks = ["天气原因可能导致延误", "末端配送人手不足"]
        recommendations = ["建议预留1天缓冲时间", "准备备选配送方案"]
    
    elif "供应商" in query:
        content = """【供应商SUP001绩效】
- 准时交付率: 94.5%
- 质量合格率: 98.2%
- 综合评分: A级

【行动】建议继续深化合作关系
【风险】单一供应商依赖度过高
【建议】开发备选供应商降低风险"""
        actions = ["继续深化与SUP001合作", "开发备选供应商"]
        risks = ["单一供应商依赖度过高", "供应商产能饱和"]
        recommendations = ["建议引入2-3家备选供应商", "签订长期合作协议锁定产能"]
    
    else:
        content = """【整体状况】
- 库存水平: 正常
- 生产进度: 按计划进行
- 物流状态: 无异常

【行动】建议定期审查供应链KPI
【风险】未发现明显风险
【建议】继续保持当前运营节奏"""
        actions = ["定期审查供应链KPI", "更新运营数据"]
        risks = ["未发现明显风险"]
        recommendations = ["继续保持当前运营节奏"]
    
    return {
        "content": content,
        "usage": {"prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550},
        "latency_ms": random.randint(800, 1500),
        "actions": actions,
        "risks": risks,
        "recommendations": recommendations
    }


demo_router = APIRouter(prefix="/demo", tags=["demo"])


@demo_router.get("/stream")
async def demo_stream_endpoint():
    return StreamingResponse(
        demo_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
