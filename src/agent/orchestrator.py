"""Agent orchestrator for end-to-end query processing.

Coordinates all components:
1. Intent classification
2. RAG retrieval
3. Memory retrieval
4. Skill matching
5. Context assembly
6. LLM generation
7. Tool execution

Reference: PRD Section 3 - System Architecture
"""

import time
import uuid
from datetime import datetime
from typing import Any

from src.config import settings
from src.context.assembler import ContextAssembler
from src.memory.offload import MemoryManager
from src.models import AgentResponse, ChatMessage
from src.rag.retriever import UnifiedRetriever
from src.agent.llm_client import get_kimi_client
from src.tools.skill_manager import get_skill_manager
from src.tools.tool_executor import get_tool_executor


class AgentOrchestrator:
    """Main orchestrator for supply chain AI assistant.
    
    Implements the complete pipeline:
    Input → Intent → RAG/Memory/Skill → Context → LLM → Output
    """
    
    def __init__(self, session_id: str | None = None):
        """Initialize agent orchestrator.
        
        Args:
            session_id: Optional session identifier
        """
        self.session_id = session_id or f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Component references
        self.retriever = UnifiedRetriever()
        self.memory_manager = MemoryManager(self.session_id)
        self.skill_manager = get_skill_manager()
        self.tool_executor = get_tool_executor()
        self.llm_client = get_kimi_client()
        
        # State
        self.conversation_history: list[ChatMessage] = []
        self.round_counter = 0
    
    async def process(
        self,
        query: str,
        mode: str = "standard",
    ) -> AgentResponse:
        """Process a user query through the complete pipeline.
        
        Args:
            query: User query text
            mode: Processing mode (standard/round)
            
        Returns:
            Agent response
        """
        start_time = time.time()
        self.round_counter += 1
        
        # Step 1: Intent Classification
        print(f"\n[Agent] Round {self.round_counter}: Processing query '{query[:50]}...'")
        intent = await self.llm_client.classify_intent(query)
        print(f"[Agent] Intent: {intent.get('primary_intent')} (confidence: {intent.get('confidence', 0):.2f})")
        
        # Step 2: RAG Retrieval
        query_type = intent.get("query_type", "hybrid")
        rag_results, rag_log = self.retriever.retrieve(
            query, top_k=5, query_type=query_type
        )
        print(f"[Agent] RAG retrieved {len(rag_results)} results")
        
        # Step 3: Memory Retrieval
        memories = self.memory_manager.retrieve_relevant_memories(query)
        short_term_memories = memories.get("short_term", [])
        long_term_memories = memories.get("long_term", [])
        print(f"[Agent] Memory: {len(short_term_memories)} short-term, {len(long_term_memories)} long-term")
        
        # Step 4: Skill Matching
        matched_skills = self.skill_manager.match_skills(query, top_k=3)
        print(f"[Agent] Matched {len(matched_skills)} skills: {[s[0].name for s in matched_skills]}")
        
        # Step 5: Context Assembly
        assembler = ContextAssembler()
        assembler.start_assembly(query_type=query_type)
        
        # Add system prompt
        system_prompt = self._build_system_prompt(matched_skills)
        assembler.add_system_prompt(system_prompt, version="2.3")
        
        # Add memory
        if short_term_memories:
            from src.models import MemoryEntry
            memory_entries = [
                MemoryEntry(
                    memory_id=f"stm-{i}",
                    memory_type="short_term",
                    content=f"{m.role}: {m.content}",
                    timestamp=m.timestamp,
                )
                for i, m in enumerate(short_term_memories[-5:])
            ]
            assembler.add_short_term_memory(memory_entries, max_turns=5)
        
        # Add RAG results
        if rag_results:
            assembler.add_rag_results(rag_results, level="L2")
        
        # Add skill index (L1)
        if matched_skills:
            assembler.add_skills_index([s[0] for s in matched_skills])
        
        # Add user query
        assembler.add_user_query(query)
        
        # Finalize context
        context, assembly_log = assembler.finalize()
        print(f"[Agent] Context assembled: {assembly_log.used_tokens}/{assembly_log.max_tokens} tokens")
        
        # Step 6: LLM Generation
        llm_response = await self.llm_client.generate_with_rag(
            query=query,
            context=context,
        )
        
        content = llm_response.get("content", "")
        
        # Step 7: Tool Execution (if needed)
        tool_calls = []
        if matched_skills and matched_skills[0][1] > 0.8:
            # High confidence skill match, consider execution
            top_skill = matched_skills[0][0]
            if top_skill.autonomy_level.value >= 2:
                # Auto or assisted execution
                tool_result = self.tool_executor.execute(
                    tool_name=top_skill.name,
                    parameters={},  # Would extract from query
                    execution_mode=top_skill.execution_mode,
                )
                tool_calls.append(tool_result)
                print(f"[Agent] Executed tool: {top_skill.name}, status: {tool_result.status}")
        
        # Step 8: Persist to memory
        self.memory_manager.add_to_short_term(
            content=query,
            source="user",
            metadata={"intent": intent.get("primary_intent")},
        )
        self.memory_manager.add_to_short_term(
            content=content,
            source="llm",
            metadata={"rag_count": len(rag_results)},
        )
        
        # Build response
        latency_ms = int((time.time() - start_time) * 1000)
        
        response = AgentResponse(
            response_id=f"resp-{uuid.uuid4().hex[:8]}",
            session_id=self.session_id,
            content=content,
            reasoning=f"Intent: {intent.get('primary_intent')}, RAG: {len(rag_results)} results",
            citations=[
                {
                    "doc_id": r.doc_id,
                    "type": r.doc_type,
                    "confidence": r.confidence,
                }
                for r in rag_results[:3]
            ],
            tool_calls=tool_calls,
            risk_level="low" if not tool_calls else "medium",
            requires_approval=any(t.status == "pending" for t in tool_calls),
            tokens_used=llm_response.get("usage", {}),
            latency_ms=latency_ms,
        )
        
        # Update conversation history
        self.conversation_history.append(ChatMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            role="user",
            content=query,
        ))
        self.conversation_history.append(ChatMessage(
            message_id=response.response_id,
            role="assistant",
            content=content,
            citations=response.citations,
            tool_calls=tool_calls,
        ))
        
        print(f"[Agent] Response generated in {latency_ms}ms\n")
        
        return response
    
    def _build_system_prompt(
        self,
        matched_skills: list,
    ) -> str:
        """Build system prompt with available tools.
        
        Args:
            matched_skills: Matched skills with scores
            
        Returns:
            System prompt text
        """
        prompt = """你是应援物品供应链智能助手，专注于以下业务领域：

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
"""
        
        if matched_skills:
            prompt += "\n【可用工具】\n"
            for skill, score in matched_skills[:3]:
                prompt += f"- {skill.name}（匹配度：{score:.2f}）: {skill.description[:50]}...\n"
        
        return prompt
    
    def get_conversation_history(self) -> list[ChatMessage]:
        """Get full conversation history."""
        return self.conversation_history
    
    def get_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "session_id": self.session_id,
            "rounds_processed": self.round_counter,
            "conversation_length": len(self.conversation_history),
            "memory_stats": self.memory_manager.short_term.get_stats(),
        }
    
    async def persist_session(self) -> None:
        """Persist session to long-term memory."""
        self.memory_manager.persist_to_long_term()


class RoundModeOrchestrator(AgentOrchestrator):
    """Extended orchestrator for round demonstration mode.
    
    Exposes intermediate steps for visualization in the UI.
    """
    
    def __init__(self, max_rounds: int = 5, session_id: str | None = None):
        """Initialize round mode orchestrator.
        
        Args:
            max_rounds: Maximum number of rounds to execute
            session_id: Optional session identifier
        """
        super().__init__(session_id)
        self.max_rounds = max_rounds
        self.current_round = 0
        self.round_states: list[dict[str, Any]] = []
    
    async def execute_round(
        self,
        query: str,
        round_idx: int,
    ) -> dict[str, Any]:
        """Execute a single round with detailed state capture.
        
        Args:
            query: User query
            round_idx: Current round index
            
        Returns:
            Round state for visualization
        """
        self.current_round = round_idx
        
        # Capture initial state
        state = {
            "round": round_idx,
            "query": query,
            "steps": [],
        }
        
        # Step 1: Intent Analysis
        intent = await self.llm_client.classify_intent(query)
        state["steps"].append({
            "name": "intent_analysis",
            "intent": intent,
            "description": f"识别意图: {intent.get('primary_intent')}",
        })
        
        # Step 2: RAG Retrieval
        query_type = intent.get("query_type", "hybrid")
        rag_results, rag_log = self.retriever.retrieve(query, top_k=5, query_type=query_type)
        state["steps"].append({
            "name": "rag_retrieval",
            "results": [r.model_dump() for r in rag_results],
            "log": rag_log.model_dump(),
            "description": f"RAG检索: BM25/HNSW/IVF-PQ融合，返回{len(rag_results)}条结果",
        })
        
        # Step 3: Memory Retrieval
        memories = self.memory_manager.retrieve_relevant_memories(query)
        state["steps"].append({
            "name": "memory_retrieval",
            "short_term_count": len(memories.get("short_term", [])),
            "long_term_count": len(memories.get("long_term", [])),
            "description": "检索短期记忆和长期记忆",
        })
        
        # Step 4: Context Assembly
        assembler = ContextAssembler()
        assembler.start_assembly()
        
        # Add components
        system_prompt = self._build_system_prompt([])
        assembler.add_system_prompt(system_prompt)
        
        if rag_results:
            assembler.add_rag_results(rag_results)
        
        assembler.add_user_query(query)
        
        context, assembly_log = assembler.finalize()
        
        state["steps"].append({
            "name": "context_assembly",
            "budget": assembly_log.max_tokens,
            "used": assembly_log.used_tokens,
            "chunks": [c.model_dump() for c in assembly_log.chunks],
            "description": f"Context组装: {assembly_log.used_tokens}/{assembly_log.max_tokens} tokens",
        })
        
        # Step 5: LLM Generation (only on final round)
        if round_idx >= self.max_rounds - 1:
            llm_response = await self.llm_client.generate_with_rag(query, context)
            state["steps"].append({
                "name": "llm_generation",
                "content": llm_response.get("content", ""),
                "tokens_used": llm_response.get("usage", {}),
                "latency_ms": llm_response.get("latency_ms", 0),
                "description": "LLM生成最终回答",
            })
        
        self.round_states.append(state)
        return state


# Global orchestrator instances
_orchestrators: dict[str, AgentOrchestrator] = {}


def get_orchestrator(session_id: str | None = None) -> AgentOrchestrator:
    """Get or create orchestrator for session."""
    if session_id is None:
        session_id = f"default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if session_id not in _orchestrators:
        if settings.is_round_mode:
            _orchestrators[session_id] = RoundModeOrchestrator(
                max_rounds=settings.MAX_ROUNDS,
                session_id=session_id,
            )
        else:
            _orchestrators[session_id] = AgentOrchestrator(session_id)
    
    return _orchestrators[session_id]
