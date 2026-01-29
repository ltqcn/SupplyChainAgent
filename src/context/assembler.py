"""Context assembler for building LLM prompts.

Assembles context from multiple sources with token budget management,
progressive disclosure, and full provenance tracking.

Reference: PRD Section 5 - Context Engineering
"""

import uuid
from datetime import datetime
from typing import Any

from src.config import settings
from src.context.progressive_disclosure import (
    ContentExpander,
    DisclosureLevel,
    ProgressiveDisclosureManager,
)
from src.context.token_budget import DynamicBudgetManager, TokenBudget
from src.models import (
    ContextAssemblyLog,
    ContextChunk,
    MemoryEntry,
    RAGResult,
    SkillDefinition,
)


class ContextAssembler:
    """Assembles context for LLM prompts with full governance.
    
    Assembly Process:
    1. Start with P0: System prompt + user query (fixed)
    2. Add P1: Short-term memory + high-confidence RAG
    3. Add P2: Tool definitions (progressive) + long-term memory
    4. Fill remaining with P3: Low-confidence RAG, examples
    
    Each step logs token usage and source provenance.
    """
    
    def __init__(self):
        """Initialize context assembler."""
        self.budget_manager = DynamicBudgetManager()
        self.disclosure_manager = ProgressiveDisclosureManager()
        self.content_expander = ContentExpander()
        
        # Assembly state
        self.budget: TokenBudget | None = None
        self.chunks: list[ContextChunk] = []
        self.assembly_log: ContextAssemblyLog | None = None
    
    def start_assembly(
        self,
        query_type: str = "general",
        max_tokens: int | None = None,
    ) -> ContextAssemblyLog:
        """Start a new context assembly.
        
        Args:
            query_type: Type of query for budget optimization
            max_tokens: Override max tokens
            
        Returns:
            Initialized assembly log
        """
        self.budget = self.budget_manager.create_budget(query_type)
        if max_tokens:
            self.budget.max_tokens = max_tokens
        
        self.chunks = []
        
        self.assembly_log = ContextAssemblyLog(
            assembly_id=f"ctx-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(),
            max_tokens=self.budget.max_tokens,
        )
        
        return self.assembly_log
    
    def add_system_prompt(
        self,
        prompt: str,
        version: str = "2.3",
    ) -> bool:
        """Add system prompt (P0 - fixed).
        
        Args:
            prompt: System prompt text
            version: Prompt template version
            
        Returns:
            True if added successfully
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started. Call start_assembly() first.")
        
        tokens = self.budget.estimate_tokens(prompt)
        
        chunk = ContextChunk(
            chunk_type="system",
            content=prompt,
            tokens=tokens,
            priority=0,
            source_id=f"system-prompt-{version}",
            source_metadata={"version": version},
        )
        
        self.budget.allocate(tokens, 0, "system_prompt")
        self.chunks.append(chunk)
        
        self.assembly_log.steps.append({
            "step": "add_system_prompt",
            "tokens": tokens,
            "priority": 0,
            "description": f"System prompt v{version}",
        })
        
        return True
    
    def add_user_query(self, query: str) -> bool:
        """Add user query (P0 - fixed).
        
        Args:
            query: User input text
            
        Returns:
            True if added successfully
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        tokens = self.budget.estimate_tokens(query)
        
        chunk = ContextChunk(
            chunk_type="user",
            content=query,
            tokens=tokens,
            priority=0,
            source_id="user-input",
        )
        
        self.budget.allocate(tokens, 0, "user_query")
        self.chunks.append(chunk)
        
        self.assembly_log.steps.append({
            "step": "add_user_query",
            "tokens": tokens,
            "priority": 0,
            "description": "Current user query",
        })
        
        return True
    
    def add_short_term_memory(
        self,
        memories: list[MemoryEntry],
        max_turns: int = 10,
    ) -> int:
        """Add short-term memory (P1 - compressible).
        
        Args:
            memories: List of memory entries
            max_turns: Maximum number of turns to include
            
        Returns:
            Number of tokens added
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        # Sort by recency
        sorted_memories = sorted(
            memories,
            key=lambda m: m.timestamp,
            reverse=True,
        )[:max_turns]
        
        # Build memory text
        memory_texts = []
        total_tokens = 0
        added_memories = []
        
        for memory in sorted_memories:
            text = f"[{memory.source.upper()}] {memory.content}"
            tokens = self.budget.estimate_tokens(text)
            
            # Check if within budget
            if not self.budget.can_allocate(tokens, 1):
                break
            
            if self.budget.allocate(tokens, 1, "short_term_memory"):
                memory_texts.append(text)
                total_tokens += tokens
                added_memories.append(memory)
        
        if memory_texts:
            content = "\n".join(reversed(memory_texts))  # Chronological order
            
            chunk = ContextChunk(
                chunk_type="memory",
                content=f"## Conversation History\n{content}",
                tokens=total_tokens,
                priority=1,
                source_id="short-term-memory",
                source_metadata={
                    "turn_count": len(added_memories),
                    "total_memories": len(memories),
                },
            )
            
            self.chunks.append(chunk)
            
            self.assembly_log.steps.append({
                "step": "add_short_term_memory",
                "tokens": total_tokens,
                "priority": 1,
                "description": f"Recent {len(added_memories)}/{len(memories)} conversation turns",
            })
        
        return total_tokens
    
    def add_rag_results(
        self,
        results: list[RAGResult],
        level: DisclosureLevel = DisclosureLevel.L2_CORE,
        min_confidence: float = 0.7,
    ) -> int:
        """Add RAG retrieval results (P1/P3 based on confidence).
        
        Args:
            results: RAG retrieval results
            level: Disclosure level for content
            min_confidence: Minimum confidence for P1 inclusion
            
        Returns:
            Number of tokens added
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        total_tokens = 0
        added_results = []
        
        for i, result in enumerate(results):
            # Determine priority based on confidence and rank
            if result.confidence >= min_confidence and i < 3:
                priority = 1  # P1: High confidence, top results
            else:
                priority = 3  # P3: Lower confidence
            
            # Create citation
            citation = (
                f"[Source: {result.doc_id}, Type: {result.doc_type}, "
                f"Confidence: {result.confidence:.2f}]"
            )
            
            # Get content based on disclosure level
            if level == DisclosureLevel.L1_METADATA:
                preview = self.disclosure_manager.create_chunk_preview(result, level)
                content = f"{citation}\n{preview.get('summary', result.content[:200])}"
            elif level == DisclosureLevel.L2_CORE:
                content = f"{citation}\n{result.content[:400]}..."
            else:
                content = f"{citation}\n{result.content}"
            
            tokens = self.budget.estimate_tokens(content)
            
            # Check budget
            if not self.budget.can_allocate(tokens, priority):
                if priority == 3:
                    continue  # Skip low priority
                # Try to truncate for P1
                if level != DisclosureLevel.L1_METADATA:
                    content = f"{citation}\n{result.content[:200]}..."
                    tokens = self.budget.estimate_tokens(content)
                    if not self.budget.can_allocate(tokens, priority):
                        continue
            
            if self.budget.allocate(tokens, priority, f"rag_result_{i}"):
                chunk = ContextChunk(
                    chunk_type="rag",
                    content=content,
                    tokens=tokens,
                    priority=priority,
                    source_id=result.doc_id,
                    source_metadata={
                        "doc_type": result.doc_type,
                        "confidence": result.confidence,
                        "retrieval_algo": result.retrieval_algo.value,
                    },
                )
                
                self.chunks.append(chunk)
                total_tokens += tokens
                added_results.append(result)
        
        if added_results:
            self.assembly_log.steps.append({
                "step": "add_rag_results",
                "tokens": total_tokens,
                "priority": "1/3",
                "description": f"Added {len(added_results)}/{len(results)} RAG results",
            })
        
        return total_tokens
    
    def add_skills_index(
        self,
        skills: list[SkillDefinition],
    ) -> int:
        """Add L1 skill index (P2 - selective).
        
        Args:
            skills: Available skills
            
        Returns:
            Number of tokens added
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        # Create L1 index
        skill_index = self.disclosure_manager.create_skill_index(skills)
        
        # Format as XML for clarity
        lines = ["## Available Tools"]
        for skill_meta in skill_index:
            lines.append(f"<tool name=\"{skill_meta['name']}\" category=\"{skill_meta['category']}\">")
            lines.append(f"  {skill_meta['description']}")
            lines.append("</tool>")
        
        content = "\n".join(lines)
        tokens = self.budget.estimate_tokens(content)
        
        if self.budget.can_allocate(tokens, 2):
            if self.budget.allocate(tokens, 2, "skills_index"):
                chunk = ContextChunk(
                    chunk_type="tool",
                    content=content,
                    tokens=tokens,
                    priority=2,
                    source_id="skills-index",
                    source_metadata={"skill_count": len(skills)},
                )
                
                self.chunks.append(chunk)
                
                self.assembly_log.steps.append({
                    "step": "add_skills_index",
                    "tokens": tokens,
                    "priority": 2,
                    "description": f"L1 index of {len(skills)} skills",
                })
                
                return tokens
        
        return 0
    
    def add_expanded_skill(
        self,
        skill: SkillDefinition,
        level: DisclosureLevel = DisclosureLevel.L2_CORE,
    ) -> int:
        """Add expanded skill definition (P2 - selective).
        
        Args:
            skill: Skill to expand
            level: Target disclosure level
            
        Returns:
            Number of tokens added
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        expanded = self.disclosure_manager.expand_skill(skill, level)
        
        # Format as JSON-like structure
        content = f"## Tool: {skill.name}\n```json\n{expanded}\n```"
        tokens = self.budget.estimate_tokens(content)
        
        if self.budget.can_allocate(tokens, 2):
            if self.budget.allocate(tokens, 2, f"skill_{skill.name}"):
                chunk = ContextChunk(
                    chunk_type="tool",
                    content=content,
                    tokens=tokens,
                    priority=2,
                    source_id=f"skill-{skill.name}",
                    source_metadata={"level": level.value},
                )
                
                self.chunks.append(chunk)
                
                self.assembly_log.steps.append({
                    "step": "add_expanded_skill",
                    "tokens": tokens,
                    "priority": 2,
                    "description": f"L{level.value} definition of {skill.name}",
                })
                
                return tokens
        
        return 0
    
    def finalize(self) -> tuple[str, ContextAssemblyLog]:
        """Finalize assembly and return complete context.
        
        Returns:
            Tuple of (assembled context string, assembly log)
        """
        if self.budget is None or self.assembly_log is None:
            raise RuntimeError("Assembly not started")
        
        # Sort chunks by priority
        sorted_chunks = sorted(self.chunks, key=lambda c: c.priority)
        
        # Assemble final context
        sections = []
        for chunk in sorted_chunks:
            if chunk.chunk_type == "system":
                sections.append(f"<system>\n{chunk.content}\n</system>")
            elif chunk.chunk_type == "user":
                sections.append(f"<user>\n{chunk.content}\n</user>")
            elif chunk.chunk_type == "memory":
                sections.append(f"<memory>\n{chunk.content}\n</memory>")
            elif chunk.chunk_type == "rag":
                sections.append(f"<reference>\n{chunk.content}\n</reference>")
            elif chunk.chunk_type == "tool":
                sections.append(chunk.content)
        
        final_context = "\n\n".join(sections)
        
        # Update assembly log
        self.assembly_log.used_tokens = self.budget.total_used
        self.assembly_log.remaining_tokens = self.budget.remaining
        self.assembly_log.chunks = sorted_chunks
        
        # Add final budget check
        self.assembly_log.steps.append({
            "step": "finalize",
            "tokens": self.budget.total_used,
            "priority": "all",
            "description": f"Final context: {self.budget.total_used}/{self.budget.max_tokens} tokens",
        })
        
        return final_context, self.assembly_log
    
    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status."""
        if self.budget is None:
            return {"error": "Assembly not started"}
        return self.budget.get_status()


# Global assembler instance
_assembler: ContextAssembler | None = None


def get_assembler() -> ContextAssembler:
    """Get or create global context assembler."""
    global _assembler
    if _assembler is None:
        _assembler = ContextAssembler()
    return _assembler
