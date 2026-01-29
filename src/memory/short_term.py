"""Short-term memory: Current session conversation history.

Manages 100+ rounds of dialogue with sliding window compression.
Provides token-efficient storage for active conversations.

Reference: PRD Section 7.1.1 - Short-term Memory
"""

import json
from datetime import datetime
from typing import Any

from src.config import settings
from src.models import MemoryEntry, MemoryType


class ShortTermMemory:
    """Short-term memory for active conversation sessions.
    
    Features:
    - Sliding window retention (recent N turns)
    - Token budget enforcement
    - Key entity preservation during compression
    """
    
    def __init__(
        self,
        session_id: str,
        max_tokens: int | None = None,
    ):
        """Initialize short-term memory.
        
        Args:
            session_id: Unique session identifier
            max_tokens: Maximum token budget (default from settings)
        """
        self.session_id = session_id
        self.max_tokens = max_tokens or settings.SHORT_TERM_MAX_TOKENS
        
        self.memories: list[MemoryEntry] = []
        self.turn_counter: int = 0
    
    def add_entry(
        self,
        content: str,
        source: str,
        tokens: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add a new memory entry.
        
        Args:
            content: Memory content text
            source: Source type (user/llm/tool/rag)
            tokens: Pre-calculated token count
            metadata: Additional metadata
            
        Returns:
            Created memory entry
        """
        self.turn_counter += 1
        
        entry = MemoryEntry(
            memory_id=f"stm-{self.session_id}-{self.turn_counter:04d}",
            memory_type=MemoryType.SHORT_TERM,
            content=content,
            tokens=tokens or self._estimate_tokens(content),
            timestamp=datetime.now(),
            session_id=self.session_id,
            turn_idx=self.turn_counter,
            source=source,  # type: ignore
            key_entities=metadata.get("key_entities", []) if metadata else [],
        )
        
        self.memories.append(entry)
        
        # Enforce token budget
        self._enforce_budget()
        
        return entry
    
    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get most recent N memory entries.
        
        Args:
            n: Number of entries to retrieve
            
        Returns:
            List of recent memory entries (oldest first)
        """
        return self.memories[-n:]
    
    def get_all(self) -> list[MemoryEntry]:
        """Get all memory entries."""
        return self.memories.copy()
    
    def get_token_count(self) -> int:
        """Calculate total token count of all memories."""
        return sum(m.tokens for m in self.memories)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4) + 1
    
    def _enforce_budget(self) -> None:
        """Enforce token budget by compressing old memories."""
        total_tokens = self.get_token_count()
        
        while total_tokens > self.max_tokens and len(self.memories) > 5:
            # Compress oldest non-essential memory
            # Keep at least 5 recent turns intact
            for i, memory in enumerate(self.memories[:-5]):
                if memory.tokens > 50:  # Can be compressed
                    # Simple compression: keep first sentence + key entities
                    compressed = self._compress_memory(memory)
                    saved_tokens = memory.tokens - compressed.tokens
                    self.memories[i] = compressed
                    total_tokens -= saved_tokens
                    break
            else:
                # No compressible memory found, remove oldest
                removed = self.memories.pop(0)
                total_tokens -= removed.tokens
    
    def _compress_memory(self, memory: MemoryEntry) -> MemoryEntry:
        """Compress a memory entry to save tokens.
        
        Args:
            memory: Memory to compress
            
        Returns:
            Compressed memory entry
        """
        # Keep first sentence + key entities
        content = memory.content
        first_sentence = content.split("。")[0] + "。"
        
        if memory.key_entities:
            entity_note = f" [涉及: {', '.join(memory.key_entities)}]"
            new_content = first_sentence + entity_note
        else:
            new_content = first_sentence
        
        new_tokens = self._estimate_tokens(new_content)
        
        return MemoryEntry(
            memory_id=memory.memory_id,
            memory_type=memory.memory_type,
            content=new_content,
            embedding=memory.embedding,
            tokens=new_tokens,
            timestamp=memory.timestamp,
            session_id=memory.session_id,
            turn_idx=memory.turn_idx,
            source=memory.source,
            key_entities=memory.key_entities,
        )
    
    def extract_key_entities(self, text: str) -> list[str]:
        """Extract key entities from text for preservation.
        
        Simple heuristic-based extraction for supply chain domain.
        
        Args:
            text: Input text
            
        Returns:
            List of key entity strings
        """
        entities = []
        
        # Pattern: uppercase codes (SKU, batch IDs)
        import re
        
        # SKU patterns (e.g., YKL-LP-001)
        sku_matches = re.findall(r'[A-Z]{2,}-[A-Z]{2}-\d{3}', text)
        entities.extend(sku_matches)
        
        # Batch patterns (e.g., F01-20260128-001)
        batch_matches = re.findall(r'F\d{2}-\d{8}-\d{3}', text)
        entities.extend(batch_matches)
        
        # Order patterns (e.g., ORD-20260128-001)
        order_matches = re.findall(r'ORD-\d{8}-\d{5}', text)
        entities.extend(order_matches)
        
        return list(set(entities))  # Deduplicate
    
    def to_context_format(self) -> str:
        """Convert memories to context-ready string format."""
        lines = []
        for memory in self.memories:
            prefix = {
                "user": "用户",
                "llm": "助手",
                "tool": "工具",
                "rag": "知识库",
            }.get(memory.source, memory.source)
            
            lines.append(f"[{prefix}] {memory.content}")
        
        return "\n".join(lines)
    
    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.turn_counter = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "total_entries": len(self.memories),
            "total_tokens": self.get_token_count(),
            "max_tokens": self.max_tokens,
            "utilization": f"{self.get_token_count() / self.max_tokens:.1%}",
            "turn_counter": self.turn_counter,
        }


class ConversationBuffer:
    """Ring buffer for efficient conversation storage."""
    
    def __init__(self, max_turns: int = 100):
        """Initialize conversation buffer.
        
        Args:
            max_turns: Maximum number of turns to retain
        """
        self.max_turns = max_turns
        self.buffer: list[dict[str, Any]] = []
    
    def add_turn(self, user_input: str, assistant_output: str) -> None:
        """Add a conversation turn.
        
        Args:
            user_input: User message
            assistant_output: Assistant response
        """
        self.buffer.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_output,
        })
        
        # Remove oldest if over limit
        if len(self.buffer) > self.max_turns:
            self.buffer.pop(0)
    
    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation turns."""
        return self.buffer[-n:]
    
    def get_all(self) -> list[dict[str, Any]]:
        """Get all conversation turns."""
        return self.buffer.copy()
