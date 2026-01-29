"""Disk-offload mechanism for memory management.

Implements automatic offloading of short-term memory to disk
when token budget is exceeded, with semantic recovery.

Reference: PRD Section 7.2.1 - Disk-offload Strategy
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.data.vectorizer import DocumentVectorizer
from src.models import MemoryEntry, MemoryType, SessionSummary
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory, PermanentKnowledge


class DiskOffloadManager:
    """Manages disk-offload for memory pressure situations.
    
    When short-term memory exceeds threshold:
    1. Summarize oldest turns using LLM
    2. Save summary to disk
    3. Remove original from memory
    4. Keep reference for potential recovery
    """
    
    def __init__(
        self,
        offload_dir: Path | None = None,
        threshold_tokens: int | None = None,
    ):
        """Initialize disk-offload manager.
        
        Args:
            offload_dir: Directory for offloaded memory files
            threshold_tokens: Token threshold to trigger offload
        """
        self.offload_dir = offload_dir or (settings.PROJECT_ROOT / "memory" / "offload")
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        
        self.threshold = threshold_tokens or settings.DISK_OFFLOAD_THRESHOLD
        # Use shared vectorizer from MemoryManager to avoid duplicate model loading
        self.vectorizer = None  # Lazy get from MemoryManager
        
        # Track offloaded sessions
        self.offloaded_files: dict[str, Path] = {}
    
    def check_and_offload(
        self,
        memory: ShortTermMemory,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Check memory size and offload if necessary.
        
        Args:
            memory: Short-term memory to check
            force: Force offload regardless of threshold
            
        Returns:
            Offload metadata if offloading occurred
        """
        token_count = memory.get_token_count()
        
        if not force and token_count < self.threshold:
            return None
        
        # Determine how much to offload (oldest 40% or at least 5 turns)
        total_turns = len(memory.memories)
        offload_count = max(5, int(total_turns * 0.4))
        
        if offload_count >= total_turns - 5:
            offload_count = max(1, total_turns - 5)  # Keep at least 5 recent
        
        if offload_count <= 0:
            return None
        
        # Get turns to offload (oldest)
        turns_to_offload = memory.memories[:offload_count]
        
        # Create summary
        summary_text = self._create_summary(turns_to_offload)
        
        # Generate embedding for recovery (use shared vectorizer)
        from src.data.vectorizer import DocumentVectorizer
        if MemoryManager._vectorizer is None:
            MemoryManager._vectorizer = DocumentVectorizer()
        embedding = MemoryManager._vectorizer.encode_single(summary_text).tolist()
        
        # Extract key entities
        all_entities = []
        for turn in turns_to_offload:
            all_entities.extend(turn.key_entities)
        key_entities = list(set(all_entities))
        
        # Build offload record
        offload_record = {
            "session_id": memory.session_id,
            "offloaded_at": datetime.now().isoformat(),
            "turn_range": f"1-{offload_count}",
            "turn_count": offload_count,
            "tokens_saved": sum(t.tokens for t in turns_to_offload),
            "summary": summary_text,
            "embedding": embedding,
            "key_entities": key_entities,
            "full_turns": [
                {
                    "turn_idx": t.turn_idx,
                    "source": t.source,
                    "content": t.content,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in turns_to_offload
            ],
        }
        
        # Save to disk
        filename = f"{memory.session_id}_rounds_1_{offload_count}.json"
        filepath = self.offload_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(offload_record, f, ensure_ascii=False, indent=2)
        
        self.offloaded_files[memory.session_id] = filepath
        
        # Remove from memory
        del memory.memories[:offload_count]
        
        print(f"[Memory] Offloaded rounds 1-{offload_count} to disk, saved {offload_record['tokens_saved']} tokens")
        
        return offload_record
    
    def _create_summary(self, turns: list[MemoryEntry]) -> str:
        """Create text summary of conversation turns.
        
        Args:
            turns: List of memory entries to summarize
            
        Returns:
            Summary text
        """
        # Simple extractive summary
        key_points = []
        
        for turn in turns:
            if turn.source == "user":
                # Extract user's question/intent
                content = turn.content
                if len(content) > 50:
                    content = content[:50] + "..."
                key_points.append(f"询问: {content}")
            elif turn.source == "llm":
                # Extract key recommendations
                if "建议" in turn.content or "推荐" in turn.content:
                    key_points.append(f"建议: {turn.content[:60]}...")
            elif turn.source == "tool":
                key_points.append(f"执行: {turn.content[:40]}...")
        
        # Take most important points
        if key_points:
            return "; ".join(key_points[:5])
        
        return "对话历史摘要"
    
    def recover_offloaded(
        self,
        session_id: str,
        query_embedding: np.ndarray | None = None,
    ) -> list[MemoryEntry] | None:
        """Recover offloaded memory based on relevance.
        
        Args:
            session_id: Session to recover from
            query_embedding: Query embedding for relevance matching
            
        Returns:
            Recovered memory entries or None
        """
        filepath = self.offloaded_files.get(session_id)
        if not filepath or not filepath.exists():
            # Search for any file matching session
            for f in self.offload_dir.glob(f"{session_id}*.json"):
                filepath = f
                break
        
        if not filepath or not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)
        
        # Check relevance if query provided
        if query_embedding is not None and "embedding" in record:
            stored_embedding = np.array(record["embedding"])
            similarity = np.dot(query_embedding, stored_embedding)
            
            if similarity < 0.5:  # Not relevant enough
                return None
        
        # Convert back to memory entries
        recovered = []
        for turn_data in record.get("full_turns", []):
            entry = MemoryEntry(
                memory_id=f"recovered-{turn_data['turn_idx']}",
                memory_type=MemoryType.SHORT_TERM,
                content=f"[已恢复] {turn_data['content']}",
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                session_id=session_id,
                turn_idx=turn_data["turn_idx"],
                source=turn_data["source"],  # type: ignore
            )
            recovered.append(entry)
        
        print(f"[Memory] Recovered {len(recovered)} offloaded turns")
        
        return recovered
    
    def get_offload_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get summary of offloaded memory without full recovery.
        
        Args:
            session_id: Session to query
            
        Returns:
            Summary metadata or None
        """
        filepath = self.offloaded_files.get(session_id)
        if not filepath or not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)
        
        return {
            "turn_range": record["turn_range"],
            "turn_count": record["turn_count"],
            "summary": record["summary"],
            "key_entities": record["key_entities"],
            "offloaded_at": record["offloaded_at"],
        }
    
    def cleanup_old_offloads(self, days: int = 30) -> int:
        """Remove offload files older than specified days.
        
        Args:
            days: Age threshold for cleanup
            
        Returns:
            Number of files removed
        """
        cutoff = datetime.now() - __import__('datetime').timedelta(days=days)
        removed = 0
        
        for filepath in self.offload_dir.glob("*.json"):
            try:
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    removed += 1
            except Exception:
                pass
        
        return removed
    
    def get_stats(self) -> dict[str, Any]:
        """Get disk-offload statistics."""
        files = list(self.offload_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "offload_dir": str(self.offload_dir),
            "offloaded_sessions": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "threshold_tokens": self.threshold,
        }


class MemoryManager:
    """Unified memory manager coordinating all memory tiers.
    
    Provides a single interface for:
    - Short-term memory operations
    - Long-term memory persistence
    - Permanent knowledge access
    - Disk-offload management
    """
    
    # Class-level shared vectorizer instance to avoid reloading model
    _vectorizer: DocumentVectorizer | None = None
    
    def __init__(self, session_id: str):
        """Initialize memory manager for a session.
        
        Args:
            session_id: Current session identifier
        """
        self.session_id = session_id
        
        self.short_term = ShortTermMemory(session_id)
        self.long_term = LongTermMemory()
        self.permanent = PermanentKnowledge()
        self.offload = DiskOffloadManager()
        
        # Initialize shared vectorizer lazily
        if MemoryManager._vectorizer is None:
            from src.data.vectorizer import DocumentVectorizer
            MemoryManager._vectorizer = DocumentVectorizer()
    
    def add_to_short_term(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Add entry to short-term memory with automatic offload check.
        
        Args:
            content: Memory content
            source: Source type
            metadata: Additional metadata
            
        Returns:
            Created memory entry
        """
        # Check if offload needed
        self.offload.check_and_offload(self.short_term)
        
        # Add entry
        return self.short_term.add_entry(content, source, metadata=metadata)
    
    def persist_to_long_term(self) -> SessionSummary | None:
        """Persist current session to long-term memory.
        
        Returns:
            Created session summary or None
        """
        if len(self.short_term.memories) < 3:
            return None
        
        # Create summary from short-term memory
        summary_parts = []
        key_entities = set()
        decisions = []
        
        for memory in self.short_term.memories:
            if memory.source in ["user", "llm"]:
                summary_parts.append(memory.content[:100])
            key_entities.update(memory.key_entities)
        
        summary = " | ".join(summary_parts[:5])
        
        return self.long_term.add_session_summary(
            session_id=self.session_id,
            summary=summary,
            key_entities=list(key_entities)[:10],
            decisions=decisions,
        )
    
    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: int = 3,
    ) -> dict[str, list]:
        """Retrieve relevant memories from all tiers.
        
        Args:
            query: Query text
            top_k: Number of results per tier
            
        Returns:
            Dictionary with memories from each tier
        """
        # Use shared vectorizer instance (avoids reloading model)
        query_embedding = MemoryManager._vectorizer.encode_single(query)
        
        # Short-term: recent memories
        short_term_memories = self.short_term.get_recent(n=5)
        
        # Long-term: semantic search
        long_term_memories = self.long_term.retrieve_relevant(
            query, top_k=top_k
        )
        
        # Try to recover offloaded if relevant
        recovered = self.offload.recover_offloaded(
            self.session_id, query_embedding
        )
        if recovered:
            short_term_memories = recovered + short_term_memories
        
        return {
            "short_term": short_term_memories,
            "long_term": long_term_memories,
            "offloaded": recovered or [],
        }
