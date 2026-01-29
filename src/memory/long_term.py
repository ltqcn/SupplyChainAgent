"""Long-term memory: Cross-session summaries and key events.

Persists important conversation summaries for future retrieval.
Uses vector embeddings for semantic similarity search.

Reference: PRD Section 7.1.2 - Long-term Memory
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.models import MemoryType, SessionSummary


class LongTermMemory:
    """Long-term memory for cross-session persistence.
    
    Stores session summaries with embeddings for semantic retrieval.
    Automatically archives old summaries into monthly summaries.
    """
    
    def __init__(self, storage_dir: Path | None = None):
        """Initialize long-term memory.
        
        Args:
            storage_dir: Directory for persistent storage
        """
        self.storage_dir = storage_dir or (settings.PROJECT_ROOT / "memory" / "long_term")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.summaries: list[SessionSummary] = []
        # Vectorizer is passed in from MemoryManager to avoid duplicate model loading
        
        # Load existing summaries
        self._load_summaries()
    
    def _load_summaries(self) -> None:
        """Load existing summaries from disk."""
        summaries_file = self.storage_dir / "summaries.pkl"
        if summaries_file.exists():
            try:
                with open(summaries_file, "rb") as f:
                    self.summaries = pickle.load(f)
                print(f"Loaded {len(self.summaries)} long-term summaries")
            except Exception as e:
                print(f"Error loading summaries: {e}")
                self.summaries = []
    
    def _save_summaries(self) -> None:
        """Save summaries to disk."""
        summaries_file = self.storage_dir / "summaries.pkl"
        with open(summaries_file, "wb") as f:
            pickle.dump(self.summaries, f)
    
    def add_session_summary(
        self,
        session_id: str,
        summary: str,
        key_entities: list[str],
        decisions: list[dict[str, Any]],
    ) -> SessionSummary:
        """Add a new session summary.
        
        Args:
            session_id: Session identifier
            summary: Text summary of the session
            key_entities: Key entities mentioned
            decisions: Important decisions made
            
        Returns:
            Created session summary
        """
        # Generate embedding for the summary (use shared vectorizer from MemoryManager)
        from src.memory.offload import MemoryManager
        if MemoryManager._vectorizer is None:
            from src.data.vectorizer import DocumentVectorizer
            MemoryManager._vectorizer = DocumentVectorizer()
        embedding = MemoryManager._vectorizer.encode_single(summary).tolist()
        
        session_summary = SessionSummary(
            session_id=session_id,
            summary=summary,
            key_entities=key_entities,
            decisions=decisions,
            embedding=embedding,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
        )
        
        self.summaries.append(session_summary)
        self._save_summaries()
        
        return session_summary
    
    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.6,
    ) -> list[SessionSummary]:
        """Retrieve relevant summaries using semantic similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant session summaries
        """
        if not self.summaries:
            return []
        
        # Encode query (use shared vectorizer from MemoryManager)
        from src.memory.offload import MemoryManager
        if MemoryManager._vectorizer is None:
            from src.data.vectorizer import DocumentVectorizer
            MemoryManager._vectorizer = DocumentVectorizer()
        query_embedding = MemoryManager._vectorizer.encode_single(query)
        
        # Calculate similarities
        results = []
        for summary in self.summaries:
            if summary.embedding:
                similarity = np.dot(query_embedding, summary.embedding)
                if similarity >= min_similarity:
                    summary.relevance_score = float(similarity)
                    results.append(summary)
        
        # Sort by similarity and return top-k
        results.sort(key=lambda s: s.relevance_score or 0, reverse=True)
        
        # Update last accessed
        for summary in results[:top_k]:
            summary.last_accessed = datetime.now()
        
        self._save_summaries()
        
        return results[:top_k]
    
    def retrieve_by_entity(self, entity: str) -> list[SessionSummary]:
        """Retrieve summaries containing a specific entity.
        
        Args:
            entity: Entity to search for
            
        Returns:
            List of matching summaries
        """
        return [
            s for s in self.summaries
            if entity in s.key_entities or entity in s.summary
        ]
    
    def merge_old_summaries(self, days_threshold: int = 30) -> int:
        """Merge summaries older than threshold into monthly summaries.
        
        Args:
            days_threshold: Age in days to trigger merge
            
        Returns:
            Number of summaries merged
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        old_summaries = [s for s in self.summaries if s.created_at < cutoff_date]
        remaining = [s for s in self.summaries if s.created_at >= cutoff_date]
        
        if len(old_summaries) < 5:
            return 0  # Not enough to merge
        
        # Group by month
        by_month: dict[str, list[SessionSummary]] = {}
        for s in old_summaries:
            month_key = s.created_at.strftime("%Y-%m")
            by_month.setdefault(month_key, []).append(s)
        
        merged_count = 0
        for month, month_summaries in by_month.items():
            if len(month_summaries) < 3:
                continue  # Keep individual if too few
            
            # Create merged summary
            all_entities = set()
            all_decisions = []
            summary_parts = []
            
            for s in month_summaries:
                summary_parts.append(s.summary)
                all_entities.update(s.key_entities)
                all_decisions.extend(s.decisions)
            
            merged_summary = f"[{month}月度摘要] 共{len(month_summaries)}次会话: " + "; ".join(summary_parts[:3])
            
            merged = SessionSummary(
                session_id=f"merged-{month}",
                summary=merged_summary,
                key_entities=list(all_entities)[:20],
                decisions=all_decisions[:10],
                created_at=month_summaries[0].created_at,
                last_accessed=datetime.now(),
            )
            
            remaining.append(merged)
            merged_count += len(month_summaries)
        
        self.summaries = remaining
        self._save_summaries()
        
        return merged_count
    
    def get_stats(self) -> dict[str, Any]:
        """Get long-term memory statistics."""
        total_size = len(self.summaries)
        
        # Calculate age distribution
        now = datetime.now()
        ages = [(now - s.created_at).days for s in self.summaries]
        
        return {
            "total_summaries": total_size,
            "avg_age_days": sum(ages) / len(ages) if ages else 0,
            "storage_dir": str(self.storage_dir),
            "oldest_summary_days": max(ages) if ages else 0,
        }


class PermanentKnowledge:
    """Permanent knowledge base for business rules and SOPs.
    
    Stores business rules, standard operating procedures, and
    historical case studies as referenceable knowledge.
    """
    
    def __init__(self, storage_dir: Path | None = None):
        """Initialize permanent knowledge base.
        
        Args:
            storage_dir: Directory for knowledge storage
        """
        self.storage_dir = storage_dir or (settings.PROJECT_ROOT / "memory" / "permanent")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.rules: dict[str, dict[str, Any]] = {}
        self.sops: dict[str, dict[str, Any]] = {}
        self.cases: list[dict[str, Any]] = []
        
        self._load_knowledge()
    
    def _load_knowledge(self) -> None:
        """Load knowledge from files."""
        # Load business rules
        rules_file = self.storage_dir / "business_rules.json"
        if rules_file.exists():
            with open(rules_file, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
        
        # Load SOPs
        sops_file = self.storage_dir / "sops.json"
        if sops_file.exists():
            with open(sops_file, "r", encoding="utf-8") as f:
                self.sops = json.load(f)
        
        # Load cases
        cases_file = self.storage_dir / "cases.json"
        if cases_file.exists():
            with open(cases_file, "r", encoding="utf-8") as f:
                self.cases = json.load(f)
    
    def get_business_rule(self, rule_id: str) -> dict[str, Any] | None:
        """Get a business rule by ID."""
        return self.rules.get(rule_id)
    
    def get_sop(self, sop_id: str) -> dict[str, Any] | None:
        """Get an SOP by ID."""
        return self.sops.get(sop_id)
    
    def search_cases(
        self,
        keywords: list[str],
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search historical cases by keywords.
        
        Args:
            keywords: Keywords to search for
            max_results: Maximum number of results
            
        Returns:
            List of matching cases
        """
        results = []
        for case in self.cases:
            case_text = json.dumps(case, ensure_ascii=False)
            score = sum(1 for kw in keywords if kw in case_text)
            if score > 0:
                results.append((score, case))
        
        # Sort by relevance
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [case for _, case in results[:max_results]]
    
    def add_case(self, case: dict[str, Any]) -> None:
        """Add a new historical case."""
        case["added_at"] = datetime.now().isoformat()
        self.cases.append(case)
        
        # Save to disk
        cases_file = self.storage_dir / "cases.json"
        with open(cases_file, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> dict[str, Any]:
        """Get permanent knowledge statistics."""
        return {
            "business_rules": len(self.rules),
            "sops": len(self.sops),
            "historical_cases": len(self.cases),
        }
