"""Token budget management for Context Engineering.

Implements hierarchical priority queues and dynamic allocation
strategies for limited context windows.

Reference: PRD Section 5.1.3 - Token Budget Management
"""

from dataclasses import dataclass, field
from typing import Literal

from src.config import settings


@dataclass
class TokenBudget:
    """Token budget allocation for context components.
    
    Priority Levels (P0 highest, P3 lowest):
        P0: System prompt, user query (fixed, non-truncatable)
        P1: Short-term memory, high-confidence RAG results
        P2: Tool descriptions (progressive), long-term memory
        P3: Low-confidence RAG, auxiliary examples
    
    Budget Distribution (8K total example):
        P0: 600 tokens (fixed)
        P1: 3,000 tokens (compressible)
        P2: 2,000 tokens (selective)
        P3: Remainder (disposable)
    """
    
    max_tokens: int = field(default=8000)
    safety_tokens: int = field(default=12000)
    
    # Current allocation
    p0_used: int = field(default=0)  # System + user (fixed)
    p1_used: int = field(default=0)  # Memory + high RAG
    p2_used: int = field(default=0)  # Tools + long-term
    p3_used: int = field(default=0)  # Low RAG + examples
    
    # Limits
    p1_limit: int = field(default=3000)
    p2_limit: int = field(default=2000)
    
    def __post_init__(self):
        """Set limits based on max_tokens."""
        self.p1_limit = int(self.max_tokens * 0.375)  # 37.5%
        self.p2_limit = int(self.max_tokens * 0.25)   # 25%
    
    @property
    def total_used(self) -> int:
        """Total tokens currently used."""
        return self.p0_used + self.p1_used + self.p2_used + self.p3_used
    
    @property
    def remaining(self) -> int:
        """Remaining tokens in budget."""
        return self.max_tokens - self.total_used
    
    @property
    def utilization_rate(self) -> float:
        """Current budget utilization rate."""
        return self.total_used / self.max_tokens if self.max_tokens > 0 else 0.0
    
    def can_allocate(self, tokens: int, priority: int) -> bool:
        """Check if tokens can be allocated at given priority.
        
        Args:
            tokens: Number of tokens to allocate
            priority: Priority level (0-3)
            
        Returns:
            True if allocation is possible
        """
        # P0 always allowed (but tracked)
        if priority == 0:
            return True
        
        # Check priority-specific limits
        if priority == 1 and self.p1_used + tokens > self.p1_limit:
            return False
        if priority == 2 and self.p2_used + tokens > self.p2_limit:
            return False
        
        # Check total budget
        return self.total_used + tokens <= self.max_tokens
    
    def allocate(self, tokens: int, priority: int, component: str) -> bool:
        """Allocate tokens and track usage.
        
        Args:
            tokens: Number of tokens to allocate
            priority: Priority level (0-3)
            component: Component name for tracking
            
        Returns:
            True if allocation succeeded
        """
        if not self.can_allocate(tokens, priority):
            return False
        
        if priority == 0:
            self.p0_used += tokens
        elif priority == 1:
            self.p1_used += tokens
        elif priority == 2:
            self.p2_used += tokens
        else:
            self.p3_used += tokens
        
        return True
    
    def get_status(self) -> dict:
        """Get current budget status for logging/display."""
        return {
            "max_tokens": self.max_tokens,
            "total_used": self.total_used,
            "remaining": self.remaining,
            "utilization": f"{self.utilization_rate:.1%}",
            "breakdown": {
                "P0 (Fixed)": f"{self.p0_used} tokens",
                "P1 (Memory/RAG)": f"{self.p1_used}/{self.p1_limit} tokens",
                "P2 (Tools)": f"{self.p2_used}/{self.p2_limit} tokens",
                "P3 (Auxiliary)": f"{self.p3_used} tokens",
            }
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses approximate heuristic: 1 token ≈ 1.5 Chinese chars or 4 English chars
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple estimation: mix of Chinese and English
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        
        # Chinese: ~1.5 chars per token, Other: ~4 chars per token
        return int(chinese_chars / 1.5 + other_chars / 4) + 1


class DynamicBudgetManager:
    """Dynamic budget manager that adjusts allocation based on query type.
    
    Different query types have different token requirements:
    - Fact query: More RAG budget
    - Tool query: More tool description budget
    - Summary: More memory budget
    """
    
    def __init__(self, base_budget: int | None = None):
        """Initialize with base budget.
        
        Args:
            base_budget: Base token budget (default from settings)
        """
        self.base_budget = base_budget or settings.MAX_CONTEXT_TOKENS
    
    def create_budget(
        self,
        query_type: Literal["fact", "tool", "summary", "general"] = "general",
    ) -> TokenBudget:
        """Create budget optimized for query type.
        
        Args:
            query_type: Type of query being processed
            
        Returns:
            Configured TokenBudget
        """
        budget = TokenBudget(max_tokens=self.base_budget)
        
        if query_type == "fact":
            # Fact queries need more RAG context
            budget.p1_limit = int(self.base_budget * 0.5)  # 50% for RAG
            budget.p2_limit = int(self.base_budget * 0.15)  # 15% for tools
        elif query_type == "tool":
            # Tool queries need more tool descriptions
            budget.p1_limit = int(self.base_budget * 0.3)
            budget.p2_limit = int(self.base_budget * 0.35)  # 35% for tools
        elif query_type == "summary":
            # Summary queries need more memory
            budget.p1_limit = int(self.base_budget * 0.45)
            budget.p2_limit = int(self.base_budget * 0.2)
        
        return budget
