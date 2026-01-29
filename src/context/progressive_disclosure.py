"""Progressive Disclosure implementation for Context Engineering.

Implements Claude Skills-inspired layered loading strategy:
L1: Metadata only (lightweight matching)
L2: Core content (high-recall chunks)
L3: Deep content (full details + best examples)

Reference: PRD Section 5.3 - Progressive Disclosure in RAG
"""

from enum import Enum
from typing import Any

from src.models import RAGResult, SkillDefinition, SkillLevel


class DisclosureLevel(Enum):
    """Progressive disclosure levels."""
    L1_METADATA = "L1"      # Metadata only (names, descriptions)
    L2_CORE = "L2"          # Core content (summaries, key fields)
    L3_DETAILED = "L3"      # Full details + best examples


class ProgressiveDisclosureManager:
    """Manages progressive disclosure for RAG and Skills.
    
    Reduces initial context size by 60-70% compared to full loading.
    """
    
    # Token estimates per level
    SKILL_TOKENS = {
        DisclosureLevel.L1_METADATA: 50,    # Name + brief desc
        DisclosureLevel.L2_CORE: 500,       # + parameters, rules
        DisclosureLevel.L3_DETAILED: 1500,  # + examples, docs
    }
    
    CHUNK_TOKENS = {
        DisclosureLevel.L1_METADATA: 80,    # Title + summary
        DisclosureLevel.L2_CORE: 300,       # + key fields
        DisclosureLevel.L3_DETAILED: 500,   # Full content
    }
    
    def __init__(self):
        """Initialize the progressive disclosure manager."""
        self.current_level: DisclosureLevel = DisclosureLevel.L1_METADATA
    
    def create_chunk_preview(
        self,
        result: RAGResult,
        level: DisclosureLevel = DisclosureLevel.L1_METADATA,
    ) -> dict[str, Any]:
        """Create lightweight chunk preview for L1/L2 loading.
        
        Args:
            result: Full RAG result
            level: Disclosure level
            
        Returns:
            Chunk preview dictionary
        """
        content = result.content
        
        # Extract first paragraph as summary
        summary = content[:150] + "..." if len(content) > 150 else content
        
        # Extract key fields using simple heuristics
        key_fields = {}
        lines = content.split("\n")
        for line in lines:
            if ":" in line and len(line) < 100:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value:
                        key_fields[key] = value
        
        preview = {
            "id": result.doc_id,
            "score": result.normalized_score,
            "doc_type": result.doc_type,
            "confidence": result.confidence,
        }
        
        if level == DisclosureLevel.L1_METADATA:
            # L1: Minimal metadata
            preview["summary"] = summary
            preview["preview_tokens"] = self.CHUNK_TOKENS[DisclosureLevel.L1_METADATA]
            
        elif level == DisclosureLevel.L2_CORE:
            # L2: Core fields
            preview["summary"] = summary
            preview["key_fields"] = dict(list(key_fields.items())[:5])
            preview["preview_tokens"] = self.CHUNK_TOKENS[DisclosureLevel.L2_CORE]
            
        else:
            # L3: Full content
            preview["content"] = content
            preview["key_fields"] = key_fields
            preview["preview_tokens"] = self.CHUNK_TOKENS[DisclosureLevel.L3_DETAILED]
        
        return preview
    
    def create_skill_index(self, skills: list[SkillDefinition]) -> list[dict[str, Any]]:
        """Create L1 skill index (metadata only).
        
        Args:
            skills: Full skill definitions
            
        Returns:
            List of L1 skill metadata
        """
        return [
            {
                "name": skill.name,
                "description": skill.description[:100] + "..." if len(skill.description) > 100 else skill.description,
                "category": skill.category,
                "autonomy_level": skill.autonomy_level.value,
                "estimated_tokens": self.SKILL_TOKENS[DisclosureLevel.L1_METADATA],
            }
            for skill in skills
        ]
    
    def expand_skill(
        self,
        skill: SkillDefinition,
        level: DisclosureLevel = DisclosureLevel.L2_CORE,
    ) -> dict[str, Any]:
        """Expand skill to specified disclosure level.
        
        Args:
            skill: Skill definition
            level: Target disclosure level
            
        Returns:
            Skill at specified level
        """
        result = {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "autonomy_level": skill.autonomy_level.value,
        }
        
        if level in [DisclosureLevel.L2_CORE, DisclosureLevel.L3_DETAILED]:
            result["parameters"] = skill.parameters
            result["business_rules"] = skill.business_rules
            result["estimated_tokens"] = self.SKILL_TOKENS[DisclosureLevel.L2_CORE]
        
        if level == DisclosureLevel.L3_DETAILED:
            result["examples"] = skill.examples[:3]  # Top 3 examples
            result["estimated_tokens"] = self.SKILL_TOKENS[DisclosureLevel.L3_DETAILED]
        
        return result
    
    def calculate_savings(
        self,
        num_skills: int = 20,
        num_rag_results: int = 10,
    ) -> dict[str, Any]:
        """Calculate token savings from progressive disclosure.
        
        Args:
            num_skills: Number of available skills
            num_rag_results: Number of RAG results
            
        Returns:
            Savings analysis dictionary
        """
        # Full loading (all L3)
        full_skill_tokens = num_skills * self.SKILL_TOKENS[DisclosureLevel.L3_DETAILED]
        full_rag_tokens = num_rag_results * self.SKILL_TOKENS[DisclosureLevel.L3_DETAILED]
        full_total = full_skill_tokens + full_rag_tokens
        
        # Progressive loading (L1 all + L2 for 3 skills + L3 for 1 skill)
        prog_skill_tokens = (
            num_skills * self.SKILL_TOKENS[DisclosureLevel.L1_METADATA] +
            3 * self.SKILL_TOKENS[DisclosureLevel.L2_CORE] +
            1 * self.SKILL_TOKENS[DisclosureLevel.L3_DETAILED]
        )
        prog_rag_tokens = (
            num_rag_results * self.SKILL_TOKENS[DisclosureLevel.L1_METADATA] +
            3 * self.SKILL_TOKENS[DisclosureLevel.L2_CORE] +
            2 * self.SKILL_TOKENS[DisclosureLevel.L3_DETAILED]
        )
        prog_total = prog_skill_tokens + prog_rag_tokens
        
        return {
            "full_loading_tokens": full_total,
            "progressive_tokens": prog_total,
            "savings_tokens": full_total - prog_total,
            "savings_percent": (full_total - prog_total) / full_total * 100,
            "skills": {
                "full": full_skill_tokens,
                "progressive": prog_skill_tokens,
            },
            "rag": {
                "full": full_rag_tokens,
                "progressive": prog_rag_tokens,
            }
        }


class ContentExpander:
    """Expands content based on LLM requests during conversation.
    
    Tracks which content has been expanded and provides
    on-demand expansion capabilities.
    """
    
    def __init__(self):
        """Initialize content expander."""
        self.expanded_content: set[str] = set()  # Set of doc_ids expanded to L3
    
    def is_expanded(self, doc_id: str) -> bool:
        """Check if document has been expanded to full content."""
        return doc_id in self.expanded_content
    
    def mark_expanded(self, doc_id: str) -> None:
        """Mark document as expanded."""
        self.expanded_content.add(doc_id)
    
    def get_expansion_request(
        self,
        doc_id: str,
        full_content: str,
    ) -> dict[str, Any] | None:
        """Get expansion content if not already expanded.
        
        Args:
            doc_id: Document identifier
            full_content: Full document content
            
        Returns:
            Expansion request or None if already expanded
        """
        if self.is_expanded(doc_id):
            return None
        
        self.mark_expanded(doc_id)
        
        return {
            "doc_id": doc_id,
            "content": full_content,
            "additional_tokens": len(full_content) // 2,  # Rough estimate
        }
    
    def reset(self) -> None:
        """Reset expansion tracking (e.g., for new conversation)."""
        self.expanded_content.clear()
