"""Skills management with progressive disclosure and embedding-based matching.

Manages tool definitions with three-level disclosure (L1/L2/L3) and
provides semantic matching between user queries and available skills.

Reference: PRD Section 6.1 - Skills Definition and Progressive Loading
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.config import settings
from src.data.vectorizer import DocumentVectorizer
from src.models import AutonomyLevel, SkillDefinition, SkillLevel


class SkillManager:
    """Manages skill definitions with progressive disclosure.
    
    Features:
    - L1/L2/L3 disclosure levels for token efficiency
    - Semantic matching using embeddings
    - Category-based organization
    - Autonomy level tracking for human-in-the-loop
    """
    
    def __init__(self, skills_dir: Path | None = None):
        """Initialize skill manager.
        
        Args:
            skills_dir: Directory containing skill definitions
        """
        self.skills_dir = skills_dir or (settings.PROJECT_ROOT / "tools" / "skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        
        self.skills: dict[str, SkillDefinition] = {}
        self.vectorizer = DocumentVectorizer()
        
        # Load built-in skills
        self._register_builtin_skills()
        
        # Load custom skills from disk
        self._load_skills()
    
    def _register_builtin_skills(self) -> None:
        """Register built-in supply chain skills."""
        
        # Inventory Query Skill
        self.register_skill(SkillDefinition(
            name="query_inventory",
            description="查询指定仓库或SKU的实时库存状态，包括可用库存、预留库存和锁定库存",
            category="warehouse",
            autonomy_level=AutonomyLevel.LEVEL_3,
            parameters={
                "warehouse_id": {
                    "type": "string",
                    "required": False,
                    "description": "仓库代码，如SH-01。不指定则查询所有仓库",
                },
                "sku": {
                    "type": "string",
                    "required": False,
                    "description": "SKU编码，如YKL-LP-001",
                },
                "include_reserved": {
                    "type": "boolean",
                    "required": False,
                    "default": False,
                    "description": "是否包含预留库存详情",
                },
            },
            business_rules=[
                "只能查询用户有权限的仓库",
                "返回结果包含库位信息",
                "低库存商品自动标记预警",
            ],
            examples=[
                {
                    "input": {"sku": "YKL-LP-001"},
                    "output": {"total_qty": 500, "available_qty": 400, "warehouse": "SH-01"},
                },
            ],
            execution_mode="internal",
        ))
        
        # Production Progress Query Skill
        self.register_skill(SkillDefinition(
            name="query_production_progress",
            description="查询生产批次的进度状态，包括各工序完成率和预计交期",
            category="production",
            autonomy_level=AutonomyLevel.LEVEL_2,
            parameters={
                "batch_id": {
                    "type": "string",
                    "required": True,
                    "description": "批次号，如F01-20260128-001",
                },
                "include_details": {
                    "type": "boolean",
                    "required": False,
                    "default": True,
                    "description": "是否包含详细工序进度",
                },
            },
            business_rules=[
                "批次号必须符合格式FACTORY-YYYYMMDD-SEQ",
                "延迟批次自动触发风险预警",
            ],
            examples=[
                {
                    "input": {"batch_id": "F01-20260128-001"},
                    "output": {
                        "overall_progress": 0.85,
                        "status": "in_production",
                        "estimated_completion": "2026-01-30",
                    },
                },
            ],
            execution_mode="internal",
        ))
        
        # Logistics Tracking Skill
        self.register_skill(SkillDefinition(
            name="track_logistics",
            description="跟踪物流单状态，获取实时位置和预计送达时间",
            category="logistics",
            autonomy_level=AutonomyLevel.LEVEL_3,
            parameters={
                "waybill_id": {
                    "type": "string",
                    "required": True,
                    "description": "物流单号",
                },
            },
            business_rules=[
                "支持顺丰、中通、韵达等主要承运商",
                "演唱会前48小时自动升级为紧急跟踪",
            ],
            examples=[],
            execution_mode="internal",
        ))
        
        # Supplier Evaluation Skill
        self.register_skill(SkillDefinition(
            name="evaluate_supplier",
            description="评估供应商绩效，生成质量、交付、成本、服务四维评分卡",
            category="procurement",
            autonomy_level=AutonomyLevel.LEVEL_1,
            parameters={
                "supplier_id": {
                    "type": "string",
                    "required": True,
                    "description": "供应商代码",
                },
                "time_period": {
                    "type": "string",
                    "required": False,
                    "default": "last_90_days",
                    "description": "评估时间段",
                },
            },
            business_rules=[
                "高风险供应商(C/D级)必须人工审核",
                "评估结果影响后续订单分配",
            ],
            examples=[],
            execution_mode="internal",
        ))
        
        # Order Fulfillment Skill
        self.register_skill(SkillDefinition(
            name="check_order_fulfillment",
            description="检查粉丝订单履约状态，关联库存和物流信息",
            category="sales",
            autonomy_level=AutonomyLevel.LEVEL_3,
            parameters={
                "order_id": {
                    "type": "string",
                    "required": True,
                    "description": "订单号",
                },
            },
            business_rules=[
                "自动关联库存预占和物流单",
                "缺货订单标记待补货",
            ],
            examples=[],
            execution_mode="internal",
        ))
        
        # Risk Alert Generation Skill
        self.register_skill(SkillDefinition(
            name="generate_risk_alert",
            description="生成供应链风险预警，包括延迟、缺货、质量异常等",
            category="risk",
            autonomy_level=AutonomyLevel.LEVEL_2,
            parameters={
                "risk_type": {
                    "type": "string",
                    "required": True,
                    "enum": ["delay", "shortage", "quality", "logistics"],
                    "description": "风险类型",
                },
                "entity_id": {
                    "type": "string",
                    "required": True,
                    "description": "关联实体ID（批次/订单/供应商）",
                },
            },
            business_rules=[
                "高风险预警必须人工确认",
                "自动生成缓解建议",
            ],
            examples=[],
            execution_mode="script",
            script_template="risk_analysis.py",
        ))
    
    def register_skill(self, skill: SkillDefinition) -> None:
        """Register a new skill.
        
        Args:
            skill: Skill definition to register
        """
        # Generate embedding for matching
        text_to_embed = f"{skill.name}: {skill.description}"
        skill.embedding = self.vectorizer.encode_single(text_to_embed).tolist()
        
        self.skills[skill.name] = skill
    
    def _load_skills(self) -> None:
        """Load skills from disk."""
        skills_file = self.skills_dir / "skills.json"
        if skills_file.exists():
            with open(skills_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for skill_data in data:
                skill = SkillDefinition(**skill_data)
                self.skills[skill.name] = skill
    
    def save_skills(self) -> None:
        """Save skills to disk."""
        skills_file = self.skills_dir / "skills.json"
        with open(skills_file, "w", encoding="utf-8") as f:
            json.dump(
                [s.model_dump() for s in self.skills.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    def get_skill(self, name: str) -> SkillDefinition | None:
        """Get skill by name.
        
        Args:
            name: Skill name
            
        Returns:
            Skill definition or None
        """
        return self.skills.get(name)
    
    def match_skills(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.6,
    ) -> list[tuple[SkillDefinition, float]]:
        """Match user query to skills using semantic similarity.
        
        Args:
            query: User query text
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (skill, similarity) tuples
        """
        query_embedding = self.vectorizer.encode_single(query)
        
        matches = []
        for skill in self.skills.values():
            if skill.embedding:
                similarity = float(np.dot(query_embedding, skill.embedding))
                if similarity >= threshold:
                    matches.append((skill, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:top_k]
    
    def get_skills_by_category(self, category: str) -> list[SkillDefinition]:
        """Get all skills in a category.
        
        Args:
            category: Skill category
            
        Returns:
            List of matching skills
        """
        return [
            skill for skill in self.skills.values()
            if skill.category == category
        ]
    
    def get_skills_by_autonomy(self, max_level: AutonomyLevel) -> list[SkillDefinition]:
        """Get skills by autonomy level.
        
        Args:
            max_level: Maximum autonomy level
            
        Returns:
            List of skills with autonomy <= max_level
        """
        return [
            skill for skill in self.skills.values()
            if skill.autonomy_level.value <= max_level.value
        ]
    
    def get_all_skills(self) -> list[SkillDefinition]:
        """Get all registered skills."""
        return list(self.skills.values())
    
    def get_l1_index(self) -> list[dict[str, Any]]:
        """Get L1 metadata index of all skills."""
        return [
            {
                "name": skill.name,
                "description": skill.description[:100] + "..." if len(skill.description) > 100 else skill.description,
                "category": skill.category,
                "autonomy_level": skill.autonomy_level.value,
            }
            for skill in self.skills.values()
        ]
    
    def get_l2_definition(self, skill_name: str) -> dict[str, Any] | None:
        """Get L2 core definition of a skill."""
        skill = self.skills.get(skill_name)
        if not skill:
            return None
        
        return {
            "name": skill.name,
            "description": skill.description,
            "category": skill.category,
            "autonomy_level": skill.autonomy_level.value,
            "parameters": skill.parameters,
            "business_rules": skill.business_rules,
        }
    
    def get_l3_full(self, skill_name: str) -> dict[str, Any] | None:
        """Get L3 full definition with examples."""
        skill = self.skills.get(skill_name)
        if not skill:
            return None
        
        return skill.model_dump()
    
    def get_stats(self) -> dict[str, Any]:
        """Get skill manager statistics."""
        categories = {}
        autonomy_levels = {1: 0, 2: 0, 3: 0}
        
        for skill in self.skills.values():
            categories[skill.category] = categories.get(skill.category, 0) + 1
            autonomy_levels[skill.autonomy_level.value] += 1
        
        return {
            "total_skills": len(self.skills),
            "categories": categories,
            "autonomy_distribution": autonomy_levels,
        }


# Global skill manager instance
_skill_manager: SkillManager | None = None


def get_skill_manager() -> SkillManager:
    """Get or create global skill manager."""
    global _skill_manager
    if _skill_manager is None:
        _skill_manager = SkillManager()
    return _skill_manager
