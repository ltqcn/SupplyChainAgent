"""Pydantic models for SupplyChainRAG data entities and interfaces.

This module defines all data structures used across the system, ensuring
type safety and clear interfaces between components.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Enums for Type Safety
# ============================================================================

class OrderStatus(str, Enum):
    """Fan order status enumeration."""
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    REFUNDING = "refunding"
    REFUNDED = "refunded"


class BatchStatus(str, Enum):
    """Production batch status enumeration."""
    PLANNED = "planned"
    IN_PRODUCTION = "in_production"
    QC_PENDING = "qc_pending"
    QC_PASSED = "qc_passed"
    QC_FAILED = "qc_failed"
    WAREHOUSED = "warehoused"


class SupplierRating(str, Enum):
    """Supplier rating enumeration."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class WarehouseType(str, Enum):
    """Warehouse type enumeration."""
    CDC = "cdc"  # Central Distribution Center
    RDC = "rdc"  # Regional Distribution Center
    FDC = "fdc"  # Front Distribution Center


class AutonomyLevel(int, Enum):
    """AI autonomy level for decision making."""
    LEVEL_1 = 1  # Human dominant
    LEVEL_2 = 2  # Assisted decision
    LEVEL_3 = 3  # Full auto


class RetrievalAlgorithm(str, Enum):
    """RAG retrieval algorithm enumeration."""
    BM25 = "BM25"
    HNSW = "HNSW"
    IVF_PQ = "IVF_PQ"
    RFF_FUSION = "RFF_FUSION"


# ============================================================================
# Core Business Entities
# ============================================================================

class Supplier(BaseModel):
    """Supplier entity representing a supply chain partner."""
    
    supplier_id: str = Field(..., description="Unique supplier code")
    name: str = Field(..., description="Supplier company name")
    location: str = Field(..., description="Geographic location")
    rating: SupplierRating = Field(default=SupplierRating.B)
    
    # Performance metrics (0-100 scale)
    quality_score: float = Field(default=85.0, ge=0, le=100)
    delivery_score: float = Field(default=85.0, ge=0, le=100)
    cost_score: float = Field(default=85.0, ge=0, le=100)
    service_score: float = Field(default=85.0, ge=0, le=100)
    
    # Certifications
    iso9001: bool = Field(default=False)
    contact_email: str = Field(default="")
    contact_phone: str = Field(default="")
    
    class Config:
        json_schema_extra = {
            "example": {
                "supplier_id": "SUP001",
                "name": "上海亚克力制品有限公司",
                "location": "上海市松江区",
                "rating": "A",
                "quality_score": 95.0,
                "delivery_score": 88.0,
            }
        }


class SKU(BaseModel):
    """Stock Keeping Unit - minimum inventory management unit."""
    
    sku_id: str = Field(..., description="Unique SKU code")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    
    # Specifications
    size: str = Field(default="", description="Product dimensions")
    weight_g: float = Field(default=0.0, description="Weight in grams")
    material: str = Field(default="", description="Primary material")
    
    # Inventory management
    safety_stock: int = Field(default=100, ge=0)
    standard_cost: float = Field(default=0.0, ge=0)
    
    # BOM reference
    bom_items: list[dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sku_id": "YKL-LP-001",
                "name": "亚克力立牌-蓝色-大号",
                "category": "立牌",
                "size": "15cm x 10cm",
                "material": "亚克力",
                "safety_stock": 200,
            }
        }


class Batch(BaseModel):
    """Production batch for traceability management."""
    
    batch_id: str = Field(..., description="Batch number: FACTORY-YYYYMMDD-SEQ")
    sku_id: str = Field(..., description="Associated SKU")
    factory_id: str = Field(..., description="Production factory code")
    
    # Quantities
    planned_qty: int = Field(..., ge=0)
    actual_qty: int = Field(default=0, ge=0)
    
    # Timeline
    planned_date: datetime
    actual_date: datetime | None = Field(default=None)
    
    # Status
    status: BatchStatus = Field(default=BatchStatus.PLANNED)
    
    # Quality metrics
    fpy_rate: float = Field(default=1.0, ge=0, le=1.0, description="First Pass Yield")
    defect_count: int = Field(default=0, ge=0)
    
    # Progress tracking (0-100%)
    cutting_progress: float = Field(default=0.0, ge=0, le=100)
    printing_progress: float = Field(default=0.0, ge=0, le=100)
    assembly_progress: float = Field(default=0.0, ge=0, le=100)
    packaging_progress: float = Field(default=0.0, ge=0, le=100)
    
    @property
    def overall_progress(self) -> float:
        """Calculate overall production progress."""
        return (
            self.cutting_progress * 0.25 +
            self.printing_progress * 0.25 +
            self.assembly_progress * 0.25 +
            self.packaging_progress * 0.25
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "F01-20260128-001",
                "sku_id": "YKL-LP-001",
                "factory_id": "F01",
                "planned_qty": 1000,
                "planned_date": "2026-01-15T00:00:00",
                "status": "in_production",
                "cutting_progress": 100.0,
                "printing_progress": 80.0,
            }
        }


class Factory(BaseModel):
    """Production factory entity."""
    
    factory_id: str = Field(..., description="Unique factory code")
    name: str = Field(..., description="Factory name")
    location: str = Field(..., description="Geographic location")
    
    # Capacity (units per day)
    daily_capacity: int = Field(..., ge=0)
    monthly_capacity: int = Field(..., ge=0)
    
    # Current status
    active_lines: int = Field(default=1, ge=0)
    status: Literal["operational", "maintenance", "closed"] = Field(default="operational")
    
    class Config:
        json_schema_extra = {
            "example": {
                "factory_id": "F01",
                "name": "松江生产基地",
                "location": "上海市松江区",
                "daily_capacity": 5000,
                "monthly_capacity": 150000,
            }
        }


class Warehouse(BaseModel):
    """Warehouse storage facility entity."""
    
    warehouse_id: str = Field(..., description="Unique warehouse code")
    name: str = Field(..., description="Warehouse name")
    warehouse_type: WarehouseType = Field(...)
    location: str = Field(..., description="Geographic location")
    
    # Capacity
    total_area_sqm: float = Field(..., ge=0)
    used_area_sqm: float = Field(default=0.0, ge=0)
    
    @property
    def available_area_sqm(self) -> float:
        """Calculate available storage area."""
        return max(0, self.total_area_sqm - self.used_area_sqm)
    
    @property
    def utilization_rate(self) -> float:
        """Calculate warehouse utilization rate."""
        if self.total_area_sqm == 0:
            return 0.0
        return self.used_area_sqm / self.total_area_sqm
    
    class Config:
        json_schema_extra = {
            "example": {
                "warehouse_id": "SH-01",
                "name": "上海中央仓",
                "warehouse_type": "cdc",
                "location": "上海市青浦区",
                "total_area_sqm": 10000.0,
                "used_area_sqm": 7500.0,
            }
        }


class InventoryRecord(BaseModel):
    """Inventory record for SKU tracking."""
    
    record_id: str = Field(..., description="Unique record ID")
    warehouse_id: str = Field(..., description="Warehouse code")
    sku_id: str = Field(..., description="SKU code")
    batch_id: str | None = Field(default=None, description="Batch number")
    
    # Quantities
    available_qty: int = Field(default=0, ge=0)
    reserved_qty: int = Field(default=0, ge=0)
    locked_qty: int = Field(default=0, ge=0)
    
    # Location
    location_code: str = Field(default="", description="Storage location")
    
    @property
    def total_qty(self) -> int:
        """Calculate total inventory quantity."""
        return self.available_qty + self.reserved_qty + self.locked_qty
    
    class Config:
        json_schema_extra = {
            "example": {
                "record_id": "INV-001",
                "warehouse_id": "SH-01",
                "sku_id": "YKL-LP-001",
                "batch_id": "F01-20260128-001",
                "available_qty": 500,
                "reserved_qty": 100,
                "location_code": "A-03-05",
            }
        }


class Waybill(BaseModel):
    """Logistics waybill for shipment tracking."""
    
    waybill_id: str = Field(..., description="Tracking number")
    carrier: str = Field(..., description="Carrier name")
    service_type: str = Field(default="standard")
    
    # Addresses
    ship_from: str = Field(...)
    ship_to: str = Field(...)
    
    # Shipment details
    weight_kg: float = Field(default=0.0, ge=0)
    volume_cbm: float = Field(default=0.0, ge=0)
    
    # Timeline
    created_at: datetime = Field(default_factory=datetime.now)
    picked_up_at: datetime | None = Field(default=None)
    delivered_at: datetime | None = Field(default=None)
    
    # Status
    status: Literal["pending", "picked_up", "in_transit", "out_for_delivery", "delivered", "exception"] = Field(default="pending")
    
    # Routing nodes
    routing_nodes: list[dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "waybill_id": "SF1234567890",
                "carrier": "顺丰",
                "ship_from": "上海中央仓",
                "ship_to": "北京市朝阳区",
                "status": "in_transit",
            }
        }


class FanOrder(BaseModel):
    """Fan order entity representing end-customer purchase."""
    
    order_id: str = Field(..., description="Order number")
    fan_id: str = Field(..., description="Encrypted fan identifier")
    
    # Event reference
    event_name: str = Field(..., description="Concert/event name")
    event_date: datetime = Field(...)
    
    # Order items
    items: list[dict[str, Any]] = Field(default_factory=list)
    total_amount: float = Field(default=0.0, ge=0)
    
    # Timeline
    order_date: datetime = Field(default_factory=datetime.now)
    expected_delivery: datetime | None = Field(default=None)
    
    # Fulfillment
    status: OrderStatus = Field(default=OrderStatus.PENDING)
    waybill_id: str | None = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "ORD-20260128-001",
                "fan_id": "FAN-***1234",
                "event_name": "XX演唱会-上海站",
                "event_date": "2026-02-14T19:00:00",
                "total_amount": 299.0,
                "status": "paid",
            }
        }


# ============================================================================
# RAG System Models
# ============================================================================

class RAGResult(BaseModel):
    """Single RAG retrieval result with full metadata."""
    
    doc_id: str = Field(..., description="Source document identifier")
    content: str = Field(..., description="Retrieved text content")
    
    # Retrieval metadata
    retrieval_algo: RetrievalAlgorithm = Field(...)
    raw_score: float = Field(..., description="Algorithm-specific score")
    normalized_score: float = Field(..., ge=0, le=1, description="Normalized score [0,1]")
    
    # Document metadata
    doc_type: str = Field(..., description="Document type (production_report, etc.)")
    timestamp: datetime | None = Field(default=None, description="Document creation time")
    source_authority: float = Field(default=0.8, ge=0, le=1)
    
    # Confidence calculation
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence score")
    
    # Citation
    chunk_idx: int = Field(default=0, description="Chunk index in source document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "PROD-2025-001",
                "content": "批次B20250120已完成生产1000件...",
                "retrieval_algo": "HNSW",
                "raw_score": 0.89,
                "normalized_score": 0.945,
                "doc_type": "production_report",
                "confidence": 0.92,
            }
        }


class RAGRetrievalLog(BaseModel):
    """Structured logging for RAG retrieval process."""
    
    query_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    query_text: str = Field(...)
    
    # Per-algorithm results
    retrieval_paths: list[dict[str, Any]] = Field(default_factory=list)
    
    # Fusion results
    fusion_method: str = Field(default="RFF")
    fusion_weights: list[float] = Field(default_factory=list)
    top_k: int = Field(default=5)
    final_results: list[RAGResult] = Field(default_factory=list)


# ============================================================================
# Context Engineering Models
# ============================================================================

class ContextChunk(BaseModel):
    """A single chunk within the assembled context."""
    
    chunk_type: Literal["system", "rag", "memory", "tool", "user"]
    content: str
    tokens: int = Field(..., ge=0)
    priority: int = Field(default=3, ge=0, le=3, description="Priority 0-3, 0 is highest")
    
    # Source tracking
    source_id: str | None = Field(default=None)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ContextAssemblyLog(BaseModel):
    """Logging for context assembly process."""
    
    assembly_id: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Budget tracking
    max_tokens: int = Field(...)
    used_tokens: int = Field(default=0)
    remaining_tokens: int = Field(default=0)
    
    # Assembly steps
    steps: list[dict[str, Any]] = Field(default_factory=list)
    
    # Final context
    chunks: list[ContextChunk] = Field(default_factory=list)


# ============================================================================
# Tool/Skill Models
# ============================================================================

class SkillLevel(str, Enum):
    """Skill definition disclosure level."""
    L1_METADATA = "L1"  # Name, description, category only
    L2_CORE = "L2"      # + Parameters, business rules
    L3_DETAILED = "L3"  # + Examples, documentation


class SkillDefinition(BaseModel):
    """Tool/Skill definition for agent capabilities."""
    
    name: str = Field(...)
    description: str = Field(...)
    category: str = Field(...)
    autonomy_level: AutonomyLevel = Field(default=AutonomyLevel.LEVEL_2)
    
    # L2: Core instructions
    parameters: dict[str, Any] = Field(default_factory=dict)
    business_rules: list[str] = Field(default_factory=list)
    
    # L3: Detailed resources
    examples: list[dict[str, Any]] = Field(default_factory=list)
    
    # Execution
    execution_mode: Literal["script", "api", "internal"] = Field(default="internal")
    script_template: str | None = Field(default=None)
    
    # Embedding for matching
    embedding: list[float] | None = Field(default=None)


class ToolCall(BaseModel):
    """Tool execution call request/result."""
    
    tool_name: str = Field(...)
    parameters: dict[str, Any] = Field(default_factory=dict)
    
    # Execution metadata
    exec_time_ms: int = Field(default=0)
    status: Literal["pending", "success", "error", "timeout"] = Field(default="pending")
    result: dict[str, Any] | None = Field(default=None)
    error_message: str | None = Field(default=None)
    container_id: str | None = Field(default=None)


# ============================================================================
# Memory Models
# ============================================================================

class MemoryType(str, Enum):
    """Memory type classification."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"


class MemoryEntry(BaseModel):
    """Single memory entry for agent memory system."""
    
    memory_id: str = Field(...)
    memory_type: MemoryType = Field(...)
    
    # Content
    content: str = Field(...)
    embedding: list[float] | None = Field(default=None)
    tokens: int = Field(default=0)
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str | None = Field(default=None)
    turn_idx: int | None = Field(default=None)
    
    # Source tracking
    source: Literal["user", "llm", "tool", "rag"] = Field(default="user")
    key_entities: list[str] = Field(default_factory=list)
    
    # Relevance (for retrieval)
    relevance_score: float | None = Field(default=None)


class SessionSummary(BaseModel):
    """Long-term memory summary of a conversation session."""
    
    session_id: str = Field(...)
    summary: str = Field(...)
    key_entities: list[str] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    embedding: list[float] | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Agent Models
# ============================================================================

class IntentClassification(BaseModel):
    """NLU intent classification result."""
    
    primary_intent: str = Field(...)
    confidence: float = Field(..., ge=0, le=1)
    intent_distribution: dict[str, float] = Field(default_factory=dict)
    
    # Entity extraction
    entities: list[dict[str, Any]] = Field(default_factory=list)
    
    # Query type for RAG routing
    query_type: Literal["exact_lookup", "semantic_search", "hybrid"] = Field(default="hybrid")


class AgentResponse(BaseModel):
    """Standard agent response format."""
    
    response_id: str = Field(...)
    session_id: str = Field(...)
    
    # Content
    content: str = Field(...)
    reasoning: str | None = Field(default=None)
    
    # Citations
    citations: list[dict[str, Any]] = Field(default_factory=list)
    
    # Tool calls made
    tool_calls: list[ToolCall] = Field(default_factory=list)
    
    # Risk assessment
    risk_level: Literal["low", "medium", "high"] = Field(default="low")
    requires_approval: bool = Field(default=False)
    
    # Metadata
    tokens_used: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================================
# UI Models
# ============================================================================

class RoundStep(BaseModel):
    """Single step in round demonstration mode."""
    
    round_idx: int = Field(...)
    step_name: str = Field(...)
    step_description: str = Field(...)
    
    # Display data
    input_data: dict[str, Any] = Field(default_factory=dict)
    context_tree: dict[str, Any] = Field(default_factory=dict)
    algorithm_traces: dict[str, Any] = Field(default_factory=dict)
    
    # Status
    status: Literal["pending", "running", "completed", "error"] = Field(default="pending")


class ChatMessage(BaseModel):
    """Chat message for UI display."""
    
    message_id: str = Field(...)
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(...)
    
    # Optional metadata
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class DashboardMetrics(BaseModel):
    """Supply chain dashboard metrics."""
    
    # Inventory
    total_sku_count: int = Field(default=0)
    low_stock_alerts: int = Field(default=0)
    warehouse_utilization: dict[str, float] = Field(default_factory=dict)
    
    # Production
    active_batches: int = Field(default=0)
    delayed_batches: int = Field(default=0)
    
    # Logistics
    in_transit_shipments: int = Field(default=0)
    delayed_shipments: int = Field(default=0)
    
    # Orders
    pending_orders: int = Field(default=0)
    today_deliveries: int = Field(default=0)
    
    # Risks
    active_risks: list[dict[str, Any]] = Field(default_factory=list)
