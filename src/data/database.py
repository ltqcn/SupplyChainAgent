"""Database management for SupplyChainRAG using SQLAlchemy.

Provides ORM models matching the Pydantic domain models and data persistence.
"""

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.config import settings
from src.models import (
    Batch,
    Factory,
    FanOrder,
    InventoryRecord,
    SKU,
    Supplier,
    Warehouse,
    Waybill,
)

Base = declarative_base()


# ====================================================================================
# SQLAlchemy ORM Models
# ====================================================================================

class SupplierORM(Base):
    """Supplier database model."""
    __tablename__ = "suppliers"
    
    supplier_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(100), nullable=False)
    rating = Column(String(1), default="B")
    quality_score = Column(Float, default=85.0)
    delivery_score = Column(Float, default=85.0)
    cost_score = Column(Float, default=85.0)
    service_score = Column(Float, default=85.0)
    iso9001 = Column(Boolean, default=False)
    contact_email = Column(String(100), default="")
    contact_phone = Column(String(20), default="")


class SKUORM(Base):
    """SKU database model."""
    __tablename__ = "skus"
    
    sku_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    size = Column(String(50), default="")
    weight_g = Column(Float, default=0.0)
    material = Column(String(50), default="")
    safety_stock = Column(Integer, default=100)
    standard_cost = Column(Float, default=0.0)
    bom_items = Column(JSON, default=list)


class FactoryORM(Base):
    """Factory database model."""
    __tablename__ = "factories"
    
    factory_id = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    location = Column(String(100), nullable=False)
    daily_capacity = Column(Integer, nullable=False)
    monthly_capacity = Column(Integer, nullable=False)
    active_lines = Column(Integer, default=1)
    status = Column(String(20), default="operational")


class BatchORM(Base):
    """Production batch database model."""
    __tablename__ = "batches"
    
    batch_id = Column(String(30), primary_key=True)
    sku_id = Column(String(20), nullable=False)
    factory_id = Column(String(10), nullable=False)
    planned_qty = Column(Integer, nullable=False)
    actual_qty = Column(Integer, default=0)
    planned_date = Column(DateTime, nullable=False)
    actual_date = Column(DateTime, nullable=True)
    status = Column(String(20), default="planned")
    fpy_rate = Column(Float, default=1.0)
    defect_count = Column(Integer, default=0)
    cutting_progress = Column(Float, default=0.0)
    printing_progress = Column(Float, default=0.0)
    assembly_progress = Column(Float, default=0.0)
    packaging_progress = Column(Float, default=0.0)


class WarehouseORM(Base):
    """Warehouse database model."""
    __tablename__ = "warehouses"
    
    warehouse_id = Column(String(10), primary_key=True)
    name = Column(String(100), nullable=False)
    warehouse_type = Column(String(10), nullable=False)
    location = Column(String(100), nullable=False)
    total_area_sqm = Column(Float, nullable=False)
    used_area_sqm = Column(Float, default=0.0)


class InventoryORM(Base):
    """Inventory record database model."""
    __tablename__ = "inventory"
    
    record_id = Column(String(20), primary_key=True)
    warehouse_id = Column(String(10), nullable=False)
    sku_id = Column(String(20), nullable=False)
    batch_id = Column(String(30), nullable=True)
    available_qty = Column(Integer, default=0)
    reserved_qty = Column(Integer, default=0)
    locked_qty = Column(Integer, default=0)
    location_code = Column(String(20), default="")


class WaybillORM(Base):
    """Waybill database model."""
    __tablename__ = "waybills"
    
    waybill_id = Column(String(30), primary_key=True)
    carrier = Column(String(20), nullable=False)
    service_type = Column(String(20), default="standard")
    ship_from = Column(String(100), nullable=False)
    ship_to = Column(String(100), nullable=False)
    weight_kg = Column(Float, default=0.0)
    volume_cbm = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    picked_up_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    status = Column(String(30), default="pending")
    routing_nodes = Column(JSON, default=list)


class FanOrderORM(Base):
    """Fan order database model."""
    __tablename__ = "fan_orders"
    
    order_id = Column(String(30), primary_key=True)
    fan_id = Column(String(20), nullable=False)
    event_name = Column(String(100), nullable=False)
    event_date = Column(DateTime, nullable=False)
    items = Column(JSON, default=list)
    total_amount = Column(Float, default=0.0)
    order_date = Column(DateTime, default=datetime.now)
    expected_delivery = Column(DateTime, nullable=True)
    status = Column(String(20), default="pending")
    waybill_id = Column(String(30), nullable=True)


# ====================================================================================
# Database Manager
# ====================================================================================

class DatabaseManager:
    """Manages database connections and CRUD operations."""
    
    def __init__(self, db_url: str | None = None):
        """Initialize database manager.
        
        Args:
            db_url: Database URL, defaults to settings.DATABASE_URL
        """
        self.db_url = db_url or settings.DATABASE_URL
        self.engine = create_engine(self.db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        print(f"Database tables created at {self.db_url}")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def insert_suppliers(self, suppliers: list[Supplier]) -> None:
        """Insert supplier records."""
        with self.get_session() as session:
            for s in suppliers:
                orm = SupplierORM(
                    supplier_id=s.supplier_id,
                    name=s.name,
                    location=s.location,
                    rating=s.rating.value,
                    quality_score=s.quality_score,
                    delivery_score=s.delivery_score,
                    cost_score=s.cost_score,
                    service_score=s.service_score,
                    iso9001=s.iso9001,
                    contact_email=s.contact_email,
                    contact_phone=s.contact_phone,
                )
                session.merge(orm)
            session.commit()
    
    def insert_skus(self, skus: list[SKU]) -> None:
        """Insert SKU records."""
        with self.get_session() as session:
            for s in skus:
                orm = SKUORM(
                    sku_id=s.sku_id,
                    name=s.name,
                    category=s.category,
                    size=s.size,
                    weight_g=s.weight_g,
                    material=s.material,
                    safety_stock=s.safety_stock,
                    standard_cost=s.standard_cost,
                    bom_items=s.bom_items,
                )
                session.merge(orm)
            session.commit()
    
    def insert_factories(self, factories: list[Factory]) -> None:
        """Insert factory records."""
        with self.get_session() as session:
            for f in factories:
                orm = FactoryORM(
                    factory_id=f.factory_id,
                    name=f.name,
                    location=f.location,
                    daily_capacity=f.daily_capacity,
                    monthly_capacity=f.monthly_capacity,
                    active_lines=f.active_lines,
                    status=f.status,
                )
                session.merge(orm)
            session.commit()
    
    def insert_batches(self, batches: list[Batch]) -> None:
        """Insert batch records."""
        with self.get_session() as session:
            for b in batches:
                orm = BatchORM(
                    batch_id=b.batch_id,
                    sku_id=b.sku_id,
                    factory_id=b.factory_id,
                    planned_qty=b.planned_qty,
                    actual_qty=b.actual_qty,
                    planned_date=b.planned_date,
                    actual_date=b.actual_date,
                    status=b.status.value,
                    fpy_rate=b.fpy_rate,
                    defect_count=b.defect_count,
                    cutting_progress=b.cutting_progress,
                    printing_progress=b.printing_progress,
                    assembly_progress=b.assembly_progress,
                    packaging_progress=b.packaging_progress,
                )
                session.merge(orm)
            session.commit()
    
    def insert_warehouses(self, warehouses: list[Warehouse]) -> None:
        """Insert warehouse records."""
        with self.get_session() as session:
            for w in warehouses:
                orm = WarehouseORM(
                    warehouse_id=w.warehouse_id,
                    name=w.name,
                    warehouse_type=w.warehouse_type.value,
                    location=w.location,
                    total_area_sqm=w.total_area_sqm,
                    used_area_sqm=w.used_area_sqm,
                )
                session.merge(orm)
            session.commit()
    
    def insert_inventory(self, inventory: list[InventoryRecord]) -> None:
        """Insert inventory records."""
        with self.get_session() as session:
            for inv in inventory:
                orm = InventoryORM(
                    record_id=inv.record_id,
                    warehouse_id=inv.warehouse_id,
                    sku_id=inv.sku_id,
                    batch_id=inv.batch_id,
                    available_qty=inv.available_qty,
                    reserved_qty=inv.reserved_qty,
                    locked_qty=inv.locked_qty,
                    location_code=inv.location_code,
                )
                session.merge(orm)
            session.commit()
    
    def insert_waybills(self, waybills: list[Waybill]) -> None:
        """Insert waybill records."""
        with self.get_session() as session:
            for w in waybills:
                orm = WaybillORM(
                    waybill_id=w.waybill_id,
                    carrier=w.carrier,
                    service_type=w.service_type,
                    ship_from=w.ship_from,
                    ship_to=w.ship_to,
                    weight_kg=w.weight_kg,
                    volume_cbm=w.volume_cbm,
                    created_at=w.created_at,
                    picked_up_at=w.picked_up_at,
                    delivered_at=w.delivered_at,
                    status=w.status,
                    routing_nodes=w.routing_nodes,
                )
                session.merge(orm)
            session.commit()
    
    def insert_fan_orders(self, orders: list[FanOrder]) -> None:
        """Insert fan order records."""
        with self.get_session() as session:
            for o in orders:
                orm = FanOrderORM(
                    order_id=o.order_id,
                    fan_id=o.fan_id,
                    event_name=o.event_name,
                    event_date=o.event_date,
                    items=o.items,
                    total_amount=o.total_amount,
                    order_date=o.order_date,
                    expected_delivery=o.expected_delivery,
                    status=o.status.value,
                    waybill_id=o.waybill_id,
                )
                session.merge(orm)
            session.commit()
    
    def load_from_synthetic_data(self, data_dir: Path) -> None:
        """Load all synthetic data from JSON files into database.
        
        Args:
            data_dir: Directory containing generated JSON files
        """
        from src.models import (
            Batch,
            Factory,
            FanOrder,
            InventoryRecord,
            SKU,
            Supplier,
            Warehouse,
            Waybill,
        )
        
        data_dir = Path(data_dir)
        
        # Load and insert each entity type
        for filename, model_class, insert_method in [
            ("suppliers.json", Supplier, self.insert_suppliers),
            ("skus.json", SKU, self.insert_skus),
            ("factories.json", Factory, self.insert_factories),
            ("batches.json", Batch, self.insert_batches),
            ("warehouses.json", Warehouse, self.insert_warehouses),
            ("inventory.json", InventoryRecord, self.insert_inventory),
            ("waybills.json", Waybill, self.insert_waybills),
            ("fan_orders.json", FanOrder, self.insert_fan_orders),
        ]:
            file_path = data_dir / filename
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping")
                continue
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            items = [model_class(**item) for item in data]
            insert_method(items)
            print(f"Loaded {len(items)} records from {filename}")


# Global database manager instance
db_manager = DatabaseManager()
