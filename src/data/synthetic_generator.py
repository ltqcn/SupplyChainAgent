"""Synthetic data generator for supply chain entities.

Generates realistic supply chain data following business rules and constraints.
Includes intentional anomalies for testing risk detection capabilities.

Reference: PRD Section 2.3 - Synthetic Data Generation Strategy
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from src.models import (
    Batch,
    BatchStatus,
    Factory,
    FanOrder,
    InventoryRecord,
    SKU,
    Supplier,
    SupplierRating,
    Warehouse,
    WarehouseType,
    Waybill,
)


class SupplyChainDataGenerator:
    """Generator for synthetic supply chain data with realistic business logic."""
    
    # Business constants
    SKU_CATEGORIES = ["灯牌", "手幅", "立牌", "应援棒", "T恤", "挂件", "海报"]
    MATERIALS = ["亚克力", "布料", "LED", "纸张", "塑料", "金属", "木材"]
    LOCATIONS = [
        "上海市松江区", "上海市青浦区", "苏州市昆山市", "杭州市萧山区",
        "深圳市宝安区", "广州市番禺区", "北京市大兴区", "成都市郫都区",
    ]
    CARRIERS = ["顺丰", "中通", "韵达", "圆通", "申通", "EMS", "京东物流"]
    EVENT_NAMES = [
        "XX演唱会-上海站", "XX演唱会-北京站", "XX见面会-广州站",
        "YY巡演-成都站", "YY巡演-杭州站", "ZZ粉丝节-深圳站",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with fixed seed for reproducibility.
        
        Args:
            seed: Random seed for deterministic data generation
        """
        self.seed = seed
        random.seed(seed)
        self.generated_data: dict[str, list] = {
            "suppliers": [],
            "skus": [],
            "factories": [],
            "batches": [],
            "warehouses": [],
            "inventory": [],
            "waybills": [],
            "fan_orders": [],
        }
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix."""
        return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"
    
    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate random date between start and end."""
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    # ====================================================================================
    # Entity Generators
    # ====================================================================================
    
    def generate_suppliers(self, count: int = 20) -> list[Supplier]:
        """Generate supplier entities with performance scorecards.
        
        Args:
            count: Number of suppliers to generate
            
        Returns:
            List of Supplier models
        """
        suppliers = []
        
        for i in range(count):
            # Generate ratings with distribution: A=20%, B=40%, C=30%, D=10%
            rating_dist = random.random()
            if rating_dist < 0.2:
                rating = SupplierRating.A
            elif rating_dist < 0.6:
                rating = SupplierRating.B
            elif rating_dist < 0.9:
                rating = SupplierRating.C
            else:
                rating = SupplierRating.D
            
            # Scores correlate with rating but have variance
            base_score = {"A": 90, "B": 80, "C": 70, "D": 60}[rating.value]
            
            supplier = Supplier(
                supplier_id=f"SUP{i+1:03d}",
                name=f"供应商{i+1:03d}-{random.choice(['材料', '制品', '科技', '贸易'])}",
                location=random.choice(self.LOCATIONS),
                rating=rating,
                quality_score=min(100, max(0, base_score + random.randint(-10, 10))),
                delivery_score=min(100, max(0, base_score + random.randint(-15, 5))),
                cost_score=min(100, max(0, base_score + random.randint(-10, 15))),
                service_score=min(100, max(0, base_score + random.randint(-10, 10))),
                iso9001=rating in [SupplierRating.A, SupplierRating.B],
                contact_email=f"contact{ i+1 }@supplier.com",
            )
            suppliers.append(supplier)
        
        self.generated_data["suppliers"] = suppliers
        return suppliers
    
    def generate_skus(self, count: int = 50) -> list[SKU]:
        """Generate SKU entities with BOM structures.
        
        Args:
            count: Number of SKUs to generate
            
        Returns:
            List of SKU models
        """
        skus = []
        
        for i in range(count):
            category = random.choice(self.SKU_CATEGORIES)
            material = random.choice(self.MATERIALS)
            
            # Generate SKU code based on category
            category_code = "".join([c[0] for c in category])
            sku_id = f"{category_code}-{i+1:03d}"
            
            # Size based on category
            size_map = {
                "灯牌": random.choice(["20cmx30cm", "30cmx40cm", "40cmx60cm"]),
                "手幅": random.choice(["15cmx45cm", "20cmx60cm"]),
                "立牌": random.choice(["10cmx15cm", "15cmx20cm", "20cmx30cm"]),
                "应援棒": random.choice(["直径3cmx长25cm", "直径4cmx长30cm"]),
                "T恤": random.choice(["S", "M", "L", "XL", "XXL"]),
                "挂件": random.choice(["5cm", "8cm", "10cm"]),
                "海报": random.choice(["A3", "A2", "A1"]),
            }
            
            # BOM items based on category
            bom_items = []
            if category == "灯牌":
                bom_items = [
                    {"material": "亚克力板", "qty": 2, "unit": "片"},
                    {"material": "LED灯珠", "qty": 20, "unit": "颗"},
                    {"material": "电池盒", "qty": 1, "unit": "个"},
                ]
            elif category == "立牌":
                bom_items = [
                    {"material": "亚克力板", "qty": 1, "unit": "片"},
                    {"material": "底座", "qty": 1, "unit": "个"},
                ]
            elif category == "T恤":
                bom_items = [
                    {"material": "纯棉布料", "qty": 0.5, "unit": "米"},
                    {"material": "印刷油墨", "qty": 0.1, "unit": "升"},
                ]
            
            sku = SKU(
                sku_id=sku_id,
                name=f"{category}-{material}-{i+1:03d}",
                category=category,
                size=size_map.get(category, ""),
                weight_g=random.randint(50, 500),
                material=material,
                safety_stock=random.randint(100, 500),
                standard_cost=random.uniform(10.0, 100.0),
                bom_items=bom_items,
            )
            skus.append(sku)
        
        self.generated_data["skus"] = skus
        return skus
    
    def generate_factories(self, count: int = 5) -> list[Factory]:
        """Generate factory entities with capacity data.
        
        Args:
            count: Number of factories to generate
            
        Returns:
            List of Factory models
        """
        factories = []
        
        for i in range(count):
            daily_cap = random.choice([1000, 2000, 3000, 5000])
            factory = Factory(
                factory_id=f"F{i+1:02d}",
                name=f"生产基地{i+1:02d}",
                location=self.LOCATIONS[i % len(self.LOCATIONS)],
                daily_capacity=daily_cap,
                monthly_capacity=daily_cap * 30,
                active_lines=random.randint(2, 5),
                status="operational",
            )
            factories.append(factory)
        
        self.generated_data["factories"] = factories
        return factories
    
    def generate_batches(
        self,
        skus: list[SKU],
        factories: list[Factory],
        count: int = 100,
        base_date: datetime | None = None,
    ) -> list[Batch]:
        """Generate production batches with progress tracking.
        
        Intentionally injects 5% delay anomalies for testing risk detection.
        
        Args:
            skus: List of SKUs to produce
            factories: List of factories for production
            count: Number of batches to generate
            base_date: Reference date for batch scheduling
            
        Returns:
            List of Batch models
        """
        if base_date is None:
            base_date = datetime.now()
        
        batches = []
        
        for i in range(count):
            sku = random.choice(skus)
            factory = random.choice(factories)
            
            # Generate batch ID: FACTORY-YYYYMMDD-SEQ
            planned_date = base_date + timedelta(days=random.randint(-30, 60))
            date_str = planned_date.strftime("%Y%m%d")
            batch_id = f"{factory.factory_id}-{date_str}-{i+1:03d}"
            
            planned_qty = random.randint(500, 2000)
            
            # Status distribution
            status_rand = random.random()
            if status_rand < 0.2:
                status = BatchStatus.PLANNED
            elif status_rand < 0.4:
                status = BatchStatus.IN_PRODUCTION
            elif status_rand < 0.6:
                status = BatchStatus.QC_PENDING
            elif status_rand < 0.8:
                status = BatchStatus.QC_PASSED
            else:
                status = BatchStatus.WAREHOUSED
            
            # Progress based on status
            progress_map = {
                BatchStatus.PLANNED: (0, 0, 0, 0),
                BatchStatus.IN_PRODUCTION: (100, random.uniform(0, 80), random.uniform(0, 50), random.uniform(0, 30)),
                BatchStatus.QC_PENDING: (100, 100, 100, 100),
                BatchStatus.QC_PASSED: (100, 100, 100, 100),
                BatchStatus.WAREHOUSED: (100, 100, 100, 100),
            }
            cutting, printing, assembly, packaging = progress_map[status]
            
            # 5% chance of delay anomaly (progress behind schedule)
            is_delayed = random.random() < 0.05
            if is_delayed and status == BatchStatus.IN_PRODUCTION:
                cutting = max(0, cutting - random.uniform(10, 30))
                printing = max(0, printing - random.uniform(10, 30))
            
            batch = Batch(
                batch_id=batch_id,
                sku_id=sku.sku_id,
                factory_id=factory.factory_id,
                planned_qty=planned_qty,
                actual_qty=int(planned_qty * random.uniform(0.95, 0.99)),
                planned_date=planned_date,
                actual_date=planned_date + timedelta(days=random.randint(-2, 5)) if status == BatchStatus.WAREHOUSED else None,
                status=status,
                fpy_rate=random.uniform(0.95, 0.99),
                defect_count=random.randint(0, 20),
                cutting_progress=cutting,
                printing_progress=printing,
                assembly_progress=assembly,
                packaging_progress=packaging,
            )
            batches.append(batch)
        
        self.generated_data["batches"] = batches
        return batches
    
    def generate_warehouses(self, count: int = 8) -> list[Warehouse]:
        """Generate warehouse entities with capacity data.
        
        Args:
            count: Number of warehouses to generate
            
        Returns:
            List of Warehouse models
        """
        warehouses = []
        
        # Create mix of warehouse types
        type_distribution = [WarehouseType.CDC] * 2 + [WarehouseType.RDC] * 4 + [WarehouseType.FDC] * 2
        
        for i in range(count):
            wh_type = type_distribution[i % len(type_distribution)]
            
            # Capacity varies by type
            capacity_map = {
                WarehouseType.CDC: 10000,
                WarehouseType.RDC: 5000,
                WarehouseType.FDC: 1000,
            }
            total_area = capacity_map[wh_type]
            used_rate = random.uniform(0.6, 0.9)
            
            warehouse = Warehouse(
                warehouse_id=f"WH{i+1:02d}",
                name=f"{['上海', '苏州', '杭州', '深圳', '广州', '北京', '成都'][i % 7]}仓库",
                warehouse_type=wh_type,
                location=self.LOCATIONS[i % len(self.LOCATIONS)],
                total_area_sqm=float(total_area),
                used_area_sqm=float(total_area * used_rate),
            )
            warehouses.append(warehouse)
        
        self.generated_data["warehouses"] = warehouses
        return warehouses
    
    def generate_inventory(
        self,
        warehouses: list[Warehouse],
        skus: list[SKU],
        batches: list[Batch],
        count: int = 200,
    ) -> list[InventoryRecord]:
        """Generate inventory records linking warehouses, SKUs, and batches.
        
        Args:
            warehouses: List of warehouses
            skus: List of SKUs
            batches: List of batches
            count: Number of inventory records to generate
            
        Returns:
            List of InventoryRecord models
        """
        inventory = []
        
        for i in range(count):
            sku = random.choice(skus)
            warehouse = random.choice(warehouses)
            batch = random.choice(batches) if random.random() < 0.7 else None
            
            total = random.randint(0, 1000)
            available = int(total * random.uniform(0.6, 0.9))
            reserved = int((total - available) * random.uniform(0.5, 0.8))
            locked = total - available - reserved
            
            record = InventoryRecord(
                record_id=f"INV-{i+1:05d}",
                warehouse_id=warehouse.warehouse_id,
                sku_id=sku.sku_id,
                batch_id=batch.batch_id if batch else None,
                available_qty=available,
                reserved_qty=reserved,
                locked_qty=locked,
                location_code=f"{random.choice('ABCDEF')}-{random.randint(1, 20):02d}-{random.randint(1, 5)}",
            )
            inventory.append(record)
        
        self.generated_data["inventory"] = inventory
        return inventory
    
    def generate_waybills(
        self,
        warehouses: list[Warehouse],
        count: int = 300,
        base_date: datetime | None = None,
    ) -> list[Waybill]:
        """Generate logistics waybills with routing information.
        
        Intentionally injects 3% logistics anomalies for testing.
        
        Args:
            warehouses: List of origin warehouses
            count: Number of waybills to generate
            base_date: Reference date
            
        Returns:
            List of Waybill models
        """
        if base_date is None:
            base_date = datetime.now()
        
        waybills = []
        destinations = [
            "北京市朝阳区", "上海市浦东新区", "广州市天河区", "深圳市南山区",
            "成都市锦江区", "杭州市西湖区", "南京市鼓楼区", "武汉市江汉区",
        ]
        
        for i in range(count):
            origin = random.choice(warehouses)
            destination = random.choice(destinations)
            carrier = random.choice(self.CARRIERS)
            
            # Service type based on urgency
            service_type = random.choice(["standard"] * 7 + ["express"] * 3)
            
            # Generate routing nodes
            created_at = base_date - timedelta(days=random.randint(0, 7))
            nodes = []
            
            # Add origin node
            nodes.append({
                "timestamp": created_at.isoformat(),
                "location": origin.location,
                "event": "已揽收",
            })
            
            # Add transit nodes
            current_time = created_at
            for j in range(random.randint(1, 3)):
                current_time += timedelta(hours=random.randint(4, 12))
                nodes.append({
                    "timestamp": current_time.isoformat(),
                    "location": f"转运中心{j+1}",
                    "event": "运输中",
                })
            
            # Determine status
            status_rand = random.random()
            if status_rand < 0.7:
                status = "delivered"
                delivered_at = current_time + timedelta(hours=random.randint(12, 48))
                nodes.append({
                    "timestamp": delivered_at.isoformat(),
                    "location": destination,
                    "event": "已签收",
                })
            elif status_rand < 0.9:
                status = "in_transit"
                delivered_at = None
            else:
                # 3% anomaly - exception status
                status = "exception"
                delivered_at = None
                nodes.append({
                    "timestamp": (current_time + timedelta(hours=24)).isoformat(),
                    "location": destination,
                    "event": "异常：地址不详",
                })
            
            waybill = Waybill(
                waybill_id=f"WB{carrier[0]}{i+1:08d}",
                carrier=carrier,
                service_type=service_type,
                ship_from=origin.location,
                ship_to=destination,
                weight_kg=random.uniform(0.5, 5.0),
                volume_cbm=random.uniform(0.001, 0.01),
                created_at=created_at,
                picked_up_at=created_at + timedelta(hours=2),
                delivered_at=delivered_at,
                status=status,
                routing_nodes=nodes,
            )
            waybills.append(waybill)
        
        self.generated_data["waybills"] = waybills
        return waybills
    
    def generate_fan_orders(
        self,
        skus: list[SKU],
        count: int = 1000,
        base_date: datetime | None = None,
    ) -> list[FanOrder]:
        """Generate fan orders with event references.
        
        Uses Poisson-like distribution for realistic order patterns around events.
        
        Args:
            skus: List of available SKUs
            count: Number of orders to generate
            base_date: Reference date
            
        Returns:
            List of FanOrder models
        """
        if base_date is None:
            base_date = datetime.now()
        
        orders = []
        
        for i in range(count):
            event = random.choice(self.EVENT_NAMES)
            # Event date is 15-45 days from now
            event_date = base_date + timedelta(days=random.randint(15, 45))
            
            # Order date is 1-30 days before event (concentrated in last 7 days)
            days_before = max(1, int(random.expovariate(1/10)))  # Poisson-like
            order_date = event_date - timedelta(days=days_before)
            
            # 1-3 items per order
            num_items = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            items = []
            total_amount = 0.0
            
            for _ in range(num_items):
                sku = random.choice(skus)
                qty = random.randint(1, 3)
                price = random.uniform(29.0, 199.0)
                items.append({
                    "sku_id": sku.sku_id,
                    "sku_name": sku.name,
                    "quantity": qty,
                    "unit_price": round(price, 2),
                    "subtotal": round(qty * price, 2),
                })
                total_amount += qty * price
            
            # Status based on order date relative to base_date
            if order_date > base_date:
                status = "pending"
            else:
                status = random.choice(["paid", "shipped", "delivered"])
            
            order = FanOrder(
                order_id=f"ORD-{order_date.strftime('%Y%m%d')}-{i+1:05d}",
                fan_id=f"FAN-***{random.randint(1000, 9999)}",
                event_name=event,
                event_date=event_date,
                items=items,
                total_amount=round(total_amount, 2),
                order_date=order_date,
                expected_delivery=event_date - timedelta(days=2),
                status=status,
                waybill_id=f"WBDEL{random.randint(10000000, 99999999)}" if status in ["shipped", "delivered"] else None,
            )
            orders.append(order)
        
        self.generated_data["fan_orders"] = orders
        return orders
    
    # ====================================================================================
    # Data Export
    # ====================================================================================
    
    def generate_all(
        self,
        scale: Literal["small", "medium", "large"] = "medium",
    ) -> dict[str, list]:
        """Generate complete supply chain dataset.
        
        Args:
            scale: Dataset size - small (fast), medium (balanced), large (comprehensive)
            
        Returns:
            Dictionary containing all generated entities
        """
        # Scale configuration
        scale_config = {
            "small": {
                "suppliers": 10, "skus": 20, "factories": 3,
                "batches": 30, "warehouses": 4, "inventory": 50,
                "waybills": 100, "fan_orders": 200,
            },
            "medium": {
                "suppliers": 20, "skus": 50, "factories": 5,
                "batches": 100, "warehouses": 8, "inventory": 200,
                "waybills": 300, "fan_orders": 1000,
            },
            "large": {
                "suppliers": 50, "skus": 100, "factories": 10,
                "batches": 300, "warehouses": 15, "inventory": 500,
                "waybills": 1000, "fan_orders": 5000,
            },
        }
        
        config = scale_config[scale]
        base_date = datetime(2026, 1, 28)
        
        # Generate in dependency order
        print(f"Generating {scale} dataset...")
        
        suppliers = self.generate_suppliers(config["suppliers"])
        print(f"  Generated {len(suppliers)} suppliers")
        
        skus = self.generate_skus(config["skus"])
        print(f"  Generated {len(skus)} SKUs")
        
        factories = self.generate_factories(config["factories"])
        print(f"  Generated {len(factories)} factories")
        
        batches = self.generate_batches(skus, factories, config["batches"], base_date)
        print(f"  Generated {len(batches)} batches")
        
        warehouses = self.generate_warehouses(config["warehouses"])
        print(f"  Generated {len(warehouses)} warehouses")
        
        inventory = self.generate_inventory(warehouses, skus, batches, config["inventory"])
        print(f"  Generated {len(inventory)} inventory records")
        
        waybills = self.generate_waybills(warehouses, config["waybills"], base_date)
        print(f"  Generated {len(waybills)} waybills")
        
        fan_orders = self.generate_fan_orders(skus, config["fan_orders"], base_date)
        print(f"  Generated {len(fan_orders)} fan orders")
        
        return self.generated_data
    
    def save_to_json(self, output_dir: Path) -> None:
        """Save all generated data to JSON files.
        
        Args:
            output_dir: Directory to save JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for entity_type, data in self.generated_data.items():
            file_path = output_dir / f"{entity_type}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    [item.model_dump(mode="json") for item in data],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Saved {len(data)} {entity_type} to {file_path}")


# ====================================================================================
# CLI Entry Point
# ====================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic supply chain data")
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default="medium",
        help="Dataset size scale",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    generator = SupplyChainDataGenerator(seed=args.seed)
    generator.generate_all(scale=args.scale)
    generator.save_to_json(args.output)
    
    print(f"\nDataset generation complete! Files saved to {args.output}")
