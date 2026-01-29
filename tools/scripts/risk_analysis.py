#!/usr/bin/env python3
"""Risk analysis script for Docker sandbox execution.

Analyzes supply chain risks and generates mitigation recommendations.
"""

import json
import sys
from datetime import datetime


def analyze_delay_risk(batch_id: str, current_progress: float, planned_progress: float) -> dict:
    """Analyze production delay risk.
    
    Args:
        batch_id: Production batch ID
        current_progress: Actual completion rate (0-1)
        planned_progress: Planned completion rate (0-1)
        
    Returns:
        Risk analysis result
    """
    delay = planned_progress - current_progress
    
    if delay < 0.05:
        risk_level = "low"
        recommendation = "正常进度，无需干预"
    elif delay < 0.15:
        risk_level = "medium"
        recommendation = "建议关注，准备备选方案"
    else:
        risk_level = "high"
        recommendation = "立即启动应急预案：1)协调加班 2)启用备选工厂 3)升级物流"
    
    return {
        "batch_id": batch_id,
        "risk_type": "delay",
        "risk_level": risk_level,
        "delay_percentage": round(delay * 100, 1),
        "current_progress": round(current_progress * 100, 1),
        "planned_progress": round(planned_progress * 100, 1),
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    """Main entry point - reads JSON from stdin, outputs JSON to stdout."""
    try:
        # Read input parameters from stdin
        params = json.load(sys.stdin)
        
        risk_type = params.get("risk_type")
        entity_id = params.get("entity_id")
        
        if risk_type == "delay":
            result = analyze_delay_risk(
                batch_id=entity_id,
                current_progress=params.get("current_progress", 0.7),
                planned_progress=params.get("planned_progress", 0.8),
            )
        else:
            result = {
                "error": f"Unsupported risk type: {risk_type}",
                "supported_types": ["delay"],
            }
        
        # Output JSON result
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
