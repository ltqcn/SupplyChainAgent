"""LLM client for Kimi API integration.

Provides unified interface for LLM calls with retry logic,
streaming support, and error handling.

Reference: PRD Section 3.1.2 - Kimi SDK Integration
"""

import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from src.config import settings


class KimiClient:
    """Client for Kimi (Moonshot AI) API.
    
    Features:
    - Compatible with OpenAI SDK
    - Streaming response support
    - Automatic retry with exponential backoff
    - Token usage tracking
    """
    
    def __init__(self):
        """Initialize Kimi client."""
        self.client = AsyncOpenAI(
            api_key=settings.KIMI_API_KEY,
            base_url=settings.KIMI_BASE_URL,
        )
        self.model = settings.KIMI_MODEL
    
    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Send chat completion request to Kimi.
        
        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stream: Whether to stream response
            tools: Optional tool definitions for function calling
            max_retries: Maximum retry attempts
            
        Returns:
            Response dict or async iterator for streaming
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                    tools=tools,  # type: ignore
                )
                
                if stream:
                    return self._stream_response(response)
                
                return {
                    "content": response.choices[0].message.content,
                    "tool_calls": [
                        {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in (response.choices[0].message.tool_calls or [])
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                }
                
            except Exception as e:
                last_error = e
                
                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # All retries failed
        raise Exception(f"LLM call failed after {max_retries} attempts: {last_error}")
    
    async def _stream_response(self, response) -> AsyncIterator[str]:
        """Stream response chunks.
        
        Args:
            response: OpenAI streaming response
            
        Yields:
            Response content chunks
        """
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def generate_with_rag(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate response with RAG context.
        
        Args:
            query: User query
            context: Assembled RAG context
            system_prompt: Optional system prompt override
            
        Returns:
            Generation result with content and metadata
        """
        if system_prompt is None:
            system_prompt = """你是应援物品供应链智能助手，专注于物料采购、生产监控、仓储管理、物流配送与售后服务的决策支持。

核心能力：
- 基于提供的引用数据回答问题，禁止推测未经验证的信息
- 识别风险并主动预警
- 区分自动化审批和需要人工确认的决策

回答要求：
1. 优先引用提供的数据来源
2. 对不确定的信息明确说明
3. 高风险建议标注需人工确认
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\n用户问题：{query}"},
        ]
        
        start_time = time.time()
        response = await self.chat_complete(
            messages,
            temperature=0.6,
            max_tokens=32768,
            top_p=1.0,
        )
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "content": response.get("content", ""),
            "usage": response.get("usage", {}),
            "latency_ms": latency_ms,
        }
    
    async def classify_intent(self, query: str) -> dict[str, Any]:
        """Classify user intent for query routing.
        
        Args:
            query: User query
            
        Returns:
            Intent classification result
        """
        system_prompt = """分析用户查询的意图，输出JSON格式：
{
  "primary_intent": "主要意图类别",
  "confidence": 0.95,
  "query_type": "exact_lookup|semantic_search|hybrid",
  "entities": [{"type": "sku|batch|order|supplier", "value": "实体值"}]
}

意图类别：
- inventory_query: 库存查询
- production_query: 生产进度查询
- logistics_query: 物流跟踪
- supplier_query: 供应商查询
- order_query: 订单查询
- risk_alert: 风险预警
- general: 一般咨询"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        
        response = await self.chat_complete(
            messages,
            temperature=0.1,
            max_tokens=200,
            top_p=1.0,
        )
        
        import json
        try:
            result = json.loads(response.get("content", "{}"))
        except json.JSONDecodeError:
            # Fallback parsing
            result = {
                "primary_intent": "general",
                "confidence": 0.5,
                "query_type": "hybrid",
                "entities": [],
            }
        
        return result


# Global client instance
_kimi_client: KimiClient | None = None


def get_kimi_client() -> KimiClient:
    """Get or create global Kimi client."""
    global _kimi_client
    if _kimi_client is None:
        _kimi_client = KimiClient()
    return _kimi_client
