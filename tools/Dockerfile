"""Docker sandbox for tool script execution.

Provides isolated execution environment with resource limits.
"""

FROM python:3.11-slim

# Install common dependencies for supply chain analysis
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    requests==2.31.0 \
    numpy==1.26.3 \
    python-dateutil==2.8.2

# Create non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-c", "print('SupplyChainRAG Sandbox ready')"]
