"""Tool executor with Docker sandbox for safe script execution.

Executes Python scripts in isolated Docker containers with
resource limits and timeout controls.

Reference: PRD Section 6.2 - Script-based Tool Invocation
"""

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound

from src.config import settings
from src.models import ToolCall


class DockerSandboxExecutor:
    """Docker sandbox executor for safe tool script execution.
    
    Security features:
    - Non-root user execution
    - Resource limits (CPU, memory)
    - Network isolation (whitelist)
    - Read-only filesystem (except /tmp)
    - Automatic timeout and cleanup
    """
    
    def __init__(self):
        """Initialize Docker sandbox executor."""
        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Warning: Docker not available: {e}")
            self.client = None
        
        self.image_name = settings.DOCKER_SANDBOX_IMAGE
        self.cpu_limit = settings.DOCKER_SANDBOX_CPU
        self.memory_limit = settings.DOCKER_SANDBOX_MEMORY
        self.timeout = settings.DOCKER_SANDBOX_TIMEOUT
    
    def _ensure_image(self) -> bool:
        """Ensure sandbox image exists.
        
        Returns:
            True if image is available
        """
        if self.client is None:
            return False
        
        try:
            self.client.images.get(self.image_name)
            return True
        except ImageNotFound:
            print(f"Building Docker image: {self.image_name}")
            return self._build_image()
    
    def _build_image(self) -> bool:
        """Build sandbox Docker image.
        
        Returns:
            True if build succeeded
        """
        if self.client is None:
            return False
        
        dockerfile_content = """
FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir pandas requests numpy

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Set working directory
WORKDIR /app

# Default command
CMD ["python", "-c", "print('Sandbox ready')"]
"""
        
        try:
            # Create temporary directory for build context
            with tempfile.TemporaryDirectory() as tmpdir:
                dockerfile_path = Path(tmpdir) / "Dockerfile"
                with open(dockerfile_path, "w") as f:
                    f.write(dockerfile_content)
                
                self.client.images.build(
                    path=tmpdir,
                    tag=self.image_name,
                    rm=True,
                )
            
            print(f"Docker image {self.image_name} built successfully")
            return True
            
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return False
    
    def execute_script(
        self,
        script_content: str,
        input_data: dict[str, Any],
    ) -> ToolCall:
        """Execute Python script in Docker sandbox.
        
        Args:
            script_content: Python script to execute
            input_data: Input data passed via stdin
            
        Returns:
            ToolCall with execution results
        """
        tool_call = ToolCall(
            tool_name="docker_script",
            parameters=input_data,
        )
        
        if self.client is None:
            tool_call.status = "error"
            tool_call.error_message = "Docker not available"
            return tool_call
        
        if not self._ensure_image():
            tool_call.status = "error"
            tool_call.error_message = "Failed to build Docker image"
            return tool_call
        
        container_id = None
        start_time = time.time()
        
        try:
            # Create temporary file for script
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Prepare input data
            input_json = json.dumps(input_data, ensure_ascii=False)
            
            # Run container
            container = self.client.containers.run(
                self.image_name,
                command=["python", "/app/script.py"],
                stdin_open=True,
                detach=True,
                cpu_quota=int(self.cpu_limit * 100000),
                mem_limit=self.memory_limit,
                network_mode="none",  # No network by default
                read_only=True,
                tmpfs={"/tmp": "noexec,nosuid,size=100m"},
                volumes={
                    script_path: {"bind": "/app/script.py", "mode": "ro"},
                },
            )
            
            container_id = container.id[:12]
            tool_call.container_id = container_id
            
            # Send input and get output
            socket = container.attach_socket(params={"stdin": 1, "stdout": 1, "stderr": 1, "stream": 1})
            
            # Send input data
            socket._sock.send(input_json.encode("utf-8"))
            socket._sock.close()
            
            # Wait for completion with timeout
            result = container.wait(timeout=self.timeout)
            
            # Get logs
            logs = container.logs(stdout=True, stderr=True).decode("utf-8")
            
            # Clean up
            container.remove(force=True)
            
            # Parse result
            exit_code = result.get("StatusCode", -1)
            
            if exit_code == 0:
                # Try to parse JSON output
                try:
                    output_data = json.loads(logs)
                    tool_call.status = "success"
                    tool_call.result = output_data
                except json.JSONDecodeError:
                    # Non-JSON output
                    tool_call.status = "success"
                    tool_call.result = {"output": logs}
            else:
                tool_call.status = "error"
                tool_call.error_message = f"Script exited with code {exit_code}: {logs[:500]}"
            
            tool_call.exec_time_ms = int((time.time() - start_time) * 1000)
            
        except ContainerError as e:
            tool_call.status = "error"
            tool_call.error_message = f"Container error: {str(e)}"
        except Exception as e:
            tool_call.status = "error"
            tool_call.error_message = f"Execution error: {str(e)}"
        finally:
            # Ensure cleanup
            if container_id:
                try:
                    container = self.client.containers.get(container_id)
                    container.remove(force=True)
                except Exception:
                    pass
            
            # Clean up temp file
            try:
                Path(script_path).unlink()
            except Exception:
                pass
        
        return tool_call
    
    def execute_command(
        self,
        command: list[str],
        working_dir: str = "/app",
    ) -> dict[str, Any]:
        """Execute a command in sandbox.
        
        Args:
            command: Command and arguments
            working_dir: Working directory
            
        Returns:
            Execution result
        """
        if self.client is None:
            return {"error": "Docker not available"}
        
        if not self._ensure_image():
            return {"error": "Failed to build Docker image"}
        
        try:
            result = self.client.containers.run(
                self.image_name,
                command=command,
                working_dir=working_dir,
                remove=True,
                cpu_quota=int(self.cpu_limit * 100000),
                mem_limit=self.memory_limit,
                network_mode="none",
                timeout=self.timeout,
            )
            
            return {
                "success": True,
                "output": result.decode("utf-8"),
            }
            
        except ContainerError as e:
            return {
                "success": False,
                "error": str(e),
                "exit_code": e.exit_status,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


class InternalToolExecutor:
    """Executor for internal Python functions (no sandbox).
    
    Used for high-frequency, low-risk operations that don't
    require sandbox isolation.
    """
    
    def __init__(self):
        """Initialize internal executor."""
        self.tools: dict[str, callable] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register built-in internal tools."""
        self.tools["query_inventory"] = self._query_inventory
        self.tools["query_production_progress"] = self._query_production_progress
        self.tools["track_logistics"] = self._track_logistics
        self.tools["evaluate_supplier"] = self._evaluate_supplier
    
    def _query_inventory(self, **kwargs) -> dict[str, Any]:
        """Query inventory (placeholder - connects to database)."""
        from src.data.database import db_manager
        
        warehouse_id = kwargs.get("warehouse_id")
        sku = kwargs.get("sku")
        
        # Query database
        # This is a simplified placeholder
        return {
            "warehouse_id": warehouse_id or "all",
            "sku": sku,
            "available_qty": 500,
            "reserved_qty": 100,
            "status": "success",
        }
    
    def _query_production_progress(self, **kwargs) -> dict[str, Any]:
        """Query production progress (placeholder)."""
        batch_id = kwargs.get("batch_id")
        
        return {
            "batch_id": batch_id,
            "overall_progress": 0.75,
            "status": "in_production",
            "estimated_completion": "2026-02-01",
        }
    
    def _track_logistics(self, **kwargs) -> dict[str, Any]:
        """Track logistics (placeholder)."""
        waybill_id = kwargs.get("waybill_id")
        
        return {
            "waybill_id": waybill_id,
            "carrier": "顺丰",
            "status": "in_transit",
            "current_location": "上海转运中心",
            "estimated_delivery": "2026-01-30",
        }
    
    def _evaluate_supplier(self, **kwargs) -> dict[str, Any]:
        """Evaluate supplier (placeholder)."""
        supplier_id = kwargs.get("supplier_id")
        
        return {
            "supplier_id": supplier_id,
            "overall_rating": "B",
            "scores": {
                "quality": 85,
                "delivery": 78,
                "cost": 82,
                "service": 88,
            },
            "recommendation": "维持合作，关注交付时效",
        }
    
    def execute(self, tool_name: str, parameters: dict[str, Any]) -> ToolCall:
        """Execute an internal tool.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            ToolCall result
        """
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
        )
        
        import time
        start_time = time.time()
        
        try:
            tool_func = self.tools.get(tool_name)
            if not tool_func:
                tool_call.status = "error"
                tool_call.error_message = f"Unknown tool: {tool_name}"
                return tool_call
            
            result = tool_func(**parameters)
            tool_call.status = "success"
            tool_call.result = result
            
        except Exception as e:
            tool_call.status = "error"
            tool_call.error_message = str(e)
        
        tool_call.exec_time_ms = int((time.time() - start_time) * 1000)
        
        return tool_call


class ToolExecutor:
    """Unified tool executor routing to appropriate backend."""
    
    def __init__(self):
        """Initialize tool executor."""
        self.docker_executor = DockerSandboxExecutor()
        self.internal_executor = InternalToolExecutor()
    
    def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        execution_mode: str = "internal",
    ) -> ToolCall:
        """Execute a tool with specified mode.
        
        Args:
            tool_name: Tool name
            parameters: Tool parameters
            execution_mode: internal/script/api
            
        Returns:
            ToolCall result
        """
        if execution_mode == "internal":
            return self.internal_executor.execute(tool_name, parameters)
        
        elif execution_mode == "script":
            # Load script template and execute in Docker
            script_path = Path(__file__).parent / "scripts" / f"{tool_name}.py"
            if not script_path.exists():
                return ToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    status="error",
                    error_message=f"Script not found: {script_path}",
                )
            
            with open(script_path, "r") as f:
                script_content = f.read()
            
            return self.docker_executor.execute_script(script_content, parameters)
        
        elif execution_mode == "api":
            # API call (placeholder)
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                status="error",
                error_message="API execution not implemented",
            )
        
        else:
            return ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                status="error",
                error_message=f"Unknown execution mode: {execution_mode}",
            )


# Global executor instance
_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    """Get or create global tool executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
