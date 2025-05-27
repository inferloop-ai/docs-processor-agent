"""
Graph Utilities for LangGraph Workflows
Common utilities and helper functions for workflow orchestration
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, TypedDict, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class NodeStatus(Enum):
    """Individual node execution status"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class NodeExecution:
    """Track individual node execution"""
    node_name: str
    status: NodeStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowExecution:
    """Track complete workflow execution"""
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time: Optional[float] = None
    node_executions: List[NodeExecution] = None
    current_node: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.node_executions is None:
            self.node_executions = []
        if self.metadata is None:
            self.metadata = {}


class WorkflowTracker:
    """Track and manage workflow executions"""
    
    def __init__(self):
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_executions: Dict[str, str] = {}  # thread_id -> workflow_id
    
    def start_workflow(self, workflow_name: str, thread_id: Optional[str] = None) -> str:
        """Start tracking a new workflow execution"""
        workflow_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.executions[workflow_id] = execution
        
        if thread_id:
            self.active_executions[thread_id] = workflow_id
        
        logger.info(f"Started workflow tracking: {workflow_name} [{workflow_id}]")
        return workflow_id
    
    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus, 
                             error_message: Optional[str] = None):
        """Update workflow status"""
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            execution.status = status
            
            if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                execution.end_time = datetime.now()
                execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            if error_message:
                execution.error_message = error_message
            
            logger.info(f"Workflow {workflow_id} status updated to: {status.value}")
    
    def start_node(self, workflow_id: str, node_name: str):
        """Start tracking node execution"""
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            execution.current_node = node_name
            
            node_execution = NodeExecution(
                node_name=node_name,
                status=NodeStatus.RUNNING,
                start_time=datetime.now()
            )
            
            # Remove any existing execution for this node (in case of retry)
            execution.node_executions = [ne for ne in execution.node_executions if ne.node_name != node_name]
            execution.node_executions.append(node_execution)
            
            logger.debug(f"Started node execution: {node_name} in workflow {workflow_id}")
    
    def complete_node(self, workflow_id: str, node_name: str, 
                     result_data: Optional[Dict[str, Any]] = None):
        """Complete node execution"""
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            
            for node_exec in execution.node_executions:
                if node_exec.node_name == node_name and node_exec.status == NodeStatus.RUNNING:
                    node_exec.status = NodeStatus.COMPLETED
                    node_exec.end_time = datetime.now()
                    node_exec.execution_time = (node_exec.end_time - node_exec.start_time).total_seconds()
                    node_exec.result_data = result_data
                    break
            
            logger.debug(f"Completed node execution: {node_name} in workflow {workflow_id}")
    
    def fail_node(self, workflow_id: str, node_name: str, error_message: str):
        """Mark node as failed"""
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            
            for node_exec in execution.node_executions:
                if node_exec.node_name == node_name and node_exec.status == NodeStatus.RUNNING:
                    node_exec.status = NodeStatus.FAILED
                    node_exec.end_time = datetime.now()
                    node_exec.execution_time = (node_exec.end_time - node_exec.start_time).total_seconds()
                    node_exec.error_message = error_message
                    break
            
            logger.warning(f"Failed node execution: {node_name} in workflow {workflow_id}: {error_message}")
    
    def retry_node(self, workflow_id: str, node_name: str):
        """Increment retry count for node"""
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            
            for node_exec in execution.node_executions:
                if node_exec.node_name == node_name:
                    node_exec.retry_count += 1
                    node_exec.status = NodeStatus.RETRYING
                    break
            
            logger.info(f"Retrying node: {node_name} in workflow {workflow_id}")
    
    def get_execution(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution details"""
        return self.executions.get(workflow_id)
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all active workflow executions"""
        return [
            execution for execution in self.executions.values()
            if execution.status == WorkflowStatus.RUNNING
        ]
    
    def get_execution_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get execution summary"""
        execution = self.executions.get(workflow_id)
        if not execution:
            return {}
        
        completed_nodes = [ne for ne in execution.node_executions if ne.status == NodeStatus.COMPLETED]
        failed_nodes = [ne for ne in execution.node_executions if ne.status == NodeStatus.FAILED]
        
        return {
            "workflow_id": execution.workflow_id,
            "workflow_name": execution.workflow_name,
            "status": execution.status.value,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "total_execution_time": execution.total_execution_time,
            "current_node": execution.current_node,
            "total_nodes": len(execution.node_executions),
            "completed_nodes": len(completed_nodes),
            "failed_nodes": len(failed_nodes),
            "error_message": execution.error_message,
            "metadata": execution.metadata
        }


class WorkflowBuilder:
    """Helper class for building LangGraph workflows"""
    
    def __init__(self, state_schema: type):
        self.state_schema = state_schema
        self.workflow = StateGraph(state_schema)
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.interrupts = []
        self.checkpointer = None
    
    def add_node(self, name: str, func: Callable, description: str = ""):
        """Add a node to the workflow"""
        self.nodes[name] = {
            "function": func,
            "description": description
        }
        self.workflow.add_node(name, func)
        return self
    
    def add_edge(self, from_node: str, to_node: str):
        """Add a direct edge between nodes"""
        self.edges.append((from_node, to_node))
        self.workflow.add_edge(from_node, to_node)
        return self
    
    def add_conditional_edge(self, from_node: str, condition_func: Callable, 
                           condition_map: Dict[str, str]):
        """Add a conditional edge"""
        self.conditional_edges.append({
            "from_node": from_node,
            "condition_func": condition_func,
            "condition_map": condition_map
        })
        self.workflow.add_conditional_edges(from_node, condition_func, condition_map)
        return self
    
    def add_interrupt(self, node_name: str):
        """Add an interrupt point"""
        self.interrupts.append(node_name)
        return self
    
    def set_entry_point(self, node_name: str):
        """Set the workflow entry point"""
        self.workflow.add_edge(START, node_name)
        return self
    
    def set_exit_point(self, node_name: str):
        """Set the workflow exit point"""
        self.workflow.add_edge(node_name, END)
        return self
    
    def enable_checkpointing(self, checkpointer=None):
        """Enable checkpointing for the workflow"""
        self.checkpointer = checkpointer or MemorySaver()
        return self
    
    def compile(self):
        """Compile the workflow"""
        if self.checkpointer:
            return self.workflow.compile(
                checkpointer=self.checkpointer,
                interrupt_before=self.interrupts
            )
        else:
            return self.workflow.compile()
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow structure"""
        return {
            "nodes": list(self.nodes.keys()),
            "edges": self.edges,
            "conditional_edges": [
                {
                    "from": ce["from_node"],
                    "conditions": list(ce["condition_map"].keys())
                }
                for ce in self.conditional_edges
            ],
            "interrupts": self.interrupts,
            "has_checkpointing": self.checkpointer is not None
        }


def create_retry_decorator(max_retries: int = 3, delay: float = 1.0, 
                         exponential_backoff: bool = True):
    """Create a retry decorator for workflow nodes"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt) if exponential_backoff else delay
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator


def create_timeout_wrapper(timeout_seconds: float):
    """Create a timeout wrapper for workflow nodes"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
        
        return wrapper
    return decorator


def create_error_handler(error_types: List[type], fallback_result: Any = None):
    """Create an error handler that catches specific exceptions"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except tuple(error_types) as e:
                logger.warning(f"Handled error in {func.__name__}: {e}")
                return fallback_result
        
        return wrapper
    return decorator


class ConditionalRouter:
    """Helper for creating conditional routing logic"""
    
    def __init__(self):
        self.conditions = []
    
    def add_condition(self, name: str, condition_func: Callable[[Any], bool]):
        """Add a condition"""
        self.conditions.append((name, condition_func))
        return self
    
    def route(self, state: Any) -> str:
        """Evaluate conditions and return route"""
        for name, condition_func in self.conditions:
            try:
                if condition_func(state):
                    return name
            except Exception as e:
                logger.warning(f"Condition {name} evaluation failed: {e}")
                continue
        
        return "default"


def create_state_validator(required_fields: List[str], 
                         field_types: Optional[Dict[str, type]] = None):
    """Create a state validator function"""
    
    def validator(state: Dict[str, Any]) -> bool:
        # Check required fields
        for field in required_fields:
            if field not in state:
                logger.error(f"Required field missing: {field}")
                return False
        
        # Check field types if specified
        if field_types:
            for field, expected_type in field_types.items():
                if field in state and not isinstance(state[field], expected_type):
                    logger.error(f"Field {field} has wrong type. Expected {expected_type}, got {type(state[field])}")
                    return False
        
        return True
    
    return validator


def create_progress_tracker(total_steps: int):
    """Create a progress tracking function"""
    
    def track_progress(current_step: int, step_name: str = "") -> Dict[str, Any]:
        progress_percentage = (current_step / total_steps) * 100
        
        progress_info = {
            "current_step": current_step,
            "total_steps": total_steps,
            "progress_percentage": progress_percentage,
            "step_name": step_name,
            "is_complete": current_step >= total_steps
        }
        
        logger.info(f"Progress: {progress_percentage:.1f}% ({current_step}/{total_steps}) - {step_name}")
        return progress_info
    
    return track_progress


class MessageBuilder:
    """Helper for building workflow messages"""
    
    @staticmethod
    def create_system_message(content: str) -> SystemMessage:
        """Create a system message"""
        return SystemMessage(content=content)
    
    @staticmethod
    def create_human_message(content: str) -> HumanMessage:
        """Create a human message"""
        return HumanMessage(content=content)
    
    @staticmethod
    def create_ai_message(content: str) -> AIMessage:
        """Create an AI message"""
        return AIMessage(content=content)
    
    @staticmethod
    def create_workflow_message(workflow_name: str, node_name: str, 
                              content: str, message_type: str = "info") -> HumanMessage:
        """Create a workflow-specific message"""
        formatted_content = f"[{workflow_name}:{node_name}] {message_type.upper()}: {content}"
        return HumanMessage(content=formatted_content)


def log_workflow_state(state: Dict[str, Any], node_name: str, 
                      include_sensitive: bool = False) -> None:
    """Log workflow state for debugging"""
    safe_state = {}
    
    for key, value in state.items():
        if not include_sensitive and any(sensitive in key.lower() 
                                       for sensitive in ['password', 'key', 'token', 'secret']):
            safe_state[key] = "[REDACTED]"
        elif isinstance(value, (dict, list)) and len(str(value)) > 1000:
            safe_state[key] = f"[LARGE_{type(value).__name__.upper()}:{len(str(value))}chars]"
        else:
            safe_state[key] = value
    
    logger.debug(f"State at {node_name}: {json.dumps(safe_state, indent=2, default=str)}")


def create_workflow_metrics_collector():
    """Create a metrics collector for workflows"""
    
    metrics = {
        "workflow_executions": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "node_executions": {},
        "execution_times": [],
        "error_counts": {}
    }
    
    def collect_workflow_start():
        metrics["workflow_executions"] += 1
    
    def collect_workflow_success(execution_time: float):
        metrics["successful_executions"] += 1
        metrics["execution_times"].append(execution_time)
    
    def collect_workflow_failure(error_type: str):
        metrics["failed_executions"] += 1
        metrics["error_counts"][error_type] = metrics["error_counts"].get(error_type, 0) + 1
    
    def collect_node_execution(node_name: str, execution_time: float):
        if node_name not in metrics["node_executions"]:
            metrics["node_executions"][node_name] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        node_metrics = metrics["node_executions"][node_name]
        node_metrics["count"] += 1
        node_metrics["total_time"] += execution_time
        node_metrics["avg_time"] = node_metrics["total_time"] / node_metrics["count"]
    
    def get_metrics() -> Dict[str, Any]:
        result = metrics.copy()
        
        if metrics["execution_times"]:
            result["average_execution_time"] = sum(metrics["execution_times"]) / len(metrics["execution_times"])
            result["min_execution_time"] = min(metrics["execution_times"])
            result["max_execution_time"] = max(metrics["execution_times"])
        
        if metrics["workflow_executions"] > 0:
            result["success_rate"] = metrics["successful_executions"] / metrics["workflow_executions"]
        
        return result
    
    return {
        "collect_workflow_start": collect_workflow_start,
        "collect_workflow_success": collect_workflow_success,
        "collect_workflow_failure": collect_workflow_failure,
        "collect_node_execution": collect_node_execution,
        "get_metrics": get_metrics
    }


# Global workflow tracker instance
global_workflow_tracker = WorkflowTracker()


def get_workflow_tracker() -> WorkflowTracker:
    """Get the global workflow tracker instance"""
    return global_workflow_tracker


# Utility functions for common workflow patterns

def create_parallel_processor(node_functions: List[Callable], 
                            max_concurrent: int = 5):
    """Create a parallel processing node"""
    
    async def parallel_node(state):
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(func):
            async with semaphore:
                return await func(state)
        
        tasks = [process_with_semaphore(func) for func in node_functions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Function {i}: {str(result)}")
            else:
                successful_results.append(result)
        
        state["parallel_results"] = successful_results
        state["parallel_errors"] = errors
        
        return state
    
    return parallel_node


def create_batch_processor(batch_size: int = 10, process_func: Callable = None):
    """Create a batch processing node"""
    
    async def batch_node(state):
        items = state.get("items_to_process", [])
        processed_items = []
        errors = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            try:
                if process_func:
                    batch_results = await process_func(batch, state)
                    processed_items.extend(batch_results)
                else:
                    processed_items.extend(batch)
            except Exception as e:
                errors.append(f"Batch {i//batch_size + 1}: {str(e)}")
        
        state["processed_items"] = processed_items
        state["batch_errors"] = errors
        
        return state
    
    return batch_node


def create_circuit_breaker(failure_threshold: int = 5, 
                         reset_timeout: int = 60):
    """Create a circuit breaker for workflow nodes"""
    
    failures = {"count": 0, "last_failure_time": None}
    
    def circuit_breaker_decorator(func):
        async def wrapper(*args, **kwargs):
            current_time = datetime.now()
            
            # Check if circuit is open
            if (failures["count"] >= failure_threshold and 
                failures["last_failure_time"] and
                (current_time - failures["last_failure_time"]).seconds < reset_timeout):
                
                raise Exception(f"Circuit breaker is OPEN. Too many failures ({failures['count']})")
            
            try:
                result = await func(*args, **kwargs)
                # Reset on success
                failures["count"] = 0
                failures["last_failure_time"] = None
                return result
                
            except Exception as e:
                failures["count"] += 1
                failures["last_failure_time"] = current_time
                raise
        
        return wrapper
    return circuit_breaker_decorator