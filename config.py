"""Configuration management for the distributed router system."""
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ClusterConfig:
    """Configuration for a single GPU cluster."""
    name: str
    num_gpus: int
    gpu_memory_gb: int
    network_bandwidth_gbps: float
    max_concurrent_jobs: int


@dataclass
class RouterConfig:
    """Configuration for the router system."""
    num_clusters: int = 3
    num_feature_dims: int = 8
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_training_steps: int = 10000
    evaluation_interval: int = 100
    
    # W&B configuration
    wandb_project: str = "distributed-router"
    wandb_entity: str = os.getenv("WANDB_ENTITY", None)
    wandb_api_key: str = os.getenv("WANDB_API_KEY", "")


@dataclass
class RequestConfig:
    """Configuration for request generation."""
    min_tokens: int = 512
    max_tokens: int = 8192
    min_priority: int = 1
    max_priority: int = 5
    arrival_rate_per_sec: float = 10.0
    models: List[str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["llama-7b", "llama-13b", "llama-70b", "mistral-7b"]


@dataclass
class ClusterMetrics:
    """Real-time metrics for a cluster."""
    gpu_utilization: float  # 0-100
    memory_used_gb: float
    memory_free_gb: float
    queue_depth: int
    avg_latency_ms: float
    tokens_per_sec: float
    network_bw_utilization: float  # 0-100
    active_jobs: int
    oom_count: int
    avg_job_duration_ms: float


def get_default_clusters() -> List[ClusterConfig]:
    """Get default cluster configurations."""
    return [
        ClusterConfig(
            name="cluster-a-t4",
            num_gpus=8,
            gpu_memory_gb=16,
            network_bandwidth_gbps=10,
            max_concurrent_jobs=32,
        ),
        ClusterConfig(
            name="cluster-b-a100",
            num_gpus=4,
            gpu_memory_gb=80,
            network_bandwidth_gbps=20,
            max_concurrent_jobs=16,
        ),
        ClusterConfig(
            name="cluster-c-h100",
            num_gpus=2,
            gpu_memory_gb=141,
            network_bandwidth_gbps=25,
            max_concurrent_jobs=8,
        ),
    ]


def get_router_config() -> RouterConfig:
    """Get router configuration from env or defaults."""
    return RouterConfig(
        wandb_entity=os.getenv("WANDB_ENTITY", ""),
        wandb_api_key=os.getenv("WANDB_API_KEY", ""),
    )
