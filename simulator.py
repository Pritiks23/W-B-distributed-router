"""GPU cluster simulator with realistic telemetry."""
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass, field
from config import ClusterConfig, ClusterMetrics
import math


@dataclass
class SimulatedJob:
    """Represents a job running on a cluster."""
    job_id: str
    tokens: int
    model_name: str
    arrival_time: float
    priority: int
    estimated_tokens_per_sec: float = 500.0
    memory_gb: float = 0.0
    
    def compute_memory_required(self) -> float:
        """Estimate memory needed based on model and tokens."""
        model_size_gb = {
            "llama-7b": 14.0,
            "llama-13b": 26.0,
            "llama-70b": 140.0,
            "mistral-7b": 14.0,
        }
        base = model_size_gb.get(self.model_name, 20.0)
        # KV cache adds overhead
        kv_cache_gb = (self.tokens * 2 * 2) / (1024**3)  # Rough approximation
        return base + kv_cache_gb
    
    def get_throughput(self) -> float:
        """Get tokens/sec for this job based on model."""
        model_tps = {
            "llama-7b": 800.0,
            "llama-13b": 600.0,
            "llama-70b": 200.0,
            "mistral-7b": 750.0,
        }
        return model_tps.get(self.model_name, 500.0)


@dataclass
class ClusterSimulator:
    """Simulates a GPU cluster with realistic dynamics."""
    config: ClusterConfig
    metrics: ClusterMetrics = field(default_factory=lambda: ClusterMetrics(
        gpu_utilization=30.0,
        memory_used_gb=64.0,
        memory_free_gb=256.0,
        queue_depth=0,
        avg_latency_ms=100.0,
        tokens_per_sec=0.0,
        network_bw_utilization=20.0,
        active_jobs=0,
        oom_count=0,
        avg_job_duration_ms=500.0,
    ))
    active_jobs: Dict[str, SimulatedJob] = field(default_factory=dict)
    completed_jobs: list = field(default_factory=list)
    queued_jobs: list = field(default_factory=list)
    time: float = 0.0
    total_throughput: float = 0.0
    
    def __post_init__(self):
        self.total_memory_gb = self.config.num_gpus * self.config.gpu_memory_gb
    
    def submit_job(self, job: SimulatedJob) -> bool:
        """Try to submit a job. Returns True if accepted, False if OOM."""
        memory_needed = job.compute_memory_required()
        
        # Check if we can accept the job
        if len(self.active_jobs) < self.config.max_concurrent_jobs:
            if self.metrics.memory_free_gb >= memory_needed:
                self.active_jobs[job.job_id] = job
                self.metrics.memory_used_gb += memory_needed
                self.metrics.memory_free_gb -= memory_needed
                self.metrics.active_jobs += 1
                return True
            else:
                self.metrics.oom_count += 1
        
        # Queue the job
        self.queued_jobs.append(job)
        self.metrics.queue_depth = len(self.queued_jobs)
        return False
    
    def step(self, dt: float = 1.0):
        """Advance simulation by dt seconds."""
        self.time += dt
        
        # Process active jobs
        completed = []
        total_tps = 0.0
        
        for job_id, job in self.active_jobs.items():
            tps = job.get_throughput()
            total_tps += tps
            
            # Simple: assume job runs to completion
            # In reality, would track progress
            completed.append(job_id)
        
        # Remove completed jobs and free memory
        for job_id in completed:
            job = self.active_jobs.pop(job_id)
            memory_freed = job.compute_memory_required()
            self.metrics.memory_used_gb -= memory_freed
            self.metrics.memory_free_gb += memory_freed
            self.completed_jobs.append(job)
        
        # Try to schedule queued jobs
        while self.queued_jobs and len(self.active_jobs) < self.config.max_concurrent_jobs:
            job = self.queued_jobs.pop(0)
            memory_needed = job.compute_memory_required()
            if self.metrics.memory_free_gb >= memory_needed:
                self.active_jobs[job.job_id] = job
                self.metrics.memory_used_gb += memory_needed
                self.metrics.memory_free_gb -= memory_needed
            else:
                # Put it back in queue
                self.queued_jobs.insert(0, job)
                break
        
        # Update metrics
        self.metrics.active_jobs = len(self.active_jobs)
        self.metrics.queue_depth = len(self.queued_jobs)
        self.metrics.tokens_per_sec = total_tps
        self.total_throughput = total_tps
        
        # Realistic utilization calculation
        if self.metrics.active_jobs > 0:
            # Utilization scales with active jobs and their throughput
            util_factor = min(1.0, self.metrics.active_jobs / self.config.num_gpus)
            self.metrics.gpu_utilization = 20.0 + (util_factor * 75.0)
        else:
            # Idle, with some baseline overhead
            self.metrics.gpu_utilization = np.random.normal(5.0, 2.0)
            self.metrics.gpu_utilization = max(0.0, self.metrics.gpu_utilization)
        
        # Network utilization scales with throughput
        max_tps = self.config.num_gpus * 1000.0  # Rough max
        network_factor = min(1.0, total_tps / max_tps)
        self.metrics.network_bw_utilization = 10.0 + (network_factor * 80.0)
        
        # Latency simulation: queue depth and active jobs increase latency
        base_latency = 50.0
        queue_latency = self.metrics.queue_depth * 10.0
        util_latency = (self.metrics.gpu_utilization / 100.0) * 100.0
        self.metrics.avg_latency_ms = base_latency + queue_latency + util_latency
        
        # Add some realistic noise
        self.metrics.gpu_utilization += np.random.normal(0, 2.0)
        self.metrics.gpu_utilization = np.clip(self.metrics.gpu_utilization, 0, 100)
        
        self.metrics.avg_latency_ms += np.random.normal(0, 10.0)
        self.metrics.avg_latency_ms = max(0, self.metrics.avg_latency_ms)
    
    def get_metrics_dict(self) -> Dict:
        """Return metrics as dictionary."""
        return {
            "gpu_utilization": self.metrics.gpu_utilization,
            "memory_used_gb": self.metrics.memory_used_gb,
            "memory_free_gb": self.metrics.memory_free_gb,
            "queue_depth": self.metrics.queue_depth,
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "tokens_per_sec": self.metrics.tokens_per_sec,
            "network_bw_utilization": self.metrics.network_bw_utilization,
            "active_jobs": self.metrics.active_jobs,
            "oom_count": self.metrics.oom_count,
        }
