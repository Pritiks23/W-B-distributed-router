"""FastAPI inference server with routing."""
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging

from config import get_default_clusters, RouterConfig
from simulator import ClusterSimulator
from features import FeatureEngineer
from router import RouterFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Models
class InferenceRequest(BaseModel):
    """Request for inference routing."""
    tokens: int = Field(default=1024, ge=1, le=8192)
    model_name: str = Field(default="llama-7b")
    priority: int = Field(default=3, ge=1, le=5)


class RoutingResponse(BaseModel):
    """Response from router."""
    cluster_name: str
    cluster_index: int
    utilization_percent: float
    queue_depth: int
    estimated_latency_ms: float
    status: str


class SystemMetrics(BaseModel):
    """Overall system metrics."""
    clusters: List[dict]
    total_utilization: float
    idle_clusters: int
    total_queue_depth: int
    overall_throughput_tps: float


class InferenceServer:
    """FastAPI server for distributed inference routing."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.clusters = get_default_clusters()
        self.router_config = RouterConfig()
        
        # Initialize simulators
        self.simulators = [
            ClusterSimulator(config) for config in self.clusters
        ]
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            len(self.clusters),
            self.router_config.num_feature_dims
        )
        
        # Initialize router
        try:
            self.router = RouterFactory.create(
                "mlp",
                self.router_config.num_feature_dims,
                self.router_config.num_clusters,
                self.router_config.hidden_dim
            )
            self.router = self.router.to(device)
            self.router.eval()
            
            if model_path:
                self.load_router(model_path)
            
            logger.info(f"Router initialized: {self.router.__class__.__name__}")
        except Exception as e:
            logger.warning(f"Could not load router model: {e}. Using untrained model.")
        
        # Request counter
        self.request_count = 0
        self.routed_requests = []
    
    def load_router(self, model_path: str):
        """Load pre-trained router weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.router.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.router.load_state_dict(checkpoint)
            logger.info(f"Loaded router from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load router: {e}")
    
    def route_request(self, request: InferenceRequest) -> RoutingResponse:
        """Route a single request to the best cluster."""
        self.request_count += 1
        
        # Convert request to dict
        request_dict = {
            "request_id": f"req-{self.request_count}",
            "tokens": request.tokens,
            "model_name": request.model_name,
            "priority": request.priority,
            "timestamp": self.request_count,
        }
        
        # Get cluster metrics
        cluster_metrics_list = [sim.metrics for sim in self.simulators]
        
        # Engineer features
        req_feat, cluster_feat = self.feature_engineer.engineer_features(
            request_dict, cluster_metrics_list
        )
        
        # Combine features
        combined_feat = self.feature_engineer.combine_features(req_feat, cluster_feat)
        
        # Convert to torch tensor
        feat_tensor = torch.from_numpy(combined_feat).float().to(self.device)
        
        # Get routing decision
        with torch.no_grad():
            if hasattr(self.router, "select_cluster"):
                cluster_idx = self.router.select_cluster(feat_tensor)
            else:
                scores = self.router(feat_tensor)
                if scores.dim() > 1:
                    scores = scores.squeeze(0)
                cluster_idx = scores.argmax(dim=-1).item()
        
        # Get selected cluster
        selected_sim = self.simulators[cluster_idx]
        selected_cluster = self.clusters[cluster_idx]
        
        # Step simulation
        selected_sim.step(dt=0.1)
        
        # Prepare response
        response = RoutingResponse(
            cluster_name=selected_cluster.name,
            cluster_index=cluster_idx,
            utilization_percent=selected_sim.metrics.gpu_utilization,
            queue_depth=selected_sim.metrics.queue_depth,
            estimated_latency_ms=selected_sim.metrics.avg_latency_ms,
            status="routed" if selected_sim.metrics.gpu_utilization < 95 else "warning_high_util",
        )
        
        # Track request
        self.routed_requests.append({
            "request_id": request_dict["request_id"],
            "routed_cluster": cluster_idx,
            "utilization": selected_sim.metrics.gpu_utilization,
        })
        
        return response
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics."""
        utilizations = [sim.metrics.gpu_utilization for sim in self.simulators]
        queue_depths = [sim.metrics.queue_depth for sim in self.simulators]
        throughputs = [sim.metrics.tokens_per_sec for sim in self.simulators]
        
        cluster_info = []
        for i, sim in enumerate(self.simulators):
            cluster_info.append({
                "name": self.clusters[i].name,
                "utilization_percent": sim.metrics.gpu_utilization,
                "queue_depth": sim.metrics.queue_depth,
                "memory_free_gb": sim.metrics.memory_free_gb,
                "throughput_tps": sim.metrics.tokens_per_sec,
                "latency_ms": sim.metrics.avg_latency_ms,
            })
        
        idle_clusters = sum(1 for u in utilizations if u < 30.0)
        total_util = np.mean(utilizations)
        total_tps = sum(throughputs)
        
        return SystemMetrics(
            clusters=cluster_info,
            total_utilization=total_util,
            idle_clusters=idle_clusters,
            total_queue_depth=sum(queue_depths),
            overall_throughput_tps=total_tps,
        )


# Create FastAPI app
app = FastAPI(
    title="Distributed GPU Router",
    description="ML-driven distributed inference routing system",
    version="1.0.0",
)

# Initialize server
server = InferenceServer(device="cpu")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "requests_processed": server.request_count,
    }


@app.post("/route", response_model=RoutingResponse)
async def route_inference(request: InferenceRequest):
    """Route an inference request."""
    try:
        response = server.route_request(request)
        return response
    except Exception as e:
        logger.error(f"Routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics."""
    try:
        return server.get_system_metrics()
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters")
async def get_clusters():
    """Get cluster information."""
    return {
        "clusters": [
            {
                "name": cluster.name,
                "num_gpus": cluster.num_gpus,
                "gpu_memory_gb": cluster.gpu_memory_gb,
                "network_bandwidth_gbps": cluster.network_bandwidth_gbps,
                "max_concurrent_jobs": cluster.max_concurrent_jobs,
            }
            for cluster in server.clusters
        ]
    }


@app.get("/routing-history")
async def get_routing_history(limit: int = 100):
    """Get recent routing decisions."""
    return {
        "routing_decisions": server.routed_requests[-limit:]
    }


@app.post("/reset-simulators")
async def reset_simulators():
    """Reset all cluster simulators."""
    for sim in server.simulators:
        sim.active_jobs.clear()
        sim.queued_jobs.clear()
        sim.completed_jobs.clear()
        sim.metrics.gpu_utilization = 30.0
        sim.metrics.queue_depth = 0
        sim.metrics.active_jobs = 0
    
    return {"status": "simulators_reset"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
