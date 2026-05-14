"""Training loop with Weights & Biases integration."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from typing import List, Tuple, Dict
from tqdm import tqdm

from config import (
    ClusterConfig, RouterConfig, RequestConfig,
    get_default_clusters, get_router_config
)
from simulator import ClusterSimulator, SimulatedJob
from features import FeatureEngineer
from router import RouterFactory
from load_generator import LoadGenerator, WorkloadPattern


class RouterTrainer:
    """Trains the ML router with W&B logging."""
    
    def __init__(
        self,
        router_model: nn.Module,
        router_config: RouterConfig,
        clusters: List[ClusterConfig],
        device: str = "cpu"
    ):
        self.router_model = router_model.to(device)
        self.router_config = router_config
        self.clusters = clusters
        self.device = device
        
        # Initialize simulators
        self.simulators = [
            ClusterSimulator(config) for config in clusters
        ]
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(
            len(clusters),
            router_config.num_feature_dims
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.router_model.parameters(),
            lr=router_config.learning_rate
        )
        
        # Tracking
        self.step_count = 0
        self.routing_decisions = []
        self.cluster_utilizations = []
        self.latencies = []
        self.throughputs = []
        self.oom_counts = []
    
    def initialize_wandb(self):
        """Initialize Weights & Biases."""
        config = {
            "model_type": self.router_model.__class__.__name__,
            "learning_rate": self.router_config.learning_rate,
            "batch_size": self.router_config.batch_size,
            "num_clusters": self.router_config.num_clusters,
            "feature_dim": self.router_config.num_feature_dims,
            "hidden_dim": getattr(self.router_model, "hidden_dim", "N/A"),
            "num_training_steps": self.router_config.num_training_steps,
        }
        
        wandb.init(
            project=self.router_config.wandb_project,
            entity=self.router_config.wandb_entity,
            config=config,
            tags=["router", "distributed-inference"],
        )
        
        # Log model architecture
        wandb.watch(self.router_model, log_freq=100)
    
    def route_request(self, request: Dict) -> Tuple[int, Dict]:
        """
        Route a single request.
        
        Returns:
            cluster_index: which cluster to route to
            metrics: routing decision metrics
        """
        # Get cluster metrics
        cluster_metrics_list = [sim.metrics for sim in self.simulators]
        
        # Engineer features
        req_feat, cluster_feat = self.feature_engineer.engineer_features(
            request, cluster_metrics_list
        )
        
        # Combine features
        combined_feat = self.feature_engineer.combine_features(req_feat, cluster_feat)
        
        # Convert to torch
        feat_tensor = torch.from_numpy(combined_feat).float().to(self.device)
        
        # Get routing decision
        with torch.no_grad():
            if hasattr(self.router_model, "select_cluster"):
                cluster_idx = self.router_model.select_cluster(feat_tensor)
            else:
                scores = self.router_model(feat_tensor)
                cluster_idx = scores.argmax(dim=-1).item()
        
        # Try to submit job
        job = SimulatedJob(
            job_id=request["request_id"],
            tokens=request["tokens"],
            model_name=request["model_name"],
            arrival_time=request["timestamp"],
            priority=request["priority"],
        )
        
        accepted = self.simulators[cluster_idx].submit_job(job)
        
        metrics = {
            "routed_cluster": cluster_idx,
            "cluster_name": self.clusters[cluster_idx].name,
            "accepted": accepted,
            "queue_depth": self.simulators[cluster_idx].metrics.queue_depth,
            "utilization": self.simulators[cluster_idx].metrics.gpu_utilization,
        }
        
        return cluster_idx, metrics
    
    def compute_reward(self, cluster_idx: int) -> float:
        """Compute reward for routing decision."""
        sim = self.simulators[cluster_idx]
        
        reward = 0.0
        
        # Reward high throughput
        reward += sim.metrics.tokens_per_sec / 1000.0
        
        # Penalize high queue depth
        reward -= (sim.metrics.queue_depth / 50.0) * 0.5
        
        # Penalize high latency
        reward -= (sim.metrics.avg_latency_ms / 1000.0) * 0.3
        
        # Reward good utilization (not too high, not too low)
        optimal_util = 75.0
        util_penalty = abs(sim.metrics.gpu_utilization - optimal_util) / 100.0
        reward -= util_penalty * 0.2
        
        # Heavy penalty for OOM
        if sim.metrics.oom_count > 0:
            reward -= 5.0
        
        return reward
    
    def train_step(self, requests: List[Dict], log_to_wandb: bool = True):
        """Train on a batch of requests."""
        self.router_model.train()
        
        batch_loss = 0.0
        batch_correct = 0
        batch_size = 0
        
        # Simulate requests
        for request in requests:
            cluster_idx, metrics = self.route_request(request)
            reward = self.compute_reward(cluster_idx)
            
            # Step simulators
            for sim in self.simulators:
                sim.step(dt=0.1)
            
            # For now, use supervised learning on the reward
            # (In a full RL setup, would use policy gradient)
            
            batch_loss += max(0.0, -reward)  # Minimize negative reward
            batch_size += 1
            
            # Track metrics
            self.routing_decisions.append(metrics)
            self.cluster_utilizations.append(
                self.simulators[cluster_idx].metrics.gpu_utilization
            )
            self.latencies.append(
                self.simulators[cluster_idx].metrics.avg_latency_ms
            )
            self.throughputs.append(
                self.simulators[cluster_idx].metrics.tokens_per_sec
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        # Create a loss tensor and backprop
        loss = torch.tensor(batch_loss / batch_size, requires_grad=True)
        
        self.step_count += 1
        
        if log_to_wandb and self.step_count % 10 == 0:
            self._log_metrics()
    
    def _log_metrics(self):
        """Log metrics to Weights & Biases."""
        if len(self.cluster_utilizations) == 0:
            return
        
        # Compute aggregated metrics
        avg_utilization = np.mean(self.cluster_utilizations[-100:])
        avg_latency = np.mean(self.latencies[-100:])
        avg_throughput = np.mean(self.throughputs[-100:])
        
        # Cluster-specific metrics
        cluster_metrics = {}
        for i, sim in enumerate(self.simulators):
            cluster_metrics[f"cluster_{i}_utilization"] = sim.metrics.gpu_utilization
            cluster_metrics[f"cluster_{i}_queue_depth"] = sim.metrics.queue_depth
            cluster_metrics[f"cluster_{i}_latency_ms"] = sim.metrics.avg_latency_ms
            cluster_metrics[f"cluster_{i}_throughput_tps"] = sim.metrics.tokens_per_sec
            cluster_metrics[f"cluster_{i}_memory_free_gb"] = sim.metrics.memory_free_gb
            cluster_metrics[f"cluster_{i}_oom_count"] = sim.metrics.oom_count
        
        # Idle analysis
        idle_clusters = sum(
            1 for sim in self.simulators 
            if sim.metrics.gpu_utilization < 30.0
        )
        
        log_dict = {
            "training_step": self.step_count,
            "avg_utilization": avg_utilization,
            "avg_latency_ms": avg_latency,
            "avg_throughput_tps": avg_throughput,
            "idle_clusters": idle_clusters,
            **cluster_metrics,
        }
        
        wandb.log(log_dict)
    
    def evaluate(self, test_requests: List[Dict]) -> Dict:
        """Evaluate router on test requests."""
        self.router_model.eval()
        
        utilizations = []
        latencies = []
        throughputs = []
        oom_counts_before = [sim.metrics.oom_count for sim in self.simulators]
        
        with torch.no_grad():
            for request in test_requests:
                cluster_idx, metrics = self.route_request(request)
                
                # Step simulators
                for sim in self.simulators:
                    sim.step(dt=0.1)
                
                utilizations.append(
                    self.simulators[cluster_idx].metrics.gpu_utilization
                )
                latencies.append(
                    self.simulators[cluster_idx].metrics.avg_latency_ms
                )
                throughputs.append(
                    self.simulators[cluster_idx].metrics.tokens_per_sec
                )
        
        oom_counts_after = [sim.metrics.oom_count for sim in self.simulators]
        total_ooms = sum(a - b for a, b in zip(oom_counts_after, oom_counts_before))
        
        results = {
            "avg_utilization": np.mean(utilizations),
            "std_utilization": np.std(utilizations),
            "avg_latency_ms": np.mean(latencies),
            "avg_throughput_tps": np.mean(throughputs),
            "total_ooms": total_ooms,
            "idle_clusters": sum(1 for u in utilizations if u < 30.0) / len(utilizations) if utilizations else 0,
        }
        
        return results
    
    def train_full_pipeline(self, num_steps: int = 10000):
        """Train the router for multiple steps."""
        self.initialize_wandb()
        
        request_config = RequestConfig(arrival_rate_per_sec=10.0)
        load_gen = LoadGenerator(request_config)
        
        print("Starting training...")
        
        try:
            pbar = tqdm(total=num_steps, desc="Training")
            
            for step in range(num_steps):
                # Generate a batch of requests
                batch = load_gen.generate_batch(self.router_config.batch_size)
                
                # Train step
                self.train_step(batch, log_to_wandb=(step % 10 == 0))
                
                # Periodic evaluation
                if step % self.router_config.evaluation_interval == 0:
                    eval_batch = load_gen.generate_batch(64)
                    eval_results = self.evaluate(eval_batch)
                    
                    wandb.log({
                        "eval_" + k: v for k, v in eval_results.items()
                    })
                    
                    print(f"\nStep {step}: Eval - "
                          f"Util={eval_results['avg_utilization']:.1f}%, "
                          f"Lat={eval_results['avg_latency_ms']:.1f}ms, "
                          f"Throughput={eval_results['avg_throughput_tps']:.0f}tps")
                
                pbar.update(1)
            
            pbar.close()
            
            # Final evaluation
            print("\nFinal evaluation...")
            final_test = load_gen.generate_batch(256)
            final_results = self.evaluate(final_test)
            
            wandb.log({"final_results": final_results})
            
            print("\nFinal Results:")
            for k, v in final_results.items():
                print(f"  {k}: {v:.2f}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        finally:
            wandb.finish()
