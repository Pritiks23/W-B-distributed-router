"""Quick demo and testing script."""
import argparse
import torch
import time
from pathlib import Path

from config import get_default_clusters, get_router_config
from simulator import ClusterSimulator, SimulatedJob
from features import FeatureEngineer
from router import RouterFactory
from load_generator import LoadGenerator, WorkloadPattern


def demo_cluster_simulator():
    """Demonstrate cluster simulator."""
    print("=" * 80)
    print("DEMO 1: Cluster Simulator")
    print("=" * 80)
    
    clusters = get_default_clusters()
    simulators = [ClusterSimulator(config) for config in clusters]
    
    print("\nInitial cluster state:")
    for i, sim in enumerate(simulators):
        metrics = sim.get_metrics_dict()
        print(f"\n  Cluster {i} ({clusters[i].name}):")
        print(f"    Utilization: {metrics['gpu_utilization']:.1f}%")
        print(f"    Memory Free: {metrics['memory_free_gb']:.1f}GB")
        print(f"    Latency: {metrics['avg_latency_ms']:.1f}ms")
    
    # Simulate some jobs
    print("\n\nSimulating 10 timesteps...")
    for t in range(10):
        # Submit random job
        job = SimulatedJob(
            job_id=f"job-{t}",
            tokens=2048,
            model_name="llama-7b",
            arrival_time=t,
            priority=3,
        )
        simulators[0].submit_job(job)
        
        # Step all simulators
        for sim in simulators:
            sim.step(dt=1.0)
    
    print("\nFinal cluster state:")
    for i, sim in enumerate(simulators):
        metrics = sim.get_metrics_dict()
        print(f"\n  Cluster {i} ({clusters[i].name}):")
        print(f"    Utilization: {metrics['gpu_utilization']:.1f}%")
        print(f"    Memory Free: {metrics['memory_free_gb']:.1f}GB")
        print(f"    Queue Depth: {metrics['queue_depth']}")
        print(f"    Latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"    Throughput: {metrics['tokens_per_sec']:.0f} tps")


def demo_feature_engineering():
    """Demonstrate feature engineering."""
    print("\n" + "=" * 80)
    print("DEMO 2: Feature Engineering")
    print("=" * 80)
    
    from config import ClusterMetrics
    
    engineer = FeatureEngineer(num_clusters=3, feature_dim=8)
    
    # Create sample request
    request = {
        "tokens": 2048,
        "priority": 4,
        "model_name": "llama-70b",
    }
    
    # Create sample cluster metrics
    metrics_list = [
        ClusterMetrics(
            gpu_utilization=85.0,
            memory_used_gb=120.0,
            memory_free_gb=80.0,
            queue_depth=10,
            avg_latency_ms=200.0,
            tokens_per_sec=1500.0,
            network_bw_utilization=75.0,
            active_jobs=4,
            oom_count=0,
            avg_job_duration_ms=800.0,
        ),
        ClusterMetrics(
            gpu_utilization=45.0,
            memory_used_gb=50.0,
            memory_free_gb=250.0,
            queue_depth=2,
            avg_latency_ms=100.0,
            tokens_per_sec=900.0,
            network_bw_utilization=30.0,
            active_jobs=1,
            oom_count=0,
            avg_job_duration_ms=500.0,
        ),
        ClusterMetrics(
            gpu_utilization=20.0,
            memory_used_gb=30.0,
            memory_free_gb=270.0,
            queue_depth=0,
            avg_latency_ms=80.0,
            tokens_per_sec=0.0,
            network_bw_utilization=10.0,
            active_jobs=0,
            oom_count=0,
            avg_job_duration_ms=300.0,
        ),
    ]
    
    print(f"\nRequest: {request}")
    print("\nCluster Metrics:")
    for i, metrics in enumerate(metrics_list):
        print(f"\n  Cluster {i}:")
        print(f"    Utilization: {metrics.gpu_utilization:.1f}%")
        print(f"    Queue Depth: {metrics.queue_depth}")
        print(f"    Latency: {metrics.avg_latency_ms:.1f}ms")
    
    # Extract features
    req_feat, cluster_feat = engineer.engineer_features(request, metrics_list)
    
    print(f"\n\nEngineered Features:")
    print(f"  Request Features: {req_feat}")
    print(f"  Cluster Context: {cluster_feat}")
    
    combined = engineer.combine_features(req_feat, cluster_feat)
    print(f"  Combined Features (normalized): {combined}")


def demo_router_models():
    """Demonstrate different router models."""
    print("\n" + "=" * 80)
    print("DEMO 3: Router Models")
    print("=" * 80)
    
    feature_dim = 8
    num_clusters = 3
    
    # Create dummy features
    features = torch.randn(feature_dim)
    
    for model_type in ["linear", "mlp", "rl"]:
        print(f"\n{model_type.upper()} Router:")
        
        router = RouterFactory.create(model_type, feature_dim, num_clusters)
        
        print(f"  Architecture: {router.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in router.parameters()):,}")
        
        with torch.no_grad():
            if model_type == "rl":
                policy_scores, value = router(features)
                print(f"  Policy Scores: {policy_scores}")
                print(f"  Value Estimate: {value.item():.3f}")
            else:
                scores = router(features)
                print(f"  Scores: {scores}")
            
            cluster_idx = router.select_cluster(features)
            print(f"  Selected Cluster: {cluster_idx}")


def demo_load_generator():
    """Demonstrate load generator."""
    print("\n" + "=" * 80)
    print("DEMO 4: Load Generator")
    print("=" * 80)
    
    from config import RequestConfig
    gen = LoadGenerator(RequestConfig())
    
    print("\nSample Generated Requests:")
    for i in range(5):
        request = gen.generate_request()
        print(f"\n  Request {i+1}:")
        print(f"    ID: {request['request_id']}")
        print(f"    Tokens: {request['tokens']}")
        print(f"    Model: {request['model_name']}")
        print(f"    Priority: {request['priority']}")
    
    print("\n\nWorkload Patterns:")
    
    # Constant load
    constant = WorkloadPattern.constant_load(5.0, 3)
    print(f"\n  Constant Load (5 req/sec, 3 sec): {len(constant)} requests")
    
    # Bursty load
    bursty = WorkloadPattern.bursty_load(
        base_rate=3.0,
        burst_rate=15.0,
        burst_duration=2,
        num_bursts=2,
        total_duration=10
    )
    print(f"  Bursty Load: {len(bursty)} requests")
    
    # Varying model sizes
    varying = WorkloadPattern.varying_model_sizes(5.0, 5)
    print(f"  Varying Model Sizes: {len(varying)} requests")
    model_dist = {}
    for req in varying:
        model = req["model_name"]
        model_dist[model] = model_dist.get(model, 0) + 1
    print(f"    Distribution: {model_dist}")


def run_all_demos():
    """Run all demos."""
    try:
        demo_cluster_simulator()
        demo_feature_engineering()
        demo_router_models()
        demo_load_generator()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Start training: python main.py")
        print("  2. Monitor in W&B: https://wandb.ai/your-entity/distributed-router")
        print("  3. Run server: python inference.py")
        print("  4. Send requests: curl -X POST http://localhost:8000/route ...")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo and test script")
    parser.add_argument(
        "--demo",
        choices=["all", "simulator", "features", "router", "load"],
        default="all",
        help="Which demo to run",
    )
    
    args = parser.parse_args()
    
    if args.demo == "all":
        run_all_demos()
    elif args.demo == "simulator":
        demo_cluster_simulator()
    elif args.demo == "features":
        demo_feature_engineering()
    elif args.demo == "router":
        demo_router_models()
    elif args.demo == "load":
        demo_load_generator()
