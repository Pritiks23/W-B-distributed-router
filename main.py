"""Main training script orchestrating the entire pipeline."""
import argparse
import torch
import os
from pathlib import Path

from config import get_default_clusters, get_router_config
from router import RouterFactory
from trainer import RouterTrainer
from load_generator import WorkloadPattern


def main():
    parser = argparse.ArgumentParser(
        description="Train distributed GPU router with W&B logging"
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp", "rl"],
        default="mlp",
        help="Router model type",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="router_model.pth",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--workload-pattern",
        choices=["constant", "bursty", "varying"],
        default="varying",
        help="Workload pattern to simulate",
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Get configuration
    clusters = get_default_clusters()
    router_config = get_router_config()
    router_config.num_training_steps = args.num_steps
    router_config.batch_size = args.batch_size
    router_config.learning_rate = args.learning_rate
    
    print(f"\nCluster Configuration:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {cluster.name}")
        print(f"    GPUs: {cluster.num_gpus}x {cluster.gpu_memory_gb}GB")
        print(f"    Network: {cluster.network_bandwidth_gbps}Gbps")
    
    print(f"\nRouter Configuration:")
    print(f"  Model Type: {args.model_type}")
    print(f"  Learning Rate: {router_config.learning_rate}")
    print(f"  Batch Size: {router_config.batch_size}")
    print(f"  Training Steps: {router_config.num_training_steps}")
    
    # Create router
    router = RouterFactory.create(
        args.model_type,
        router_config.num_feature_dims,
        router_config.num_clusters,
        router_config.hidden_dim,
    )
    
    print(f"\nRouter Architecture:")
    print(router)
    
    # Create trainer
    trainer = RouterTrainer(
        router,
        router_config,
        clusters,
        device=args.device
    )
    
    # Train
    print(f"\nStarting training with {args.model_type} model...")
    print(f"W&B Project: {router_config.wandb_project}")
    print(f"W&B Entity: {router_config.wandb_entity if router_config.wandb_entity else '(default)'}")
    print()
    
    try:
        trainer.train_full_pipeline(num_steps=args.num_steps)
        
        # Save model
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / args.save_model
        
        checkpoint = {
            "model_type": args.model_type,
            "model_state_dict": router.state_dict(),
            "config": {
                "feature_dim": router_config.num_feature_dims,
                "num_clusters": router_config.num_clusters,
                "hidden_dim": router_config.hidden_dim,
            },
        }
        
        torch.save(checkpoint, model_path)
        print(f"\nModel saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise


if __name__ == "__main__":
    main()
