#!/usr/bin/env python
"""
Interactive setup wizard for Distributed GPU Router with W&B integration.
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_wandb_auth():
    """Check if W&B is authenticated."""
    auth_path = Path.home() / ".wandb"
    return auth_path.exists()


def prompt_wandb_setup():
    """Prompt user to setup W&B."""
    print_header("Weights & Biases Setup")
    
    print("This system logs training to Weights & Biases (free tier available).")
    print("\nOptions:")
    print("1. Login to W&B (recommended)")
    print("2. Skip for now (offline mode)")
    print("3. Exit")
    
    choice = input("\nChoice [1-3]: ").strip()
    
    if choice == "1":
        print("\n📱 Opening W&B login page...")
        print("Visit: https://wandb.ai/site")
        print("\nThen run: wandb login")
        
        # Try to open browser
        try:
            webbrowser.open("https://wandb.ai/site")
        except:
            pass
        
        input("\nPress Enter after you've signed up...")
        
        # Try to login
        try:
            subprocess.run(["wandb", "login"], check=False)
            print("\n✅ W&B authenticated!")
            return True
        except Exception as e:
            print(f"\n⚠️ Could not authenticate: {e}")
            print("Running in offline mode instead.")
            os.environ["WANDB_MODE"] = "offline"
            return False
    
    elif choice == "2":
        print("\n⚠️ Running in offline mode.")
        print("W&B runs will be saved locally and can be synced later.")
        os.environ["WANDB_MODE"] = "offline"
        return True
    
    else:
        print("\nExiting.")
        sys.exit(0)


def show_training_options():
    """Show training options."""
    print_header("Training Configuration")
    
    print("Select router model:")
    print("1. Linear (fastest, interpretable) - 1-2 min")
    print("2. MLP (recommended) - 5-10 min")
    print("3. RL (most advanced) - 15-30 min")
    print("4. Custom")
    
    model_choice = input("\nChoice [1-4]: ").strip()
    
    model_type_map = {"1": "linear", "2": "mlp", "3": "rl"}
    model_type = model_type_map.get(model_choice, "mlp")
    
    if model_choice == "4":
        model_type = input("Model type (linear/mlp/rl): ").strip() or "mlp"
    
    steps_map = {"1": "1000", "2": "2000", "3": "5000"}
    num_steps = steps_map.get(model_choice, "2000")
    
    print(f"\nTraining Configuration:")
    print(f"  Model: {model_type}")
    print(f"  Steps: {num_steps}")
    
    confirm = input("\nProceed? [y/n]: ").strip().lower()
    if confirm != "y":
        return None
    
    return {"model_type": model_type, "num_steps": num_steps}


def run_training(config):
    """Run training."""
    print_header("Starting Training")
    
    cmd = [
        "python", "main.py",
        "--model-type", config["model_type"],
        "--num-steps", config["num_steps"],
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, cwd="/workspaces/W-B-distributed-router")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")


def show_next_steps():
    """Show what to do after training."""
    print_header("Training Complete!")
    
    print("Your router has been trained and saved to: models/router_model.pth\n")
    print("Next steps:\n")
    
    print("1. View Results on W&B:")
    print("   https://wandb.ai/your-entity/distributed-router\n")
    
    print("2. Run the Inference Server:")
    print("   python inference.py\n")
    
    print("3. Send Test Requests:")
    print('   curl -X POST http://localhost:8000/route \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"tokens": 2048, "model_name": "llama-7b", "priority": 3}\'\n')
    
    print("4. View System Metrics:")
    print("   curl http://localhost:8000/metrics\n")
    
    print("5. Explore the Codebase:")
    print("   - trainer.py: Training loop")
    print("   - router.py: ML models")
    print("   - simulator.py: GPU cluster simulation")
    print("   - inference.py: Production API")
    print("   - features.py: Feature engineering\n")


def main():
    """Main setup wizard."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  Distributed GPU Router with Weights & Biases Integration  ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nThis setup wizard will help you:")
    print("  ✓ Configure Weights & Biases logging")
    print("  ✓ Run the demo")
    print("  ✓ Train an ML router")
    print("  ✓ Monitor results")
    
    # Step 1: W&B setup
    if not check_wandb_auth():
        if not prompt_wandb_setup():
            print("\nContinuing in offline mode...")
    else:
        print_header("W&B Authentication")
        print("✅ W&B is already configured!")
    
    # Step 2: Run demo
    print_header("Running Demo")
    print("Testing all components...\n")
    
    try:
        subprocess.run(
            ["python", "demo.py"],
            cwd="/workspaces/W-B-distributed-router",
            check=True
        )
        print("\n✅ Demo completed successfully!")
    except subprocess.CalledProcessError:
        print("\n❌ Demo failed. Please check the error above.")
        sys.exit(1)
    
    # Step 3: Training configuration
    config = show_training_options()
    
    if config is None:
        print("\nSetup cancelled.")
        sys.exit(0)
    
    # Step 4: Run training
    run_training(config)
    
    # Step 5: Show next steps
    show_next_steps()
    
    print("=" * 80)
    print("Setup complete! Happy training! 🚀")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
