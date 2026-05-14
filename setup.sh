#!/bin/bash
# Setup and Training Script

set -e

echo "=========================================="
echo "Distributed GPU Router - Setup & Training"
echo "=========================================="
echo ""

# Check if W&B API key is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "⚠️  WANDB_API_KEY not set. Running in offline mode."
    echo ""
    echo "To connect to Weights & Biases:"
    echo "  1. Get free tier at: https://wandb.ai/site"
    echo "  2. Login with: wandb login"
    echo "  3. Copy your API key when prompted"
    echo ""
    export WANDB_MODE=offline
else
    echo "✅ W&B API key detected. Running in online mode."
    echo ""
fi

echo "Running demo to verify setup..."
python demo.py
echo ""

echo "=========================================="
echo "Demo completed successfully!"
echo "=========================================="
echo ""
echo "Ready to train. Options:"
echo ""
echo "Quick start (recommended):"
echo "  python main.py --model-type mlp --num-steps 2000"
echo ""
echo "Other options:"
echo "  python main.py --model-type linear --num-steps 1000  # Simplest"
echo "  python main.py --model-type rl --num-steps 5000      # Most advanced"
echo ""
echo "View training on W&B:"
echo "  https://wandb.ai/your-entity/distributed-router"
echo ""
