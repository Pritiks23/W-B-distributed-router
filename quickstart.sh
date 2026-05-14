#!/usr/bin/env bash

# Quick Start Script - Run this first!

set -e

echo "🚀 Distributed GPU Router - Quick Start"
echo "======================================"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; import wandb; import fastapi" || {
    echo "❌ Dependencies not installed. Run: pip install -r requirements.txt"
    exit 1
}
echo "✅ All dependencies ready"
echo ""

# Check if in correct directory
if [ ! -f "demo.py" ]; then
    echo "❌ Please run from /workspaces/W-B-distributed-router"
    exit 1
fi

echo "Running demo..."
python demo.py
echo ""

echo "✅ Demo passed!"
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Set up Weights & Biases (free tier):"
echo "   wandb login"
echo "   (Get free account at: https://wandb.ai/site)"
echo ""
echo "2️⃣  Start training:"
echo "   python main.py --model-type mlp --num-steps 2000"
echo ""
echo "3️⃣  Monitor on W&B:"
echo "   https://wandb.ai/your-entity/distributed-router"
echo ""
echo "4️⃣  Run inference server (in separate terminal):"
echo "   python inference.py"
echo ""
echo "📚 Full documentation: README.md"
echo "🎯 Quick reference: QUICKSTART.md"
echo "📖 Detailed guide: GETTING_STARTED.md"
echo ""
echo "======================================"
