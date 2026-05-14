# Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Setup W&B (2 minutes)

```bash
# Create free account at: https://wandb.ai/site
# Get your API key from https://wandb.ai/authorize

# Login to W&B
wandb login

# Paste your API key when prompted
# (Your key will be saved locally)
```

### Step 2: Run Demo (1 minute)

```bash
# Test that everything works
python demo.py

# You should see all 4 demos complete successfully
```

### Step 3: Start Training (2 minutes)

```bash
# Train with MLP router (recommended balance of performance)
python main.py --model-type mlp --num-steps 2000

# Open W&B dashboard to watch training in real-time
# https://wandb.ai/your-entity/distributed-router
```

---

## 📊 What You'll See in W&B

Once training starts, open your W&B project dashboard to see:

### Real-Time Metrics
- **avg_utilization** - Average GPU utilization across clusters
- **avg_latency_ms** - Average request latency
- **avg_throughput_tps** - Tokens processed per second
- **idle_clusters** - Number of underutilized clusters

### Per-Cluster Details
- `cluster_0_utilization` - T4 cluster metrics
- `cluster_1_utilization` - A100 cluster metrics
- `cluster_2_utilization` - H100 cluster metrics
- Memory, queue depth, and throughput for each

### Training Progress
- Step count
- Model parameter updates
- Loss trends

---

## 🔧 Training Options

### Model Types
```bash
# Linear: Fast, interpretable baseline
python main.py --model-type linear --num-steps 1000

# MLP: Recommended, good balance
python main.py --model-type mlp --num-steps 2000

# RL: Most sophisticated, best performance
python main.py --model-type rl --num-steps 5000
```

### Hyperparameters
```bash
# Custom learning rate
python main.py --learning-rate 5e-4

# Larger batches
python main.py --batch-size 64

# Longer training
python main.py --num-steps 10000

# Save model with custom name
python main.py --save-model my_custom_router.pth
```

### Workload Patterns
```bash
# Different workload scenarios
python main.py --workload-pattern constant   # Steady load
python main.py --workload-pattern bursty     # Spiky load
python main.py --workload-pattern varying    # Mixed model sizes
```

---

## 📡 Run Inference Server

In a separate terminal:

```bash
# Start the API server
python inference.py

# In another terminal, test it
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "tokens": 2048,
    "model_name": "llama-7b",
    "priority": 3
  }'
```

---

## 🌐 API Endpoints

Once `inference.py` is running:

### Route a Request
```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "tokens": 2048,
    "model_name": "llama-7b",
    "priority": 3
  }'
```

Response:
```json
{
  "cluster_name": "cluster-b-a100",
  "cluster_index": 1,
  "utilization_percent": 68.5,
  "queue_depth": 4,
  "estimated_latency_ms": 245.3,
  "status": "routed"
}
```

### Get System Metrics
```bash
curl http://localhost:8000/metrics
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Cluster Info
```bash
curl http://localhost:8000/clusters
```

---

## 📚 Understanding Results

### Good Performance
- **Utilization**: 70-80% (balanced load, not overloaded)
- **Latency**: <200ms (responsive)
- **Idle Clusters**: 0-1 (efficient routing)
- **Queue Depth**: <5 per cluster (no bottleneck)

### What to Optimize
If you see:
- **High Idle Clusters** (>1): Router isn't spreading load well
- **High Latency** (>500ms): Queues are building up
- **Uneven Utilization**: Try longer training or better features

---

## 🎯 Next Steps

1. **Train a Router**: `python main.py`
2. **Monitor Training**: Open W&B dashboard
3. **Run Server**: `python inference.py`
4. **Send Requests**: Use curl or write a client script
5. **Experiment**: Try different model types and workload patterns
6. **Deploy**: Container the inference.py for production

---

## ❓ Troubleshooting

### "No W&B API key"
```bash
# Option 1: Set environment variable
export WANDB_API_KEY="your-key-here"
python main.py

# Option 2: Login first
wandb login
python main.py

# Option 3: Offline mode
export WANDB_MODE=offline
python main.py
```

### "Port 8000 already in use"
```bash
# Change the port in inference.py or use:
python -m uvicorn inference:app --port 8001
```

### Training is slow
- Use `--model-type linear` for faster training
- Reduce `--num-steps`
- Increase `--batch-size`

---

## 📖 Learn More

- [W&B Documentation](https://docs.wandb.ai/)
- [PyTorch Basics](https://pytorch.org/tutorials/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Distributed Systems](https://en.wikipedia.org/wiki/Distributed_computing)
