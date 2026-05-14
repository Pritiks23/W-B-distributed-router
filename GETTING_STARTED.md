# Getting Started with Distributed GPU Router

## 📦 What You Have

A complete, production-ready distributed inference scheduler with:

- **3 ML Router Models**: Linear, MLP, and Reinforcement Learning
- **GPU Cluster Simulator**: Realistic simulation of T4/A100/H100 clusters
- **Feature Engineering**: Intelligent request + cluster context vectorization
- **Weights & Biases Integration**: Real-time monitoring and visualization
- **FastAPI Server**: Production-ready HTTP inference endpoint
- **Load Generator**: Multiple workload patterns (constant, bursty, varying)

**Total**: ~1,400 lines of production-quality code

---

## 🎯 3-Step Getting Started

### Step 1: Authenticate with Weights & Biases (2 minutes)

```bash
# Get free tier: https://wandb.ai/site
# Create account and get API key

# Authenticate in terminal
wandb login

# Paste your API key when prompted
# (Saved locally, never shared)
```

**Alternative**: Run offline mode (local logging only)
```bash
export WANDB_MODE=offline
```

### Step 2: Run Demo (30 seconds)

```bash
cd /workspaces/W-B-distributed-router

# Test all components
python demo.py
```

Expected output: All 4 demos pass ✅

### Step 3: Start Training (5-30 minutes depending on model)

```bash
# Option A: Interactive setup (recommended)
python setup_wizard.py

# Option B: Quick start (MLP model)
python main.py --model-type mlp --num-steps 2000

# Option C: Custom
python main.py --model-type rl --num-steps 5000 --learning-rate 1e-3
```

---

## 🖥️ Project Structure

```
W-B-distributed-router/
├── config.py              # Configuration & dataclasses
├── simulator.py           # GPU cluster simulation engine
├── features.py            # Feature engineering + vectorization
├── router.py              # ML models (Linear/MLP/RL)
├── load_generator.py      # Request generation & workload patterns
├── trainer.py             # Training loop with W&B logging
├── inference.py           # FastAPI production server
├── main.py                # Training orchestration
├── demo.py                # Component testing
├── setup_wizard.py        # Interactive setup guide
├── utils.py               # Helper utilities
├── requirements.txt       # Python dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # Quick reference
└── GETTING_STARTED.md     # This file
```

---

## 📊 What Happens During Training

### Real-Time Monitoring on W&B

Once training starts, your W&B dashboard shows:

#### Key Metrics
- **avg_utilization** - GPU cluster utilization (target: 70-80%)
- **avg_latency_ms** - Average request latency
- **avg_throughput_tps** - Tokens processed per second
- **idle_clusters** - Count of underutilized clusters

#### Per-Cluster Metrics
- Cluster 0 (T4): Budget GPU cluster
- Cluster 1 (A100): High-performance cluster  
- Cluster 2 (H100): Premium cluster

Each shows:
- GPU utilization %
- Queue depth
- Latency
- Memory remaining
- Throughput
- OOM incidents

### Training Progress
- Step counter
- Loss trends
- Model parameter updates

---

## 🚀 After Training Completes

### Option 1: View Results on W&B

Go to: `https://wandb.ai/your-entity/distributed-router`

You'll see:
- Complete training history
- Hyperparameter settings
- Performance curves
- Cluster metrics over time
- Model comparison (if you trained multiple models)

### Option 2: Run Inference Server

```bash
# In terminal 1
python inference.py
# Server runs on http://localhost:8000

# In terminal 2, test it
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

### Option 3: Analyze the Learned Model

```python
import torch
from router import RouterFactory

# Load trained model
checkpoint = torch.load("models/router_model.pth")
router = RouterFactory.create(
    checkpoint["model_type"],
    checkpoint["config"]["feature_dim"],
    checkpoint["config"]["num_clusters"],
)
router.load_state_dict(checkpoint["model_state_dict"])

# Inspect weights (for linear model)
weights = router.linear.weight
print("Router learned weights:")
print(weights)
# Each weight corresponds to a feature's importance
```

---

## 🎓 Training Details

### What the Router Learns

The router learns a routing policy that:

```python
score = model(
    [
        request_tokens,
        request_priority,
        request_model_size,
        gpu_utilization,
        queue_depth,
        latency,
        memory_free,
        network_bandwidth_utilization,
    ]
)
best_cluster = argmax(score)
```

### Reward Function

The training loop optimizes for:

```python
reward = (
    + throughput                    # More work done = good
    - queue_depth                   # Less queueing = good
    - latency                       # Faster = good
    - utilization_variance          # Balanced load = good
    - oom_penalty                   # No crashes = critical
)
```

### Model Types

| Model | Pros | Cons | Training Time |
|-------|------|------|---------------|
| Linear | Interpretable, fast, simple | Less flexible | 1-2 min |
| MLP | Good balance, non-linear | Still simple | 5-10 min |
| RL | Most sophisticated, learns complex policies | Slower, harder to interpret | 15-30 min |

---

## 📈 Interpreting Results

### Good Performance Indicators
- ✅ Utilization 70-80% (balanced)
- ✅ Latency <200ms (responsive)
- ✅ 0-1 idle clusters (efficient routing)
- ✅ Queue depth <5 per cluster
- ✅ No OOMs

### Things to Improve
- ❌ Multiple idle clusters → Router needs to learn better load distribution
- ❌ High latency (>500ms) → Queues building up, routing too aggressive
- ❌ Uneven utilization → Need better clustering or feature engineering
- ❌ OOMs → Router not respecting memory constraints

---

## 🔬 Experiments to Try

### Experiment 1: Compare Model Types
```bash
# Run each and compare on W&B
python main.py --model-type linear --num-steps 1000
python main.py --model-type mlp --num-steps 2000  
python main.py --model-type rl --num-steps 5000
```

Compare metrics on W&B dashboard to see which generalizes best.

### Experiment 2: Different Workloads
```bash
# Test on different traffic patterns
python main.py --workload-pattern constant   # Steady
python main.py --workload-pattern bursty     # Spiky
python main.py --workload-pattern varying    # Mixed models
```

### Experiment 3: Hyperparameter Sweep
```bash
# Try different learning rates
for lr in 1e-4 5e-4 1e-3 5e-3; do
  python main.py --learning-rate $lr
done
```

### Experiment 4: Cost-Aware Routing
Edit `trainer.py` reward function:
```python
# Add cost penalty to prefer cheaper clusters
reward -= (cost_per_gpu_hour[cluster_idx] * 0.1)
```

---

## 🌐 Inference Server API

Once running (`python inference.py`):

### Route a Request
```bash
POST /route
Content-Type: application/json

{
  "tokens": 2048,           # 1-8192
  "model_name": "llama-7b", # llama-7b, llama-13b, llama-70b, mistral-7b
  "priority": 3             # 1-5
}

Returns: routing decision + cluster metrics
```

### Get System Metrics
```bash
GET /metrics

Returns: all cluster utilization, queues, latencies, etc.
```

### Health Check
```bash
GET /health

Returns: server status and requests processed
```

### Get Cluster Info
```bash
GET /clusters

Returns: configuration of all clusters
```

### View Routing History
```bash
GET /routing-history?limit=100

Returns: last N routing decisions
```

### Reset Simulators
```bash
POST /reset-simulators

Clears all jobs and resets cluster state
```

---

## 💾 Model Checkpoints

Trained models are saved to `models/`:

```bash
models/
├── router_model.pth          # Default (MLP)
├── linear_model.pth          # Linear router
├── rl_model.pth              # RL router
└── my_custom_model.pth       # Custom save
```

Load and use:
```python
from inference import InferenceServer

server = InferenceServer(model_path="models/router_model.pth")
# Use server.route_request()
```

---

## 📱 Production Deployment

### Local API
```bash
python inference.py &
curl http://localhost:8000/metrics
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "inference.py"]
```

```bash
docker build -t distributed-router .
docker run -p 8000:8000 distributed-router
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: distributed-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: router
  template:
    metadata:
      labels:
        app: router
    spec:
      containers:
      - name: router
        image: distributed-router:latest
        ports:
        - containerPort: 8000
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb-secret
              key: api-key
```

---

## 🔗 Useful Links

- **W&B Docs**: https://docs.wandb.ai/
- **PyTorch Docs**: https://pytorch.org/docs/
- **FastAPI Guide**: https://fastapi.tiangolo.com/
- **This Repo**: See README.md for full documentation

---

## ❓ Common Issues

### "No API key" Error
```bash
# Option 1: Login
wandb login

# Option 2: Set environment variable
export WANDB_API_KEY="your-key-here"

# Option 3: Offline mode
export WANDB_MODE=offline
```

### Port 8000 in use
```bash
# Use different port
python -m uvicorn inference:app --port 8001

# Or find and kill process
lsof -i :8000
kill -9 <PID>
```

### Training is slow
```bash
# Use faster model and fewer steps
python main.py --model-type linear --num-steps 500

# Or increase batch size
python main.py --batch-size 128
```

### CUDA out of memory
```bash
# Use CPU (still works fine)
python main.py --device cpu
```

---

## 🎯 Next Milestones

1. ✅ **Train Router** - Learn optimal routing policy
2. ✅ **Monitor on W&B** - See real-time metrics
3. ✅ **Run Inference Server** - Production API
4. ✅ **Send Requests** - Test routing in action
5. 🔲 **Optimize Hyperparameters** - Fine-tune performance
6. 🔲 **Deploy to Production** - Container + Kubernetes
7. 🔲 **Integrate with Real Clusters** - Replace simulator

---

## 📝 For Your Resume

**System Design & Implementation:**
> "Built an ML-driven distributed inference router that dynamically routes requests across heterogeneous GPU clusters. Implemented three router models (Linear, MLP, RL) and trained them to optimize utilization, latency, and throughput. Integrated with Weights & Biases for real-time monitoring and created a production-ready FastAPI server."

**Skills Demonstrated:**
- Distributed Systems
- Machine Learning (PyTorch)
- Systems Design
- Feature Engineering
- API Development (FastAPI)
- MLOps/Monitoring (W&B)
- Simulation & Modeling
- Python Engineering

---

**You're all set! Start with: `python setup_wizard.py`** 🚀
