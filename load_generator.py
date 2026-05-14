"""Load generator for simulating incoming inference requests."""
import numpy as np
from typing import Iterator, Dict, List
from config import RequestConfig
import uuid


class LoadGenerator:
    """Generates realistic inference requests."""
    
    def __init__(self, config: RequestConfig):
        self.config = config
        self.request_count = 0
        self.rng = np.random.RandomState(42)
    
    def generate_request(self) -> Dict:
        """Generate a single request."""
        self.request_count += 1
        
        tokens = self.rng.randint(
            self.config.min_tokens,
            self.config.max_tokens + 1
        )
        
        priority = self.rng.randint(
            self.config.min_priority,
            self.config.max_priority + 1
        )
        
        model = self.rng.choice(self.config.models)
        
        return {
            "request_id": f"req-{self.request_count}",
            "tokens": tokens,
            "priority": priority,
            "model_name": model,
            "timestamp": self.request_count,  # Simplified time
        }
    
    def generate_batch(self, batch_size: int = 32) -> List[Dict]:
        """Generate a batch of requests."""
        return [self.generate_request() for _ in range(batch_size)]
    
    def generate_stream(self, num_requests: int = 1000) -> Iterator[Dict]:
        """Generate requests as a stream."""
        for _ in range(num_requests):
            yield self.generate_request()
    
    def generate_with_varying_load(
        self,
        base_requests_per_sec: float,
        duration_sec: int = 100,
        spike_factor: float = 3.0,
        spike_duration: int = 10
    ) -> Iterator[Dict]:
        """
        Generate requests with time-varying load (includes spike scenarios).
        
        Args:
            base_requests_per_sec: baseline arrival rate
            duration_sec: total duration to simulate
            spike_factor: multiplier during spike period
            spike_duration: how long spike lasts
        """
        time = 0
        total_requests = int(base_requests_per_sec * duration_sec)
        
        spike_start = duration_sec // 3
        spike_end = spike_start + spike_duration
        
        requests_generated = 0
        
        for t in range(int(duration_sec)):
            # Determine current arrival rate
            if spike_start <= t < spike_end:
                current_rate = base_requests_per_sec * spike_factor
            else:
                current_rate = base_requests_per_sec
            
            # Generate requests for this second
            num_requests_this_step = int(
                current_rate + self.rng.exponential(current_rate * 0.1)
            )
            
            for _ in range(num_requests_this_step):
                requests_generated += 1
                if requests_generated > total_requests:
                    return
                yield self.generate_request()


class WorkloadPattern:
    """Predefined workload patterns."""
    
    @staticmethod
    def constant_load(arrival_rate: float, duration: int) -> List[Dict]:
        """Constant arrival rate."""
        config = RequestConfig(arrival_rate_per_sec=arrival_rate)
        gen = LoadGenerator(config)
        requests = []
        for _ in range(int(arrival_rate * duration)):
            requests.append(gen.generate_request())
        return requests
    
    @staticmethod
    def bursty_load(
        base_rate: float,
        burst_rate: float,
        burst_duration: int,
        num_bursts: int,
        total_duration: int
    ) -> List[Dict]:
        """Bursty workload with peaks."""
        config = RequestConfig(arrival_rate_per_sec=base_rate)
        gen = LoadGenerator(config)
        requests = []
        
        interval_between_bursts = total_duration // num_bursts
        
        for t in range(total_duration):
            # Check if we're in a burst period
            burst_index = t // interval_between_bursts
            offset_in_burst = t % interval_between_bursts
            
            in_burst = offset_in_burst < burst_duration
            current_rate = burst_rate if in_burst else base_rate
            
            # Generate requests for this second
            num_requests = int(
                current_rate + np.random.exponential(current_rate * 0.1)
            )
            
            for _ in range(num_requests):
                requests.append(gen.generate_request())
        
        return requests
    
    @staticmethod
    def varying_model_sizes(base_rate: float, duration: int) -> List[Dict]:
        """Requests with varying model sizes (realistic production scenario)."""
        config = RequestConfig(arrival_rate_per_sec=base_rate)
        gen = LoadGenerator(config)
        requests = []
        
        # 70% small models, 20% medium, 10% large
        models = ["llama-7b"] * 70 + ["llama-13b"] * 20 + ["llama-70b"] * 10
        
        for t in range(int(base_rate * duration)):
            request = gen.generate_request()
            request["model_name"] = np.random.choice(models)
            
            # Larger models get fewer tokens to fit in context
            if request["model_name"] == "llama-70b":
                request["tokens"] = np.random.randint(512, 2048)
            elif request["model_name"] == "llama-13b":
                request["tokens"] = np.random.randint(1024, 4096)
            else:
                request["tokens"] = np.random.randint(512, 8192)
            
            requests.append(request)
        
        return requests
