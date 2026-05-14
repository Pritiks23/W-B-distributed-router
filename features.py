"""Feature engineering for routing decisions."""
import numpy as np
from typing import List, Dict, Tuple
from config import ClusterMetrics
import torch


class FeatureEngineer:
    """Converts cluster metrics and request features to ML features."""
    
    def __init__(self, num_clusters: int, feature_dim: int = 8):
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        
        # Running statistics for normalization
        # Total features: 4 request + (4 cluster features per cluster)
        total_input_dim = 4 + (4 * num_clusters)
        self.feature_mean = np.zeros(total_input_dim)
        self.feature_std = np.ones(total_input_dim)
        self.feature_count = 0
        self.total_input_dim = total_input_dim
    
    def extract_request_features(self, request: Dict) -> np.ndarray:
        """Extract features from a request."""
        return np.array([
            request.get("tokens", 1024) / 8192.0,  # normalize to 0-1
            request.get("priority", 3) / 5.0,
            1.0 if request.get("model_name", "llama-7b") == "llama-70b" else 0.5,
            0.0,  # placeholder for additional features
        ], dtype=np.float32)
    
    def extract_cluster_features(self, metrics: ClusterMetrics) -> np.ndarray:
        """Extract features from cluster metrics."""
        return np.array([
            metrics.gpu_utilization / 100.0,
            metrics.memory_used_gb / 200.0,  # normalize assuming max ~200GB
            metrics.queue_depth / 50.0,  # normalize assuming queue < 50
            metrics.avg_latency_ms / 1000.0,  # normalize to seconds
        ], dtype=np.float32)
    
    def get_cluster_context(self, all_metrics: List[ClusterMetrics]) -> np.ndarray:
        """Get context vector from all cluster metrics."""
        features = []
        for metrics in all_metrics:
            features.extend(self.extract_cluster_features(metrics))
        
        return np.array(features, dtype=np.float32)
    
    def engineer_features(
        self,
        request: Dict,
        cluster_metrics_list: List[ClusterMetrics]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features for routing decision.
        
        Returns:
            request_features: shape (4,)
            cluster_context: shape (num_clusters * 4,)
        """
        req_feat = self.extract_request_features(request)
        cluster_feat = self.get_cluster_context(cluster_metrics_list)
        
        return req_feat, cluster_feat
    
    def combine_features(
        self,
        request_features: np.ndarray,
        cluster_context: np.ndarray
    ) -> np.ndarray:
        """Combine request and cluster features for the model."""
        # Simple concatenation
        combined = np.concatenate([request_features, cluster_context])
        
        # Ensure correct dimensionality
        if len(combined) < self.total_input_dim:
            # Pad with zeros if needed
            combined = np.pad(
                combined,
                (0, self.total_input_dim - len(combined)),
                'constant'
            )
        else:
            # Truncate if too long
            combined = combined[:self.total_input_dim]
        
        # Normalize
        combined = (combined - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Reduce to feature_dim via mean pooling over clusters
        if len(combined) > self.feature_dim:
            # Average pool: request features + averaged cluster features
            request_part = combined[:4]
            cluster_part = combined[4:]
            cluster_avg = np.mean(cluster_part.reshape(-1, 4), axis=0)
            combined = np.concatenate([request_part, cluster_avg])[:self.feature_dim]
        
        return combined.astype(np.float32)
    
    def update_statistics(self, features: np.ndarray):
        """Update running mean and std for normalization."""
        # Ensure features are the right size
        if len(features) < self.total_input_dim:
            features = np.pad(
                features,
                (0, self.total_input_dim - len(features)),
                'constant'
            )
        else:
            features = features[:self.total_input_dim]
        
        n = self.feature_count
        new_count = n + 1
        
        delta = features - self.feature_mean
        self.feature_mean += delta / new_count
        delta2 = features - self.feature_mean
        self.feature_std = np.sqrt(
            ((self.feature_std ** 2 * n + delta * delta2) / new_count)
        )
        self.feature_count = new_count
    
    def to_torch(self, features: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.from_numpy(features).float()
