"""ML-based router models for cluster selection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class LinearRouter(nn.Module):
    """Simple linear router: score = Wx + b."""
    
    def __init__(self, feature_dim: int, num_clusters: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        
        self.linear = nn.Linear(feature_dim, num_clusters)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Route a request.
        
        Args:
            features: shape (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            scores: shape (batch_size, num_clusters) or (num_clusters,)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        scores = self.linear(features)
        return scores
    
    def select_cluster(self, features: torch.Tensor) -> int:
        """Select best cluster for a single request."""
        scores = self.forward(features)
        if scores.dim() > 1:
            scores = scores.squeeze(0)
        return scores.argmax(dim=-1).item()


class MLPRouter(nn.Module):
    """Multi-layer perceptron router."""
    
    def __init__(self, feature_dim: int, num_clusters: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_clusters),
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Route a request.
        
        Args:
            features: shape (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            scores: shape (batch_size, num_clusters) or (num_clusters,)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        scores = self.net(features)
        return scores
    
    def select_cluster(self, features: torch.Tensor) -> int:
        """Select best cluster for a single request."""
        scores = self.forward(features)
        if scores.dim() > 1:
            scores = scores.squeeze(0)
        return scores.argmax(dim=-1).item()
    
    def get_routing_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities for exploration."""
        scores = self.forward(features)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        return F.softmax(scores, dim=-1)


class RLRouter(nn.Module):
    """Reinforcement Learning-based router using PPO-style training."""
    
    def __init__(self, feature_dim: int, num_clusters: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters),
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize weights
        for net in [self.policy_net, self.value_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy scores and value.
        
        Args:
            features: shape (batch_size, feature_dim) or (feature_dim,)
        
        Returns:
            policy_scores: shape (batch_size, num_clusters) or (num_clusters,)
            value: shape (batch_size, 1) or (1,)
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        policy_scores = self.policy_net(features)
        value = self.value_net(features)
        
        if squeeze_output:
            policy_scores = policy_scores.squeeze(0)
            value = value.squeeze(0)
        
        return policy_scores, value
    
    def get_action_distribution(self, features: torch.Tensor) -> torch.distributions.Categorical:
        """Get action distribution for sampling."""
        policy_scores, _ = self.forward(features)
        if policy_scores.dim() > 1:
            policy_scores = policy_scores.squeeze(0)
        
        probs = F.softmax(policy_scores, dim=-1)
        return torch.distributions.Categorical(probs)
    
    def select_cluster(self, features: torch.Tensor, deterministic: bool = True) -> int:
        """
        Select cluster.
        
        Args:
            features: input features
            deterministic: if True, use argmax; if False, sample from distribution
        """
        policy_scores, _ = self.forward(features)
        if policy_scores.dim() > 1:
            policy_scores = policy_scores.squeeze(0)
        
        if deterministic:
            return policy_scores.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(
                logits=policy_scores
            )
            return dist.sample().item()


class RouterFactory:
    """Factory for creating different router models."""
    
    @staticmethod
    def create(
        model_type: str,
        feature_dim: int,
        num_clusters: int,
        hidden_dim: int = 64
    ) -> nn.Module:
        """Create a router model."""
        if model_type == "linear":
            return LinearRouter(feature_dim, num_clusters)
        elif model_type == "mlp":
            return MLPRouter(feature_dim, num_clusters, hidden_dim)
        elif model_type == "rl":
            return RLRouter(feature_dim, num_clusters, hidden_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
