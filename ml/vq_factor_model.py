#!/usr/bin/env python3
"""
Vector-Quantized Latent Factors (VQ-VAE) for Stock Ranking
Implements VQ-VAE with financial priors for cross-sectional stock ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import json

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer that maps continuous latent vectors to discrete embeddings
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize embedding vectors
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        """
        Args:
            inputs: Continuous latent vectors of shape (batch, embedding_dim)
            
        Returns:
            quantized: Quantized latent vectors
            loss: VQ loss (commitment + codebook)
            perplexity: Measure of codebook usage
            encoding_indices: Indices of selected embeddings
        """
        # Flatten input except last dimension
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to embeddings
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Get encoding indices (closest embedding)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Get quantized latent vectors
        quantized = self.embedding(encoding_indices).view(input_shape)
        
        # Calculate VQ loss
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        encoding_indices_flat = encoding_indices.view(-1)  # Flatten to 1D
        avg_probs = torch.bincount(encoding_indices_flat, minlength=self.num_embeddings).float()
        avg_probs = avg_probs / avg_probs.sum()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.squeeze(-1)


class VectorQuantizedVAE(nn.Module):
    """
    VQ-VAE model for learning discrete latent factors aligned with financial characteristics
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 latent_dim: int = 64,
                 num_embeddings: int = 512,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Vector quantization layer
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Ranking head: maps latent factors to stock scores
        self.ranking_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Financial priors head: predicts key financial metrics from latents
        self.financial_priors_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # ROE, Debt/Equity, Margin, Growth, Size
        )
        
    def forward(self, x):
        """
        Forward pass through VQ-VAE
        
        Args:
            x: Input features of shape (batch, input_dim)
            
        Returns:
            reconstruction: Reconstructed input features
            ranking_score: VQ-based ranking score for each stock
            vq_loss: Vector quantization loss
            perplexity: Codebook usage measure
            financial_priors: Predicted financial metrics from latents
        """
        # Encode to continuous latent space
        z = self.encoder(x)
        
        # Vector quantization
        z_q, vq_loss, perplexity, encoding_indices = self.vq_layer(z)
        
        # Decode back to input space
        reconstruction = self.decoder(z_q)
        
        # Generate ranking score from quantized latents
        ranking_score = self.ranking_head(z_q).squeeze(-1)
        
        # Predict financial priors (for auxiliary loss)
        financial_priors = self.financial_priors_head(z_q)
        
        return reconstruction, ranking_score, vq_loss, perplexity, financial_priors, encoding_indices
    
    def get_latent_factors(self, x):
        """
        Get the discrete latent factors for interpretability
        
        Args:
            x: Input features
            
        Returns:
            latent_factors: Discrete latent factor indices
            quantized_latents: Quantized continuous latents
        """
        with torch.no_grad():
            z = self.encoder(x)
            z_q, _, _, encoding_indices = self.vq_layer(z)
            return encoding_indices, z_q


class VQFactorTrainer:
    """
    Trainer for VQ-VAE model with financial priors
    """
    
    def __init__(self, 
                 model: VectorQuantizedVAE,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.reconstruction_criterion = nn.MSELoss()
        self.financial_priors_criterion = nn.MSELoss()
        
    def train_step(self, 
                   batch_features: torch.Tensor,
                   financial_targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch_features: Input features batch
            financial_targets: Target financial metrics (optional)
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        reconstruction, ranking_score, vq_loss, perplexity, financial_priors, _ = self.model(batch_features)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_criterion(reconstruction, batch_features)
        
        # Financial priors loss (if targets provided)
        priors_loss = 0.0
        if financial_targets is not None:
            priors_loss = self.financial_priors_criterion(financial_priors, financial_targets)
        
        # Total loss
        total_loss = recon_loss + vq_loss + 0.1 * priors_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'vq_loss': vq_loss.item(),
            'perplexity': perplexity.item(),
            'financial_priors_loss': priors_loss.item() if financial_targets is not None else 0.0
        }
    
    def evaluate(self, 
                 batch_features: torch.Tensor,
                 financial_targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluation step
        
        Args:
            batch_features: Input features batch
            financial_targets: Target financial metrics (optional)
            
        Returns:
            Dictionary of losses
        """
        self.model.eval()
        with torch.no_grad():
            reconstruction, ranking_score, vq_loss, perplexity, financial_priors, _ = self.model(batch_features)
            
            recon_loss = self.reconstruction_criterion(reconstruction, batch_features)
            
            priors_loss = 0.0
            if financial_targets is not None:
                priors_loss = self.financial_priors_criterion(financial_priors, financial_targets)
            
            total_loss = recon_loss + vq_loss + 0.1 * priors_loss
            
            return {
                'total_loss': total_loss.item(),
                'reconstruction_loss': recon_loss.item(),
                'vq_loss': vq_loss.item(),
                'perplexity': perplexity.item(),
                'financial_priors_loss': priors_loss.item() if financial_targets is not None else 0.0
            }


def create_vq_factor_model(input_dim: int, 
                          hidden_dim: int = 128,
                          latent_dim: int = 64,
                          num_embeddings: int = 512) -> VectorQuantizedVAE:
    """
    Factory function to create VQ-VAE model
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        latent_dim: Latent dimension
        num_embeddings: Number of embedding vectors in codebook
        
    Returns:
        Initialized VQ-VAE model
    """
    model = VectorQuantizedVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_embeddings=num_embeddings
    )
    logger.info(f"Created VQ-VAE model: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                f"latent_dim={latent_dim}, num_embeddings={num_embeddings}")
    return model


if __name__ == "__main__":
    # Test the VQ-VAE model
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    batch_size = 32
    input_dim = 50  # Example feature dimension
    
    # Create model
    model = create_vq_factor_model(input_dim=input_dim)
    
    # Create sample input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    reconstruction, ranking_score, vq_loss, perplexity, financial_priors, encoding_indices = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Ranking score shape: {ranking_score.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.4f}")
    print(f"Financial priors shape: {financial_priors.shape}")
    print(f"Encoding indices shape: {encoding_indices.shape}")
    print(f"Unique latents used: {len(torch.unique(encoding_indices))}/{model.num_embeddings}")