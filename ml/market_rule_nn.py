"""
Market-Rule Informed Neural Network for Buffett Monitor
Embeds market microstructure rules and Buffett-style constraints as model priors
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRulePrior(nn.Module):
    """
    Encodes market microstructure rules as neural network priors
    """
    
    def __init__(self, input_dim: int, rule_config: Dict):
        super().__init__()
        self.input_dim = input_dim
        self.rule_config = rule_config
        
        # Rule embedding layers
        self.tick_size_rule = nn.Linear(1, 8)
        self.circuit_breaker_rule = nn.Linear(1, 8)
        self.short_sale_rule = nn.Linear(1, 8)
        self.liquidity_rule = nn.Linear(1, 8)
        
        # Buffett constraint layers
        self.roe_constraint = nn.Linear(1, 8)
        self.debt_constraint = nn.Linear(1, 8)
        self.margin_constraint = nn.Linear(1, 8)
        self.earnings_stability = nn.Linear(1, 8)
        
        # Fusion layer
        self.fusion = nn.Linear(8 * 6, input_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, market_features: torch.Tensor, 
                buffett_features: torch.Tensor) -> torch.Tensor:
        """
        Apply market rules and Buffett constraints as priors
        
        Args:
            market_features: Market microstructure features [batch_size, market_dim]
            buffett_features: Buffett-style fundamental features [batch_size, buffett_dim]
            
        Returns:
            Prior-adjusted features [batch_size, input_dim]
        """
        batch_size = market_features.size(0)
        
        # Extract rule features (assuming they're in the input)
        tick_size = market_features[:, 0:1]  # First feature: tick size info
        circuit_breaker = market_features[:, 1:2]  # Second feature: circuit breaker proximity
        short_sale = market_features[:, 2:3]  # Third feature: short sale restrictions
        liquidity = market_features[:, 3:4]  # Fourth feature: liquidity measure
        
        # Extract Buffett features
        roe = buffett_features[:, 0:1]  # Return on Equity
        debt_to_equity = buffett_features[:, 1:2]  # Debt to Equity ratio
        profit_margin = buffett_features[:, 2:3]  # Profit margin
        earnings_stability = buffett_features[:, 3:4]  # Earnings stability measure
        
        # Apply rule transformations
        tick_emb = F.relu(self.tick_size_rule(tick_size))
        circuit_emb = F.relu(self.circuit_breaker_rule(circuit_breaker))
        short_emb = F.relu(self.short_sale_rule(short_sale))
        liquid_emb = F.relu(self.liquidity_rule(liquidity))
        
        # Apply Buffett constraint transformations
        roe_emb = F.relu(self.roe_constraint(roe))
        debt_emb = F.relu(self.debt_constraint(debt_to_equity))
        margin_emb = F.relu(self.margin_constraint(profit_margin))
        stability_emb = F.relu(self.earnings_stability(earnings_stability))
        
        # Concatenate all embeddings
        combined = torch.cat([
            tick_emb, circuit_emb, short_emb, liquid_emb,
            roe_emb, debt_emb, margin_emb, stability_emb
        ], dim=1)
        
        # Generate prior adjustment
        prior_adjustment = torch.tanh(self.fusion(combined))
        prior_adjustment = self.dropout(prior_adjustment)
        
        # Apply to market features (residual connection)
        adjusted_features = market_features + 0.1 * prior_adjustment
        
        return adjusted_features


class MarketRuleNN(nn.Module):
    """
    Market-Rule Informed Neural Network that combines traditional features
    with market microstructure rules and Buffett constraints
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 market_rule_config: Optional[Dict] = None,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Default market rule configuration
        if market_rule_config is None:
            market_rule_config = {
                'tick_size_sensitivity': 1.0,
                'circuit_breaker_threshold': 0.1,
                'short_sale_enabled': True,
                'min_liquidity_threshold': 1000000,
                'min_roe_threshold': 0.15,
                'max_debt_to_equity': 0.5,
                'min_profit_margin': 0.1,
                'earnings_stability_window': 5
            }
        
        self.market_rule_config = market_rule_config
        
        # Market rule prior
        self.market_prior = MarketRulePrior(input_dim // 2, market_rule_config)
        self.buffett_prior = MarketRulePrior(input_dim // 2, market_rule_config)
        
        # Main network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (ranking score)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Feature splitter for market vs Buffett features
        self.feature_split = input_dim // 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with market rule priors
        
        Args:
            x: Input features [batch_size, input_dim]
                First half: market microstructure features
                Second half: Buffett-style fundamental features
                
        Returns:
            Ranking scores [batch_size, 1]
        """
        # Split features
        market_features = x[:, :self.feature_split]
        buffett_features = x[:, self.feature_split:]
        
        # Apply market rule priors
        priored_market = self.market_prior(market_features, buffett_features)
        priored_buffett = self.buffett_prior(buffett_features, market_features)
        
        # Recombine features
        priored_features = torch.cat([priored_market, priored_buffett], dim=1)
        
        # Pass through main network
        output = self.network(priored_features)
        
        return output


def create_market_rule_nn(input_dim: int, 
                         hidden_dims: List[int] = [256, 128, 64],
                         config: Optional[Dict] = None) -> MarketRuleNN:
    """
    Factory function to create a Market-Rule Informed Neural Network
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        config: Market rule and Buffett constraint configuration
        
    Returns:
        Configured MarketRuleNN instance
    """
    model = MarketRuleNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        market_rule_config=config
    )
    
    logger.info(f"Created Market-Rule NN with input_dim={input_dim}, "
                f"hidden_dims={hidden_dims}")
    
    return model


def apply_market_rules_to_signal(base_signal: float,
                                market_features: Dict,
                                buffett_features: Dict,
                                rule_config: Dict) -> float:
    """
    Apply market rules and Buffett constraints to adjust a base signal
    
    Args:
        base_signal: Original signal strength (-1 to 1)
        market_features: Market microstructure data
        buffett_features: Buffett-style fundamental data
        rule_config: Rule configuration dictionary
        
    Returns:
        Rule-adjusted signal strength
    """
    adjusted_signal = base_signal
    
    # Apply market microstructure rules
    tick_size = market_features.get('tick_size', 0.01)
    if tick_size > rule_config.get('max_tick_size', 0.1):
        # Penalize stocks with excessively large tick sizes
        adjusted_signal *= 0.9
    
    circuit_breaker_prox = market_features.get('circuit_breaker_proximity', 1.0)
    if circuit_breaker_prox < rule_config.get('circuit_breaker_warning', 0.2):
        # Reduce position near circuit breakers
        adjusted_signal *= 0.7
    
    short_sale_restricted = market_features.get('short_sale_restricted', False)
    if short_sale_restricted and rule_config.get('respect_short_sale_rules', True):
        # Adjust for short sale restrictions
        adjusted_signal *= 0.8 if base_signal < 0 else 1.0
    
    liquidity_score = market_features.get('liquidity_score', 1.0)
    min_liquidity = rule_config.get('min_liquidity_threshold', 1000000)
    if liquidity_score < min_liquidity:
        # Penalize illiquid stocks
        liquidity_penalty = min(liquidity_score / min_liquidity, 1.0)
        adjusted_signal *= (0.5 + 0.5 * liquidity_penalty)
    
    # Apply Buffett-style constraints
    roe = buffett_features.get('roe', 0.0)
    min_roe = rule_config.get('min_roe_threshold', 0.15)
    if roe < min_roe:
        # Penalize low ROE stocks
        roe_penalty = min(roe / min_roe, 1.0) if min_roe > 0 else 0.5
        adjusted_signal *= (0.3 + 0.7 * roe_penalty)
    
    debt_to_equity = buffett_features.get('debt_to_equity', 0.0)
    max_debt = rule_config.get('max_debt_to_equity', 0.5)
    if debt_to_equity > max_debt:
        # Penalize high debt stocks
        debt_penalty = max(0.0, 1.0 - (debt_to_equity - max_debt) / max_debt)
        adjusted_signal *= (0.4 + 0.6 * debt_penalty)
    
    profit_margin = buffett_features.get('profit_margin', 0.0)
    min_margin = rule_config.get('min_profit_margin', 0.1)
    if profit_margin < min_margin:
        # Penalize low margin stocks
        margin_penalty = min(profit_margin / min_margin, 1.0) if min_margin > 0 else 0.5
        adjusted_signal *= (0.4 + 0.6 * margin_penalty)
    
    earnings_stability = buffett_features.get('earnings_stability', 0.0)
    min_stability = rule_config.get('min_earnings_stability', 0.6)
    if earnings_stability < min_stability:
        # Penalize unstable earnings
        stability_penalty = min(earnings_stability / min_stability, 1.0) if min_stability > 0 else 0.5
        adjusted_signal *= (0.5 + 0.5 * stability_penalty)
    
    # Ensure signal stays in bounds
    adjusted_signal = max(-1.0, min(1.0, adjusted_signal))
    
    return adjusted_signal


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample configuration
    config = {
        'tick_size_sensitivity': 1.0,
        'circuit_breaker_threshold': 0.1,
        'short_sale_enabled': True,
        'min_liquidity_threshold': 1000000,
        'min_roe_threshold': 0.15,
        'max_debt_to_equity': 0.5,
        'min_profit_margin': 0.1,
        'earnings_stability_window': 5,
        'max_tick_size': 0.1,
        'circuit_breaker_warning': 0.2,
        'respect_short_sale_rules': True,
        'min_earnings_stability': 0.6
    }
    
    # Create model
    model = create_market_rule_nn(input_dim=50, config=config)
    
    # Test forward pass
    batch_size = 32
    sample_input = torch.randn(batch_size, 50)
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs: {output.squeeze()[:5]}")
    
    # Test signal adjustment function
    base_signal = 0.8
    market_features = {
        'tick_size': 0.01,
        'circuit_breaker_proximity': 0.8,
        'short_sale_restricted': False,
        'liquidity_score': 5000000
    }
    buffett_features = {
        'roe': 0.20,
        'debt_to_equity': 0.3,
        'profit_margin': 0.15,
        'earnings_stability': 0.8
    }
    
    adjusted = apply_market_rules_to_signal(base_signal, market_features, buffett_features, config)
    print(f"\nSignal adjustment example:")
    print(f"Base signal: {base_signal}")
    print(f"Adjusted signal: {adjusted:.4f}")