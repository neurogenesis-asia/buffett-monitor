"""
Nexus Agentic Framework for Time Series Forecasting
Combines time series with news/fundamentals via agentic framework for macro/earnings-aware positioning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

# Simple tokenizer for news text
class SimpleNewsTokenizer:
    """Simple word-level tokenizer for news headlines"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_freq = defaultdict(int)
        self.is_fitted = False
    
    def fit_on_texts(self, texts: List[str]):
        """Fit tokenizer on list of texts"""
        # Count word frequencies
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_freq[word] += 1
        
        # Sort by frequency and assign indices
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words):
            if i + 2 >= self.vocab_size:  # Reserve 0 for PAD, 1 for UNK
                break
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word
        
        self.is_fitted = True
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        # Convert to lowercase and split
        text = text.lower()
        # Keep letters, numbers, and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split on whitespace and filter empty
        tokens = [token for token in text.split() if token]
        return tokens
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert texts to sequences of indices"""
        if not self.is_fitted:
            # Fit on the texts if not already fitted
            self.fit_on_texts(texts)
        
        sequences = []
        for text in texts:
            tokens = self._tokenize(text)
            sequence = []
            for token in tokens:
                sequence.append(self.word_to_idx.get(token, self.word_to_idx["<UNK>"]))
            sequences.append(sequence)
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], maxlen: int = 50) -> List[List[int]]:
        """Pad sequences to same length"""
        padded = []
        for seq in sequences:
            if len(seq) >= maxlen:
                padded.append(seq[:maxlen])
            else:
                padded.append(seq + [self.word_to_idx["<PAD>"]] * (maxlen - len(seq)))
        return padded


class EventType(Enum):
    """Types of events that the Nexus framework can detect and respond to"""
    MACRO_ANNOUNCEMENT = "macro_announcement"
    EARNINGS_RELEASE = "earnings_release"
    FEDERAL_RESERVE = "federal_reserve"
    GEOPOLITICAL = "geopolitical"
    SECTOR_ROTATION = "sector_rotation"
    MARKET_REGIME_SHIFT = "market_regime_shift"
    LIQUIDITY_EVENT = "liquidity_event"


@dataclass
class EventSignal:
    """Represents a detected event signal"""
    event_type: EventType
    timestamp: float
    confidence: float  # 0-1
    impact_score: float  # -1 to 1 (negative = bearish, positive = bullish)
    affected_sectors: List[str]
    affected_tickers: List[str]
    metadata: Dict[str, Any]


class NewsEncoder(nn.Module):
    """Encodes news/text data for event detection"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                     dim_feedforward=hidden_dim, dropout=0.1),
            num_layers=4
        )
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(embed_dim, 64)
        
    def forward(self, text_tokens: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode news/text data
        
        Args:
            text_tokens: Tokenized text [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            News features [batch_size, 64]
        """
        embedded = self.embedding(text_tokens)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        if attention_mask is not None:
            # Convert attention mask for transformer (False where we want to attend)
            attention_mask = attention_mask.transpose(0, 1)  # [seq_len, batch_size]
        
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=attention_mask)
        encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Pool sequence dimension
        pooled = self.pooling(encoded.transpose(1, 2)).squeeze(-1)  # [batch_size, embed_dim]
        output = self.output_proj(pooled)  # [batch_size, 64]
        
        return output


class FundamentalEncoder(nn.Module):
    """Encodes fundamental/factor data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64)
        )
        
    def forward(self, fundamentals: torch.Tensor) -> torch.Tensor:
        """
        Encode fundamental data
        
        Args:
            fundamentals: Fundamental features [batch_size, input_dim]
            
        Returns:
            Fundamental features [batch_size, 64]
        """
        return self.encoder(fundamentals)


class TimeSeriesEncoder(nn.Module):
    """Encodes time series/price data"""
    
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1)
        self.output_proj = nn.Linear(hidden_dim, 64)
        
    def forward(self, ts_data: torch.Tensor) -> torch.Tensor:
        """
        Encode time series data
        
        Args:
            ts_data: Time series data [batch_size, seq_len, feature_dim]
            
        Returns:
            Time series features [batch_size, 64]
        """
        # Convert for CNN: [batch_size, feature_dim, seq_len]
        cnn_input = ts_data.transpose(1, 2)
        cnn_features = self.conv_layers(cnn_input)  # [batch_size, hidden_dim, seq_len]
        
        # Convert for LSTM: [batch_size, seq_len, hidden_dim]
        lstm_input = cnn_features.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)  # [batch_size, seq_len, hidden_dim]
        
        # Self-attention
        attn_input = lstm_out.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # [batch_size, hidden_dim]
        output = self.output_proj(pooled)  # [batch_size, 64]
        
        return output


class EventDetector(nn.Module):
    """Detects macro/earnings events from multi-modal data"""
    
    def __init__(self, news_dim: int = 64, fundamental_dim: int = 64, 
                 ts_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(news_dim + fundamental_dim + ts_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Event type classifier
        self.event_classifier = nn.Linear(hidden_dim, len(EventType))
        
        # Confidence and impact predictors
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.impact_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, news_features: torch.Tensor,
                fundamental_features: torch.Tensor,
                ts_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect events from multi-modal data
        
        Args:
            news_features: Encoded news features [batch_size, news_dim]
            fundamental_features: Encoded fundamental features [batch_size, fundamental_dim]
            ts_features: Encoded time series features [batch_size, ts_dim]
            
        Returns:
            event_logits: Raw event type logits [batch_size, num_event_types]
            confidence: Event confidence scores [batch_size, 1]
            impact: Impact scores [-1, 1] [batch_size, 1]
        """
        # Fuse all modalities
        fused = torch.cat([news_features, fundamental_features, ts_features], dim=1)
        fused = self.fusion(fused)
        
        # Predictions
        event_logits = self.event_classifier(fused)
        confidence = self.confidence_predictor(fused)
        impact = self.impact_predictor(fused)
        
        return event_logits, confidence, impact


class NexusAgenticFramework(nn.Module):
    """
    Nexus Agentic Framework for Time Series Forecasting
    Combines traditional TS forecasting with news/fundamentals via event detection
    """
    
    def __init__(self, 
                 ts_seq_len: int = 60,
                 ts_feature_dim: int = 10,
                 news_vocab_size: int = 10000,
                 fundamental_dim: int = 20,
                 hidden_dim: int = 128,
                 forecast_horizon: int = 5):
        super().__init__()
        
        self.ts_seq_len = ts_seq_len
        self.ts_feature_dim = ts_feature_dim
        self.fundamental_dim = fundamental_dim
        self.forecast_horizon = forecast_horizon
        
        # Initialize tokenizer for raw news text
        self.news_tokenizer = SimpleNewsTokenizer(vocab_size=news_vocab_size)
        
        # Encoders for each modality
        self.ts_encoder = TimeSeriesEncoder(ts_seq_len, ts_feature_dim, hidden_dim)
        self.news_encoder = NewsEncoder(vocab_size=news_vocab_size, embed_dim=64, 
                                       hidden_dim=hidden_dim//2)
        self.fundamental_encoder = FundamentalEncoder(fundamental_dim, hidden_dim)
        
        # Event detector
        self.event_detector = EventDetector(news_dim=64, fundamental_dim=64, 
                                          ts_dim=64, hidden_dim=hidden_dim)
        
        # Forecasting components
        self.forecast_lstm = nn.LSTM(64 * 3 + len(EventType) + 2, hidden_dim, 
                                   batch_first=True, dropout=0.1)  # +event_type +confidence+impact
        self.forecast_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )
        
        # Position sizing adjustment based on events
        self.position_adjuster = nn.Sequential(
            nn.Linear(64 * 3 + len(EventType) + 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 2 (1 = no change)
        )

    def _prepare_news_input(self, news_data: Union[List[str], torch.Tensor], maxlen: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare news input for the encoder, handling both raw text and pre-tokenized tensors.
        
        Args:
            news_data: Either list of raw news strings OR pre-tokenized tensor [batch_size, seq_len]
            maxlen: Maximum sequence length for padding/truncation
            
        Returns:
            Tuple of (news_tokens, news_attention_mask) both as torch.Tensor
        \"\"\"
        if isinstance(news_data, list):
            # Handle raw text: tokenize and pad
            sequences = self.news_tokenizer.texts_to_sequences(news_data)
            padded = self.news_tokenizer.pad_sequences(sequences, maxlen=maxlen)
            news_tokens = torch.tensor(padded, dtype=torch.long)
            # Create attention mask: 1 for real tokens, 0 for padding
            news_attention_mask = (news_tokens != self.news_tokenizer.word_to_idx["<PAD>"]).long()
        else:
            # Handle pre-tokenized tensor
            news_tokens = news_data.long()
            # Truncate or pad to maxlen
            if news_tokens.size(1) > maxlen:
                news_tokens = news_tokens[:, :maxlen]
            elif news_tokens.size(1) < maxlen:
                padding = torch.full((news_tokens.size(0), maxlen - news_tokens.size(1)), 
                                   self.news_tokenizer.word_to_idx["<PAD>"], 
                                   dtype=torch.long)
                news_tokens = torch.cat([news_tokens, padding], dim=1)
            # Create attention mask: 1 for non-padding tokens
            news_attention_mask = (news_tokens != self.news_tokenizer.word_to_idx["<PAD>"]).long()
        
        return news_tokens, news_attention_mask
        
    def forward(self, 
                ts_data: torch.Tensor,
                news_tokens: torch.Tensor,
                news_attention_mask: Optional[torch.Tensor],
                fundamentals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Nexus framework
        
        Args:
            ts_data: Time series data [batch_size, ts_seq_len, ts_feature_dim]
            news_tokens: Tokenized news data [batch_size, news_seq_len]
            news_attention_mask: Attention mask for news [batch_size, news_seq_len]
            fundamentals: Fundamental data [batch_size, fundamental_dim]
            
        Returns:
            Dictionary containing:
                - forecast: Price forecasts [batch_size, forecast_horizon]
                - event_logits: Event type predictions [batch_size, num_event_types]
                - confidence: Event confidence [batch_size, 1]
                - impact: Event impact [-1, 1] [batch_size, 1]
                - position_multiplier: Position sizing adjustment [batch_size, 1]
        """
        # Encode each modality
        ts_features = self.ts_encoder(ts_data)  # [batch_size, 64]
        news_features = self.news_encoder(news_tokens, news_attention_mask)  # [batch_size, 64]
        fundamental_features = self.fundamental_encoder(fundamentals)  # [batch_size, 64]
        
        # Detect events
        event_logits, confidence, impact = self.event_detector(
            news_features, fundamental_features, ts_features
        )
        
        # Prepare features for forecasting and position adjustment
        event_probs = F.softmax(event_logits, dim=-1)  # [batch_size, num_event_types]
        combined_features = torch.cat([
            ts_features, news_features, fundamental_features,
            event_probs, confidence, impact
        ], dim=1)  # [batch_size, 64*3 + num_event_types + 2]
        
        # Generate forecast
        # Expand to sequence for LSTM (repeat same features for each time step)
        seq_features = combined_features.unsqueeze(1).repeat(1, self.ts_seq_len, 1)
        lstm_out, _ = self.forecast_lstm(seq_features)  # [batch_size, ts_seq_len, hidden_dim]
        # Use last time step for forecast
        forecast = self.forecast_fc(lstm_out[:, -1, :])  # [batch_size, forecast_horizon]
        
        # Adjust position sizing based on events
        position_multiplier = self.position_adjuster(combined_features)  # [batch_size, 1]
        position_multiplier = position_multiplier * 2.0  # Scale to [0, 2] range
        
        return {
            'forecast': forecast,
            'event_logits': event_logits,
            'confidence': confidence,
            'impact': impact,
            'position_multiplier': position_multiplier
        }
    
    def detect_events(self, 
                     ts_data: torch.Tensor,
                     news_tokens: torch.Tensor,
                     news_attention_mask: Optional[torch.Tensor],
                     fundamentals: torch.Tensor) -> List[EventSignal]:
        """Detect events and return as structured EventSignal objects

        Args:
            ts_data: Time series data [batch_size, ts_seq_len, ts_feature_dim]
            news_tokens: Tokenized news data [batch_size, news_seq_len]
            news_attention_mask: Attention mask for news [batch_size, news_seq_len]
            fundamentals: Fundamental data [batch_size, fundamental_dim]

        Returns:
            List of detected EventSignal objects
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(ts_data, news_tokens, news_attention_mask, fundamentals)
            
            event_logits = outputs['event_logits']
            confidence = outputs['confidence']
            impact = outputs['impact']
            
            # Get predicted event types
            event_probs = F.softmax(event_logits, dim=-1)
            predicted_event_types = torch.argmax(event_probs, dim=-1)
            max_probs = torch.max(event_probs, dim=-1)[0]
            
            events = []
            batch_size = ts_data.size(0)
            
            for i in range(batch_size):
                event_type_idx = predicted_event_types[i].item()
                event_type = list(EventType)[event_type_idx]
                conf = confidence[i].item()
                imp = impact[i].item()
                
                # Only return events with sufficient confidence
                if conf > 0.5 and abs(imp) > 0.1:
                    event_signal = EventSignal(
                        event_type=event_type,
                        timestamp=float(np.datetime64('now').astype('datetime64[s]').astype('float')),
                        confidence=conf,
                        impact_score=imp,
                        affected_sectors=[],  # Would be filled by sector detection logic
                        affected_tickers=[],  # Would be filled by ticker-specific logic
                        metadata={
                            'raw_event_logits': event_logits[i].cpu().numpy().tolist(),
                            'event_probabilities': event_probs[i].cpu().numpy().tolist()
                        }
                    )
                    events.append(event_signal)
            
            return events


def create_nexus_framework(ts_seq_len: int = 60,
                          ts_feature_dim: int = 10,
                          news_vocab_size: int = 10000,
                          fundamental_dim: int = 20,
                          hidden_dim: int = 128,
                          forecast_horizon: int = 5) -> NexusAgenticFramework:
    """
    Factory function to create a Nexus Agentic Framework
    
    Args:
        ts_seq_len: Length of time series sequence
        ts_feature_dim: Number of features in time series data
        news_vocab_size: Vocabulary size for news text
        fundamental_dim: Number of fundamental features
        hidden_dim: Hidden dimension size
        forecast_horizon: Number of steps to forecast ahead
        
    Returns:
        Configured NexusAgenticFramework instance
    """
    framework = NexusAgenticFramework(
        ts_seq_len=ts_seq_len,
        ts_feature_dim=ts_feature_dim,
        news_vocab_size=news_vocab_size,
        fundamental_dim=fundamental_dim,
        hidden_dim=hidden_dim,
        forecast_horizon=forecast_horizon
    )
    
    logger.info(f"Created Nexus Agentic Framework with:")
    logger.info(f"  TS sequence length: {ts_seq_len}")
    logger.info(f"  TS feature dimension: {ts_feature_dim}")
    logger.info(f"  News vocabulary size: {news_vocab_size}")
    logger.info(f"  Fundamental dimension: {fundamental_dim}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Forecast horizon: {forecast_horizon}")
    
    return framework


def apply_nexus_adjustments(base_signal: float,
                           nexus_output: Dict[str, torch.Tensor],
                           adjustment_config: Dict) -> float:
    """
    Apply Nexus framework adjustments to a base trading signal
    
    Args:
        base_signal: Original signal strength (-1 to 1)
        nexus_output: Output from NexusAgenticFramework.forward()
        adjustment_config: Configuration for how to apply adjustments
        
    Returns:
        Nexus-adjusted signal strength
    """
    adjusted_signal = base_signal
    
    # Extract components from Nexus output
    confidence = nexus_output['confidence'].item() if torch.is_tensor(nexus_output['confidence']) else nexus_output['confidence']
    impact = nexus_output['impact'].item() if torch.is_tensor(nexus_output['impact']) else nexus_output['impact']
    position_multiplier = nexus_output['position_multiplier'].item() if torch.is_tensor(nexus_output['position_multiplier']) else nexus_output['position_multiplier']
    event_logits = nexus_output['event_logits']
    
    # Apply event-based signal adjustment
    event_weight = adjustment_config.get('event_weight', 0.3)
    if confidence > adjustment_config.get('min_confidence_threshold', 0.5):
        # Adjust signal based on event impact and confidence
        event_adjustment = impact * confidence * event_weight
        adjusted_signal += event_adjustment
    
    # Apply position sizing adjustment
    position_weight = adjustment_config.get('position_weight', 0.2)
    position_adjustment = (position_multiplier - 1.0) * position_weight  # Convert multiplier to adjustment
    adjusted_signal *= (1.0 + position_adjustment)
    
    # Apply event-type specific adjustments
    event_probs = F.softmax(event_logits, dim=-1)
    event_type_weights = adjustment_config.get('event_type_weights', {
        EventType.MACRO_ANNOUNCEMENT: 0.4,
        EventType.EARNINGS_RELEASE: 0.3,
        EventType.FEDERAL_RESERVE: 0.5,
        EventType.GEOPOLITICAL: 0.35,
        EventType.SECTOR_ROTATION: 0.25,
        EventType.MARKET_REGIME_SHIFT: 0.45,
        EventType.LIQUIDITY_EVENT: 0.3
    })
    
    # Calculate weighted event adjustment
    event_adjustment = 0.0
    for i, event_type in enumerate(EventType):
        prob = event_probs[i].item() if torch.is_tensor(event_probs[i]) else event_probs[i]
        weight = event_type_weights.get(event_type, 0.2)
        event_adjustment += prob * weight * impact * confidence
    
    adjusted_signal += event_adjustment * adjustment_config.get('event_type_weight', 0.25)
    
    # Ensure signal stays in bounds
    max_signal = adjustment_config.get('max_signal', 1.0)
    min_signal = adjustment_config.get('min_signal', -1.0)
    adjusted_signal = max(min_signal, min(max_signal, adjusted_signal))
    
    return adjusted_signal


