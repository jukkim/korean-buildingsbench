"""BB LoadForecastingTransformer — Standalone 구현

BuildingsBench Transformer-with-Gaussian 모델 재현.
external/BuildingsBench 의존성 없이 독립 실행.

Architecture:
  - Encoder-decoder Transformer (PyTorch nn.Transformer)
  - Embeddings: power, lat/lon, building_type, day_of_year/week/hour, positional
  - Gaussian NLL output head: (mean, log_sigma) → F.gaussian_nll_loss
  - Teacher forcing (학습), autoregressive decoding (추론)

Reference:
  external/BuildingsBench/buildings_bench/models/transformers.py
  external/BuildingsBench/buildings_bench/models/base_model.py
"""
import abc
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Base Model
# ============================================================

class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """BB BaseModel 인터페이스"""

    def __init__(self, context_len: int, pred_len: int, continuous_loads: bool):
        super().__init__()
        self.context_len = context_len
        self.pred_len = pred_len
        self.continuous_loads = continuous_loads

    @abc.abstractmethod
    def forward(self, x: Dict) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        raise NotImplementedError


# ============================================================
# Embedding Modules
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (BB 원본)"""

    def __init__(self, emb_size: int, dropout: float, maxlen: int = 500):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pe = torch.zeros((maxlen, emb_size))
        pe[:, 0::2] = torch.sin(pos * den)
        pe[:, 1::2] = torch.cos(pos * den)
        pe = pe.unsqueeze(-2)  # (maxlen, 1, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) → permute → add PE → permute back
        return self.dropout(
            x.permute(1, 0, 2) + self.pe[:x.size(1), :]
        ).permute(1, 0, 2)


class TimeSeriesSinusoidalPeriodicEmbedding(nn.Module):
    """Periodic embedding: [-1, +1] → sin/cos → linear projection"""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        with torch.no_grad():
            x = torch.cat([torch.sin(np.pi * x), torch.cos(np.pi * x)], dim=2)
        return self.linear(x)


# ============================================================
# LoadForecastingTransformer
# ============================================================

class LoadForecastingTransformer(BaseModel):
    """BB Encoder-Decoder Transformer with Gaussian NLL head

    학습: teacher forcing (ground truth를 decoder 입력으로 사용)
    추론: autoregressive decoding (이전 예측을 다음 입력으로)

    Args:
        context_len: context window (default 168 = 1주)
        pred_len: prediction horizon (default 24 = 1일)
        num_encoder_layers: encoder Transformer layers
        num_decoder_layers: decoder Transformer layers
        d_model: model dimension
        nhead: attention heads
        dim_feedforward: FFN dimension
        dropout: dropout rate
        activation: 'gelu' or 'relu'
        continuous_loads: True (continuous values)
        continuous_head: 'gaussian_nll' or 'mse'
        ignore_spatial: ignore lat/lon features
    """

    def __init__(
        self,
        context_len: int = 168,
        pred_len: int = 24,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        activation: str = 'gelu',
        continuous_loads: bool = True,
        continuous_head: str = 'gaussian_nll',
        ignore_spatial: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        decomp_kernel: int = 25,
        **kwargs,  # 설정 파일의 extra keys 무시
    ):
        super().__init__(context_len, pred_len, continuous_loads)

        self.continuous_head = continuous_head
        self.ignore_spatial = ignore_spatial
        self.use_revin = use_revin
        self.use_seasonal_decomp = use_seasonal_decomp
        self.decomp_kernel = decomp_kernel
        self._revin_mean: Optional[torch.Tensor] = None
        self._revin_std: Optional[torch.Tensor] = None
        self._trend: Optional[torch.Tensor] = None
        s = d_model // 256  # 스케일 팩터

        # Transformer core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        # Output head
        out_dim = 1 if continuous_head == 'mse' else 2
        self.logits = nn.Linear(d_model, out_dim)

        # Power embedding (continuous)
        self.power_embedding = nn.Linear(1, 64 * s)

        # Causal mask for decoder
        self.tgt_mask = self.transformer.generate_square_subsequent_mask(pred_len)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Metadata embeddings
        self.building_embedding = nn.Embedding(2, 32 * s)  # residential/commercial
        self.lat_embedding = nn.Linear(1, 32 * s)
        self.lon_embedding = nn.Linear(1, 32 * s)
        if ignore_spatial:
            # Zero embeddings
            self.lat_embedding = nn.Linear(1, 32 * s)
            nn.init.zeros_(self.lat_embedding.weight)
            nn.init.zeros_(self.lat_embedding.bias)
            self.lat_embedding.weight.requires_grad_(False)
            self.lat_embedding.bias.requires_grad_(False)

        self.day_of_year_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s)
        self.day_of_week_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s)
        self.hour_of_day_encoding = TimeSeriesSinusoidalPeriodicEmbedding(32 * s)

    def to(self, device):
        self.tgt_mask = self.tgt_mask.to(device)
        return super().to(device)

    def _moving_avg(self, x: torch.Tensor) -> torch.Tensor:
        """Moving average trend extraction (replicate padding)

        Args:
            x: (batch, seq_len, 1) — load values
        Returns:
            trend: (batch, seq_len, 1)
        """
        k = self.decomp_kernel
        b, t, c = x.shape
        x_t = x.permute(0, 2, 1)            # (batch, 1, seq_len)
        pad = k // 2
        x_padded = F.pad(x_t, (pad, pad), mode='replicate')
        trend = F.avg_pool1d(x_padded, kernel_size=k, stride=1, padding=0)
        return trend.permute(0, 2, 1)       # (batch, seq_len, 1)

    def _embed(self, x: Dict) -> torch.Tensor:
        """입력 텐서들을 임베딩하여 concat"""
        parts = [
            self.lat_embedding(x['latitude']),
            self.lon_embedding(x['longitude']),
            self.building_embedding(x['building_type']).squeeze(2),
            self.day_of_year_encoding(x['day_of_year']),
            self.day_of_week_encoding(x['day_of_week']),
            self.hour_of_day_encoding(x['hour_of_day']),
            self.power_embedding(x['load']).squeeze(2),
        ]
        return torch.cat(parts, dim=2)  # (batch, seq_len, d_model)

    def forward(self, x: Dict) -> torch.Tensor:
        """Teacher forcing forward pass

        Returns:
            (batch, pred_len, 2) for gaussian_nll: [mean, log_sigma] — normalized space if use_revin
            (batch, pred_len, 1) for mse
        """
        if self.use_seasonal_decomp:
            ctx = x['load'][:, :self.context_len, :]        # (batch, 168, 1)
            context_trend = self._moving_avg(ctx)           # (batch, 168, 1)
            trend_last = context_trend[:, -1:, :]           # (batch, 1, 1)
            n_rest = x['load'].shape[1] - self.context_len
            trend_full = torch.cat(
                [context_trend, trend_last.expand(-1, n_rest, -1)], dim=1
            )                                               # (batch, 192, 1)
            self._trend = trend_last                        # save for loss()
            x = dict(x)
            x['load'] = x['load'] - trend_full             # seasonal residual

        if self.use_revin:
            ctx = x['load'][:, :self.context_len, :]   # (batch, 168, 1)
            self._revin_mean = ctx.mean(dim=1, keepdim=True)              # (batch, 1, 1)
            self._revin_std = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
            x = dict(x)
            x['load'] = (x['load'] - self._revin_mean) / self._revin_std

        embed = self._embed(x)

        # Encoder: context window
        src = embed[:, :self.context_len, :]
        # Decoder: shifted target (last context token + target[:-1])
        tgt = embed[:, self.context_len - 1:-1, :]

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        memory = self.transformer.encoder(src)
        out = self.transformer.decoder(tgt, memory, tgt_mask=self.tgt_mask)
        return self.logits(out)

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Gaussian NLL or MSE loss

        Args:
            preds: (batch, pred_len, 2) for gaussian_nll or (batch, pred_len, 1) for mse
            targets: (batch, pred_len, 1) — original Box-Cox space
        """
        if self.use_seasonal_decomp and self._trend is not None:
            # targets → seasonal residual (subtract constant trend extrapolation)
            targets = targets - self._trend.expand(-1, self.pred_len, -1)

        if self.use_revin and self._revin_mean is not None:
            # Normalize targets to same space as preds (which are in RevIN-normalized space)
            targets = (targets - self._revin_mean) / self._revin_std

        if self.continuous_head == 'gaussian_nll':
            mean = preds[:, :, 0].unsqueeze(2)
            var = F.softplus(preds[:, :, 1].unsqueeze(2)) ** 2
            return F.gaussian_nll_loss(mean, targets, var)
        else:
            return F.mse_loss(preds, targets)

    def predict(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy autoregressive prediction"""
        return self._generate(x, greedy=True)

    @torch.no_grad()
    def _generate(self, x: Dict, greedy: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation

        Returns:
            predictions: (batch, pred_len, 1) — point predictions (mean), Box-Cox space
            dist_params: (batch, pred_len, 2) — [mean, sigma], Box-Cox space
        """
        trend_last = None
        trend_future = None

        if self.use_seasonal_decomp:
            ctx = x['load'][:, :self.context_len, :]        # (batch, 168, 1)
            context_trend = self._moving_avg(ctx)           # (batch, 168, 1)
            trend_last = context_trend[:, -1:, :]           # (batch, 1, 1)
            trend_future = trend_last.expand(-1, self.pred_len, -1)  # (batch, 24, 1)
            n_rest = x['load'].shape[1] - self.context_len
            trend_full = torch.cat(
                [context_trend, trend_last.expand(-1, n_rest, -1)], dim=1
            )
            x = dict(x)
            x['load'] = x['load'] - trend_full

        if self.use_revin:
            ctx = x['load'][:, :self.context_len, :]   # (batch, 168, 1)
            inst_mean = ctx.mean(dim=1, keepdim=True)              # (batch, 1, 1)
            inst_std = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
            x = dict(x)
            x['load'] = (x['load'] - inst_mean) / inst_std

        embed = self._embed(x)
        src = embed[:, :self.context_len, :]
        tgt_full = embed[:, self.context_len - 1:-1, :]  # teacher targets

        src_embed = self.positional_encoding(src)
        encoder_output = self.transformer.encoder(src_embed)

        # Start: last context token
        decoder_input = tgt_full[:, 0, :].unsqueeze(1)
        all_preds = []
        all_logits = []

        for k in range(1, self.pred_len + 1):
            dec_embed = self.positional_encoding(decoder_input)
            tgt_mask = self.transformer.generate_square_subsequent_mask(k).to(
                encoder_output.device)
            dec_out = self.transformer.decoder(dec_embed, encoder_output, tgt_mask=tgt_mask)
            output = self.logits(dec_out[:, -1, :])  # (batch, 2) or (batch, 1)
            all_logits.append(output.unsqueeze(1))

            if self.continuous_head == 'gaussian_nll':
                pred = output[:, 0].unsqueeze(1)  # mean only
                all_preds.append(pred)
                output_for_embed = pred
            else:
                all_preds.append(output)
                output_for_embed = output

            # 다음 step: 예측값을 embed하여 decoder input에 추가
            if k < self.pred_len:
                next_input = tgt_full[:, k, :]  # metadata는 ground truth 사용
                embedded_pred = self.power_embedding(output_for_embed)
                # power embedding 부분만 교체
                next_input = torch.cat([
                    next_input[:, :-embedded_pred.shape[-1]],
                    embedded_pred
                ], dim=1)
                decoder_input = torch.cat([
                    decoder_input, next_input.unsqueeze(1)
                ], dim=1)

        predictions = torch.stack(all_preds, 1)  # (batch, pred_len, 1) — normalized space
        logits = torch.cat(all_logits, 1)         # (batch, pred_len, 2)

        if self.continuous_head == 'gaussian_nll':
            means = logits[:, :, 0]               # (batch, pred_len)
            sigma = F.softplus(logits[:, :, 1])   # (batch, pred_len)
            if self.use_revin:
                # Denormalize: mean_actual = mean_norm * std + mean
                #              sigma_actual = sigma_norm * std  (linear scale)
                means = means * inst_std[:, 0, :] + inst_mean[:, 0, :]
                sigma = sigma * inst_std[:, 0, :]
                predictions = predictions * inst_std + inst_mean
            if self.use_seasonal_decomp and trend_last is not None:
                # Add trend back: seasonal → original Box-Cox space
                means = means + trend_last[:, 0, :].expand(-1, self.pred_len)
                predictions = predictions + trend_future
            dist_params = torch.stack([means, sigma], dim=2)
        else:
            if self.use_revin:
                predictions = predictions * inst_std + inst_mean
                logits = logits * inst_std + inst_mean
            if self.use_seasonal_decomp and trend_last is not None:
                predictions = predictions + trend_future
                logits = logits + trend_future
            dist_params = logits

        return predictions, dist_params

    def load_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
        # DDP 'module.' prefix 제거 + BB SOTA key 호환 (pos_embedding → pe)
        new_state = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            k = k.replace('positional_encoding.pos_embedding', 'positional_encoding.pe')
            new_state[k] = v
        self.load_state_dict(new_state)

    def unfreeze_and_get_parameters_for_finetuning(self):
        return self.parameters()


# ============================================================
# Model Factory
# ============================================================

def model_factory(model_args: dict) -> LoadForecastingTransformer:
    """TOML config dict → model instance"""
    return LoadForecastingTransformer(**model_args)
