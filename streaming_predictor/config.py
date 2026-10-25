from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    backbone: str = "lru"
    d_model: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    feature_dim: int = 512
    num_tokens: int = 24
    window_sec: float = 20.0
    ramp_W0: float = 20.0
    ramp_scale: float = 10.0
    ramp_floor: float = 0.30
    content_gate: bool = True
    transformer_heads: int = 4
    transformer_layers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    groupdro: bool = True
    groupdro_eta: float = 0.01
    calib_frac: float = 0.2
    seed: int = 0
    focal_gamma: float = 2.0
    alarm_margin_weight: float = 0.5
    alarm_margin_beta: float = 4.0
    alarm_margin_margin: float = 1.0
    cvar_weight: float = 0.1
    cvar_q: float = 0.1
    alarm_loss_weight: float = 0.5
    alarm_loss_margin: float = 2.0
    hazard_loss_weight: float = 0.3
    hazard_tau: float = 10.0
    hazard_k: float = 0.3
    feat_channel_drop_p: float = 0.2


@dataclass
class AlarmConfig:
    mode: str = "hmm4_early_skip_ssem"
    ssem_blend: float = 0.3


@dataclass
class EvalConfig:
    n_folds: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)
