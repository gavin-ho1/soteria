import torch
import torch.nn.functional as F
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        # Left padding only, padding="same" would inflate lead times
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=0, dilation=dilation)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_t = F.pad(x.transpose(1, 2), (self.causal_pad, 0))
        x = self.conv(x_t).transpose(1, 2)
        x = self.norm(x)
        x = F.relu(x)
        x = self.drop(x)
        return x + residual


class StreamGRU(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int,
                 dropout: float, content_gate: bool):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.content_gate = nn.Linear(d_model, d_model) if content_gate else None
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        # nn.GRU ignores dropout at num_layers=1, apply output dropout explicitly
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.norm(self.proj(x)))
        if self.content_gate is not None:
            x = x * torch.sigmoid(self.content_gate(x))
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = self.out_drop(x)
        return self.head(x).squeeze(-1)


class StreamTCN(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size=3, dilation=d, dropout=dropout)
            for d in [1, 2, 4, 8]
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.drop(self.norm(self.proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class StreamLSTM(nn.Module):
    """Same interface as StreamGRU, without the content gate"""
    def __init__(self, input_dim: int, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
        # nn.LSTM ignores dropout at num_layers=1, apply output dropout explicitly
        self.out_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.norm(self.proj(x)))
        self.lstm.flatten_parameters()
        h, _ = self.lstm(x)
        return self.head(self.out_drop(h)).squeeze(-1)


class LRUCell(nn.Module):
    """Linear Recurrent Unit (Orvieto et al. 2023)"""
    def __init__(self, d_model: int):
        super().__init__()
        d_state = d_model
        self.d_state = d_state
        self.nu_log   = nn.Parameter(torch.log(torch.ones(d_state) * 0.5))
        self.theta_log = nn.Parameter(torch.log(torch.ones(d_state) * 0.1 +
                                                 torch.rand(d_state) * 0.4))
        scale_in  = d_model ** -0.5
        scale_out = d_state ** -0.5
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) * scale_in)
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) * scale_in)
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) * scale_out)
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) * scale_out)
        self.D    = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B, T, _ = x.shape
        r      = torch.exp(-torch.exp(self.nu_log))   # (d_state,)
        theta  = torch.exp(self.theta_log)
        A_re   = r * torch.cos(theta)
        A_im   = r * torch.sin(theta)

        h_re = x.new_zeros(B, self.d_state)
        h_im = x.new_zeros(B, self.d_state)
        outs = []
        for t in range(T):
            u = x[:, t]                               # (B, d_model)
            Bu_re = u @ self.B_re.T                   # (B, d_state)
            Bu_im = u @ self.B_im.T
            new_re = A_re * h_re - A_im * h_im + Bu_re
            new_im = A_re * h_im + A_im * h_re + Bu_im
            h_re, h_im = new_re, new_im
            y = 2.0 * (h_re @ self.C_re.T - h_im @ self.C_im.T) + self.D * u
            outs.append(y)
        return torch.stack(outs, dim=1)               # (B, T, d_model)


class StreamLRU(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float):
        super().__init__()
        self.proj     = nn.Linear(input_dim, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.drop     = nn.Dropout(dropout)
        self.lru      = LRUCell(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.drop(self.norm(self.proj(x)))
        h = self.out_drop(self.out_norm(self.lru(x)))
        return self.head(h).squeeze(-1)


class StreamTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float,
                 n_heads: int, n_layers: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        # Pre-register causal mask for N=24 (re-generated on the fly if N differs)
        N = 24
        mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        self.register_buffer("_causal_mask_24", mask)

    def _causal_mask(self, N: int):
        if N == 24:
            return self._causal_mask_24
        return torch.triu(torch.ones(N, N, device=self._causal_mask_24.device), diagonal=1).bool()

    def forward(self, x):
        x = self.drop(self.norm(self.proj(x)))
        mask = self._causal_mask(x.size(1))
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.out_norm(x)).squeeze(-1)


# Unused by every backbone's forward pass, it is still constructed because its initialization draws from the torch RNG after the backbone's, so removing it would change every later random draw (dropout, shuffling) of the paper runs
_UNUSED_HEAD_DIM = 32


def build_model(cfg):
    m = cfg.model
    if m.backbone == "gru":
        model = StreamGRU(m.feature_dim, m.d_model, m.num_layers, m.dropout,
                          m.content_gate)
    elif m.backbone == "lstm":
        model = StreamLSTM(m.feature_dim, m.d_model, m.num_layers, m.dropout)
    elif m.backbone == "tcn":
        model = StreamTCN(m.feature_dim, m.d_model, m.dropout)
    elif m.backbone == "lru":
        model = StreamLRU(m.feature_dim, m.d_model, m.dropout)
    elif m.backbone == "transformer":
        model = StreamTransformer(m.feature_dim, m.d_model, m.dropout,
                                  m.transformer_heads, m.transformer_layers)
    else:
        raise ValueError(f"unknown backbone: {m.backbone}")
    model.alarm_head = nn.Sequential(
        nn.Linear(m.d_model, _UNUSED_HEAD_DIM),
        nn.ReLU(),
        nn.Linear(_UNUSED_HEAD_DIM, 1),
    )
    return model
