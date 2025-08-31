import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from athena.encoders.input_encoders.action_tokenizer import ActionTokenizer
from utils.device_selector import device_selector

# ---------------------------------------------------------
# 1. A single Mamba block (post-norm, residual)
# ---------------------------------------------------------


class MambaLayer(nn.Module):
    """
    A drop-in replacement for the Transformer decoder layer that
    uses the Mamba SSM mixer (linear-time, no attention).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mixer = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )  # (B, L, C) → (B, L, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C)
        return x + self.dropout(self.mixer(self.ln(x)))


# ---------------------------------------------------------
# 2. Athena-Mamba model
# ---------------------------------------------------------
class AthenaMamba(nn.Module):
    """
    Sequence-to-scalar value head that feeds a {FEN-tokens + action-token}
    sequence through stacked Mamba layers.

    Args:
        dim        – model width
        depth      – number of Mamba layers
        d_state    – hidden state size inside each Mamba (≈ 16-64)
        d_conv     – convolution width inside each Mamba (≈ 4-8)
        expand     – channel expansion factor inside each Mamba feed-forward
        K, M       – identical meaning to your Transformer version
    """

    def __init__(self, cfg):
        assert cfg.architecture.type == "mamba", "Expected Mamba architecture"

        super().__init__()

        # Get configuration parameters
        self.action_tokenizer = ActionTokenizer(cfg)
        self.width = cfg.architecture.width
        self.depth = cfg.architecture.depth
        self.d_state = cfg.architecture.d_state
        self.d_conv = cfg.architecture.d_conv
        self.expand = cfg.architecture.expand
        self.K = cfg.K
        self.M = cfg.M

        self.device = device_selector(label="AthenaMamba")

        # --- embeddings ----------------------------------------------------
        self.token_emb = nn.Embedding(self.action_tokenizer.vocab_size, self.width)
        self.action_emb = nn.Embedding(len(self.action_tokenizer.uci_moves), self.width)

        # Positional information
        # Mamba can work *without* explicit positions, but absolute
        # embeddings still help on short fixed-length inputs.
        self.pos_emb = nn.Embedding(80, self.width)  # 77 + action + CLS buffer

        # --- stack of Mamba layers -----------------------------------------
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    self.width,
                    d_state=self.d_state,
                    d_conv=self.d_conv,
                    expand=self.expand,
                )
                for _ in range(self.depth)
            ]
        )

        # --- projection head -----------------------------------------------
        self.norm = nn.LayerNorm(self.width)
        self.output_bins = self.K + 2 * self.M + 1
        self.head = nn.Linear(self.width, self.output_bins)

    # ----------------------------------------------------------------------
    def forward(self, fen_tokens: torch.LongTensor, action_idx: torch.LongTensor):
        """
        Args
        ----
        fen_tokens : (B, 77) long
        action_idx : (B,)     long   in [0, 1967]

        Returns
        -------
        logits : (B, K + 2*M + 1)
        """
        B, L = fen_tokens.shape
        device = fen_tokens.device

        # Build sequence = [FEN77, action]
        seq = torch.cat(
            [self.token_emb(fen_tokens), self.action_emb(action_idx).unsqueeze(1)],
            dim=1,
        )  # (B, 78, C)

        # Add (absolute) position embedding
        pos_ids = torch.arange(seq.size(1), device=device).unsqueeze(0)
        seq = seq + self.pos_emb(pos_ids)

        # --- stacked Mamba mixing -----------------------------------------
        for layer in self.layers:
            seq = layer(seq)

        cls_out = self.norm(seq[:, 0])  # CLS token
        return self.head(cls_out)

    # ----------------------------------------------------------------------
    # Copy of your helper for completeness (unchanged)
    def encode_win_prob(self, win_prob, mate, K=128, M=20):
        tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)
        if isinstance(mate, str):
            index = -1 if mate == "#" else M + int(round(win_prob * (K - 1)))
        else:
            assert mate != 0
            index = K + 2 * M - min(mate, M) if mate > 0 else M - min(-mate, M)
        tensor[index] = 1.0
        return tensor

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
