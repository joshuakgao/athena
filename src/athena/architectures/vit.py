"""Module used to implement ViT architecture."""

import math

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from athena.utils.device_selector import device_selector


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """Init PatchEmbedding."""
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )  # This conv layer will both split into patches and do the embedding

    def forward(self, x):
        """Forward function for PatchEmbedding.

        Input shape: (batch_size, channels, height, width)
        Output shape: (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention mechanism."""

    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        """Initialize the MultiHeadSelfAttention module."""
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert self.head_dim * n_heads == embed_dim, (
            "Embedding dimension needs to be divisible by number of heads"
        )

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # For queries, keys, values
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """Forward function for MultiHeadSelfAttention.

        Input shape: (batch_size, n_patches + 1, embed_dim)
        Output shape: (batch_size, n_patches + 1, embed_dim)
        """
        batch_size, n_tokens, embed_dim = x.shape

        # Generate q, k, v
        qkv = self.qkv(x)  # (batch_size, n_tokens, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # Compute attention output
        out = attn_probs @ v  # (batch_size, n_heads, n_tokens, head_dim)
        out = out.transpose(1, 2)  # (batch_size, n_tokens, n_heads, head_dim)
        out = out.reshape(batch_size, n_tokens, embed_dim)  # Concatenate heads

        # Project to original dimension
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out


class MLP(nn.Module):
    """Simple MLP with GELU activation and dropout."""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        """Initialize the MLP."""
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass for the MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with layer normalization, multi-head attention, and MLP."""

    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        """Initializes a Transformer-style encoder block with multi-head self-attention and a feed-forward MLP.

        Args:
            embed_dim (int, optional): Dimension of input embeddings. Default is 768.
            n_heads (int, optional): Number of attention heads. Default is 12.
            mlp_ratio (float, optional): Expansion factor for the hidden layer in the MLP. Default is 4.0.
            dropout (float, optional): Dropout probability applied after attention and MLP layers. Default is 0.1.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x):
        """Forward pass for the Transformer block."""
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class AthenaViT(nn.Module):
    """Vision Transformer adapted for chess position evaluation.

    Takes 8x8x28 input tensor and outputs win probability distribution.
    """

    def __init__(self, cfg):
        """Initialize the Vision Transformer."""
        assert cfg.architecture.type == "vit", "Expected architecture type to be 'vit'"

        super().__init__()

        # Get config params
        self.depth = cfg.architecture.depth
        self.width = cfg.architecture.width
        self.patch_size = cfg.architecture.patch_size
        self.heads = cfg.architecture.heads
        self.mlp_ratio = cfg.architecture.mlp_ratio
        self.dropout = cfg.architecture.dropout
        self.emb_dropout = cfg.architecture.emb_dropout
        self.K = cfg.K
        self.M = cfg.M
        self.input_channels = cfg.encoder.input_encoder.input_channels
        self.board_size = 8

        self.device = device_selector(cfg.device, label="AthenaViT")
        self.output_bins = (
            self.K + 2 * self.M + 1
        )  # K for win probs, 2*M for mate-for and mate-against, 1 for checkmate

        # Verify patch size divides board size
        assert self.board_size % self.patch_size == 0, "Board size must be divisible by patch size"

        # Patch embedding - we'll use 1x1 patches to preserve all spatial info
        self.patch_embed = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Calculate number of patches
        self.n_patches = (self.board_size // self.patch_size) ** 2

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.width))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.width))
        self.pos_dropout = nn.Dropout(self.emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(self.width, self.heads, self.mlp_ratio, self.dropout)
                for _ in range(self.depth)
            ]
        )

        # Classification head for win probability bins
        self.norm = nn.LayerNorm(self.width)
        self.head = nn.Linear(
            self.width, self.output_bins
        )  # Now matches encode_win_prob dimensions

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        """Forward function for the Vision Transformer.

        Input shape: (batch_size, channels, height, width)
        Output shape: (batch_size, output_bins)
        """
        batch_size = x.shape[0]

        # Patch embedding - output is (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = self.patch_embed(x)

        # Flatten spatial dimensions and transpose to (batch_size, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head - use only class token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        win_prob_logits = self.head(cls_token_final)

        return win_prob_logits

    def encode_action_value(self, fen, move_uci, input_channels=24):
        """Convert a FEN and move into an AlphaZero-style input tensor with move encoding.

        Args:
            fen (str): The FEN string representing the chess position.
            move_uci (str): The UCI string representing the move to be made.
            input_channels (int): The number of input channels for the tensor.

        Returns:
            np.ndarray: A 8x8xN tensor where:
                    - Planes 0-17: Board state encoding (as before)
                    - Plane 18: 'From' square of the move (1 where piece moves from)
                    - Plane 19: 'To' square of the move (1 where piece moves to)
                    - Plane 20: Promote to Queen (entire plane 1 if promotion)
                    - Plane 21: Promote to Rook
                    - Plane 22: Promote to Bishop
                    - Plane 23: Promote to Knight
        """
        board = chess.Board(fen)
        color_to_move = board.turn

        # Initialize tensor with extra planes for move encoding
        board_tensor = np.zeros((8, 8, input_channels), dtype=np.float32)

        # Split the FEN into its components
        parts = fen.split()
        board_part = parts[0]
        color_part = parts[1]
        castling_part = parts[2]
        en_passant_part = parts[3]
        halfmove_part = int(parts[4])
        # fullmove_part = int(parts[5])

        # Piece encoding (planes 0-11)
        piece_to_plane = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,
        }

        # Parse the board
        row = 0
        col = 0
        for c in board_part:
            if c == "/":
                row += 1
                col = 0
            elif c.isdigit():
                col += int(c)
            else:
                plane = piece_to_plane[c]
                board_tensor[row, col, plane] = 1
                col += 1

        # Set castling rights (planes 12-15: K, Q, k, q)
        castling_map = {
            "w": {"K": 12, "Q": 13, "k": 14, "q": 15},
            "b": {"k": 12, "q": 13, "K": 14, "Q": 15},
        }
        for right, plane in castling_map[color_part].items():
            if right in castling_part:
                board_tensor[:, :, plane] = 1

        # 50-move rule (plane 16)
        board_tensor[:, :, 16] = min(halfmove_part / 50.0, 1.0)

        # En passant (plane 17)
        if en_passant_part != "-":
            ep_col = ord(en_passant_part[0]) - ord("a")
            ep_row = 8 - int(en_passant_part[1])
            board_tensor[ep_row, ep_col, 17] = 1

        # Move encoding (planes 18-23)
        move = chess.Move.from_uci(move_uci)

        # Compute coordinates without perspective flip
        from_row = 7 - (move.from_square // 8)
        from_col = move.from_square % 8
        to_row = 7 - (move.to_square // 8)
        to_col = move.to_square % 8

        board_tensor[from_row, from_col, 18] = 1  # From square
        board_tensor[to_row, to_col, 19] = 1  # To square

        # Promotion planes (20-23)
        if move.promotion:
            if move.promotion == chess.QUEEN:
                board_tensor[:, :, 20] = 1
            elif move.promotion == chess.ROOK:
                board_tensor[:, :, 21] = 1
            elif move.promotion == chess.BISHOP:
                board_tensor[:, :, 22] = 1
            elif move.promotion == chess.KNIGHT:
                board_tensor[:, :, 23] = 1

        # Flip the board if black to move (only rows)
        if color_to_move == chess.BLACK:
            board_tensor = np.flip(board_tensor, axis=0).copy()

        return board_tensor

    def encode_win_prob(self, win_prob, mate, K=128, M=20):
        """Encode win probability and mate information into a tensor with K + 2*M + 1 bins.

        We need the extra bin for the move that checkmates the opponent.
        This extra bin isn't shown on the negative side of the tensor, since a move that checkmates yourself is illegal.
        Bin structure:
        [-M1, ..., -M20, win_prob_bins..., M20, ..., M1, M0 (Move that checkmates opponent)]

        Args:
            win_prob (float): Win probability in [0, 1].
            mate (int): Number of plies to mate. Positive = mate for, negative = mate against.
            K (int): Number of win probability bins.
            M (int): Number of mate bins on each side.

        Returns:
            np.ndarray: One-hot encoded tensor of shape (K + 2*M,)
        """
        assert 0.0 <= win_prob <= 1.0

        tensor = np.zeros((K + 2 * M + 1,), dtype=np.float32)

        if isinstance(mate, str):
            assert mate in ("#", "-"), f"Unrecognized mate string: {mate}"
            if mate == "#":
                # assert win_prob == 1.0
                index = -1
            elif mate == "-":
                index = M + int(round(win_prob * (K - 1)))
        else:
            # assert mate != 0
            if mate > 0:
                # assert win_prob == 1.0
                index = K + 2 * M - min(mate, M)
            elif mate < 0:
                # assert win_prob == 0.0
                index = M - min(-mate, M)
            else:
                raise ValueError(f"Invalid mate value: {mate}")

        # assert -1 <= index < len(tensor), f"Index {index} out of bounds"
        tensor[index] = 1.0
        return tensor

    def count_parameters(self):
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
