import torch
import torch.nn as nn
from dataset import SCALAR_DIM, PATRON_DIM, ZONE_KEYS

CARD_DIM  = 32
NUM_ZONES = len(ZONE_KEYS)                                   # 10
INPUT_DIM = NUM_ZONES * CARD_DIM + SCALAR_DIM + PATRON_DIM  # 360


class CardEmbedder(nn.Module):
    """
    One learned D-dimensional vector per card name (the Dominion AI approach).
    The network discovers what the dimensions mean from game outcomes.
    padding_idx keeps the dummy embedding frozen at zero so it never
    contributes to the zone sum.
    """
    def __init__(self, vocab_size: int, card_dim: int = CARD_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, card_dim, padding_idx=vocab_size)

    def forward(self, name_idxs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # name_idxs : (B, max_cards)
        # mask      : (B, max_cards)  bool — True for real cards
        x = self.embed(name_idxs)              # (B, max_cards, card_dim)
        x = x * mask.unsqueeze(-1).float()    # zero out padding positions
        return x.sum(dim=1)                   # (B, card_dim)


class TalesValueNet(nn.Module):
    """
    Value network: game state → win probability in [0, 1].

    One shared CardEmbedder pools each zone into a CARD_DIM vector.
    All zone vectors + game scalars + patron features are concatenated
    and passed through an MLP.
    """
    def __init__(self, vocab_size: int, card_dim: int = CARD_DIM, dropout: float = 0.3):
        super().__init__()
        self.embedder = CardEmbedder(vocab_size, card_dim)
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, zones: dict, scalars: torch.Tensor, patrons: torch.Tensor) -> torch.Tensor:
        zone_vecs = []
        for key in ZONE_KEYS:
            name_idxs, mask = zones[key]
            zone_vecs.append(self.embedder(name_idxs, mask))
        x = torch.cat(zone_vecs + [scalars, patrons], dim=-1)  # (B, INPUT_DIM)
        return self.mlp(x).squeeze(-1)                          # (B,)