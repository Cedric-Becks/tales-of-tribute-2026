import os
import glob
import pickle
import torch
from torch.utils.data import Dataset

ALL_PATRONS = ['ANSEI', 'DUKE_OF_CROWS', 'HLAALU', 'PELIN', 'PSIJIC', 'RAJHIN', 'SAINT_ALESSIA', 'TREASURY']

ZONE_KEYS = [
    'hand', 'draw', 'cooldown', 'played', 'upcoming',
    'my_agents', 'enemy_cards', 'enemy_agents',
    'tavern_available', 'tavern_deck',
]
ZONE_MAX = {
    'hand': 12, 'draw': 40, 'cooldown': 40, 'played': 20, 'upcoming': 8,
    'my_agents': 5, 'enemy_cards': 60, 'enemy_agents': 5,
    'tavern_available': 6, 'tavern_deck': 30,
}

SCALAR_DIM = 16  # game-level scalars (raw + derived)
PATRON_DIM = 24  # 8 patrons × 3 (mine / enemy / neutral)


def build_vocab(catalog: dict) -> dict:
    """Build {card_name: index} from the merged card catalog."""
    names = sorted({attrs['name'] for attrs in catalog.values()})
    return {name: i for i, name in enumerate(names)}


def encode_zone(card_ids: list, catalog: dict, vocab: dict, pad_idx: int, max_size: int):
    """
    Encode a list of card unique_ids into (name_idxs, mask), both padded to max_size.
    Unknown cards (not in catalog or vocab) use pad_idx.
    """
    name_idxs, mask = [], []
    for uid in card_ids[:max_size]:
        attrs = catalog.get(uid)
        if attrs:
            name_idxs.append(vocab.get(attrs['name'], pad_idx))
        else:
            name_idxs.append(pad_idx)
        mask.append(1)

    pad = max_size - len(name_idxs)
    name_idxs += [pad_idx] * pad
    mask      += [0] * pad

    return (
        torch.tensor(name_idxs, dtype=torch.long),
        torch.tensor(mask,      dtype=torch.bool),
    )


def encode_scalars(state: dict, my_player_id: str) -> torch.Tensor:
    me      = state['me']
    en      = state['enemy']
    patrons = state.get('patrons', {})

    my_p   = float(me.get('prestige', 0))
    opp_p  = float(en.get('prestige', 0))
    my_pw  = float(me.get('power',    0))
    opp_pw = float(en.get('power',    0))
    my_c   = float(me.get('coins',    0))
    pc     = float(me.get('patron_calls', 0))

    enemy_pid = 'PLAYER2' if my_player_id == 'PLAYER1' else 'PLAYER1'
    my_pats  = float(sum(1 for v in patrons.values() if v == my_player_id))
    opp_pats = float(sum(1 for v in patrons.values() if v == enemy_pid))

    prestige_diff = (my_p  - opp_p)  / 40.0
    power_diff    = (my_pw - opp_pw) / 10.0
    p_to_win      = max(0.0, 40.0 - my_p)  / 40.0
    opp_p_to_win  = max(0.0, 40.0 - opp_p) / 40.0
    patron_lead   = (my_pats - opp_pats) / 4.0
    i_near        = 1.0 if my_pats >= 3 and pc >= 1 else 0.0
    opp_near      = 1.0 if opp_pats >= 3 else 0.0

    total_p     = my_p + opp_p
    phase_early = 1.0 if total_p < 15  else 0.0
    phase_mid   = 1.0 if 15 <= total_p < 45 else 0.0
    phase_late  = 1.0 if total_p >= 45 else 0.0

    return torch.tensor([
        my_p / 40.0, opp_p / 40.0,
        my_pw / 10.0, opp_pw / 10.0,
        my_c / 10.0, pc / 3.0,
        prestige_diff, power_diff,
        p_to_win, opp_p_to_win, patron_lead,
        i_near, opp_near,
        phase_early, phase_mid, phase_late,
    ], dtype=torch.float32)


def encode_patrons(patrons_dict: dict, my_player_id: str) -> torch.Tensor:
    enemy_pid = 'PLAYER2' if my_player_id == 'PLAYER1' else 'PLAYER1'
    features = []
    for patron in ALL_PATRONS:
        owner = patrons_dict.get(patron)
        if   owner == my_player_id:         features += [1.0, 0.0, 0.0]
        elif owner == enemy_pid:            features += [0.0, 1.0, 0.0]
        elif owner == 'NO_PLAYER_SELECTED': features += [0.0, 0.0, 1.0]
        else:                               features += [0.0, 0.0, 0.0]
    return torch.tensor(features, dtype=torch.float32)


def encode_state(state: dict, catalog: dict, vocab: dict, pad_idx: int, my_player_id: str):
    """Encode a single state dict into (zones, scalars, patrons) tensors."""
    me = state['me']
    en = state['enemy']

    my_agent_ids    = [uid for uid, _ in me.get('agents', [])]
    enemy_agent_ids = [uid for uid, _ in en.get('agents', [])]

    zones = {
        'hand':             encode_zone(me.get('hand',     []), catalog, vocab, pad_idx, ZONE_MAX['hand']),
        'draw':             encode_zone(me.get('draw',     []), catalog, vocab, pad_idx, ZONE_MAX['draw']),
        'cooldown':         encode_zone(me.get('cooldown', []), catalog, vocab, pad_idx, ZONE_MAX['cooldown']),
        'played':           encode_zone(me.get('played',   []), catalog, vocab, pad_idx, ZONE_MAX['played']),
        'upcoming':         encode_zone(me.get('upcoming', []), catalog, vocab, pad_idx, ZONE_MAX['upcoming']),
        'my_agents':        encode_zone(my_agent_ids,           catalog, vocab, pad_idx, ZONE_MAX['my_agents']),
        'enemy_cards':      encode_zone(en.get('cards',   []), catalog, vocab, pad_idx, ZONE_MAX['enemy_cards']),
        'enemy_agents':     encode_zone(enemy_agent_ids,        catalog, vocab, pad_idx, ZONE_MAX['enemy_agents']),
        'tavern_available': encode_zone(state.get('tavern_available', []), catalog, vocab, pad_idx, ZONE_MAX['tavern_available']),
        'tavern_deck':      encode_zone(state.get('tavern_deck',      []), catalog, vocab, pad_idx, ZONE_MAX['tavern_deck']),
    }
    return zones, encode_scalars(state, my_player_id), encode_patrons(state.get('patrons', {}), my_player_id)


class TalesDataset(Dataset):
    def __init__(self, data_dir: str = 'data'):
        self.catalog: dict = {}
        raw: list = []

        for path in glob.glob(os.path.join(data_dir, '*.pkl')):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            self.catalog.update(d.get('card_catalog', {}))
            my_pid  = d.get('my_player_id', 'PLAYER1')
            outcome = float(d['outcome'])
            for s in d['samples']:
                raw.append((s['state'], outcome, my_pid))

        self.vocab     = build_vocab(self.catalog)
        self.vocab_size = len(self.vocab)
        self.pad_idx   = self.vocab_size  # one past the end
        self.raw       = raw

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        state, outcome, my_pid = self.raw[idx]
        zones, scalars, patrons = encode_state(
            state, self.catalog, self.vocab, self.pad_idx, my_pid
        )
        return zones, scalars, patrons, torch.tensor(outcome, dtype=torch.float32)


def collate_fn(batch):
    zones_batch = {}
    for key in ZONE_KEYS:
        zones_batch[key] = (
            torch.stack([item[0][key][0] for item in batch]),  # name_idxs
            torch.stack([item[0][key][1] for item in batch]),  # mask
        )
    return (
        zones_batch,
        torch.stack([item[1] for item in batch]),
        torch.stack([item[2] for item in batch]),
        torch.stack([item[3] for item in batch]),
    )
