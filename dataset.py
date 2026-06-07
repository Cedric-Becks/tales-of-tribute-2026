#This conversion has been done by a llm.
import os
import glob
import pickle
import torch
from torch.utils.data import Dataset

ALL_DECKS   = ['ANSEI', 'DUKE_OF_CROWS', 'HLAALU', 'PELIN', 'PSIJIC', 'RAJHIN', 'SAINT_ALESSIA', 'TREASURY']
ALL_TYPES   = ['ACTION', 'AGENT', 'CONTRACT_ACTION', 'CONTRACT_AGENT', 'CURSE', 'STARTER']
ALL_PATRONS = ALL_DECKS

DECK2IDX = {d: i for i, d in enumerate(ALL_DECKS)}
TYPE2IDX = {t: i for i, t in enumerate(ALL_TYPES)}
PAD_DECK = len(ALL_DECKS)
PAD_TYPE = len(ALL_TYPES)

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

CARD_SCALAR_DIM = 13  # cost, hp_norm, is_agent, taunt + 9 effect buckets
SCALAR_DIM      = 16  # game-level scalars (raw + derived)
PATRON_DIM      = 24  # 8 patrons × 3 (mine / enemy / neutral)


def _parse_effects(effects: list) -> list:
    coin = power = prestige = draw = knockout = toss = tavern = patron_call = utility = 0.0
    for e in effects:
        e = str(e).strip()
        if not e:
            continue
        for part in e.upper().replace(' OR ', ' AND ').split(' AND '):
            tokens = part.strip().split()
            if not tokens:
                continue
            name = tokens[0]
            try:
                val = float(tokens[1]) if len(tokens) > 1 else 1.0
            except ValueError:
                val = 1.0
            if   'GAIN_COIN'     in name: coin        += val
            elif 'GAIN_POWER'    in name: power       += val
            elif 'GAIN_PRESTIGE' in name: prestige    += val
            elif 'DRAW'          in name: draw        += val
            elif 'KNOCKOUT'      in name: knockout    += val
            elif 'TOSS'          in name: toss        += val
            elif 'TAVERN' in name or 'ACQUIRE' in name: tavern += val
            elif 'PATRON_CALL'   in name: patron_call += val
            else:                         utility     += val
    return [coin, power, prestige, draw, knockout, toss, tavern, patron_call, utility]


def _card_scalars(attrs: dict, current_hp: int = None) -> list:
    cost     = float(attrs.get('cost', 0)) / 15.0
    max_hp   = int(attrs.get('hp', -1))
    is_agent = 1.0 if max_hp > 0 else 0.0
    hp_val   = current_hp if current_hp is not None else max_hp
    hp_norm  = float(max(hp_val, 0)) / 10.0
    taunt    = 1.0 if attrs.get('taunt', False) else 0.0
    return [cost, hp_norm, is_agent, taunt] + _parse_effects(attrs.get('effects', []))


def encode_zone(card_ids: list, catalog: dict, max_size: int,
                agent_hps: dict = None):
    """
    Encode a list of card unique_ids into padded tensors.
    agent_hps: optional {unique_id: current_hp} used for agents on the board
               so we use live HP instead of the card's max HP.
    Returns (deck_idxs, type_idxs, card_scalars, mask) each of length max_size.
    """
    deck_idxs, type_idxs, scalars, mask = [], [], [], []
    for uid in card_ids[:max_size]:
        attrs  = catalog.get(uid)
        cur_hp = agent_hps.get(uid) if agent_hps else None
        if attrs:
            deck_idxs.append(DECK2IDX.get(attrs.get('deck', ''), PAD_DECK))
            type_idxs.append(TYPE2IDX.get(attrs.get('type', ''), PAD_TYPE))
            scalars.append(_card_scalars(attrs, cur_hp))
        else:
            deck_idxs.append(PAD_DECK)
            type_idxs.append(PAD_TYPE)
            scalars.append([0.0] * CARD_SCALAR_DIM)
        mask.append(1)

    pad = max_size - len(deck_idxs)
    deck_idxs += [PAD_DECK] * pad
    type_idxs += [PAD_TYPE] * pad
    scalars   += [[0.0] * CARD_SCALAR_DIM] * pad
    mask      += [0] * pad

    return (
        torch.tensor(deck_idxs, dtype=torch.long),
        torch.tensor(type_idxs, dtype=torch.long),
        torch.tensor(scalars,   dtype=torch.float32),
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
        else:                               features += [0.0, 0.0, 0.0]  # not in this game
    return torch.tensor(features, dtype=torch.float32)


def encode_state(state: dict, catalog: dict, my_player_id: str):
    """Encode a single state dict into (zones, scalars, patrons) tensors."""
    me = state['me']
    en = state['enemy']

    my_agent_hps    = {uid: hp for uid, hp in me.get('agents', [])}
    enemy_agent_hps = {uid: hp for uid, hp in en.get('agents', [])}

    zones = {
        'hand':             encode_zone(me.get('hand',     []), catalog, ZONE_MAX['hand']),
        'draw':             encode_zone(me.get('draw',     []), catalog, ZONE_MAX['draw']),
        'cooldown':         encode_zone(me.get('cooldown', []), catalog, ZONE_MAX['cooldown']),
        'played':           encode_zone(me.get('played',   []), catalog, ZONE_MAX['played']),
        'upcoming':         encode_zone(me.get('upcoming', []), catalog, ZONE_MAX['upcoming']),
        'my_agents':        encode_zone(list(my_agent_hps.keys()),    catalog, ZONE_MAX['my_agents'],    my_agent_hps),
        'enemy_cards':      encode_zone(en.get('cards',   []), catalog, ZONE_MAX['enemy_cards']),
        'enemy_agents':     encode_zone(list(enemy_agent_hps.keys()), catalog, ZONE_MAX['enemy_agents'], enemy_agent_hps),
        'tavern_available': encode_zone(state.get('tavern_available', []), catalog, ZONE_MAX['tavern_available']),
        'tavern_deck':      encode_zone(state.get('tavern_deck',      []), catalog, ZONE_MAX['tavern_deck']),
    }
    return zones, encode_scalars(state, my_player_id), encode_patrons(state.get('patrons', {}), my_player_id)


class TalesDataset(Dataset):
    def __init__(self, data_dir: str = 'data'):
        self.catalog: dict = {}
        self.raw: list = []  # (state_dict, outcome, my_player_id)

        for path in glob.glob(os.path.join(data_dir, '*.pkl')):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            self.catalog.update(d.get('card_catalog', {}))
            my_pid  = d.get('my_player_id', 'PLAYER1')
            outcome = float(d['outcome'])
            for s in d['samples']:
                self.raw.append((s['state'], outcome, my_pid))

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int):
        state, outcome, my_pid = self.raw[idx]
        zones, scalars, patrons = encode_state(state, self.catalog, my_pid)
        return zones, scalars, patrons, torch.tensor(outcome, dtype=torch.float32)


def collate_fn(batch):
    zones_batch = {}
    for key in ZONE_KEYS:
        zones_batch[key] = (
            torch.stack([item[0][key][0] for item in batch]),
            torch.stack([item[0][key][1] for item in batch]),
            torch.stack([item[0][key][2] for item in batch]),
            torch.stack([item[0][key][3] for item in batch]),
        )
    return (
        zones_batch,
        torch.stack([item[1] for item in batch]),
        torch.stack([item[2] for item in batch]),
        torch.stack([item[3] for item in batch]),
    )
