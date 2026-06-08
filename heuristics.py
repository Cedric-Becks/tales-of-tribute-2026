import itertools
import os
from typing import List
from typing import Optional
from typing import Dict
from math import tanh
from collections import Counter
from scripts_of_tribute.board import GameState, EnemyPlayer, PatronId
from scripts_of_tribute.enums import CardType

class TierEnum:
    S = 50
    A = 25
    B = 10
    C = 5
    D = 1
    UNKNOWN = 0

CARD_TIERS: Dict[str, int] = {
    "Currency Exchange": TierEnum.S, "Luxury Exports": TierEnum.S, "Oathman": TierEnum.A,
    "Ebony Mine": TierEnum.B, "Hlaalu Councilor": TierEnum.B, "Hlaalu Kinsman": TierEnum.B,
    "House Embassy": TierEnum.B, "House Marketplace": TierEnum.B, "Hireling": TierEnum.C,
    "Hostile Takeover": TierEnum.C, "Kwama Egg Mine": TierEnum.C, "Customs Seizure": TierEnum.D,
    "Goods Shipment": TierEnum.D, "Midnight Raid": TierEnum.S, "Blood Sacrifice": TierEnum.S,
    "Bloody Offering": TierEnum.S, "Bonfire": TierEnum.C, "Briarheart Ritual": TierEnum.C,
    "Clan-Witch": TierEnum.C, "Elder Witch": TierEnum.B, "Hagraven": TierEnum.B,
    "Hagraven Matron": TierEnum.A, "Imperial Plunder": TierEnum.A, "Imperial Spoils": TierEnum.B,
    "Karth Man-Hunter": TierEnum.C, "War Song": TierEnum.D, "Blackfeather Knave": TierEnum.S,
    "Plunder": TierEnum.S, "Toll of Flesh": TierEnum.S, "Toll of Silver": TierEnum.S,
    "Murder of Crows": TierEnum.A, "Pilfer": TierEnum.A, "Squawking Oratory": TierEnum.A,
    "Law of Sovereign Roost": TierEnum.B, "Pool of Shadow": TierEnum.B, "Scratch": TierEnum.B,
    "Blackfeather Brigand": TierEnum.C, "Blackfeather Knight": TierEnum.C, "Peck": TierEnum.D,
    "Conquest": TierEnum.S, "Grand Oratory": TierEnum.S, "Hira's End": TierEnum.S,
    "Hel Shira Herald": TierEnum.A, "March on Hattu": TierEnum.A, "Shehai Summoning": TierEnum.A,
    "Warrior Wave": TierEnum.A, "Ansei Assault": TierEnum.B, "Ansei's Victory": TierEnum.B,
    "Battle Meditation": TierEnum.B, "No Shira Poet": TierEnum.C, "Way of the Sword": TierEnum.D,
    "Prophesy": TierEnum.S, "Scrying Globe": TierEnum.S, "The Dreaming Cave": TierEnum.S,
    "Augur's Counsel": TierEnum.B, "Psijic Relicmaster": TierEnum.A, "Sage Counsel": TierEnum.A,
    "Prescience": TierEnum.B, "Psijic Apprentice": TierEnum.B, "Ceporah's Insight": TierEnum.C,
    "Psijic's Insight": TierEnum.C, "Time Mastery": TierEnum.D, "Mainland Inquiries": TierEnum.D,
    "Rally": TierEnum.S, "Siege Weapon Volley": TierEnum.S, "The Armory": TierEnum.S,
    "Banneret": TierEnum.A, "Knight Commander": TierEnum.A, "Reinforcements": TierEnum.A,
    "Archers' Volley": TierEnum.B, "Legion's Arrival": TierEnum.B, "Shield Bearer": TierEnum.B,
    "Bangkorai Sentries": TierEnum.C, "Knights of Saint Pelin": TierEnum.C, "The Portcullis": TierEnum.D,
    "Fortify": TierEnum.D, "Bag of Tricks": TierEnum.B, "Bewilderment": TierEnum.D,
    "Grand Larceny": TierEnum.A, "Jarring Lullaby": TierEnum.S, "Jeering Shadow": TierEnum.B,
    "Moonlit Illusion": TierEnum.A, "Pounce and Profit": TierEnum.S, "Prowling Shadow": TierEnum.B,
    "Ring's Guile": TierEnum.B, "Shadow's Slumber": TierEnum.A, "Slight of Hand": TierEnum.B,
    "Stubborn Shadow": TierEnum.B, "Swipe": TierEnum.D, "Twilight Revelry": TierEnum.S,
    "Ghostscale Sea Serpent": TierEnum.B, "King Orgnum's Command": TierEnum.C, "Maormer Boarding Party": TierEnum.B,
    "Maormer Cutter": TierEnum.B, "Pyandonean War Fleet": TierEnum.B, "Sea Elf Raid": TierEnum.C,
    "Sea Raider's Glory": TierEnum.C, "Sea Serpent Colossus": TierEnum.B, "Serpentguard Rider": TierEnum.A,
    "Serpentprow Schooner": TierEnum.B, "Snakeskin Freebooter": TierEnum.S, "Storm Shark Wavecaller": TierEnum.B,
    "Summerset Sacking": TierEnum.B, "Ambush": TierEnum.B, "Barterer": TierEnum.C,
    "Black Sacrament": TierEnum.B, "Blackmail": TierEnum.B, "Gold": TierEnum.UNKNOWN,
    "Harvest Season": TierEnum.C, "Imprisonment": TierEnum.C, "Ragpicker": TierEnum.C,
    "Tithe": TierEnum.C, "Writ of Coin": TierEnum.D, "Unknown": TierEnum.UNKNOWN,
    "Alessian Rebel": TierEnum.D, "Ayleid Defector": TierEnum.A, "Ayleid Quartermaster": TierEnum.A,
    "Chainbreaker Captain": TierEnum.A, "Chainbreaker Sergeant": TierEnum.B, "Morihaus, Sacred Bull": TierEnum.S,
    "Morihaus, the Archer": TierEnum.A, "Pelinal Whitestrake": TierEnum.S, "Priestess of the Eight": TierEnum.B,
    "Saint's Wrath": TierEnum.B, "Soldier of the Empire": TierEnum.C, "Whitestrake Ascendant": TierEnum.S
}

def get_card_tier(card_name: str) -> int:
    return CARD_TIERS.get(card_name, TierEnum.UNKNOWN)

class Apriori:
    """Frequent itemset mining for optimal Patron selection."""
    
    def __init__(self):
        self.all_patrons = ["ANSEI", "DUKE_OF_CROWS", "RAJHIN", "PSIJIC", "ORGNUM", "HLAALU", "PELIN", "RED_EAGLE"]
        self.support: Dict[frozenset, int] = {}
        self.rule_length = 4

    def _read_support_from_data(self, historical_data_path: str, support_threshold: int):
        self.support.clear()
        if not os.path.exists(historical_data_path):
            return

        with open(historical_data_path, 'r') as f:
            # Parse lines and ignore empty ones
            historical_data = [line.strip().split(',') for line in f if line.strip()]

        # Convert to sets for much faster subset matching
        historical_sets = [set(data) for data in historical_data]

        for i in range(1, self.rule_length + 1):
            for combo in itertools.combinations(self.all_patrons, i):
                combo_set = set(combo)
                # Count how many historical records contain this combination
                count = sum(1 for data_set in historical_sets if combo_set.issubset(data_set))
                if count >= support_threshold:
                    self.support[frozenset(combo)] = count

    def _get_good_fitting_patron(self, patron_context: List[str], confidence_threshold: float) -> str:
        best_patron = "random"
        supportx = -1
        patron_set = frozenset(patron_context)

        # If no patrons have been picked yet, pick the single most frequent one
        if not patron_set:
            for k, v in self.support.items():
                if len(k) == 1 and v > supportx:
                    supportx = v
                    best_patron = next(iter(k))
            return best_patron

        supportx = self.support.get(patron_set, -1)
        if supportx == -1:
            return best_patron

        max_confidence_found = 0.0

        # Look for rules of size len(context) + 1 where context is a subset
        for k, v in self.support.items():
            if len(k) == len(patron_set) + 1 and patron_set.issubset(k):
                # The 'effect' is the one element in k that isn't in our context
                effect = next(iter(k - patron_set))
                confidence = v / supportx
                
                if confidence > max_confidence_found:
                    max_confidence_found = confidence
                    best_patron = effect

        if max_confidence_found >= confidence_threshold:
            return best_patron
            
        return "random"

    def _id_from_string(self, patron_str: str) -> Optional[PatronId]:
        try:
            return PatronId[patron_str]
        except KeyError:
            return None

    def apriori_best_choice(self, available_patrons: List[PatronId], historical_data_path: str, support_threshold: int, confidence_threshold: float) -> Optional[PatronId]:
        self._read_support_from_data(historical_data_path, support_threshold)
        
        available_strs = [p.name for p in available_patrons]
        
        # 'patrons' represents the context of what has already been chosen/removed from the pool
        context_patrons = [p for p in self.all_patrons if p not in available_strs]
        
        selected_patron_str = self._get_good_fitting_patron(context_patrons, confidence_threshold)
        return self._id_from_string(selected_patron_str)

def max_prestige(gameState: GameState) -> float:
    player = gameState.current_player
    return player.prestige + player.power


def max_patrons(gameState: GameState) -> float:
    player = gameState.current_player.player_id
    patrons = gameState.patron_states.patrons
    return sum([1 for x in patrons if patrons[x] == player])


def _game_phase(me_prestige: int, enemy_prestige: int) -> str:
    total_prestige = me_prestige + enemy_prestige
    if total_prestige < 15:
        return "early"
    elif total_prestige < 30:
        return "mid"
    else:
        return "late"


def _card_score(card, my_patron_counts: Counter, enemy_patron_counts: Counter, phase: str) -> float:
    score = float(card.cost)

    # Combo bonus
    score += my_patron_counts[card.deck] * 1.5

    if enemy_patron_counts[card.deck] >= 2:
        score += 3.0

    if phase == "early" and card.deck == PatronId.TREASURY:
        score += 2.0

    if card.type in (CardType.AGENT, CardType.CONTRACT_AGENT):
        score += {"early": 3.0, "mid": 1.0, "late": -1.0}[phase]
    if card.type in (CardType.CONTRACT_AGENT, CardType.CONTRACT_ACTION):
        score += {"early": -2.0, "mid": 1.0, "late": 3.0}[phase]
    if card.type == CardType.ACTION:
        score += {"early": 2.0, "mid": 1.0, "late": -1.0}[phase]
    if card.type == CardType.CURSE:
        score -= 2.0
    if card.type == CardType.STARTER:
        score -= 1.0

    return score


def _deck_quality(cards, my_patron_counts: Counter, enemy_patron_counts: Counter, phase: str) -> float:
    scored = [_card_score(c, my_patron_counts, enemy_patron_counts, phase) for c in cards]
    return sum(scored) / len(scored) if scored else 0.0


def _patron_score(patrons: dict, my_id, enemy_id, patron_calls: int) -> float:
    # Treasury does not count toward the 4-patron win condition
    win_patrons = {k: v for k, v in patrons.items() if k != PatronId.TREASURY}

    my_win_patrons = sum(1 for v in win_patrons.values() if v == my_id)
    enemy_win_patrons = sum(1 for v in win_patrons.values() if v == enemy_id)

    score = 0.0

    if enemy_win_patrons >= 3:
        score -= 60.0
    elif enemy_win_patrons >= 2:
        score -= 20.0

    if my_win_patrons >= 3:
        score += 60.0
        if patron_calls >= 1:
            score += 30.0
    elif my_win_patrons >= 2 and patron_calls >= 2:
        score += 25.01

    return score


def greedy_heuristic(gameState: GameState) -> float:
    me = gameState.current_player
    enemy = gameState.enemy_player
    patrons = gameState.patron_states.patrons
    my_id = me.player_id
    enemy_id = enemy.player_id

    phase = _game_phase(me.prestige, enemy.prestige)

    # Prestige matters more as the game goes on
    prestige_weight = {"early": 10.0, "mid": 20.0, "late": 30.0}[phase]
    prestige_diff = (me.prestige - enemy.prestige) * prestige_weight

    power_diff = (me.power - enemy.power) * 4.0

    agent_diff = (
        sum(a.currentHP for a in me.agents)
        - sum(a.currentHP for a in enemy.agents)
    ) * 2.0

    all_my_cards = me.hand + me.cooldown_pile + me.played + me.draw_pile + me.known_upcoming_draws
    # In real game states enemy is EnemyPlayer (hand_and_draw combined).
    # In seeded/simulated states IS-MCTS gives full info so enemy is CurrentPlayer.
    if isinstance(enemy, EnemyPlayer):
        all_enemy_cards = enemy.hand_and_draw + enemy.played + enemy.cooldown_pile
    else:
        all_enemy_cards = enemy.hand + enemy.draw_pile + enemy.played + enemy.cooldown_pile + enemy.known_upcoming_draws

    my_patron_counts = Counter(c.deck for c in all_my_cards)
    enemy_patron_counts = Counter(c.deck for c in all_enemy_cards)

    deck_quality = _deck_quality(all_my_cards, my_patron_counts, enemy_patron_counts, phase)
    patron_score = _patron_score(patrons, my_id, enemy_id, me.patron_calls)

    urgency = 0.0
    if me.prestige >= 35:
        urgency += 20.0
    if enemy.prestige >= 35:
        urgency -= 20.0

    return tanh((
        prestige_diff
        + power_diff
        + agent_diff
        + deck_quality * 3.0
        + patron_score
        + urgency
    )/100.)
