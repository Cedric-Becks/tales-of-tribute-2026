from collections import Counter
from scripts_of_tribute.board import GameState, EnemyPlayer, PatronId
from scripts_of_tribute.enums import CardType


def max_prestige(gameState: GameState) -> int:
    player = gameState.current_player
    return player.prestige + player.power


def max_patrons(gameState: GameState) -> int:
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

    return (
        prestige_diff
        + power_diff
        + agent_diff
        + deck_quality * 3.0
        + patron_score
        + urgency
    )
