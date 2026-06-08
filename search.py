
from __future__ import annotations
from typing import Dict
from heuristics import get_card_tier
from scripts_of_tribute.enums import PatronId
from functools import total_ordering
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple

from random import randrange, Random
from math import log, sqrt
from abc import ABC, abstractmethod

from scripts_of_tribute.move import BasicMove
from scripts_of_tribute.enums import MoveEnum, PlayerEnum
from scripts_of_tribute.board import GameState

@total_ordering
class Search(ABC):
    children: dict[BasicMove, Optional[Search]]
    visits: int = 0
    fully_expanded: bool = False
    max_score: float = 1
    min_score: float = -1
    score: float = 0
    leaf: bool = False

    game_state: GameState
    
    @abstractmethod
    def __init__(self, gameState: GameState, moves: List[BasicMove]):
        pass

    def __lt__(self, other: object):
        if not hasattr(other, "score"):
            return False
        return self.score < other.score

    def __eq__(self, other: object):
        if not hasattr(other, "score"):
            return False
        return self.score == other.score

    @abstractmethod
    def simulate(self, heuristic: Callable[[GameState], float], root_player_id: PlayerEnum) -> Tuple[float, int]:
        pass

    @abstractmethod
    def expand(self, move: BasicMove) -> None:
        pass

    def ucb_score(self, simulations: int) -> float:
        if self.visits == 0:
            return self.max_score
        return self.score/self.visits + 1.414 * sqrt(log(simulations)/self.visits)
    
    def update(self, score: float):
        self.visits += 1
        self.score += score
    
    def select_winner(self, player_id: PlayerEnum) -> BasicMove:
        if player_id == self.game_state.current_player.player_id:
            score: float = self.min_score
        else:
            score: float = self.max_score
        best_move: BasicMove  = list(self.children.keys())[0]
        for move in self.children:
            if self.children[move] is None:
                self.expand(move)
            # pyrefly: ignore [missing-attribute]
            _score = self.children[move].ucb_score(self.visits)
            if player_id != self.game_state.current_player.player_id:
                _score = -_score
            if _score > score:
                best_move = move
                score = _score

        return best_move

class MCTSNode(Search):
    real_moves: List[BasicMove]

    def __init__(self, gameState: GameState, moves: List[BasicMove]):
        self.game_state = gameState
        if moves is None or len(moves) <= 1:
            self.real_moves = moves
            self.leaf = True
            self.fully_expanded = True
            self.children = {}
        else:
            self.children = {move: None for move in moves}
            self.real_moves = moves

    def expand(self, move: BasicMove) -> None:
        if move.command == MoveEnum.END_TURN:
            self.children[move] = MCTSNode(self.game_state, [move])
        else:
            game_state, moves = self.game_state.apply_move(move)
            self.children[move] = MCTSNode(game_state, moves)
    
    def base_score(self, heuristic: Callable[[GameState], float])-> float:
        return heuristic(self.game_state)

    def simulate(self, heuristic: Callable[[GameState], float], root_player_id: PlayerEnum) -> Tuple[float, int]:
        if self.leaf:
            score: float = heuristic(self.game_state)
            if self.game_state.current_player.player_id != root_player_id:
                score = -score
            return score, 0
        moveCount = 0
        moves: list[BasicMove] = self.real_moves
        move: BasicMove = moves[randrange(len(moves))]
        state: GameState = self.game_state
        while (move.command != MoveEnum.END_TURN):
            moveCount += 1
            state, moves = state.apply_move(move)
            if (len(moves) == 1 and moves[0].command == MoveEnum.END_TURN):
                break
            move = moves[randrange(len(moves))]
        score: float = heuristic(state)
        if state.current_player.player_id != root_player_id:
            score = -score
        return score, moveCount


class LLMNode:
    """
    Represents a single state node in the Monte Carlo Tree Search.
    """
    # Heuristic Weights
    PATRON_FAVOUR = 50
    PATRON_NEUTRAL = 10
    PATRON_UNFAVOUR = -50
    POWER_VALUE = 40
    PRESTIGE_VALUE = 50
    AGENT_ON_BOARD_VALUE = 30
    HP_VALUE = 3
    OPPONENT_AGENTS_PENALTY_VALUE = 40
    POTENTIAL_COMBO_VALUE = 3
    CARD_VALUE = 10
    PENALTY_FOR_HIGH_TIER_IN_TAVERN = 2

    HEURISTIC_MAX = 40000
    HEURISTIC_MIN = -10000
    C_PARAM = sqrt(2)

    def __init__(self, father_game_state: GameState, node_move: Optional[BasicMove], father_orig: Optional['LLMNode'], possible_moves: List[BasicMove] = None, seed: int = 42):
        self.wins = 0.0
        self.visits = 0
        self.move = node_move
        self.father = father_orig
        self.children: List['LLMNode'] = []
        self.seed = seed

        if node_move is not None and node_move.command != MoveEnum.END_TURN:
            new_game_state, new_moves = father_game_state.apply_move(node_move, self.seed)
            self.node_game_state = new_game_state
            self.possible_moves = new_moves
        else:
            self.node_game_state = father_game_state
            self.possible_moves = possible_moves if possible_moves is not None else []

    def create_children(self):
        """Expands the tree by creating a child node for every possible move."""
        for child_move in self.possible_moves:
            self.children.append(LLMNode(self.node_game_state, child_move, self, seed=self.seed))

    def ucb_score(self) -> float:
        """Calculates the Upper Confidence Bound for Trees (UCT) score."""
        if self.visits < 1:
            return float('inf')
        
        if self.father is not None:
            return self.wins + self.C_PARAM * sqrt(log(self.father.visits) / self.visits)
        return self.wins + self.C_PARAM * sqrt(log(self.visits) / self.visits)

    def select_best_child(self) -> 'LLMNode':
        """Selects the child node with the highest UCB score."""
        if not self.children:
            return self
            
        best_child = self.children[0]
        best_score = float('-inf')

        for child in self.children:
            score = child.wins # Uses pure wins for final selection, mirroring C#
            if score >= best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _not_end_turn_moves(self, possible_moves: List[BasicMove]) -> List[BasicMove]:
        return [m for m in possible_moves if m.command != MoveEnum.END_TURN]

    def _draw_next_move(self, possible_moves: List[BasicMove], rng: Random) -> BasicMove:
        """Draws the next random move for the simulation phase."""
        not_end_turn_moves = self._not_end_turn_moves(possible_moves)
        if not_end_turn_moves:
            # Random chance to end turn early to prevent infinite loops in neutral states
            if rng.randint(0, 10000) == 0: 
                # We need to find the actual end turn move object
                end_turns = [m for m in possible_moves if m.command == MoveEnum.END_TURN]
                return end_turns[0] if end_turns else rng.choice(not_end_turn_moves)
            return rng.choice(not_end_turn_moves)
        
        end_turns = [m for m in possible_moves if m.command == MoveEnum.END_TURN]
        if end_turns:
             return end_turns[0]
             
        # Fallback if no moves are theoretically left (should not happen in valid state)
        return possible_moves[0] 

    def simulate(self, rng: Random) -> float:
        """Simulates a random rollout to the end of the turn and evaluates the board."""
        if self.move and self.move.command == MoveEnum.END_TURN:
            return self.heuristic(self.node_game_state)

        next_move = self._draw_next_move(self.possible_moves, rng)
        current_state = self.node_game_state

        while next_move.command != MoveEnum.END_TURN:
            new_seed_game_state, new_possible_moves = current_state.apply_move(next_move, self.seed)
            if not new_possible_moves:
                break
            next_move = self._draw_next_move(new_possible_moves, rng)
            current_state = new_seed_game_state

        return self.heuristic(self.node_game_state)

    def heuristic(self, game_state: GameState) -> float:
        """Evaluates the board state and returns a normalized score."""
        final_value = 0
        enemy_patron_favour = 0
        
        # 1. Patron Evaluation
        for patron_id, player_enum in game_state.patron_states.patrons.items():
            if patron_id == PatronId.TREASURY:
                continue
            if player_enum == game_state.current_player.player_id:
                final_value += self.PATRON_FAVOUR
            elif player_enum == PlayerEnum.NO_PLAYER_SELECTED:
                final_value += self.PATRON_NEUTRAL
            else:
                final_value += self.PATRON_UNFAVOUR
                enemy_patron_favour += 1

        if enemy_patron_favour >= 2:
            final_value -= 100

        # 2. Base Stats Evaluation
        final_value += game_state.current_player.power * self.POWER_VALUE
        final_value += game_state.current_player.prestige * self.PRESTIGE_VALUE

        # 3. Board & Deck Evaluation (if game is still competitive)
        if game_state.current_player.prestige < 30:
            for agent in game_state.current_player.agents:
                tier = get_card_tier(agent.representing_card.name)
                final_value += self.AGENT_ON_BOARD_VALUE * tier + agent.currentHP * self.HP_VALUE

            for agent in game_state.enemy_player.agents:
                tier = get_card_tier(agent.representing_card.name)
                final_value -= (self.AGENT_ON_BOARD_VALUE * tier + agent.currentHP * self.HP_VALUE + self.OPPONENT_AGENTS_PENALTY_VALUE)

            # Analyze Combos
            all_cards = game_state.current_player.hand + game_state.current_player.played + game_state.current_player.cooldown_pile + game_state.current_player.draw_pile
            potential_combo_number: Dict[PatronId, int] = {}
            
            for card in all_cards:
                tier = get_card_tier(card.name)
                final_value += tier * self.CARD_VALUE
                if card.deck != PatronId.TREASURY:
                    potential_combo_number[card.deck] = potential_combo_number.get(card.deck, 0) + 1

            for patron, count in potential_combo_number.items():
                final_value += int(pow(count, self.POTENTIAL_COMBO_VALUE))

            for card in game_state.tavern_available_cards:
                tier = get_card_tier(card.name)
                final_value -= self.PENALTY_FOR_HIGH_TIER_IN_TAVERN * tier

        return self._normalize_heuristic(final_value)

    def _normalize_heuristic(self, value: int) -> float:
        normalized_value = (float(value) - self.HEURISTIC_MIN) / (self.HEURISTIC_MAX - self.HEURISTIC_MIN)
        return max(0.0, normalized_value)

