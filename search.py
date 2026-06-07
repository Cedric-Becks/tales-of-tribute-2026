
from __future__ import annotations
from functools import total_ordering
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple

from random import randrange
from math import log, sqrt, inf
from abc import ABC, abstractmethod

from scripts_of_tribute.move import BasicMove
from scripts_of_tribute.enums import MoveEnum
from scripts_of_tribute.board import GameState

@total_ordering
class Search(ABC):
    children: dict[BasicMove, Optional[Search]]
    visits: int = 0
    fullyExpanded: bool = False
    maxScore: float = inf
    minScore: float = -inf
    score: float = 0
    leaf: bool = False

    gameState: GameState
    
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
    def simulate(self, heuristic: Callable[[GameState], float]) -> Tuple[float, int]:
        pass

    @abstractmethod
    def expand(self, move: BasicMove) -> None:
        pass

    def ucb_score(self, simulations: int) -> float:
        if self.visits == 0:
            return self.maxScore
        return self.score/self.visits + 1.414 * sqrt(log(simulations)/self.visits)
    
    def update(self, score: float):
        self.visits += 1
        self.score += score
    
    def select_winner(self) -> BasicMove:
        score: float = self.minScore
        best_move: BasicMove  = list(self.children.keys())[0]
        for move in self.children:
            if self.children[move] is None:
                self.expand(move)
            # pyrefly: ignore [missing-attribute]
            if self.children[move].ucb_score(self.visits) > score:
                best_move= move

        return best_move




class MCTSNode(Search):
    real_moves: List[BasicMove]

    def __init__(self, gameState: GameState, moves: List[BasicMove]):
        self.gameState = gameState
        if moves is None or len(moves) <= 1:
            self.real_moves = moves
            self.leaf = True
            self.fullyExpanded = True
            self.children = {}
        else:
            self.children = {move: None for move in moves}
            self.real_moves = [x for x in moves if x.command != MoveEnum.END_TURN]

    def expand(self, move: BasicMove) -> None:
        if move.command == MoveEnum.END_TURN:
            self.children[move] = MCTSNode(self.gameState, [move])
        else:
            game_state, moves = self.gameState.apply_move(move)
            self.children[move] = MCTSNode(game_state, moves)
    
    def base_score(self, heuristic: Callable[[GameState], float])-> float:
        return heuristic(self.gameState)

    def simulate(self, heuristic: Callable[[GameState], float]) -> Tuple[float, int]:
        if self.leaf:
            return heuristic(self.gameState), 0
        moveCount = 0
        moves: list[BasicMove] = self.real_moves
        move: BasicMove = moves[randrange(len(moves))]
        state: GameState = self.gameState
        while (move.command != MoveEnum.END_TURN):
            moveCount += 1
            state, moves = state.apply_move(move)
            moves = [x for x in moves if x.command != MoveEnum.END_TURN]
            if (len(moves) > 0):
                move = moves[randrange(len(moves))]
            else:
                break
        return heuristic(state), moveCount