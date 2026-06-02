
from __future__ import annotations
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


class Search(ABC):
    children: dict[BasicMove, Optional[Search]] = {}
    visits: int = 0
    fullyExpanded: bool = False
    maxScore: float = inf
    minScore: float = -inf
    score: int = 0
    leaf: bool = False

    gameState: GameState
    
    @abstractmethod
    def __init__(self, gameState: GameState, moves: List[BasicMove]):
        pass

    @abstractmethod
    def simulate(self, heuristic: Callable[[GameState], int]) -> Tuple[int, int]:
        pass

    @abstractmethod
    def expand(self, move: BasicMove) -> None:
        pass

    def ucbScore(self, simulations: int) -> float:
        if self.visits == 0:
            return self.maxScore
        if self.fullyExpanded:
            return self.minScore
        return self.score + 1.41 * sqrt(log(simulations)/self.visits)



class MCTSNode(Search):
    real_moves: List[BasicMove]

    def __init__(self, gameState: GameState, moves: List[BasicMove]):
        self.gameState = gameState
        if moves is None or len(moves) <= 1:
            self.real_moves = moves
            self.leaf = True
            self.fullyExpanded = True
        else:
            self.children = {move: None for move in moves}
            self.real_moves = [x for x in moves if x.command != MoveEnum.END_TURN]

    def expand(self, move: BasicMove) -> None:
        if move.command == MoveEnum.END_TURN:
            self.children[move] = MCTSNode(self.gameState, [move])
        else:
            game_state, moves = self.gameState.apply_move(move)
            self.children[move] = MCTSNode(game_state, moves)

    def simulate(self, heuristic: Callable[[GameState], int]) -> Tuple[int, int]:
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