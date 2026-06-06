from random import randrange
from scripts_of_tribute.enums import MoveEnum
from random import randint
from scripts_of_tribute.board import SeededGameState
from search import MCTSNode
from heuristics import max_prestige, greedy_heuristic
from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.enums import PatronId
from scripts_of_tribute.move import BasicMove
from scripts_of_tribute.board import GameState, EndGameState

from search import Search

from time import time
from typing import List
from typing import Tuple
from typing import Callable

class Context:
    moves: List[BasicMove]
    states: List[GameState]
    result: str = ""
    reason: str = ""
    def __init__(self):
        self.moves = []
        self.states = []

    def refresh(self):
        self.moves = []
        self.states = []
        self.result = ""
        self.reason = ""
        
    def add_move(self, move: BasicMove):
        self.moves.append(move)
    
    def add_state(self, state: GameState):
        if len(self.states) == 0 or len(self.states) == len(self.moves): 
            self.states.append(state)

    def __str__(self):
        return str(len(self.moves)) + ", " + str(len(self.states))
class ISMCTSBot(BaseAI):
    context: Context
    searchTree: type[Search]  = MCTSNode
    heuristic: Callable[[GameState], int] = max_prestige
    turnTimeout = 10.
    moveTimeout = 0.8
    repeats = 20

    turnStart = True
    treeNumber = 1
    tree: Search

    def __init__(self, name: str, search_tree: type[Search], heuristic: Callable[[GameState], int], context: Context) -> None:
        super().__init__(bot_name=name)
        self.searchTree = search_tree
        self.heuristic = heuristic
        self.seed = randint(0,1000000)
        self.context = context

    def convert_gamestate(self, game_state: GameState) -> SeededGameState:
        return SeededGameState(
            game_state.state_id, 
            game_state.patron_states, 
            game_state.tavern_available_cards, 
            game_state.board_state, 
            game_state.upcoming_effects, 
            game_state.start_of_next_turn_effects, 
            game_state.current_player, 
            game_state.enemy_player, 
            game_state.completed_actions, 
            game_state.tavern_cards, 
            # pyrefly: ignore [bad-argument-type]
            game_state.pending_choice, 
            game_state.end_game_state, 
            self.seed, 
            self.seed, 
            game_state._engine_service_stub
            )

    def run(self, node: Search, num_moves: int)-> Tuple[int, int]:
        if (node.leaf or node.visits == 0):
            score, tmp_moves = node.simulate(self.heuristic)
            node.visits += 1
            node.score = score
            return score, num_moves+tmp_moves
        num_moves += 1
        max_score = node.minScore
        max_move: BasicMove = None
        for move in node.children:
            child = node.children[move]
            score: float = node.maxScore
            if child is not None:
                score = child.ucbScore(node.visits)
            if score >= max_score:
                max_move = move
                max_score = score
        
        if node.children[max_move] is None:
            node.expand(max_move)

        # pyrefly: ignore [bad-argument-type]
        result, num_moves = self.run(node.children[max_move], num_moves)

        fully_expanded = True
        for move in node.children:
            child= node.children[move]
            fully_expanded &= (child is not None and child.fullyExpanded)
        node.fullyExpanded = fully_expanded
        node.visits += 1
        node.score = max(node.score, result)
        return result, num_moves

    def pregame_prepare(self) -> None:
            """Optional: Prepare your bot before the game starts."""
            pass

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        priority: int = randrange(len(available_patrons))
        return available_patrons[priority]

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        self.context.add_state(game_state)
        try:
            return self._play(game_state, possible_moves, remaining_time)
        except Exception:
            import traceback
            traceback.print_exc()
            return possible_moves[0]

    def _play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        if len(possible_moves) == 1:
            self.context.add_move(possible_moves[0])
            return possible_moves[0]

        # Always rebuild from the authoritative game_state. Reusing simulated
        # states across play() calls causes KeyNotFoundErrors because the engine
        # discards simulation states once the real game advances.
        self.tree = self.searchTree(self.convert_gamestate(game_state), possible_moves)

        bestMove: BasicMove = possible_moves[randrange(len(possible_moves))]
        runtime = time()
        num_moves = 0
        for i in range(self.repeats):
            _, single_num_moves = self.run(self.tree, num_moves)
            num_moves = max(single_num_moves, num_moves)

        if num_moves == 0:
            return bestMove

        runtime = min((time() - runtime) / num_moves, self.moveTimeout)

        if runtime < 0.1:
            self.context.add_move(bestMove)
            return bestMove

        ref_time = time()
        while not self.tree.fullyExpanded and time() - ref_time < runtime:
            self.run(self.tree, 0)

        max_score = self.tree.minScore
        for move in self.tree.children:
            child = self.tree.children[move]
            if child is not None and child.score >= max_score:
                bestMove = move
                max_score = child.score

        if bestMove.command == MoveEnum.END_TURN:
            self.turnStart = True

        return bestMove

    def game_end(self, end_game_state: EndGameState, final_state: GameState) -> None:
        self.context.add_state(final_state)
        print("Results " + self.bot_name + ":" + str(self.context))