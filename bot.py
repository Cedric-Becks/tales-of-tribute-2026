from typing import List
from typing import Tuple
from typing import Callable

from scripts_of_tribute.base_ai import BaseAI
from scripts_of_tribute.board import GameState, SeededGameState, EndGameState
from scripts_of_tribute.enums import PatronId, MoveEnum
from scripts_of_tribute.move import BasicMove

from search import Search, MCTSNode
from cmath import inf
from random import randint, randrange, shuffle, choice
from time import time

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
        self.states.append(state)

    def get_features(self):
        features = []
        for i in range(len(self.moves)):
            features.append((self.states[i],self.moves[i],self.states[i+1], self.result))


    def __str__(self):
        return str(len(self.moves)) + ", " + str(len(self.states))
class ISMCTSBot(BaseAI):
    context: Context
    searchTree: type[Search]  = MCTSNode
    heuristic: Callable[[GameState], float]
    turnTimeout = 10.
    turn_timeout = 10.
    moveTimeout = 0.8
    move_timeout = 0.8
    max_depth = 20
    repeats = 20

    turnStart = True
    treeNumber = 1
    tree: Search

    time_used: float = 0.

    def __init__(self, name: str, search_tree: type[Search], heuristic: Callable[[GameState], float]) -> None:
        super().__init__(bot_name=name)
        self.searchTree = search_tree
        self.heuristic = heuristic
        self.seed = randint(0,1000000)
        self.context = Context()

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

    def run(self, node: Search, num_moves: int)-> Tuple[float, int]:
        if (node.leaf or node.visits == 0):
            score, tmp_moves = node.simulate(self.heuristic)
            node.visits += 1
            node.score = score
            return score, num_moves+tmp_moves
        num_moves += 1
        max_score = node.minScore
        # pyrefly: ignore [bad-assignment]
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
            node.expand(max_move, self.heuristic)

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
        self.seed = randint(0,1000000)
        self.context = Context()

    def select_patron(self, available_patrons: List[PatronId]) -> PatronId:
        priority: int = randrange(len(available_patrons))
        return available_patrons[priority]

    def play(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        self.context.add_state(game_state)
        start_time: float = time()
        try:
            move: BasicMove = self._play_saccarina(game_state, possible_moves, remaining_time)
        except Exception:
            import traceback
            traceback.print_exc()
            move =  possible_moves[0]
        self.time_used += time() - start_time
        if move.command == MoveEnum.END_TURN:
            self.time_used = 0
        self.context.add_move(move)
        return move

    def trivial_move(self, possible_moves: List[BasicMove]) -> None | BasicMove:
        if len(possible_moves) == 1:
            return possible_moves[0]
        # TODO: See if the list contains instant-play cards i.e. Writ of Coin and play them
        return None

    def _play_saccarina(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:

        best_move: BasicMove | None = self.trivial_move(possible_moves)
        if best_move is not None:
            return best_move
        shuffle(possible_moves)
        sgs = self.convert_gamestate(game_state)
        self.tree = self.searchTree(sgs, possible_moves)
        
        score: float = -inf
        for move in possible_moves:
            self.tree.expand(move, self.heuristic)
            # pyrefly: ignore [missing-attribute]
            if self.tree.children[move].score > score:
                best_move = move

        if remaining_time < self.turn_timeout:
            # pyrefly: ignore [bad-return]
            return best_move

        move_compute_time = min(self.move_timeout, (self.turn_timeout-self.time_used)/4)
        think_start = time()
        while (time()-think_start < move_compute_time):
            self.run_saccarina(self.tree, 0)
        self.time_used += time()-think_start
        
        score = -inf
        for move in possible_moves:
            # pyrefly: ignore [missing-attribute]
            if self.tree.children[move].score > score:
                best_move = move
        
        # pyrefly: ignore [bad-return]
        return best_move

    def run_saccarina(self, tree: Search, depth: int) -> float:
        # Bandit child?
        
        value = 0.
        if tree.leaf:
            value = tree.score
        elif tree.visits < 1 and depth > self.max_depth:
            value = tree.score
        else:
            move = choice(list(tree.children.keys()))
            if tree.children[move] is None:
                tree.expand(move, self.heuristic)
            # pyrefly: ignore [bad-argument-type]
            value = self.run_saccarina(tree.children[move], depth+1)
        return value

    def _play_bestmcts3(self, game_state: GameState, possible_moves: List[BasicMove], remaining_time: int) -> BasicMove:
        if len(possible_moves) == 1:
            return possible_moves[0]

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
        self.context.reason = end_game_state.reason
        self.context.result = end_game_state.winner
        print("Results " + self.bot_name + ":" + str(self.context))