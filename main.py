from search import MCTSNode
from heuristics import greedy_heuristic, max_prestige, max_patrons
from scripts_of_tribute.game import Game
from bot import ISMCTSBot, Context

BASE_CLIENT_PORT=50000
BASE_SERVER_PORT=49000
THREADS=6
RUNS=6

def main(run_game: bool):
    game = Game()
    context = Context()
    if run_game:
        player_one = ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige, context)
        player_two = ISMCTSBot("patronmaxxer", MCTSNode, max_patrons, context)
        greedy_bot = ISMCTSBot("greedy", MCTSNode, greedy_heuristic, context)
        game.register_bot(player_one)
        game.register_bot(player_two)
        game.register_bot(greedy_bot)

        game.run(
        "MCTSBot",
        "greedy",
        start_game_runner=True,
        runs=RUNS,
        threads=THREADS,
        )
    else:
        game._run_bot_instances(
            ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige, context),
            ISMCTSBot("patronmaxxer", MCTSNode, max_patrons, context),
            num_threads=1,
            base_client_port=BASE_CLIENT_PORT,
            base_server_port=BASE_SERVER_PORT)
        while True:
            pass

if __name__ == "__main__":
    main(True)