from search import MCTSNode
from heuristics import max_prestige, max_patrons
from scripts_of_tribute.game import Game
from bot import ISMCTSBot

BASE_CLIENT_PORT=50000
BASE_SERVER_PORT=49000
THREADS=1
RUNS=10

def main(run_game: bool):
    game = Game()
    if run_game:
        player_one = ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige)
        player_two = ISMCTSBot("patronmaxxer", MCTSNode, max_patrons)
        game.register_bot(player_one)
        game.register_bot(player_two)
    
        game.run(
            "prestigemaxxer",
            "patronmaxxer",
            start_game_runner=True,
            runs=RUNS,
            threads=THREADS,
        )
    else:
        game._run_bot_instances(
            ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige),
            ISMCTSBot("patronmaxxer", MCTSNode, max_patrons),
            num_threads=1,
            base_client_port=BASE_CLIENT_PORT,
            base_server_port=BASE_SERVER_PORT)
        while True:
            pass

if __name__ == "__main__":
    main(False)