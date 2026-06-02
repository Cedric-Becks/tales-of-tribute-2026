from search import MCTSNode
from heuristics import max_prestige, max_patrons
from scripts_of_tribute.game import Game
from bot import ISMCTSBot

def main():
    player_one = ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige)
    player_two = ISMCTSBot("patronmaxxer", MCTSNode, max_patrons)
    game = Game()
    game.register_bot(player_one)
    game.register_bot(player_two)
    
    game.run(
        "prestigemaxxer",
        "patronmaxxer",
        start_game_runner=True,
        runs=10,
        threads=1,
    )

if __name__ == "__main__":
    main()