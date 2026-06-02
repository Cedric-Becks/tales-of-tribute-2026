
from random import randint
from scripts_of_tribute.board import GameState
def max_prestige(gameState: GameState) -> int:
    player = gameState.current_player
    return player.prestige + player.power

def max_patrons(gameState: GameState) -> int:
    player = gameState.current_player.player_id
    patrons = gameState.patron_states.patrons
    return sum([1 for x in patrons if patrons[x] == player])