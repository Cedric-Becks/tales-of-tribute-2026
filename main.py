import multiprocessing
from search import MCTSNode
from heuristics import greedy_heuristic, max_prestige, max_patrons
from scripts_of_tribute.game import Game
from scripts_of_tribute.server import Server, AIService
from bot import ISMCTSBot

import grpc
from concurrent import futures

from scripts_of_tribute.protos import main_pb2_grpc
from scripts_of_tribute.base_ai import BaseAI

BASE_CLIENT_PORT=50000
BASE_SERVER_PORT=49000
THREADS=6
RUNS=6

def run_grpc_server(
        bot: BaseAI,
        base_client_port: int,
        base_server_port: int,
        debug_prints=False
    ):
    server1 = Server()
    server1.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    server1.add_bot()
    ai_service1 = AIService(bot, server1, base_server_port)
    server1.server.add_insecure_port(f"0.0.0.0:{base_client_port}")
    main_pb2_grpc.add_AIServiceServicer_to_server(ai_service1, server1.server)
    if debug_prints:
        print(f"Bot {bot.bot_name} listening on localhost:{base_client_port}, channel for engine service open on: {base_server_port}")
    server1.server.start()
    server1.server.wait_for_termination()

def main(run_game: bool):
    game = Game()
    if run_game:
        player_one = ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige)
        player_two = ISMCTSBot("patronmaxxer", MCTSNode, max_patrons)
        greedy_bot = ISMCTSBot("greedy", MCTSNode, greedy_heuristic)
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
        processes = []
        for i in range(THREADS*2):
            client_port = BASE_CLIENT_PORT + i
            server_port = BASE_SERVER_PORT + i
            p = multiprocessing.Process(
                target=run_grpc_server,
                name=f"Bot {i+1} listening on {client_port} and serves {server_port}",
                args=(ISMCTSBot("prestigemaxxer", MCTSNode, max_prestige), client_port, server_port),
                #daemon=True
            )
            p.start()
            processes.append(p)
        
        while True:
            pass

if __name__ == "__main__":
    main(False)