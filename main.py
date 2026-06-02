from ScriptsOfTribute.game import Game

def main():
    ISMCTSBot = ISMCTSBot(bot_name="ISMCTSBot")
    
    game = Game()
    game.register_bot(ISMCTSBot)
    
    game.run(
        "ISMCTSBot",
        start_game_runner=True,
        runs=10,
        threads=1,
    )

if __name__ == "__main__":
    main()