import os
import platform

from stockfish import Stockfish as Stockfishy

from utils.chess_utils import is_fen_valid


class Stockfish:
    def __init__(self):
        # Get OS. OS determines the path where stockfish is stored
        os_name = platform.system()
        if os_name == "Linux":
            stockfish_path = "/usr/games/stockfish"
        elif os_name == "Darwin":
            stockfish_path = "/opt/homebrew/bin/stockfish"
        else:
            raise f"This is an unknown or unsupported system: {os_name}"

        self.stockfish = Stockfishy(
            stockfish_path,
            depth=20,
            # These parameters are set to maximize Stockfish speed
            parameters={"Threads": os.cpu_count()},
        )

    def get_fen_position(self):
        return self.stockfish.get_fen_position()

    def set_fen_position(self, fen):
        assert is_fen_valid(fen)
        self.stockfish.set_fen_position(fen)

    def get_board_visual(self):
        return self.stockfish.get_board_visual()

    def get_best_move(self, fen, time_constraint=None):
        self.set_fen_position(fen)

        if time_constraint:
            return self.stockfish.get_best_move_time(time_constraint)
        else:
            return self.stockfish.get_best_move()

    def get_top_moves(self, fen, num_top_moves=5):
        self.set_fen_position(fen)
        return self.stockfish.get_top_moves(num_top_moves)


stockfish = Stockfish()

if __name__ == "__main__":
    print(stockfish.get_board_visual())
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    stockfish.set_fen_position(fen)
    print(stockfish.get_board_visual())
    print(stockfish.get_best_move(fen))
    print(str(stockfish.get_top_moves(fen, 5)))
