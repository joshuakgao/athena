"""Module to test stockfish integration."""

import chess
import chess.engine

from athena.datasets.chessbenchmate.add_mating_data import stockfish_evaluate

ENGINE_PATH = "models/stockfish"
ENGINE_LIMIT = chess.engine.Limit(time=0.05)

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)


def test_stockfish_labels():
    """Test if Stockfish engine is working correctly."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    win_prob, mate_label = stockfish_evaluate(board, move, engine)
    assert 0.0 < win_prob < 1.0, "Win probability should be between 0 and 1"
    assert mate_label in ("#", "-"), "Invalid mate label"

    # Test a forced mate position
    board.set_fen("rnbqkbnr/ppp2ppp/3p4/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 3")
    move = chess.Move.from_uci("a7a6")  # This move leads white to mate in 1
    win_prob, mate_label = stockfish_evaluate(board, move, engine)
    assert win_prob == 0.0, "Win probability should be 1.0 for forced mate"
    assert mate_label == -1, "Mate label should indicate mate in 1"

    board.set_fen("rnbqkbnr/1pp2ppp/p2p4/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 4")
    move = chess.Move.from_uci("h5f7")  # This move is checkmate
    win_prob, mate_label = stockfish_evaluate(board, move, engine)
    assert win_prob == 1.0, "Win probability should be 1.0 for forced mate"
    assert mate_label == "#", "Mate label should indicate mate in 1"

    board.set_fen("1k6/1P5Q/8/7B/8/5K2/8/8 w - - 0 1")
    move = chess.Move.from_uci("h5e8")  # This move leads to mate in 2 for white
    win_prob, mate_label = stockfish_evaluate(board, move, engine)
    assert win_prob == 1.0, "Win probability should be 1.0 for forced mate"
    assert mate_label == 2, "Mate label should indicate mate in 2"

    board.set_fen("1r4k1/5ppp/8/3N4/2Pp4/qP1PnQ2/P3P2P/RK5R b - - 0 25")
    move = chess.Move.from_uci("b8b3")  # This move leads to mate in 2 for black
    win_prob, mate_label = stockfish_evaluate(board, move, engine)
    assert win_prob == 1.0, "Win probability should be 1.0 for forced mate"
    assert mate_label == 2, "Mate label should indicate mate in 2"

    engine.close()
