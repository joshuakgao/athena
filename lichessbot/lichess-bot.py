"""Starting point for lichess-bot."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lichessbot.lib.lichess_bot import start_program

if __name__ == "__main__":
    start_program()
