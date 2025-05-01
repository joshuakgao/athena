"""Constants, interfaces, and types."""

from apache_beam import coders

CODERS = {
    "fen": coders.StrUtf8Coder(),
    "move": coders.StrUtf8Coder(),
    "count": coders.BigIntegerCoder(),
    "win_prob": coders.FloatCoder(),
}
CODERS["state_value"] = coders.TupleCoder(
    (
        CODERS["fen"],
        CODERS["win_prob"],
    )
)
CODERS["action_value"] = coders.TupleCoder(
    (
        CODERS["fen"],
        CODERS["move"],
        CODERS["win_prob"],
    )
)
CODERS["behavioral_cloning"] = coders.TupleCoder(
    (
        CODERS["fen"],
        CODERS["move"],
    )
)
