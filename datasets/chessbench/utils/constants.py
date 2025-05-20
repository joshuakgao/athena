from apache_beam import coders


class IntOrCharCoder(coders.Coder):
    def encode(self, value):
        if isinstance(value, int):
            return b"\x00" + coders.VarIntCoder().encode(value)
        elif isinstance(value, str) and len(value) == 1:
            return b"\x01" + value.encode("utf-8")
        else:
            raise ValueError("Value must be an int or a single character string.")

    def decode(self, encoded):
        if encoded[0:1] == b"\x00":
            return coders.VarIntCoder().decode(encoded[1:])
        elif encoded[0:1] == b"\x01":
            return encoded[1:].decode("utf-8")
        else:
            raise ValueError("Invalid encoding prefix.")

    def is_deterministic(self):
        return True


CODERS = {
    "fen": coders.StrUtf8Coder(),
    "move": coders.StrUtf8Coder(),
    "count": coders.BigIntegerCoder(),
    "win_prob": coders.FloatCoder(),
    "mate": IntOrCharCoder(),  # Use the custom coder here
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
CODERS["action_value_with_mate"] = coders.TupleCoder(
    (
        CODERS["fen"],
        CODERS["move"],
        CODERS["win_prob"],
        CODERS["mate"],
    )
)
CODERS["behavioral_cloning"] = coders.TupleCoder(
    (
        CODERS["fen"],
        CODERS["move"],
    )
)
