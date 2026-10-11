"""Small numeric tennis datasets. Optional runtime imports stay in stage modules."""

GENERATOR_VERSION = "1.0"

KINDS = ("toss", "serve", "return", "volley", "lob", "slice", "out")
EVENT_CODES = {"bounce": 1, "hit": 2, "net_crossing": 3, "out": 4}
