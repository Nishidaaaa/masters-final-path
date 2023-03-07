from enum import Enum
PHASES = ["train", "val", "test"]


class Phase(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "TEST"
