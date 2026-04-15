# src/runtime.py

from enum import Enum


class RunMode(str, Enum):
    TRAIN = "train"
    BACKTEST = "backtest"
    LIVE = "live"


def set_run_mode(mode: RunMode):
    global CURRENT_RUN_MODE
    CURRENT_RUN_MODE = mode


# Default is LIVE unless explicitly overridden
CURRENT_RUN_MODE: RunMode = RunMode.LIVE
