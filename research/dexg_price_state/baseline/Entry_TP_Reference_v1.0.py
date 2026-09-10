from dataclasses import dataclass
from enum import Enum, auto

class State(Enum):
    WAIT = auto()
    ARMED = auto()
    ENTRY = auto()
    HOLD = auto()
    ADD = auto()
    TARGET_TAKE = auto()
    REASSESS = auto()
    PARTIAL = auto()
    EXIT = auto()

@dataclass
class EntrySignal:
    side: str
    move_prob: float
    direction_prob: float
    velocity: float
    acceleration: float
    destination_score: float
    destination_reach_prob: float
    structure_confidence: float
    mtf_state: str

def should_enter(x: EntrySignal, th_move=0.65, th_dir=0.60, th_dest=0.60):
    return (
        x.side in ("LONG", "SHORT")
        and x.move_prob >= th_move
        and x.direction_prob >= th_dir
        and x.destination_score >= th_dest
        and x.structure_confidence > 0.0
    )

def should_add(direction_valid: bool, velocity_valid: bool, structure_valid: bool,
               next_target_reach_prob: float, th_add=0.70):
    return all([
        direction_valid,
        velocity_valid,
        structure_valid,
        next_target_reach_prob >= th_add,
    ])

def dynamic_atr_k(regime: str):
    if regime == "STRONG_DISPLACEMENT":
        return 2.25
    if regime == "POST_TARGET_DECELERATION":
        return 0.95
    return 1.55
