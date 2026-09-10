# DAIGO second-scalp extract

Source: DAIGO Entry_TP_Module_v1.0 (Drive 11Uc9LHBWWZOtQaFuEzNF8fF0XMB74g8n)
Mode extracted: FAST only. MEDIUM/SLOW and sleep-hold are out of this extract.
Signal source for DexG: Python ML Builder feature_order only. No G75. No Hydra66 mix.
Status: SPEC EXTRACT. Not adopted EDGE.

## What second-scalp is in this spec

A trade whose first-passage time to d_min is <= 1.5 seconds.

T_move = inf{ tau > 0 : |P_{t+tau}-P_t| >= d_min }
FAST if T_move <= 1.5s
MEDIUM 1.5-5s and SLOW >5s are not this extract.

This matches DexG observation language: 0.5-1.5s fill-and-go vs 3-5s sticky tape.

## Gate before entry

ARMED only if all three hold:
- P_move > 0.65   (a move of size d_min is likely inside the horizon)
- C_D > 0.60      (|P_up - P_down|)
- Score(D*) > 0.60 (there is a destination worth running toward)

No destination => no second-scalp. Spike with no target is skip.

## Tick pressure (the second-scalp engine)

TI = (N_up - N_down) / (N_up + N_down + eps)
V  = (P_t - P_{t-dt}) / dt
A  = (V_t - V_{t-dt}) / dt
Q  = w1*TI + w2*V + w3*A - w4*Spread

LONG entry extra: V > theta_V and A > theta_A and structure valid and MTF not BearishDominant
SHORT entry extra: V < -theta_V and A < -theta_A and structure valid and MTF not BullishDominant

theta_V / theta_A are symbol-optimized, not frozen numbers.

## Hold vs flatten on the second clock

HOLD if structure still valid AND direction prob still above hold theta AND D* still valid.
PriceReversal != HypothesisInvalidation.
But FAST mode must reassess inside 1.5s. If T_move forecast was FAST and price has not traveled d_min by 1.5s, the timing hypothesis is dead -> EXIT. Do not silently promote FAST into MEDIUM inside the same ticket.

ADD only same-direction, velocity still valid, next destination reach prob >= 0.70. No averaging down.

## Exit priority (do not invert)

1. Hard invalidation (direction prob or structure confidence collapse)
2. Destination flip
3. Structure break (micro swing against the position)
4. Dynamic ATR only after profit threshold or first destination consumed

Partial split in the parent spec is 50/30/20. For FAST second-scalp the practical default is full flatten at first destination or hard invalidation. Runner 20% belongs to MEDIUM/SLOW, not this extract.

## State machine for FAST only

WAIT -> ARMED -> ENTRY -> HOLD <-> ADD -> TARGET_TAKE -> EXIT

No SLEEP_HOLD in this extract.

## What this extract refuses

- Mixing FAST tickets with sleep-FX 0.2 holds in one KPI table
- Using G75 / Hydra66 as X_t
- Averaging down
- Entering because price already jumped with no prep (P_move was not armed before the jump)
- Calling ATR trail the entry thesis
