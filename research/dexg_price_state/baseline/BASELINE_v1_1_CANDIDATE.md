# Baseline v1.1-candidate

Frozen core remains Entry_TP_Module_v1.0.
This file folds public DexG + Nercia techniques as NAMED MODULES.
Default BOT path = CORE only. Overlays off until Raw Tick A/B.
G75 / Hydra66 still excluded.

## Layer 0 CORE (always on) — DAIGO FAST

ARMED iff P_move>0.65 AND C_D>0.60 AND Score(D*)>0.60
ENTRY after V/A same-sign, S_t=1, MTF not opposite-dominant
FAST clock: T_move>1.5s => EXIT, no silent promote to MEDIUM
ADD: same-price only, no averaging down, next reach>=0.70
EXIT priority: hard invalidation > dest flip > structure break > ATR
FAST default flatten at first dest or hard invalidation

## Layer 1 DEX_G (on for v1.1-candidate FAST)

D1 PrevBarPrep: previous M1 candle body+wick is the prep feature for P_move
D2 ModeLock: ticket.mode in {FAST,M1,SLEEP}; cannot change without flatten
D3 MinakiriSamePrice: add only inside entry-band (symbol opt), never distance-martingale
D4 SkipOverChase: if move already exceeded d_min before ARMED, skip
D5 FlatThenReverse: no flip while open; flatten first

## Layer 2 NERCIA (overlay, default OFF on FAST tickets)

Enable only when ticket.mode!=FAST or flag nercia_overlay=true
N1 SessionNY: optional score bonus, not a hard AND on FAST
N2 Stack: Sweep(session high/low) + MSS + (OB or BPR) + (FVG or OTE) + IDM as Score(D*) components
N3 DualFVG: if FVG_A dies keep FVG_B; do not zero destination set
N4 SplitTarget: inner 50% vs external-range liquidity as dest_1 / dest_2
N5 OutlookLog: TF, side, price, SL, TP, RR, reasons[]
N6 SlShortBias: structure stop, not fixed 40 pips on FAST (40/60-100 is NERCIA_ICT lane only)

## Wiring into Score(D*)

Do not AND all Nercia items onto ARMED.
Map N2 labels onto existing q_liq, q_fvg, q_mtf, q_tick.
PrevBarPrep feeds P_move, not a fourth ARMED gate.

## A/B required before freeze as v1.1

A = CORE
B = CORE+DEX_G
C = CORE+DEX_G+NERCIA overlay on M1 only
Raw Tick + explicit spread. No D1 OHLC path.
Promote a module only if PF/RF hold and DD does not worsen vs A.
