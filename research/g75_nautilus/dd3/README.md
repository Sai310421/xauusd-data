# G75 Adaptive Distance DD3 v1.0

Adaptive Distance版のA/B実測Max DD **1.7700%** を基準に、DD予算 **3.00%** へリスクを再配分する保存版です。

## Initial DD3 calibration

```text
risk_multiplier = 3.00 / 1.7700
                = 1.6949152542
```

Base lot 0.05の場合:

```text
scaled_lot = 0.05 × 1.6949152542
           = 0.0847457627
```

## Adaptive geometry

```text
scale_t = clip(
    EMA60(TrueRange_t) / Median1440(TrueRange up to t),
    0.50,
    2.00
)

Trigger_t  = 0.12  × scale_cycle
Add_t      = 0.025 × scale_cycle
Reversal_t = 0.20  × scale_cycle
```

Cycle開始時のscaleをBasket Exitまで固定します。

## G75 state machine

```text
PRICE FOLLOW
→ TRIGGER
→ ENTRY
→ PROFIT-DIRECTION ADD
→ RUNNING EXTREME
→ REVERSAL FROM EXTREME
→ BASKET EXIT
→ IMMEDIATE RE-ARM
```

## Important

`risk_multiplier=1.694915` はDD 1.7700%から3.00%へ合わせる一次近似です。DDは非線形になり得るため、Nautilus再BT後に、

```text
k_next = k_current × 3.00 / DD_measured
```

で再校正してください。

EMA9はこのv1.0ではEntry/Add/Exit条件に使用していません。既存G75 EDGEを壊さないため、別A/B層として扱います。
