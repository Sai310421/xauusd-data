# RAGE-DCH-H1

| ID | Claim | Status in this tree |
|----|--------|---------------------|
| H1-A | Main geometric monthly net ≥ 10% | BLOCKED |
| H1-B | Hedge geometric monthly net ≥ 10% | BLOCKED |
| H1-C | Hedge covers main losses | CONDITIONAL (scenario math only) |
| H1-D | Main covers hedge losses | BLOCKED |
| H1-E | Simultaneous losses are controlled | CONDITIONAL (paired method only) |
| H1-F | Combined book compounds positively | BLOCKED |
| H1-G | The above survive OOS, WFO, Monte Carlo | BLOCKED |

"Mean month above 10%" and "every month above 10%" are different statements. Neither has been measured on ticks.

Identity under test:

`P_total = P_A + P_R + P_I - C`

The identity is an accounting definition. It is not a proof that the total is positive.
