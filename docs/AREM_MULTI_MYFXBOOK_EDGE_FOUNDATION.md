# AREM Multi-Myfxbook EDGE Foundation v0.1

## Goal
Build a reusable EDGE-learning system from many Myfxbook / statement strategies plus mathematical EDGE candidates.

The goal is not to average many EAs into one weak model. The goal is to preserve each strategy's behavior, identify the market states where each strategy has positive edge, extract reusable representations, and route the current regime to the best compatible expert.

## Core pipeline

Myfxbook/Statement_i
→ Behavioral Clone_i
→ EDGE Embedding_i
→ EDGE Family / Regime
→ Mixture-of-Experts Router
→ Master Policy
→ Deterministic Risk Supervisor

Mathematical EDGE enters as features, priors, constraints, or specialist experts rather than as unverified profit labels.

## Required behavior models
Each strategy source should produce independent models for:
- ENTRY / NO_ENTRY
- BUY / SELL
- HOLD / CLOSE

TRAIL is treated as an enhanced exit action unless directly observable from the source data.

## Why preserve strategy identity
A trend-following strategy and a mean-reversion strategy can both be profitable but issue opposite actions in the same state definition. A single averaged classifier can destroy both edges. Therefore every training row retains:
- strategy_id
- edge_family
- regime label or inferred regime
- source confidence / quality

The master model uses routing rather than naive averaging.

## Mathematical EDGE layer
Candidate families include:
- First-passage / hitting-time structure
- Optimal stopping
- Kelly / fractional Kelly
- HJB / stochastic control
- Singular / reflected control
- Drawdown-constrained control
- Ruin probability
- Regime switching
- EVT / tail risk
- Robust / distributionally robust control

Mathematics can contribute in four ways:
1. Feature: e.g. distance-to-boundary, first-passage probability, drawdown state.
2. Constraint: e.g. ruin probability or DD bound.
3. Label modifier: e.g. classify an original action as economically dominated under cost-aware optimal stopping.
4. Specialist expert: e.g. an Exit or Risk expert independent of any one Myfxbook source.

## Training hierarchy
### Level 1 — Individual Clone
Reproduce each source before mixing it with other sources.

### Level 2 — EDGE Extraction
Compare wins/losses, ENTRY/NO_ENTRY, MFE/MAE and OOS states to identify positive/negative behavior regions.

### Level 3 — EDGE Family Embedding
Represent strategies and market states in a shared feature/embedding space while retaining source identity.

### Level 4 — Regime Router
Select or weight experts by current regime. Opposing experts are not allowed to cancel blindly.

### Level 5 — Master Policy
Outputs:
- NO_ENTRY / ENTRY
- BUY / SELL
- HOLD / CLOSE

The deterministic Risk Supervisor can veto any AI action.

## OOS design
Three different OOS tests are required:
1. Time OOS — unseen dates for a known strategy.
2. Strategy OOS — hold out one complete strategy source and test whether shared EDGE generalizes.
3. Regime OOS — hold out a volatility/trend/range regime where possible.

This is important: a model that only memorizes the source EAs is not a foundation model.

## Promotion
A foundation candidate is not promoted because aggregate accuracy is high. It must show positive economic edge after costs, class-level Direction quality including SELL, robust NO_ENTRY control, regime stability, and Reality BT.

Promotion path:
Training → Walk Forward → Time OOS → Strategy OOS → Regime OOS → Reality BT → Shadow → Demo → Live

## First source
Scalpers Circle becomes Source #001 and establishes the first behavioral-clone baseline. Additional Myfxbook sources should be ingested through the same schema rather than custom code.
