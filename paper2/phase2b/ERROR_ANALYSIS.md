# Phase 2B failure analysis

## Category definitions

Every saved outcome has boolean diagnostic flags. A single primary category is
assigned in this precedence order: missed recovery, regression avoided,
unnecessary continuation, invalid-extraction uncertainty, unstable trajectory,
stable wrong trajectory, safe early stop, other. “Unnecessary continuation”
requires reaching N=16 after the final plurality had already stabilized.
“Unstable” means at least two plurality switches from t=2 to 16; “stable wrong”
means a wrong N=16 outcome with no such switch. Invalid uncertainty means at
least one invalid extraction before stopping and either delayed stopping or a
nonzero invalid rate at the stop.

| Primary category | Nested margin | Primary hybrid |
|---|---:|---:|
| Safe early stop | 349 | 347 |
| Premature stop with missed recovery | 11 | 9 |
| Unnecessary continuation | 56 | 68 |
| Regression avoided | 9 | 6 |
| Unstable trajectory | 3 | 2 |
| Stable wrong trajectory | 23 | 22 |
| Invalid-extraction-driven uncertainty | 49 | 46 |

The flags remain available in `results/error_cases.csv` for analyses where
overlap is preferable to the primary-category partition.

## Successful interventions and new failures

The hybrid successfully overrides margin on two recoverable questions:

| Question | Margin stop/outcome | Hybrid stop/outcome | N=16 outcome |
|---:|---|---|---|
| 234 | t=2, wrong | t=8, correct | correct |
| 380 | t=2, wrong | t=16, correct | correct |

These are the requested cases where margin would stop, the hybrid continues,
and later computation recovers correctness. They are genuine paired gains.

However, questions 412, 460, and 494 move in the opposite direction. Margin
stops correctly at t=4, t=4, and t=2 respectively, while the risk check
continues each to N=16 and returns a wrong final plurality. Thus a future-change
risk detector can prevent premature stopping yet sacrifice regression
avoidance. This three-versus-two exchange explains the net -0.2-point result.

The primary hybrid stops at t=2 for 296 questions (59.2%), at t=3 for 70
(14.0%), at t=4 for 31 (6.2%), and reaches N=16 for 71 (14.2%). Margin stops at
t=2 for 386 (77.2%) and reaches N=16 for 58 (11.6%). The hybrid's additional
continuation is therefore broad relative to its two successful interventions.

